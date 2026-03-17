use anyhow::Result;
use cubecl::prelude::*;
use cubecl::{
    Runtime,
    server::Handle,
    wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async},
};
use cubecl_runtime::client::ComputeClient;
use std::num::NonZeroU64;
use std::time::Instant;

use crate::backend::ImageProcessor;
use crate::nodes::gpu_warp::perspective_warp_kernel;

/// Per-call timing summary returned by [`ImageProcessor::warp_perspective`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WarpRunTimings {
    /// Host-to-device upload duration in milliseconds.
    pub h2d_ms: f64,
    /// Kernel execution duration in milliseconds.
    pub kernel_ms: f64,
    /// Device-to-host readback duration in milliseconds.
    pub d2h_ms: f64,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Persistent CubeCL/WGPU backend for perspective warp.
///
/// The backend initializes GPU resources once and reuses them across calls.
pub struct CubeCLBackend {
    client: ComputeClient<WgpuRuntime>,
    queue: wgpu::Queue,
    width: usize,
    height: usize,
    pixel_count: usize,
    expected_gpu_bytes: usize,
    input_copy_size_aligned: u64,
    input_handle: Handle,
    output_handle: Handle,
    matrix_handle: Handle,
    input_buffer: wgpu::Buffer,
    input_offset: u64,
    input_size: u64,
    matrix_buffer: wgpu::Buffer,
    matrix_offset: u64,
    matrix_size: u64,
    img_shape: [usize; 2],
    img_strides: [usize; 2],
    mat_shape: [usize; 1],
    mat_strides: [usize; 1],
    cube_dim: CubeDim,
    cube_count: CubeCount,
}

impl CubeCLBackend {
    /// Create a backend instance for fixed image dimensions and an initial homography matrix.
    pub async fn new(width: usize, height: usize, matrix: [f32; 9]) -> Result<Self> {
        let device = WgpuDevice::default();
        let setup = init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
        let client = WgpuRuntime::client(&device);
        let queue = setup.queue;

        let pixel_count = width * height;
        let expected_gpu_bytes = pixel_count * std::mem::size_of::<u32>();

        let input_handle = client.empty(expected_gpu_bytes);
        let output_handle = client.empty(expected_gpu_bytes);
        let matrix_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            f32::as_bytes(&matrix).to_vec(),
        ));

        // HACK: Bypassing CubeCL abstraction to access raw WGPU buffers for persistent,
        // zero-allocation host writes. Replace this path when CubeCL exposes a stable
        // persistent-buffer update API that avoids direct backend resource extraction.
        let input_resource = client.get_resource(input_handle.clone().binding());
        let input_buffer = input_resource.resource().buffer.clone();
        let input_offset = input_resource.resource().offset;
        let input_size = input_resource.resource().size;

        let matrix_resource = client.get_resource(matrix_handle.clone().binding());
        let matrix_buffer = matrix_resource.resource().buffer.clone();
        let matrix_offset = matrix_resource.resource().offset;
        let matrix_size = matrix_resource.resource().size;

        let input_copy_size_aligned = (expected_gpu_bytes as u64).next_multiple_of(4u64);

        let img_shape = [height, width];
        let img_strides = [width, 1usize];
        let mat_shape = [9usize];
        let mat_strides = [1usize];
        let cube_dim = CubeDim::new_2d(16, 16);
        let cube_count = CubeCount::new_2d(
            (width as u32).div_ceil(cube_dim.x),
            (height as u32).div_ceil(cube_dim.y),
        );

        Ok(Self {
            client,
            queue,
            width,
            height,
            pixel_count,
            expected_gpu_bytes,
            input_copy_size_aligned,
            input_handle,
            output_handle,
            matrix_handle,
            input_buffer,
            input_offset,
            input_size,
            matrix_buffer,
            matrix_offset,
            matrix_size,
            img_shape,
            img_strides,
            mat_shape,
            mat_strides,
            cube_dim,
            cube_count,
        })
    }

    /// Update the homography matrix buffer used by subsequent kernel launches.
    fn update_matrix(&self, matrix: &[f32; 9]) -> Result<()> {
        let matrix_bytes = bytemuck::cast_slice::<f32, u8>(matrix);
        anyhow::ensure!(
            self.matrix_size >= matrix_bytes.len() as u64,
            "matrix buffer too small: got {} bytes, need {} bytes",
            self.matrix_size,
            matrix_bytes.len()
        );

        self.queue
            .write_buffer(&self.matrix_buffer, self.matrix_offset, matrix_bytes);
        Ok(())
    }
}

impl ImageProcessor for CubeCLBackend {
    async fn warp_perspective(
        &self,
        input_packed_rgb: &[u32],
        output_packed_rgb: &mut [u32],
        matrix: &[f32; 9],
    ) -> Result<WarpRunTimings> {
        anyhow::ensure!(
            input_packed_rgb.len() == self.pixel_count,
            "input pixel count mismatch: got {}, expected {}",
            input_packed_rgb.len(),
            self.pixel_count
        );
        anyhow::ensure!(
            output_packed_rgb.len() == self.pixel_count,
            "output pixel count mismatch: got {}, expected {}",
            output_packed_rgb.len(),
            self.pixel_count
        );

        anyhow::ensure!(
            self.width > 0 && self.height > 0,
            "image dimensions must be non-zero"
        );

        // Update per-call homography matrix.
        self.update_matrix(matrix)?;

        // Upload packed image to device buffer.
        let t_h2d = Instant::now();
        anyhow::ensure!(
            self.input_size >= self.input_copy_size_aligned,
            "input buffer too small: got {} bytes, need {} bytes",
            self.input_size,
            self.input_copy_size_aligned
        );

        let input_bytes = bytemuck::cast_slice::<u32, u8>(input_packed_rgb);
        if self.input_copy_size_aligned == self.expected_gpu_bytes as u64 {
            self.queue
                .write_buffer(&self.input_buffer, self.input_offset, input_bytes);
        } else {
            let Some(mut view) = self.queue.write_buffer_with(
                &self.input_buffer,
                self.input_offset,
                NonZeroU64::new(self.input_copy_size_aligned)
                    .expect("input_copy_size_aligned must be non-zero"),
            ) else {
                anyhow::bail!(
                    "failed to stage-write {} bytes into input buffer",
                    self.input_copy_size_aligned
                );
            };
            view[0..input_bytes.len()].copy_from_slice(input_bytes);
            view[input_bytes.len()..].fill(0u8);
        }

        self.queue.submit([]);
        self.client.sync().await?;
        let h2d_ms = elapsed_ms(t_h2d);

        // Launch the perspective warp kernel.
        let t_kernel = Instant::now();
        unsafe {
            perspective_warp_kernel::launch::<WgpuRuntime>(
                &self.client,
                self.cube_count.clone(),
                self.cube_dim,
                TensorArg::from_raw_parts::<u32>(
                    &self.input_handle,
                    &self.img_strides,
                    &self.img_shape,
                    1,
                ),
                TensorArg::from_raw_parts::<u32>(
                    &self.output_handle,
                    &self.img_strides,
                    &self.img_shape,
                    1,
                ),
                TensorArg::from_raw_parts::<f32>(
                    &self.matrix_handle,
                    &self.mat_strides,
                    &self.mat_shape,
                    1,
                ),
            )?;
        }

        self.client.sync().await?;
        let kernel_ms = elapsed_ms(t_kernel);

        // Read output buffer back to host memory.
        let t_d2h = Instant::now();
        // NOTE: CubeCL read_async may materialize host-side byte chunks internally.
        // The hot path below avoids additional Vec allocations in this crate.
        let mut output_chunks = self
            .client
            .read_async(vec![self.output_handle.clone()])
            .await?;
        let output_bytes = output_chunks.remove(0);
        let d2h_ms = elapsed_ms(t_d2h);

        let output_raw: &[u8] = output_bytes.as_ref();
        let output_packed = bytemuck::try_cast_slice::<u8, u32>(output_raw)
            .map_err(|_| anyhow::anyhow!("output buffer has invalid alignment/length"))?;

        anyhow::ensure!(
            output_packed.len() == self.pixel_count,
            "output pixel count mismatch: got {}, expected {}",
            output_packed.len(),
            self.pixel_count
        );

        output_packed_rgb.copy_from_slice(output_packed);

        Ok(WarpRunTimings {
            h2d_ms,
            kernel_ms,
            d2h_ms,
        })
    }
}
