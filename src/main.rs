//! Zenoh node that applies a GPU homography warp (CubeCL/WGPU) to incoming frames.
//!
//! # Topics
//! - Subscribes: `camera/front/frames`
//! - Publishes: `gpu-bev-node/frames/birdseye` (JPEG by default; raw RGB optional)
//!
//! # Input payload
//! The subscriber expects either:
//! - JPEG bytes (detected via SOI marker), decoded to RGB8, or
//! - PNG bytes (detected via signature), decoded to RGB8, or
//! - Raw RGB8 bytes with length `CAM_WIDTH * CAM_HEIGHT * 3`.
//!
//! # IPM configuration
//! Set camera intrinsics and pose via environment variables:
//! `CAM_FX`, `CAM_FY`, `CAM_CX`, `CAM_CY` (or `CAM_FOV_X_DEG`, `CAM_FOV_Y_DEG`), and
//! `CAM_HEIGHT_M`, `CAM_ROLL_DEG`, `CAM_PITCH_DEG`, `CAM_YAW_DEG`.
//! The BEV region of interest is controlled by: `BEV_X_MIN_M`, `BEV_X_MAX_M`,
//! `BEV_Y_MIN_M`, `BEV_Y_MAX_M`.
//!
//! # Output encoding
//! - `BEV_PUBLISH_JPEG` (default: `true`)
//! - `BEV_JPEG_QUALITY` (default: `80`)

mod math;
mod nodes;

use anyhow::Result;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::time::Instant;
use tokio::time::{Duration, sleep};
use zenoh::Config;

use cubecl::prelude::*;
use nodes::gpu_warp::perspective_warp_kernel;

use cubecl::{
    Runtime,
    wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async},
};
use kornia_image::ImageSize;
use kornia_image::allocator::CpuAllocator;
use kornia_image::color_spaces::Rgb8;
use kornia_io::{jpeg, png};

fn looks_like_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

fn looks_like_png(bytes: &[u8]) -> bool {
    bytes.len() >= 8
        && bytes[0] == 0x89
        && bytes[1] == b'P'
        && bytes[2] == b'N'
        && bytes[3] == b'G'
        && bytes[4] == 0x0D
        && bytes[5] == 0x0A
        && bytes[6] == 0x1A
        && bytes[7] == 0x0A
}

fn parse_env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => true,
            "0" | "false" | "no" | "n" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn parse_env_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(default)
        .min(100)
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchStats {
    count: u64,
    sum_ms: f64,
    max_ms: f64,
}

impl BenchStats {
    fn record(&mut self, ms: f64) {
        self.count += 1;
        self.sum_ms += ms;
        self.max_ms = self.max_ms.max(ms);
    }

    fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ms / self.count as f64
        }
    }
}

#[derive(Debug)]
struct PipelineBench {
    enabled: bool,
    warmup_frames: u64,
    report_every: u64,
    frame_idx: u64,
    window_frames: u64,
    jpeg_decode: BenchStats,
    pack: BenchStats,
    h2d: BenchStats,
    kernel: BenchStats,
    d2h: BenchStats,
    unpack: BenchStats,
    jpeg_encode: BenchStats,
    total: BenchStats,
}

struct FrameTimings {
    jpeg_decode_ms: f64,
    pack_ms: f64,
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    unpack_ms: f64,
    jpeg_encode_ms: f64,
    total_ms: f64,
}

impl PipelineBench {
    fn from_env() -> Self {
        Self {
            enabled: parse_env_bool("BEV_BENCH", false),
            warmup_frames: parse_env_u64("BEV_BENCH_WARMUP", 10),
            report_every: parse_env_u64("BEV_BENCH_EVERY", 60).max(1),
            frame_idx: 0,
            window_frames: 0,
            jpeg_decode: BenchStats::default(),
            pack: BenchStats::default(),
            h2d: BenchStats::default(),
            kernel: BenchStats::default(),
            d2h: BenchStats::default(),
            unpack: BenchStats::default(),
            jpeg_encode: BenchStats::default(),
            total: BenchStats::default(),
        }
    }

    fn record(&mut self, timings: FrameTimings) {
        self.frame_idx += 1;
        if !self.enabled || self.frame_idx <= self.warmup_frames {
            return;
        }

        self.window_frames += 1;
        self.jpeg_decode.record(timings.jpeg_decode_ms);
        self.pack.record(timings.pack_ms);
        self.h2d.record(timings.h2d_ms);
        self.kernel.record(timings.kernel_ms);
        self.d2h.record(timings.d2h_ms);
        self.unpack.record(timings.unpack_ms);
        self.jpeg_encode.record(timings.jpeg_encode_ms);
        self.total.record(timings.total_ms);

        if self.window_frames.is_multiple_of(self.report_every) {
            eprintln!(
                "[GPU-BEV][BENCH] n={} avg(ms): decode={:.3} pack={:.3} h2d={:.3} kernel={:.3} d2h={:.3} unpack={:.3} encode={:.3} total={:.3} | max(ms): decode={:.3} pack={:.3} h2d={:.3} kernel={:.3} d2h={:.3} unpack={:.3} encode={:.3} total={:.3}",
                self.window_frames,
                self.jpeg_decode.avg_ms(),
                self.pack.avg_ms(),
                self.h2d.avg_ms(),
                self.kernel.avg_ms(),
                self.d2h.avg_ms(),
                self.unpack.avg_ms(),
                self.jpeg_encode.avg_ms(),
                self.total.avg_ms(),
                self.jpeg_decode.max_ms,
                self.pack.max_ms,
                self.h2d.max_ms,
                self.kernel.max_ms,
                self.d2h.max_ms,
                self.unpack.max_ms,
                self.jpeg_encode.max_ms,
                self.total.max_ms,
            );

            self.window_frames = 0;
            self.jpeg_decode = BenchStats::default();
            self.pack = BenchStats::default();
            self.h2d = BenchStats::default();
            self.kernel = BenchStats::default();
            self.d2h = BenchStats::default();
            self.unpack = BenchStats::default();
            self.jpeg_encode = BenchStats::default();
            self.total = BenchStats::default();
        }
    }
}

fn pack_rgb24_to_u32(rgb: &[u8], out: &mut [u32]) {
    debug_assert_eq!(rgb.len(), out.len() * 3);
    for (i, px) in out.iter_mut().enumerate() {
        let base = i * 3;
        let r = rgb[base] as u32;
        let g = rgb[base + 1] as u32;
        let b = rgb[base + 2] as u32;
        *px = (r << 16) | (g << 8) | b;
    }
}

fn unpack_u32_to_rgb24(pixels: &[u32], out_rgb: &mut [u8]) {
    debug_assert_eq!(out_rgb.len(), pixels.len() * 3);
    for (i, &px) in pixels.iter().enumerate() {
        let base = i * 3;
        out_rgb[base] = ((px >> 16) & 0xFF) as u8;
        out_rgb[base + 1] = ((px >> 8) & 0xFF) as u8;
        out_rgb[base + 2] = (px & 0xFF) as u8;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let source_topic = "camera/front/frames";
    let dest_topic = "gpu-bev-node/frames/birdseye";
    let img_size = ImageSize {
        width: std::env::var("CAM_WIDTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1920),
        height: std::env::var("CAM_HEIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1080),
    };

    println!("[GPU-BEV] Initializing pure Zenoh node...");
    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(Config::DEFAULT_CONFIG_PATH_ENV).is_ok() {
                eprintln!("[GPU-BEV] WARN: failed to load ZENOH_CONFIG: {e}. Using defaults.");
            }
            Config::default()
        }
    };
    let session = zenoh::open(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to open Zenoh session: {}", e))?;

    println!("[GPU-BEV] Initializing WGPU/CubeCL hardware...");
    let device = WgpuDevice::default();
    let setup = init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
    let client = WgpuRuntime::client(&device);
    let queue = setup.queue;
    println!("[GPU-BEV] Hardware bound successfully.");

    let pixel_count = img_size.width * img_size.height;
    let expected_rgb_bytes = pixel_count * 3;
    let expected_gpu_bytes = pixel_count * std::mem::size_of::<u32>();
    let publish_jpeg = parse_env_bool("BEV_PUBLISH_JPEG", true);
    let jpeg_quality = parse_env_u8("BEV_JPEG_QUALITY", 80);
    let loop_delay_ms = parse_env_u64("BEV_LOOP_DELAY_MS", 16);
    let mut bench = PipelineBench::from_env();

    let ipm_cfg = math::homography::IpmConfig::from_env(img_size, img_size);
    let h_matrix_cpu = math::homography::bev_out_to_img_homography(&ipm_cfg);
    let h_matrix_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&h_matrix_cpu).to_vec(),
    ));

    println!("[GPU-BEV] Node running. Subscribing to: {}", source_topic);

    let subscriber = session
        .declare_subscriber(source_topic)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create subscriber: {}", e))?;

    let publisher = session
        .declare_publisher(dest_topic)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create publisher: {}", e))?;

    let input_handle = client.empty(expected_gpu_bytes);
    let output_handle = client.empty(expected_gpu_bytes);
    let input_resource = client.get_resource(input_handle.clone().binding());
    let input_buffer = input_resource.resource().buffer.clone();
    let input_offset = input_resource.resource().offset;
    let input_size = input_resource.resource().size;
    let input_copy_size_aligned = (expected_gpu_bytes as u64).next_multiple_of(4u64);

    let img_shape = [img_size.height, img_size.width];
    let img_strides = [img_size.width, 1usize];
    let mat_shape = [9usize];
    let mat_strides = [1usize];
    let cube_dim = CubeDim::new_2d(16, 16);
    let width_u32 = img_size.width as u32;
    let height_u32 = img_size.height as u32;
    let cube_count = CubeCount::new_2d(
        width_u32.div_ceil(cube_dim.x),
        height_u32.div_ceil(cube_dim.y),
    );

    let mut jpeg_decode_scratch = Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
        .map_err(|e| anyhow::anyhow!("Image alloc failed: {:?}", e))?;
    let mut jpeg_encode_buffer = Vec::new();
    let mut input_packed = vec![0u32; pixel_count];

    loop {
        let sample = match subscriber.recv_async().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[GPU-BEV] WARN: failed to receive sample: {e}");
                sleep(Duration::from_millis(10)).await;
                continue;
            }
        };

        let payload = sample.payload().to_bytes();
        let t_decode = Instant::now();
        let frame_bytes: Cow<'_, [u8]> = if looks_like_jpeg(payload.as_ref()) {
            if let Err(e) = jpeg::decode_image_jpeg_rgb8(payload.as_ref(), &mut jpeg_decode_scratch)
            {
                eprintln!("[GPU-BEV] WARN: JPEG decode failed: {e:?}");
                continue;
            }
            Cow::Borrowed(jpeg_decode_scratch.as_slice())
        } else if looks_like_png(payload.as_ref()) {
            if let Err(e) = png::decode_image_png_rgb8(payload.as_ref(), &mut jpeg_decode_scratch) {
                eprintln!(
                    "[GPU-BEV] WARN: PNG decode failed: {e:?} (check CAM_WIDTH/CAM_HEIGHT match the PNG resolution)"
                );
                continue;
            }
            Cow::Borrowed(jpeg_decode_scratch.as_slice())
        } else {
            if payload.len() != expected_rgb_bytes {
                eprintln!(
                    "[GPU-BEV] WARN: raw RGB frame size mismatch: got {}, expected {}",
                    payload.len(),
                    expected_rgb_bytes
                );
                continue;
            }
            payload
        };
        let jpeg_decode_ms = ms(t_decode);

        if frame_bytes.len() != expected_rgb_bytes {
            eprintln!(
                "[GPU-BEV] WARN: decoded frame size mismatch: got {}, expected {}",
                frame_bytes.len(),
                expected_rgb_bytes
            );
            continue;
        }

        let t_pack = Instant::now();
        pack_rgb24_to_u32(frame_bytes.as_ref(), &mut input_packed);
        let pack_ms = ms(t_pack);

        let t_h2d = Instant::now();
        if input_size < input_copy_size_aligned {
            eprintln!(
                "[GPU-BEV] WARN: input buffer too small: got {input_size} bytes, need {input_copy_size_aligned} bytes"
            );
            continue;
        }

        let input_bytes = bytemuck::cast_slice::<u32, u8>(&input_packed);
        if input_copy_size_aligned == expected_gpu_bytes as u64 {
            queue.write_buffer(&input_buffer, input_offset, input_bytes);
        } else {
            let Some(mut view) = queue.write_buffer_with(
                &input_buffer,
                input_offset,
                NonZeroU64::new(input_copy_size_aligned)
                    .expect("input_copy_size_aligned must be non-zero"),
            ) else {
                eprintln!(
                    "[GPU-BEV] WARN: failed to stage-write {input_copy_size_aligned} bytes into input buffer"
                );
                continue;
            };
            view[0..input_bytes.len()].copy_from_slice(input_bytes);
            view[input_bytes.len()..].fill(0u8);
        }

        if bench.enabled {
            queue.submit([]);
            if let Err(e) = client.sync().await {
                eprintln!("[GPU-BEV] WARN: H2D sync failed: {e:?}");
                sleep(Duration::from_millis(50)).await;
                continue;
            }
        }
        let h2d_ms = ms(t_h2d);

        let t_kernel = Instant::now();
        unsafe {
            if let Err(e) = perspective_warp_kernel::launch::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                TensorArg::from_raw_parts::<u32>(&input_handle, &img_strides, &img_shape, 1),
                TensorArg::from_raw_parts::<u32>(&output_handle, &img_strides, &img_shape, 1),
                TensorArg::from_raw_parts::<f32>(&h_matrix_handle, &mat_strides, &mat_shape, 1),
            ) {
                eprintln!("[GPU-BEV] WARN: perspective_warp_kernel launch failed: {e}");
                sleep(Duration::from_millis(50)).await;
                continue;
            }
        }

        if bench.enabled
            && let Err(e) = client.sync().await
        {
            eprintln!("[GPU-BEV] WARN: kernel sync failed: {e:?}");
            sleep(Duration::from_millis(50)).await;
            continue;
        }
        let kernel_ms = ms(t_kernel);

        let t_d2h = Instant::now();
        let mut output_chunks = match client.read_async(vec![output_handle.clone()]).await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[GPU-BEV] WARN: failed to read VRAM buffer: {e}");
                sleep(Duration::from_millis(10)).await;
                continue;
            }
        };
        let output_bytes = output_chunks.remove(0);
        let d2h_ms = ms(t_d2h);

        let output_packed = match output_bytes.try_into_vec::<u32>() {
            Ok(v) => v,
            Err(bytes) => {
                let raw = bytes.to_vec();
                let Ok(words) = bytemuck::try_cast_slice::<u8, u32>(&raw) else {
                    eprintln!("[GPU-BEV] WARN: output buffer has invalid alignment/length");
                    continue;
                };
                words.to_vec()
            }
        };

        if output_packed.len() != pixel_count {
            eprintln!(
                "[GPU-BEV] WARN: output pixel count mismatch: got {}, expected {}",
                output_packed.len(),
                pixel_count
            );
            continue;
        }

        let t_unpack = Instant::now();
        unpack_u32_to_rgb24(&output_packed, jpeg_decode_scratch.as_slice_mut());
        let unpack_ms = ms(t_unpack);

        let mut jpeg_encode_ms = 0.0;
        let publish_payload: Vec<u8> = if publish_jpeg {
            let t_encode = Instant::now();
            jpeg_encode_buffer.clear();
            if let Err(e) = jpeg::encode_image_jpeg_rgb8(
                &jpeg_decode_scratch,
                jpeg_quality,
                &mut jpeg_encode_buffer,
            ) {
                eprintln!("[GPU-BEV] WARN: JPEG encode failed: {e:?}");
                continue;
            }
            jpeg_encode_ms = ms(t_encode);
            jpeg_encode_buffer.clone()
        } else {
            jpeg_decode_scratch.as_slice().to_vec()
        };

        if let Err(e) = publisher.put(publish_payload).await {
            eprintln!("[GPU-BEV] WARN: Zenoh publish failed: {e}");
            sleep(Duration::from_millis(10)).await;
            continue;
        }

        if !bench.enabled {
            if publish_jpeg {
                println!("[GPU-BEV] Published BEV frame (JPEG quality={jpeg_quality})");
            } else {
                println!("[GPU-BEV] Published 6.2MB BEV frame to network!");
            }
        }

        bench.record(FrameTimings {
            jpeg_decode_ms,
            pack_ms,
            h2d_ms,
            kernel_ms,
            d2h_ms,
            unpack_ms,
            jpeg_encode_ms,
            total_ms: jpeg_decode_ms
                + pack_ms
                + h2d_ms
                + kernel_ms
                + d2h_ms
                + unpack_ms
                + jpeg_encode_ms,
        });

        if loop_delay_ms > 0 {
            sleep(Duration::from_millis(loop_delay_ms)).await;
        }
    }
}
