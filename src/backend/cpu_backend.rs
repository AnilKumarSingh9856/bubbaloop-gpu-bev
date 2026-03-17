use anyhow::Result;
use std::time::Instant;

use crate::backend::{ImageProcessor, WarpRunTimings};

/// CPU nearest-neighbor reference implementation of perspective warp.
pub struct CpuBackend {
    width: usize,
    height: usize,
}

impl CpuBackend {
    /// Create a CPU backend for fixed image dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl ImageProcessor for CpuBackend {
    async fn warp_perspective(
        &self,
        input_packed_rgb: &[u32],
        output_packed_rgb: &mut [u32],
        matrix: &[f32; 9],
    ) -> Result<WarpRunTimings> {
        let pixel_count = self.width * self.height;
        anyhow::ensure!(
            input_packed_rgb.len() == pixel_count,
            "input pixel count mismatch: got {}, expected {}",
            input_packed_rgb.len(),
            pixel_count
        );
        anyhow::ensure!(
            output_packed_rgb.len() == pixel_count,
            "output pixel count mismatch: got {}, expected {}",
            output_packed_rgb.len(),
            pixel_count
        );

        let t_kernel = Instant::now();
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                output_packed_rgb[idx] = 0;

                let target_x = x as f32;
                let target_y = y as f32;

                let src_x_h = matrix[0] * target_x + matrix[1] * target_y + matrix[2];
                let src_y_h = matrix[3] * target_x + matrix[4] * target_y + matrix[5];
                let weight = matrix[6] * target_x + matrix[7] * target_y + matrix[8];

                if weight == 0.0 {
                    continue;
                }

                let src_x_f = src_x_h / weight;
                let src_y_f = src_y_h / weight;

                if src_x_f >= 0.0
                    && src_y_f >= 0.0
                    && src_x_f < self.width as f32
                    && src_y_f < self.height as f32
                {
                    let src_x = src_x_f as usize;
                    let src_y = src_y_f as usize;
                    let in_idx = src_y * self.width + src_x;
                    output_packed_rgb[idx] = input_packed_rgb[in_idx];
                }
            }
        }
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

        Ok(WarpRunTimings {
            h2d_ms: 0.0,
            kernel_ms,
            d2h_ms: 0.0,
        })
    }
}
