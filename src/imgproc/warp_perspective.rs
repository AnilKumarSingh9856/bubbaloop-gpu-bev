use anyhow::Result;

use crate::backend::{ImageProcessor, WarpRunTimings};

/// Backend-dispatched perspective warp for packed RGB images.
///
/// Inputs and outputs are flat arrays of packed `0x00RRGGBB` pixels.
/// The concrete execution backend is selected by the `backend` argument.
pub fn warp_perspective_packed_rgb<B: ImageProcessor>(
    backend: &B,
    input_packed_rgb: &[u32],
    output_packed_rgb: &mut [u32],
    matrix: &[f32; 9],
) -> Result<WarpRunTimings> {
    backend.warp_perspective(input_packed_rgb, output_packed_rgb, matrix)
}
