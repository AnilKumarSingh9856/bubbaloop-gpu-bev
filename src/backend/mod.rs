mod cpu_backend;
mod cubecl_backend;

/// CPU reference backend.
pub use cpu_backend::CpuBackend;
/// CubeCL/WGPU backend implementation.
pub use cubecl_backend::{CubeCLBackend, WarpRunTimings};

/// Image processing backend interface for perspective warping.
pub trait ImageProcessor {
    /// Warp a packed RGB image using a 3x3 homography matrix.
    ///
    /// The image is represented as a `[height * width]` slice of packed `0x00RRGGBB` pixels.
    /// Implementations are expected to write all output pixels.
    fn warp_perspective(
        &self,
        input_packed_rgb: &[u32],
        output_packed_rgb: &mut [u32],
        matrix: &[f32; 9],
    ) -> anyhow::Result<WarpRunTimings>;
}
