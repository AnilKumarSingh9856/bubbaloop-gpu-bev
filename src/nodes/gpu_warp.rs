use cubecl::prelude::*;

/// Warp an RGB image using a 3×3 homography.
///
/// # Tensor layouts
/// - `input`/`output`: `u32` tensor with shape `[height, width]`, where each element stores a
///   packed pixel value in the form `0x00RRGGBB`.
/// - `matrix`: `f32` tensor with 9 elements in row-major order (3×3).
///
/// # Behavior
/// For each output pixel `(x, y)`, the kernel computes the corresponding source coordinate using
/// homogeneous division and samples the input with nearest-neighbor indexing.
///
/// Pixels that map out of bounds are filled with `0` (black).
#[cube(launch)]
pub fn perspective_warp_kernel(
    input: &Tensor<u32>,
    output: &mut Tensor<u32>,
    matrix: &Tensor<f32>,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let width = u32::cast_from(output.shape(1));
    let height = u32::cast_from(output.shape(0));
    let in_width = u32::cast_from(input.shape(1));
    let in_height = u32::cast_from(input.shape(0));

    if x < width && y < height {
        let out_idx = usize::cast_from(y * width + x);
        output[out_idx] = 0u32;

        let target_x = f32::cast_from(x);
        let target_y = f32::cast_from(y);

        let src_x_h = matrix[0] * target_x + matrix[1] * target_y + matrix[2];
        let src_y_h = matrix[3] * target_x + matrix[4] * target_y + matrix[5];
        let weight = matrix[6] * target_x + matrix[7] * target_y + matrix[8];

        if weight != 0.0 {
            let src_x_f = src_x_h / weight;
            let src_y_f = src_y_h / weight;

            if src_x_f >= 0.0 && src_y_f >= 0.0 {
                let src_x = u32::cast_from(src_x_f);
                let src_y = u32::cast_from(src_y_f);

                if src_x < in_width && src_y < in_height {
                    let in_idx = usize::cast_from(src_y * in_width + src_x);
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}
