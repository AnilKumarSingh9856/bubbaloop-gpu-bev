//! Benchmark utility for comparing Rust GPU warp output against an OpenCV reference image.
//!
//! The benchmark reports latency and pixel-difference metrics, and can save the Rust GPU output
//! image for visual inspection.

use anyhow::Result;
use gpu_bev_node::backend::CubeCLBackend;
use gpu_bev_node::imgproc::warp_perspective_packed_rgb;
use gpu_bev_node::math;
use kornia_image::ImageSize;
use kornia_image::allocator::CpuAllocator;
use kornia_image::color_spaces::Rgb8;
use kornia_io::{jpeg, png};

/// Return true when the payload starts with the JPEG SOI marker.
fn looks_like_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

/// Return true when the payload starts with the PNG signature.
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

/// Pack `RGBRGB...` bytes into `0x00RRGGBB` pixels.
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

/// Unpack `0x00RRGGBB` pixels into `RGBRGB...` bytes.
fn unpack_u32_to_rgb24(pixels: &[u32], out_rgb: &mut [u8]) {
    debug_assert_eq!(out_rgb.len(), pixels.len() * 3);
    for (i, &px) in pixels.iter().enumerate() {
        let base = i * 3;
        out_rgb[base] = ((px >> 16) & 0xFF) as u8;
        out_rgb[base + 1] = ((px >> 8) & 0xFF) as u8;
        out_rgb[base + 2] = (px & 0xFF) as u8;
    }
}

/// Parse a boolean environment variable with common true/false aliases.
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

/// Compute the arithmetic mean for a sequence of values.
fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[tokio::main]
async fn main() -> Result<()> {
    // Read runtime configuration.
    let image_path =
        std::env::var("BENCH_IMAGE_PATH").unwrap_or_else(|_| "images/frame.jpg".into());
    let width = std::env::var("CAM_WIDTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1280);
    let height = std::env::var("CAM_HEIGHT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(720);
    let warmup = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10);
    let iters = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50);
    let save_outputs = parse_env_bool("BENCH_SAVE_OUTPUTS", true);
    let opencv_ref_path = std::env::var("BENCH_OPENCV_REF")
        .unwrap_or_else(|_| "images/opencv_baseline_bev.png".into());

    let img_size = ImageSize { width, height };
    let pixel_count = width * height;
    let expected_rgb_bytes = pixel_count * 3;

    // 1) Load benchmark input image.
    let mut decoded = Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
        .map_err(|e| anyhow::anyhow!("image alloc failed: {e:?}"))?;

    let bytes = std::fs::read(&image_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", image_path))?;

    if looks_like_jpeg(&bytes) {
        jpeg::decode_image_jpeg_rgb8(&bytes, &mut decoded)
            .map_err(|e| anyhow::anyhow!("jpeg decode failed: {e:?}"))?;
    } else if looks_like_png(&bytes) {
        png::decode_image_png_rgb8(&bytes, &mut decoded)
            .map_err(|e| anyhow::anyhow!("png decode failed: {e:?}"))?;
    } else {
        anyhow::bail!("unsupported input format: expected JPEG or PNG");
    }

    anyhow::ensure!(
        decoded.as_slice().len() == expected_rgb_bytes,
        "decoded image size mismatch: got {}, expected {}. Set CAM_WIDTH/CAM_HEIGHT correctly",
        decoded.as_slice().len(),
        expected_rgb_bytes
    );

    // 2) Build homography matrix.
    let ipm_cfg = math::homography::IpmConfig::from_env(img_size, img_size);
    let h_matrix = math::homography::bev_out_to_img_homography(&ipm_cfg);

    let mut input_packed = vec![0u32; pixel_count];
    let mut gpu_output = vec![0u32; pixel_count];
    pack_rgb24_to_u32(decoded.as_slice(), &mut input_packed);

    // 3) Initialize GPU backend.
    let gpu_backend = CubeCLBackend::new(width, height, h_matrix).await?;

    // 4) Run warmup iterations.
    for _ in 0..warmup {
        let _ =
            warp_perspective_packed_rgb(&gpu_backend, &input_packed, &mut gpu_output, &h_matrix)?;
    }

    // 5) Timed benchmark loop.
    let mut gpu_totals = Vec::with_capacity(iters);

    for _ in 0..iters {
        let g =
            warp_perspective_packed_rgb(&gpu_backend, &input_packed, &mut gpu_output, &h_matrix)?;
        gpu_totals.push(g.h2d_ms + g.kernel_ms + g.d2h_ms);
    }

    let gpu_avg = avg(&gpu_totals);

    // 6) Load OpenCV reference image and compute accuracy deltas.
    let mut gpu_rgb = vec![0u8; expected_rgb_bytes];
    unpack_u32_to_rgb24(&gpu_output, &mut gpu_rgb);

    let mut opencv_ref = Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
        .map_err(|e| anyhow::anyhow!("opencv ref alloc failed: {e:?}"))?;
    let opencv_ref_bytes = std::fs::read(&opencv_ref_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", opencv_ref_path))?;
    if looks_like_jpeg(&opencv_ref_bytes) {
        jpeg::decode_image_jpeg_rgb8(&opencv_ref_bytes, &mut opencv_ref)
            .map_err(|e| anyhow::anyhow!("opencv ref jpeg decode failed: {e:?}"))?;
    } else if looks_like_png(&opencv_ref_bytes) {
        png::decode_image_png_rgb8(&opencv_ref_bytes, &mut opencv_ref)
            .map_err(|e| anyhow::anyhow!("opencv ref png decode failed: {e:?}"))?;
    } else {
        anyhow::bail!("unsupported OpenCV reference format: expected JPEG or PNG");
    }

    anyhow::ensure!(
        opencv_ref.as_slice().len() == expected_rgb_bytes,
        "opencv ref size mismatch: got {}, expected {}",
        opencv_ref.as_slice().len(),
        expected_rgb_bytes
    );

    let mean_abs_diff = gpu_rgb
        .iter()
        .zip(opencv_ref.as_slice().iter())
        .map(|(g, c)| (*g as f64 - *c as f64).abs())
        .sum::<f64>()
        / expected_rgb_bytes as f64;

    let max_abs_diff = gpu_rgb
        .iter()
        .zip(opencv_ref.as_slice().iter())
        .map(|(g, c)| (*g as i32 - *c as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    let mut within_1: usize = 0;
    let mut within_2: usize = 0;
    let mut within_4: usize = 0;
    for (g, c) in gpu_rgb.iter().zip(opencv_ref.as_slice().iter()) {
        let d = (*g as i32 - *c as i32).unsigned_abs();
        if d <= 1 {
            within_1 += 1;
        }
        if d <= 2 {
            within_2 += 1;
        }
        if d <= 4 {
            within_4 += 1;
        }
    }
    let total = expected_rgb_bytes as f64;
    let within_1_pct = within_1 as f64 * 100.0 / total;
    let within_2_pct = within_2 as f64 * 100.0 / total;
    let within_4_pct = within_4 as f64 * 100.0 / total;

    // 7) Optionally persist GPU output image.
    if save_outputs {
        let mut gpu_img = Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
            .map_err(|e| anyhow::anyhow!("gpu image alloc failed: {e:?}"))?;
        gpu_img.as_slice_mut().copy_from_slice(&gpu_rgb);

        let mut gpu_jpeg = Vec::new();
        jpeg::encode_image_jpeg_rgb8(&gpu_img, 90, &mut gpu_jpeg)
            .map_err(|e| anyhow::anyhow!("gpu jpeg encode failed: {e:?}"))?;
        std::fs::write("images/bench_output_gpu.jpg", gpu_jpeg)
            .map_err(|e| anyhow::anyhow!("failed writing images/bench_output_gpu.jpg: {e}"))?;
    }

    println!(
        "[WARP-BENCH] image={} size={}x{} warmup={} iters={}",
        image_path, width, height, warmup, iters
    );
    println!("[WARP-BENCH] GPU avg total: {:.3} ms", gpu_avg);
    println!("[WARP-BENCH] OpenCV ref image: {}", opencv_ref_path);
    println!(
        "[WARP-BENCH] Mean abs pixel diff (GPU vs OpenCV): {:.4}",
        mean_abs_diff
    );
    println!(
        "[WARP-BENCH] Max abs pixel diff (GPU vs OpenCV): {}",
        max_abs_diff
    );
    println!(
        "[WARP-BENCH] Match <=1 intensity diff: {:.2}%",
        within_1_pct
    );
    println!(
        "[WARP-BENCH] Match <=2 intensity diff: {:.2}%",
        within_2_pct
    );
    println!(
        "[WARP-BENCH] Match <=4 intensity diff: {:.2}%",
        within_4_pct
    );
    if save_outputs {
        println!("[WARP-BENCH] Saved: images/bench_output_gpu.jpg");
    }

    Ok(())
}
