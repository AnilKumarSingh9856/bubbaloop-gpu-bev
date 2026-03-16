//! Minimal Zenoh subscriber that visualizes BEV frames published by `gpu-bev-node`.
//!
//! The GPU node publishes frames on `gpu-bev-node/frames/birdseye` as either:
//! - JPEG (default), or
//! - raw `Vec<u8>` in HWC order (RGB, 3 channels).
//!
//! This viewer subscribes to that topic, detects the payload format, decodes/reshapes using the
//! configured width/height, and displays it in a window.
//!
//! # Configuration
//! - `BEV_TOPIC` (default: `gpu-bev-node/frames/birdseye`)
//! - `BEV_WIDTH` / `BEV_HEIGHT` (fallback: `CAM_WIDTH` / `CAM_HEIGHT`, default: 1920×1080)
//!
//! # Run
//! `cargo run --features viewer --bin bev_viewer`

use anyhow::Result;
use kornia_image::allocator::CpuAllocator;
use kornia_image::{Image, ImageSize};
use kornia_io::jpeg;
use minifb::{Key, Window, WindowOptions};
use std::borrow::Cow;
use tokio::time::{Duration, interval};
use zenoh::Config;

fn looks_like_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

fn frame_dims() -> (usize, usize) {
    let width = env_usize("BEV_WIDTH")
        .or_else(|| env_usize("CAM_WIDTH"))
        .unwrap_or(1920);
    let height = env_usize("BEV_HEIGHT")
        .or_else(|| env_usize("CAM_HEIGHT"))
        .unwrap_or(1080);
    (width, height)
}

fn rgb24_to_u32(rgb: &[u8], out: &mut [u32]) {
    debug_assert_eq!(rgb.len(), out.len() * 3);
    for (i, px) in out.iter_mut().enumerate() {
        let base = i * 3;
        let r = rgb[base] as u32;
        let g = rgb[base + 1] as u32;
        let b = rgb[base + 2] as u32;
        *px = (r << 16) | (g << 8) | b;
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() -> Result<()> {
    let topic =
        std::env::var("BEV_TOPIC").unwrap_or_else(|_| "gpu-bev-node/frames/birdseye".into());
    let (width, height) = frame_dims();
    let img_size = ImageSize { width, height };
    let expected_len = width * height * 3;

    eprintln!("[BEV-VIEWER] Subscribing to `{topic}` ({width}×{height} RGB)");
    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(Config::DEFAULT_CONFIG_PATH_ENV).is_ok() {
                eprintln!("[BEV-VIEWER] WARN: failed to load ZENOH_CONFIG: {e}. Using defaults.");
            }
            Config::default()
        }
    };
    let session = zenoh::open(config)
        .await
        .map_err(|e| anyhow::anyhow!("zenoh open failed: {e}"))?;
    let subscriber = session
        .declare_subscriber(&topic)
        .await
        .map_err(|e| anyhow::anyhow!("declare_subscriber failed: {e}"))?;

    let mut window = Window::new(
        "gpu-bev-node | BEV Viewer (ESC to quit)",
        width,
        height,
        WindowOptions::default(),
    )?;
    window.limit_update_rate(Some(Duration::from_micros(16_000)));

    let mut pixels = vec![0u32; width * height];
    let mut jpeg_decode_scratch =
        Image::<u8, 3, CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)?;
    let mut tick = interval(Duration::from_millis(16));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        tokio::select! {
            sample = subscriber.recv_async() => {
                let sample = match sample {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[BEV-VIEWER] recv error: {e}");
                        continue;
                    }
                };

                let payload = sample.payload().to_bytes();

                let frame_bytes: Cow<'_, [u8]> = if looks_like_jpeg(payload.as_ref()) {
                    if let Err(e) = jpeg::decode_image_jpeg_rgb8(payload.as_ref(), &mut jpeg_decode_scratch) {
                        eprintln!("[BEV-VIEWER] WARN: JPEG decode failed: {e:?}");
                        continue;
                    }
                    Cow::Borrowed(jpeg_decode_scratch.as_slice())
                } else {
                    if payload.len() != expected_len {
                        eprintln!(
                            "[BEV-VIEWER] unexpected payload size: got {}, expected {}",
                            payload.len(),
                            expected_len
                        );
                        continue;
                    }
                    payload
                };

                rgb24_to_u32(frame_bytes.as_ref(), &mut pixels);
                window.update_with_buffer(&pixels, width, height)?;
            }
            _ = tick.tick() => {
                window.update();
            }
        }
    }

    Ok(())
}
