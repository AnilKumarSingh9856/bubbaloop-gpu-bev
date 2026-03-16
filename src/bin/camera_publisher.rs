//! Simple Zenoh camera publisher for demos/benchmarks.
//!
//! Publishes JPEG frames on `camera/front/frames` (by default) so `gpu-bev-node` can be benchmarked
//! without needing a real camera driver.
//!
//! # Usage
//! - Publish a single JPEG repeatedly:
//!   `CAM_PUB_PATH=./frame.jpg cargo run --bin camera_publisher`
//! - Publish a directory of JPEGs (sorted by filename):
//!   `CAM_PUB_PATH=./frames CAM_PUB_FPS=30 cargo run --bin camera_publisher`
//!
//! # Environment
//! - `CAM_PUB_TOPIC` (default: `camera/front/frames`)
//! - `CAM_PUB_PATH` (optional): file or directory containing `.jpg`/`.jpeg`/`.png` (otherwise generates frames)
//! - `CAM_PUB_FPS` (default: `30`)
//! - `CAM_PUB_LOOP` (default: `1`) loop directory playback
//! - `CAM_PUB_WIDTH` / `CAM_PUB_HEIGHT` (fallback: `CAM_WIDTH` / `CAM_HEIGHT`, default: 1920×1080)
//! - `CAM_PUB_GEN_FRAMES` (default: `10`)
//! - `CAM_PUB_GEN_QUALITY` (default: `85`)

use anyhow::Result;
use kornia_image::allocator::CpuAllocator;
use kornia_image::{Image, ImageSize};
use kornia_io::jpeg;
use std::path::{Path, PathBuf};
use tokio::time::{Duration, interval};
use zenoh::Config;
use zenoh::bytes::ZBytes;

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

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

fn parse_env_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(default)
        .min(100)
}

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

fn is_supported_image_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
}

fn load_frames(path: &Path) -> Result<Vec<ZBytes>> {
    if path.is_file() {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            looks_like_jpeg(&bytes) || looks_like_png(&bytes),
            "unsupported image format for `{}` (expected JPEG/PNG bytes)",
            path.display()
        );
        return Ok(vec![ZBytes::from(bytes)]);
    }

    if !path.is_dir() {
        anyhow::bail!(
            "CAM_PUB_PATH is not a file or directory: {}",
            path.display()
        );
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_supported_image_path(p))
        .collect();
    entries.sort();

    let mut frames = Vec::with_capacity(entries.len());
    for p in entries {
        let bytes = std::fs::read(&p)?;
        if looks_like_jpeg(&bytes) || looks_like_png(&bytes) {
            frames.push(ZBytes::from(bytes));
        } else {
            eprintln!(
                "[CAM-PUB] WARN: skipping `{}` (not JPEG/PNG bytes)",
                p.display()
            );
        }
    }

    anyhow::ensure!(
        !frames.is_empty(),
        "no JPEG/PNG frames found in {}",
        path.display()
    );

    Ok(frames)
}

fn generate_frames(size: ImageSize, num_frames: usize, quality: u8) -> Result<Vec<ZBytes>> {
    anyhow::ensure!(num_frames > 0, "CAM_PUB_GEN_FRAMES must be > 0");

    let mut frames = Vec::with_capacity(num_frames);
    let mut image = Image::<u8, 3, CpuAllocator>::from_size_val(size, 0u8, CpuAllocator)?;
    let w = size.width;
    let h = size.height;

    for t in 0..num_frames {
        let t_u32 = t as u32;
        let data = image.as_slice_mut();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let r = ((x as u32 + t_u32 * 3) % 256) as u8;
                let g = ((y as u32 + t_u32 * 5) % 256) as u8;
                let b = (((x as u32 / 2 + y as u32 / 2) + t_u32 * 7) % 256) as u8;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        let mut buffer = Vec::new();
        jpeg::encode_image_jpeg_rgb8(&image, quality, &mut buffer)
            .map_err(|e| anyhow::anyhow!("JPEG encode failed: {e:?}"))?;
        frames.push(ZBytes::from(buffer));
    }

    Ok(frames)
}

#[tokio::main]
async fn main() -> Result<()> {
    let topic = std::env::var("CAM_PUB_TOPIC").unwrap_or_else(|_| "camera/front/frames".into());
    let fps = parse_env_u64("CAM_PUB_FPS", 30).max(1);
    let loop_playback = parse_env_bool("CAM_PUB_LOOP", true);

    let width = env_usize("CAM_PUB_WIDTH")
        .or_else(|| env_usize("CAM_WIDTH"))
        .unwrap_or(1920);
    let height = env_usize("CAM_PUB_HEIGHT")
        .or_else(|| env_usize("CAM_HEIGHT"))
        .unwrap_or(1080);
    let gen_frames = parse_env_u64("CAM_PUB_GEN_FRAMES", 10) as usize;
    let gen_quality = parse_env_u8("CAM_PUB_GEN_QUALITY", 85);

    let frames = match std::env::var("CAM_PUB_PATH").ok().map(PathBuf::from) {
        Some(path) => {
            let frames = load_frames(&path)?;
            let total = frames.len();
            eprintln!(
                "[CAM-PUB] Loaded {total} frame(s) from `{}`",
                path.display()
            );
            frames
        }
        None => {
            let size = ImageSize { width, height };
            eprintln!(
                "[CAM-PUB] CAM_PUB_PATH not set; generating {gen_frames} JPEG frame(s) ({width}×{height}, quality={gen_quality})"
            );
            generate_frames(size, gen_frames, gen_quality)?
        }
    };
    let total = frames.len();
    eprintln!("[CAM-PUB] Publishing to `{topic}` at {fps} FPS (loop={loop_playback})");

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(Config::DEFAULT_CONFIG_PATH_ENV).is_ok() {
                eprintln!("[CAM-PUB] WARN: failed to load ZENOH_CONFIG: {e}. Using defaults.");
            }
            Config::default()
        }
    };
    let session = zenoh::open(config)
        .await
        .map_err(|e| anyhow::anyhow!("zenoh open failed: {e}"))?;
    let publisher = session
        .declare_publisher(&topic)
        .await
        .map_err(|e| anyhow::anyhow!("declare_publisher failed: {e}"))?;

    let mut tick = interval(Duration::from_secs_f64(1.0 / fps as f64));
    let mut idx: usize = 0;
    let mut sent: u64 = 0;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("[CAM-PUB] Ctrl+C received, stopping.");
                break;
            }
            _ = tick.tick() => {
                if idx >= total {
                    if loop_playback {
                        idx = 0;
                    } else {
                        eprintln!("[CAM-PUB] Reached end of frames, stopping.");
                        break;
                    }
                }

                if let Err(e) = publisher.put(frames[idx].clone()).await {
                    eprintln!("[CAM-PUB] WARN: publish failed: {e}");
                } else {
                    sent += 1;
                    if sent.is_multiple_of(120) {
                        eprintln!("[CAM-PUB] sent={sent}");
                    }
                }
                idx += 1;
            }
        }
    }

    Ok(())
}
