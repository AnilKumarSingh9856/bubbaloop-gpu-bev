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

use anyhow::Result;
use std::time::Instant;
use tokio::task;
use tokio::time::{Duration, sleep};
use zenoh::Config;

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

/// Parse a JPEG quality value from env and clamp to 0..=100.
fn parse_env_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(default)
        .min(100)
}

/// Parse an unsigned integer environment variable.
fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Convert elapsed time to milliseconds.
fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Aggregated statistics for one measured stage.
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

/// Rolling benchmark state for pipeline stage timing.
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

/// Per-frame stage timing values.
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

/// Pack `RGBRGB...` bytes into `0x00RRGGBB` pixels.
fn pack_rgb24_to_u32(rgb: &[u8], out: &mut [u32]) {
    debug_assert_eq!(rgb.len(), out.len() * 3);
    for (px, rgb_chunk) in out.iter_mut().zip(rgb.chunks_exact(3)) {
        let r = rgb_chunk[0] as u32;
        let g = rgb_chunk[1] as u32;
        let b = rgb_chunk[2] as u32;
        *px = (r << 16) | (g << 8) | b;
    }
}

/// Unpack `0x00RRGGBB` pixels into `RGBRGB...` bytes.
fn unpack_u32_to_rgb24(pixels: &[u32], out_rgb: &mut [u8]) {
    debug_assert_eq!(out_rgb.len(), pixels.len() * 3);
    for (rgb_chunk, &px) in out_rgb.chunks_exact_mut(3).zip(pixels.iter()) {
        rgb_chunk[0] = ((px >> 16) & 0xFF) as u8;
        rgb_chunk[1] = ((px >> 8) & 0xFF) as u8;
        rgb_chunk[2] = (px & 0xFF) as u8;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Source and destination topic names.
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
    // Initialize Zenoh session from environment configuration.
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

    let pixel_count = img_size.width * img_size.height;
    let expected_rgb_bytes = pixel_count * 3;
    let publish_jpeg = parse_env_bool("BEV_PUBLISH_JPEG", true);
    let jpeg_quality = parse_env_u8("BEV_JPEG_QUALITY", 80);
    let loop_delay_ms = parse_env_u64("BEV_LOOP_DELAY_MS", 16);
    let mut bench = PipelineBench::from_env();

    // Build inverse-perspective homography matrix.
    let ipm_cfg = math::homography::IpmConfig::from_env(img_size, img_size);
    let h_matrix_cpu = math::homography::bev_out_to_img_homography(&ipm_cfg);

    println!("[GPU-BEV] Initializing WGPU/CubeCL backend...");
    let backend = CubeCLBackend::new(img_size.width, img_size.height, h_matrix_cpu).await?;
    println!("[GPU-BEV] Hardware bound successfully.");

    println!("[GPU-BEV] Node running. Subscribing to: {}", source_topic);

    let subscriber = session
        .declare_subscriber(source_topic)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create subscriber: {}", e))?;

    let publisher = session
        .declare_publisher(dest_topic)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create publisher: {}", e))?;

    // Allocate scratch image for decode and encode stages.
    let mut jpeg_decode_scratch = Some(
        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
            .map_err(|e| anyhow::anyhow!("Image alloc failed: {:?}", e))?,
    );
    let mut jpeg_encode_buffer = Vec::with_capacity(expected_rgb_bytes);
    let mut input_packed = vec![0u32; pixel_count];
    let mut output_packed = vec![0u32; pixel_count];

    loop {
        // Receive one frame payload.
        let sample = match subscriber.recv_async().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[GPU-BEV] WARN: failed to receive sample: {e}");
                sleep(Duration::from_millis(10)).await;
                continue;
            }
        };

        let payload = sample.payload().to_bytes();

        // Decode payload into RGB bytes.
        let t_decode = Instant::now();
        let frame_bytes: &[u8] = if looks_like_jpeg(payload.as_ref()) {
            let payload_owned = payload.as_ref().to_vec();
            let decoded = jpeg_decode_scratch
                .take()
                .ok_or_else(|| anyhow::anyhow!("decode scratch buffer missing"))?;
            let decode_task = task::spawn_blocking(move || -> Result<Rgb8<CpuAllocator>> {
                let mut decoded = decoded;
                jpeg::decode_image_jpeg_rgb8(&payload_owned, &mut decoded)
                    .map_err(|e| anyhow::anyhow!("JPEG decode failed: {e:?}"))?;
                Ok(decoded)
            })
            .await;

            match decode_task {
                Ok(Ok(decoded)) => {
                    jpeg_decode_scratch = Some(decoded);
                    jpeg_decode_scratch
                        .as_ref()
                        .expect("decode scratch set")
                        .as_slice()
                }
                Ok(Err(e)) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator).map_err(
                            |alloc_err| {
                                anyhow::anyhow!(
                                    "JPEG decode failed: {e}; alloc failed: {alloc_err:?}"
                                )
                            },
                        )?,
                    );
                    eprintln!("[GPU-BEV] WARN: {e}");
                    continue;
                }
                Err(e) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
                            .map_err(|alloc_err| {
                                anyhow::anyhow!("JPEG decode worker join failed: {e}; alloc failed: {alloc_err:?}")
                            })?,
                    );
                    eprintln!("[GPU-BEV] WARN: JPEG decode worker join failed: {e}");
                    continue;
                }
            }
        } else if looks_like_png(payload.as_ref()) {
            let payload_owned = payload.as_ref().to_vec();
            let decoded = jpeg_decode_scratch
                .take()
                .ok_or_else(|| anyhow::anyhow!("decode scratch buffer missing"))?;
            let decode_task = task::spawn_blocking(move || -> Result<Rgb8<CpuAllocator>> {
                let mut decoded = decoded;
                png::decode_image_png_rgb8(&payload_owned, &mut decoded)
                    .map_err(|e| anyhow::anyhow!("PNG decode failed: {e:?}"))?;
                Ok(decoded)
            })
            .await;

            match decode_task {
                Ok(Ok(decoded)) => {
                    jpeg_decode_scratch = Some(decoded);
                    jpeg_decode_scratch
                        .as_ref()
                        .expect("decode scratch set")
                        .as_slice()
                }
                Ok(Err(e)) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator).map_err(
                            |alloc_err| {
                                anyhow::anyhow!(
                                    "PNG decode failed: {e}; alloc failed: {alloc_err:?}"
                                )
                            },
                        )?,
                    );
                    eprintln!(
                        "[GPU-BEV] WARN: {e} (check CAM_WIDTH/CAM_HEIGHT match the PNG resolution)"
                    );
                    continue;
                }
                Err(e) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
                            .map_err(|alloc_err| {
                                anyhow::anyhow!("PNG decode worker join failed: {e}; alloc failed: {alloc_err:?}")
                            })?,
                    );
                    eprintln!("[GPU-BEV] WARN: PNG decode worker join failed: {e}");
                    continue;
                }
            }
        } else {
            if payload.len() != expected_rgb_bytes {
                eprintln!(
                    "[GPU-BEV] WARN: raw RGB frame size mismatch: got {}, expected {}",
                    payload.len(),
                    expected_rgb_bytes
                );
                continue;
            }
            payload.as_ref()
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

        // Pack RGB bytes into backend input format.
        let t_pack = Instant::now();
        pack_rgb24_to_u32(frame_bytes, &mut input_packed);
        let pack_ms = ms(t_pack);

        // Execute perspective warp.
        let run_timings = match warp_perspective_packed_rgb(
            &backend,
            &input_packed,
            &mut output_packed,
            &h_matrix_cpu,
        )
        .await
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[GPU-BEV] WARN: backend warp failed: {e}");
                sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        let h2d_ms = run_timings.h2d_ms;
        let kernel_ms = run_timings.kernel_ms;
        let d2h_ms = run_timings.d2h_ms;

        // Unpack backend output into RGB bytes.
        let t_unpack = Instant::now();
        unpack_u32_to_rgb24(
            &output_packed,
            jpeg_decode_scratch
                .as_mut()
                .expect("decode scratch available")
                .as_slice_mut(),
        );
        let unpack_ms = ms(t_unpack);

        // Encode outbound payload and publish.
        let mut jpeg_encode_ms = 0.0;
        if publish_jpeg {
            let t_encode = Instant::now();
            let encode_quality = jpeg_quality;
            let encode_buffer = std::mem::take(&mut jpeg_encode_buffer);
            let encode_img = jpeg_decode_scratch
                .take()
                .ok_or_else(|| anyhow::anyhow!("encode scratch buffer missing"))?;
            let encode_task =
                task::spawn_blocking(move || -> Result<(Vec<u8>, Rgb8<CpuAllocator>)> {
                    let mut out = encode_buffer;
                    out.clear();
                    jpeg::encode_image_jpeg_rgb8(&encode_img, encode_quality, &mut out)
                        .map_err(|e| anyhow::anyhow!("JPEG encode failed: {e:?}"))?;
                    Ok((out, encode_img))
                })
                .await;

            let (encoded, decode_img) = match encode_task {
                Ok(Ok((buffer, decode_img))) => (buffer, decode_img),
                Ok(Err(e)) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator).map_err(
                            |alloc_err| {
                                anyhow::anyhow!(
                                    "JPEG encode failed: {e}; alloc failed: {alloc_err:?}"
                                )
                            },
                        )?,
                    );
                    jpeg_encode_buffer = Vec::with_capacity(expected_rgb_bytes);
                    eprintln!("[GPU-BEV] WARN: {e}");
                    continue;
                }
                Err(e) => {
                    jpeg_decode_scratch = Some(
                        Rgb8::<CpuAllocator>::from_size_val(img_size, 0u8, CpuAllocator)
                            .map_err(|alloc_err| {
                                anyhow::anyhow!("JPEG encode worker join failed: {e}; alloc failed: {alloc_err:?}")
                            })?,
                    );
                    jpeg_encode_buffer = Vec::with_capacity(expected_rgb_bytes);
                    eprintln!("[GPU-BEV] WARN: JPEG encode worker join failed: {e}");
                    continue;
                }
            };
            jpeg_decode_scratch = Some(decode_img);
            jpeg_encode_ms = ms(t_encode);
            if let Err(e) = publisher.put(encoded.as_slice()).await {
                eprintln!("[GPU-BEV] WARN: Zenoh publish failed: {e}");
                sleep(Duration::from_millis(10)).await;
                jpeg_encode_buffer = encoded;
                continue;
            }
            jpeg_encode_buffer = encoded;
        } else {
            let raw_rgb = jpeg_decode_scratch
                .as_ref()
                .expect("decode scratch available")
                .as_slice();
            if let Err(e) = publisher.put(raw_rgb).await {
                eprintln!("[GPU-BEV] WARN: Zenoh publish failed: {e}");
                sleep(Duration::from_millis(10)).await;
                continue;
            }
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
