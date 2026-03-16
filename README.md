## gpu-bev-node

Zenoh node that receives camera frames, warps them on the GPU (CubeCL/WGPU) using an IPM homography,
and republishes a bird’s-eye-view (BEV) frame.

### Run (node)

```bash
cargo run --bin gpu-bev-node
```

Env:

- `CAM_WIDTH` / `CAM_HEIGHT` (default: `1920` / `1080`)
- `BEV_PUBLISH_JPEG` (default: `1`) and `BEV_JPEG_QUALITY` (default: `80`)
- Camera/IPM config: `CAM_FX`, `CAM_FY`, `CAM_CX`, `CAM_CY` (or `CAM_FOV_X_DEG`, `CAM_FOV_Y_DEG`),
  `CAM_HEIGHT_M`, `CAM_ROLL_DEG`, `CAM_PITCH_DEG`, `CAM_YAW_DEG`,
  `BEV_X_MIN_M`, `BEV_X_MAX_M`, `BEV_Y_MIN_M`, `BEV_Y_MAX_M`

### Run (viewer)

```bash
cargo run --features viewer --bin bev_viewer
```

Env:

- `BEV_TOPIC` (default: `gpu-bev-node/frames/birdseye`)
- `BEV_WIDTH` / `BEV_HEIGHT` (fallback: `CAM_WIDTH` / `CAM_HEIGHT`)

### Run (camera publisher for demos)

If you don’t have a real camera node, use the built-in demo publisher to feed JPEG frames into
`camera/front/frames`.

```bash
# Generate synthetic JPEG frames (no assets needed)
cargo run --bin camera_publisher

# Publish a single JPEG repeatedly
CAM_PUB_PATH=./frame.jpg CAM_PUB_FPS=30 cargo run --bin camera_publisher

# Publish a single PNG repeatedly (set CAM_WIDTH/CAM_HEIGHT to match its resolution)
CAM_PUB_PATH=./frame.png CAM_PUB_FPS=30 cargo run --bin camera_publisher

# Or publish a directory of JPEG frames (sorted by filename)
CAM_PUB_PATH=./frames CAM_PUB_FPS=30 cargo run --bin camera_publisher
```

Extract frames from a video (example):

```bash
mkdir -p frames
ffmpeg -i input.mp4 -vf "fps=30,scale=1920:1080" -q:v 2 frames/frame_%05d.jpg
```

Make sure `CAM_WIDTH`/`CAM_HEIGHT` match the frame resolution (otherwise JPEG decode will fail with a
resolution mismatch).

Env:

- `CAM_PUB_TOPIC` (default: `camera/front/frames`)
- `CAM_PUB_PATH` (optional): JPEG file or directory of `.jpg`/`.jpeg` (otherwise generates frames)
- `CAM_PUB_FPS` (default: `30`)
- `CAM_PUB_LOOP` (default: `1`)
- `CAM_PUB_WIDTH` / `CAM_PUB_HEIGHT` (fallback: `CAM_WIDTH` / `CAM_HEIGHT`, default: 1920×1080)
- `CAM_PUB_GEN_FRAMES` (default: `10`)
- `CAM_PUB_GEN_QUALITY` (default: `85`)

### Benchmark (stage latencies)

This prints per-stage latency (ms) for:
JPEG decode (CPU), RGB pack, H2D, kernel, D2H, RGB unpack, JPEG encode (CPU).

```bash
BEV_BENCH=1 BEV_BENCH_WARMUP=30 BEV_BENCH_EVERY=60 BEV_LOOP_DELAY_MS=0 cargo run --release --bin gpu-bev-node
```

Notes:

- In benchmark mode the node forces sync points to isolate H2D/kernel/D2H timings.
- Reported `total` is the sum of measured stages (does not include Zenoh publish).

### Demo capture (screenshots + video)

Suggested setup (3 terminals):
1. Publisher: `CAM_PUB_PATH=./frames CAM_PUB_FPS=30 cargo run --bin camera_publisher`
2. Node (bench on): `BEV_BENCH=1 BEV_BENCH_WARMUP=30 BEV_BENCH_EVERY=60 BEV_LOOP_DELAY_MS=0 cargo run --release --bin gpu-bev-node`
3. Viewer: `cargo run --release --features viewer --bin bev_viewer`

Record 30–60 seconds showing:
- the BEV window updating, and
- the terminal printing `[GPU-BEV][BENCH] ...` latency lines.

### Troubleshooting

- **Node not receiving frames** (publisher prints `sent=...` but `gpu-bev-node` shows no bench lines):
  Zenoh default discovery uses multicast. If multicast is blocked on your network, run everything via
  an explicit router and config:

  1. Start a router (if installed): `zenohd -l tcp/127.0.0.1:7447`
  2. Create a config file `zenoh_client.json5`:

     ```json5
     { mode: "client", connect: { endpoints: ["tcp/127.0.0.1:7447"] } }
     ```

  3. Run all binaries with `ZENOH_CONFIG=zenoh_client.json5 ...`

- **JPEG decode “resolution mismatch”**: ensure your input JPEG resolution matches `CAM_WIDTH`/`CAM_HEIGHT`.
- **Viewer doesn’t open**: it requires a desktop session; run `cargo run --features viewer --bin bev_viewer`.

### Known bottleneck (CPU JPEG)

In the current PoC, JPEG decode/encode runs on the CPU. At 1080p this can dominate the per-frame
latency, even when the GPU kernel itself executes in under ~1ms. As a result, the GPU may spend a
large fraction of time idle waiting for CPU-side compression.

This is intentional for the PoC and should be called out in the GSoC proposal as the primary
pipeline bottleneck and an opportunity for future work:

- Add double-buffering/pipelining to overlap CPU encode/decode of frame `N-1` with GPU warp of frame `N`.
- Offload encoding/decoding to hardware accelerators (e.g., NVENC/NVJPEG, VAAPI) where available.
- Use Zenoh shared memory for raw frames to avoid network re-serialization/compression when feasible.
