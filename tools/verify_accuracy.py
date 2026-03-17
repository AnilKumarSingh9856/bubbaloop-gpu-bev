import cv2
import numpy as np
import time
import os
from pathlib import Path


def env_f32(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def deg_to_rad(deg: float) -> float:
    return deg * np.pi / 180.0


def rot_x(a: float) -> np.ndarray:
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)


def rot_z(a: float) -> np.ndarray:
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def rust_bev_out_to_img_homography(width: int, height: int) -> np.ndarray:
    fov_x_deg = np.clip(env_f32("CAM_FOV_X_DEG", 90.0), 1.0, 179.0)
    fov_y_deg = np.clip(env_f32("CAM_FOV_Y_DEG", 60.0), 1.0, 179.0)

    fx_default = width / (2.0 * np.tan(deg_to_rad(fov_x_deg) * 0.5))
    fy_default = height / (2.0 * np.tan(deg_to_rad(fov_y_deg) * 0.5))
    fx = env_f32("CAM_FX", float(fx_default))
    fy = env_f32("CAM_FY", float(fy_default))
    cx = env_f32("CAM_CX", width * 0.5)
    cy = env_f32("CAM_CY", height * 0.5)

    height_m = max(env_f32("CAM_HEIGHT_M", 1.5), 0.01)
    roll = deg_to_rad(env_f32("CAM_ROLL_DEG", 0.0))
    pitch = deg_to_rad(env_f32("CAM_PITCH_DEG", -10.0))
    yaw = deg_to_rad(env_f32("CAM_YAW_DEG", 0.0))

    x_min = env_f32("BEV_X_MIN_M", 0.0)
    x_max = env_f32("BEV_X_MAX_M", 30.0)
    y_min = env_f32("BEV_Y_MIN_M", -10.0)
    y_max = env_f32("BEV_Y_MAX_M", 10.0)

    k = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    r_align_wc = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    r_align_cw = r_align_wc.T

    r_world = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    r_cw = r_world @ r_align_cw
    r_wc = r_cw.T

    c_world = np.array([0.0, 0.0, height_m], dtype=np.float32)
    t = r_wc @ (-c_world)

    h_plane = np.array([
        [r_wc[0, 0], r_wc[0, 1], t[0]],
        [r_wc[1, 0], r_wc[1, 1], t[1]],
        [r_wc[2, 0], r_wc[2, 1], t[2]],
    ], dtype=np.float32)
    h_world_to_img = k @ h_plane

    w = float(max(width, 2))
    h = float(max(height, 2))
    x_span = x_max - x_min
    y_span = y_max - y_min
    sx = -x_span / (h - 1.0)
    sy = -y_span / (w - 1.0)

    a_out_to_world = np.array([
        [0.0, sx, x_max],
        [sy, 0.0, y_max],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    h_out_to_img = h_world_to_img @ a_out_to_world
    if h_out_to_img[2, 2] != 0.0:
        h_out_to_img = h_out_to_img / h_out_to_img[2, 2]

    return h_out_to_img.astype(np.float32)

# 1. Load the test image from workspace input_images/
input_path = Path('input_images/frame.jpg')
output_path = Path('output_images/opencv_baseline_frame1.png')

img = cv2.imread(str(input_path))
if img is None:
    raise RuntimeError(f'Failed to read image: {input_path}')

h, w = img.shape[:2]

# 2. Use the same Rust IPM homography (BEV output pixel -> input pixel)
matrix = rust_bev_out_to_img_homography(w, h)

# 3. OpenCV benchmark using inverse mapping flag to match Rust kernel semantics
start_time = time.time()
cpu_warp = cv2.warpPerspective(
    img,
    matrix,
    (w, h),
    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
)
cpu_time = (time.time() - start_time) * 1000

print(f"OpenCV Warp Time (Rust matrix): {cpu_time:.2f} ms")

# Save the OpenCV baseline to compare with Rust output
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), cpu_warp)
print(f"Saved OpenCV baseline: {output_path}")
print("Reference baseline generation complete.")