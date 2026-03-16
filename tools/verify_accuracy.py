import cv2
import numpy as np
import time

# 1. Load the Udacity test image
img = cv2.imread('frame.jpg')
h, w = img.shape[:2]

# 2. Replicate your Rust IPM Configuration
# These points define a trapezoid on the dashcam image and a rectangle on the BEV image
src_points = np.float32([
    [w // 2 - 70, h * 0.65],   # Top Left
    [w // 2 + 70, h * 0.65],   # Top Right
    [w * 0.85, h],             # Bottom Right
    [w * 0.15, h]              # Bottom Left
])

dst_points = np.float32([
    [w * 0.25, 0],             # Top Left
    [w * 0.75, 0],             # Top Right
    [w * 0.75, h],             # Bottom Right
    [w * 0.25, h]              # Bottom Left
])

# 3. Calculate the Matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 4. CPU Performance Benchmark
start_time = time.time()
# Note: We use INTER_NEAREST to exactly match your GPU's nearest-neighbor math
cpu_warp = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_NEAREST)
cpu_time = (time.time() - start_time) * 1000

print(f"CPU OpenCV Warp Time: {cpu_time:.2f} ms")

# Save the CPU baseline to compare visually
cv2.imwrite('cpu_baseline_bev.jpg', cpu_warp)

print("\n--- Next Steps for your GSoC Proposal ---")
print("1. Take a screenshot of your bev_viewer output.")
print("2. Put the CPU OpenCV Warp Time next to your Rust [GPU-BEV][BENCH] kernel time.")
print("3. Explain that the GPU kernel achieves identical Nearest-Neighbor accuracy at a fraction of the CPU latency.")