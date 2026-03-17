use kornia_image::ImageSize;

#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    /// Camera roll in radians (right-hand rule about world X).
    pub roll_rad: f32,
    /// Camera pitch in radians (right-hand rule about world Y).
    pub pitch_rad: f32,
    /// Camera yaw in radians (right-hand rule about world Z).
    pub yaw_rad: f32,
    /// Camera height above the ground plane in meters.
    pub height_m: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct BevRoi {
    /// Forward range in meters.
    pub x_min_m: f32,
    pub x_max_m: f32,
    /// Lateral range in meters (positive = left).
    pub y_min_m: f32,
    pub y_max_m: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct IpmConfig {
    pub output_size: ImageSize,
    pub intrinsics: CameraIntrinsics,
    pub pose: CameraPose,
    pub roi: BevRoi,
}

impl IpmConfig {
    pub fn from_env(input_size: ImageSize, output_size: ImageSize) -> Self {
        let fx = env_f32("CAM_FX").unwrap_or_else(|| {
            let fov_x_deg = env_f32("CAM_FOV_X_DEG").unwrap_or(90.0).clamp(1.0, 179.0);
            let fov_x = fov_x_deg.to_radians();
            input_size.width as f32 / (2.0 * (fov_x * 0.5).tan())
        });
        let fy = env_f32("CAM_FY").unwrap_or_else(|| {
            let fov_y_deg = env_f32("CAM_FOV_Y_DEG").unwrap_or(60.0).clamp(1.0, 179.0);
            let fov_y = fov_y_deg.to_radians();
            input_size.height as f32 / (2.0 * (fov_y * 0.5).tan())
        });
        let cx = env_f32("CAM_CX").unwrap_or(input_size.width as f32 * 0.5);
        let cy = env_f32("CAM_CY").unwrap_or(input_size.height as f32 * 0.5);

        let height_m = env_f32("CAM_HEIGHT_M").unwrap_or(1.5).max(0.01);
        let roll_rad = env_f32("CAM_ROLL_DEG").unwrap_or(0.0).to_radians();
        let pitch_rad = env_f32("CAM_PITCH_DEG").unwrap_or(-10.0).to_radians();
        let yaw_rad = env_f32("CAM_YAW_DEG").unwrap_or(0.0).to_radians();

        let x_min_m = env_f32("BEV_X_MIN_M").unwrap_or(0.0);
        let x_max_m = env_f32("BEV_X_MAX_M").unwrap_or(30.0);
        let y_min_m = env_f32("BEV_Y_MIN_M").unwrap_or(-10.0);
        let y_max_m = env_f32("BEV_Y_MAX_M").unwrap_or(10.0);

        Self {
            output_size,
            intrinsics: CameraIntrinsics { fx, fy, cx, cy },
            pose: CameraPose {
                roll_rad,
                pitch_rad,
                yaw_rad,
                height_m,
            },
            roi: BevRoi {
                x_min_m,
                x_max_m,
                y_min_m,
                y_max_m,
            },
        }
    }
}

/// Compute a homography that maps BEV output pixel coordinates `(u, v)` to input image pixel
/// coordinates `(x, y)` (inverse mapping suitable for texture sampling).
///
/// # Conventions
/// - World frame: X forward, Y left, Z up. Ground plane is `Z = 0`.
/// - Camera frame (OpenCV): x right, y down, z forward.
/// - Camera center is at `(0, 0, height_m)` in the world frame.
/// - Output image coordinates: `u` increases to the right, `v` increases downward.
///
/// This implementation is intentionally minimal and assumes a pinhole camera (no distortion).
pub fn bev_out_to_img_homography(cfg: &IpmConfig) -> [f32; 9] {
    let k = [
        cfg.intrinsics.fx,
        0.0,
        cfg.intrinsics.cx,
        0.0,
        cfg.intrinsics.fy,
        cfg.intrinsics.cy,
        0.0,
        0.0,
        1.0,
    ];

    let r_align_wc = [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0];
    let r_align_cw = mat3_transpose(r_align_wc);

    let r_world = mat3_mul(
        mat3_mul(rot_z(cfg.pose.yaw_rad), rot_y(cfg.pose.pitch_rad)),
        rot_x(cfg.pose.roll_rad),
    );

    let r_cw = mat3_mul(r_world, r_align_cw);
    let r_wc = mat3_transpose(r_cw);

    let c_world = [0.0f32, 0.0f32, cfg.pose.height_m];
    let t = mat3_vec_mul(r_wc, [-c_world[0], -c_world[1], -c_world[2]]);

    let h_plane = [
        r_wc[0], r_wc[1], t[0], r_wc[3], r_wc[4], t[1], r_wc[6], r_wc[7], t[2],
    ];
    let h_world_to_img = mat3_mul(k, h_plane);

    let w = cfg.output_size.width.max(2) as f32;
    let h = cfg.output_size.height.max(2) as f32;
    let x_span = cfg.roi.x_max_m - cfg.roi.x_min_m;
    let y_span = cfg.roi.y_max_m - cfg.roi.y_min_m;

    let sx = -x_span / (h - 1.0);
    let sy = -y_span / (w - 1.0);
    let a_out_to_world = [
        0.0,
        sx,
        cfg.roi.x_max_m,
        sy,
        0.0,
        cfg.roi.y_max_m,
        0.0,
        0.0,
        1.0,
    ];

    let mut h_out_to_img = mat3_mul(h_world_to_img, a_out_to_world);

    let w33 = h_out_to_img[8];
    if w33 != 0.0 {
        for v in &mut h_out_to_img {
            *v /= w33;
        }
    }

    h_out_to_img
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok()?.parse::<f32>().ok()
}

fn mat3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for row in 0..3 {
        let row_base = row * 3;
        for col in 0..3 {
            out[row_base + col] =
                a[row_base] * b[col] + a[row_base + 1] * b[3 + col] + a[row_base + 2] * b[6 + col];
        }
    }
    out
}

fn mat3_transpose(a: [f32; 9]) -> [f32; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

fn mat3_vec_mul(a: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

fn rot_x(angle_rad: f32) -> [f32; 9] {
    let (s, c) = angle_rad.sin_cos();
    [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
}

fn rot_y(angle_rad: f32) -> [f32; 9] {
    let (s, c) = angle_rad.sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

fn rot_z(angle_rad: f32) -> [f32; 9] {
    let (s, c) = angle_rad.sin_cos();
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}
