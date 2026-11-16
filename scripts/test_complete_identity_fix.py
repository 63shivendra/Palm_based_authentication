import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.generators.procedural_palm import CleanRoomPalmGenerator

gen = CleanRoomPalmGenerator(base_seed=42)

# Update apply_3d_pose_warp with centered rotation in memory
def apply_3d_pose_warp_centered(self, image, pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0, d=300.0):
    if abs(pitch_deg) < 0.5 and abs(roll_deg) < 0.5 and abs(yaw_deg) < 0.5:
        return image
    h, w = self.height, self.width
    cx, cy = w * 0.5, h * 0.5
    rx = np.radians(pitch_deg)
    ry = np.radians(roll_deg)
    rz = np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    R_inv = (Rz @ Ry @ Rx).T
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    X = x_coords - cx
    Y = y_coords - cy
    pts_dest = np.stack([X.flatten(), Y.flatten(), np.zeros(h*w, dtype=np.float32)], axis=0)
    pts_src = R_inv @ pts_dest
    Z_src = pts_src[2]
    src_x = (pts_src[0] / (1.0 + Z_src / d) + cx).reshape(h, w)
    src_y = (pts_src[1] / (1.0 + Z_src / d) + cy).reshape(h, w)
    valid_mask = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 2)
    wx1 = np.clip(src_x - x0, 0.0, 1.0)
    wy1 = np.clip(src_y - y0, 0.0, 1.0)
    if image.ndim == 3:
        wx0 = (1.0 - wx1)[:, :, None]
        wx1 = wx1[:, :, None]
        wy0 = (1.0 - wy1)[:, :, None]
        wy1 = wy1[:, :, None]
        valid_mask = valid_mask[:, :, None]
        warped = wy0 * (wx0 * image[y0, x0, :] + wx1 * image[y0, x0+1, :]) + wy1 * (wx0 * image[y0+1, x0, :] + wx1 * image[y0+1, x0+1, :])
        fill = np.median(image, axis=(0, 1), keepdims=True)
        return np.where(valid_mask, warped, fill).astype(image.dtype)
    else:
        warped = (1.0 - wy1) * ((1.0 - wx1) * image[y0, x0] + wx1 * image[y0, x0+1]) + wy1 * ((1.0 - wx1) * image[y0+1, x0] + wx1 * image[y0+1, x0+1])
        fill = float(np.median(image))
        return np.where(valid_mask, warped, fill).astype(image.dtype)

CleanRoomPalmGenerator.apply_3d_pose_warp = apply_3d_pose_warp_centered

# Generate 3 samples of ID 0 with different 3D tilts
id0_seed = 100
rng_id0 = np.random.RandomState(id0_seed)
mel0 = float(rng_id0.uniform(0.20, 0.45))
p0_a = gen.generate_pair(identity_seed=id0_seed, presentation_seed=1, melanin_fraction=mel0, pitch_deg=8.0, roll_deg=5.0)
p0_b = gen.generate_pair(identity_seed=id0_seed, presentation_seed=2, melanin_fraction=mel0, pitch_deg=-8.0, roll_deg=-5.0)

# Generate sample of ID 1
id1_seed = 200
rng_id1 = np.random.RandomState(id1_seed)
mel1 = float(rng_id1.uniform(0.20, 0.45))
p1_a = gen.generate_pair(identity_seed=id1_seed, presentation_seed=3, melanin_fraction=mel1, pitch_deg=4.0, roll_deg=2.0)

s0_a = p0_a["rgb"]
s0_b = p0_b["rgb"]
s1_a = p1_a["rgb"]

corr_genuine = np.corrcoef(s0_a.flatten(), s0_b.flatten())[0, 1]
corr_impostor = np.corrcoef(s0_a.flatten(), s1_a.flatten())[0, 1]

print("=" * 60)
print(f"GENUINE (Same Person across 3D tilts +8° vs -8°): {corr_genuine:.4f}")
print(f"IMPOSTOR (Different People):                      {corr_impostor:.4f}")
print(f"Separation Margin (Genuine - Impostor):          {corr_genuine - corr_impostor:.4f}")
print("=" * 60)

