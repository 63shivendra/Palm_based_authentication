import numpy as np

c1 = np.ones((128, 128), dtype=np.float32) * 0.95
c1[58:70, 20:108] = 0.5

def warp_tilt(c, pitch, roll):
    h, w = 128, 128
    cx, cy = w * 0.5, h * 0.5
    rx = np.radians(pitch)
    ry = np.radians(roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    R = Ry @ Rx
    f = 300.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    H_inv = K @ R.T @ np.linalg.inv(K)
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dest_pts = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=0).reshape(3, -1)
    src_pts = H_inv @ dest_pts
    src_x = (src_pts[0] / (src_pts[2] + 1e-8)).reshape(h, w)
    src_y = (src_pts[1] / (src_pts[2] + 1e-8)).reshape(h, w)
    valid_mask = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 2)
    wx1 = np.clip(src_x - x0, 0.0, 1.0)
    wy1 = np.clip(src_y - y0, 0.0, 1.0)
    warped = (1.0 - wy1) * ((1.0 - wx1) * c[y0, x0] + wx1 * c[y0, x0+1]) + wy1 * ((1.0 - wx1) * c[y0+1, x0] + wx1 * c[y0+1, x0+1])
    return np.where(valid_mask, warped, 0.95)

w_ref = warp_tilt(c1, 0.0, 0.0)
w_pos = warp_tilt(c1, 4.0, 3.0)
w_neg = warp_tilt(c1, -4.0, -3.0)

print("Correlation at realistic natural tilt (+4° vs ref 0°):", np.corrcoef(w_ref.flatten(), w_pos.flatten())[0, 1])
print("Correlation between (+4° vs -4°):                    ", np.corrcoef(w_pos.flatten(), w_neg.flatten())[0, 1])

