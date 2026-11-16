import numpy as np
from scipy.ndimage import gaussian_filter

c1 = np.ones((128, 128), dtype=np.float32) * 0.95
# draw a crease
c1[60:68, :] = 0.5

# Test pose warp with 0.0 vs skin_val
def warp_test(fill_val):
    h, w = 128, 128
    cx, cy = w * 0.5, h * 0.5
    rx = np.radians(10.0)
    ry = np.radians(8.0)
    rz = 0.0
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
    warped = (1.0 - wy1) * ((1.0 - wx1) * c1[y0, x0] + wx1 * c1[y0, x0+1]) + wy1 * ((1.0 - wx1) * c1[y0+1, x0] + wx1 * c1[y0+1, x0+1])
    return np.where(valid_mask, warped, fill_val)

w_zero = warp_test(0.0)
w_skin = warp_test(0.95)

# Correlation with another sample of the same line with different tilt (e.g. -8 deg)
def warp_test2(fill_val):
    h, w = 128, 128
    cx, cy = w * 0.5, h * 0.5
    rx = np.radians(-8.0)
    ry = np.radians(4.0)
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
    warped = (1.0 - wy1) * ((1.0 - wx1) * c1[y0, x0] + wx1 * c1[y0, x0+1]) + wy1 * ((1.0 - wx1) * c1[y0+1, x0] + wx1 * c1[y0+1, x0+1])
    return np.where(valid_mask, warped, fill_val)

w_zero2 = warp_test2(0.0)
w_skin2 = warp_test2(0.95)

print("Correlation with 0.0 fill (current bug):", np.corrcoef(w_zero.flatten(), w_zero2.flatten())[0, 1])
print("Correlation with skin reflection fill: ", np.corrcoef(w_skin.flatten(), w_skin2.flatten())[0, 1])

