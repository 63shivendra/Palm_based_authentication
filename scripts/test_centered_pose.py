import numpy as np

# Centered 3D palm rotation around palm center (cx, cy, d)
def warp_centered_3d(image, pitch_deg, roll_deg, yaw_deg, d=300.0):
    h, w = image.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    
    rx = np.radians(pitch_deg)
    ry = np.radians(roll_deg)
    rz = np.radians(yaw_deg)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    
    # 3D points on palm surface relative to palm center: (X, Y, 0)
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    X = x_coords - cx
    Y = y_coords - cy
    Z = np.zeros_like(X)
    
    # Rotate around palm center: P' = R @ [X, Y, 0]^T
    # We want backward mapping: for each destination pixel (xd, yd), where did it come from?
    # R_inv = R.T
    R_inv = R.T
    
    # Perspective projection: x_proj = d * X' / (d + Z') + cx, y_proj = d * Y' / (d + Z') + cy
    # Or affine approximation with perspective foreshortening:
    # Under backward mapping:
    pts_dest = np.stack([X.flatten(), Y.flatten(), np.zeros(h*w, dtype=np.float32)], axis=0)
    pts_src = R_inv @ pts_dest
    
    # Foreshortened coordinates
    Z_src = pts_src[2]
    src_x = (pts_src[0] / (1.0 + Z_src / d) + cx).reshape(h, w)
    src_y = (pts_src[1] / (1.0 + Z_src / d) + cy).reshape(h, w)
    
    valid_mask = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 2)
    wx1 = np.clip(src_x - x0, 0.0, 1.0)
    wy1 = np.clip(src_y - y0, 0.0, 1.0)
    
    if image.ndim == 3:
        pass
    else:
        warped = (1.0 - wy1) * ((1.0 - wx1) * image[y0, x0] + wx1 * image[y0, x0+1]) + wy1 * ((1.0 - wx1) * image[y0+1, x0] + wx1 * image[y0+1, x0+1])
        fill_val = float(np.median(image))
        return np.where(valid_mask, warped, fill_val)

c1 = np.ones((128, 128), dtype=np.float32) * 0.95
c1[58:70, 20:108] = 0.5 # horizontal line

w0 = warp_centered_3d(c1, 0, 0, 0)
w1 = warp_centered_3d(c1, 15, 10, 5)
w2 = warp_centered_3d(c1, -15, -10, -5)

print("Correlation with centered 3D rotation (+15° vs 0°): ", np.corrcoef(w0.flatten(), w1.flatten())[0, 1])
print("Correlation between (+15° vs -15°):                 ", np.corrcoef(w1.flatten(), w2.flatten())[0, 1])
print("Center (64, 64) value in w0:", w0[64, 64], "in w1:", w1[64, 64], "in w2:", w2[64, 64])

