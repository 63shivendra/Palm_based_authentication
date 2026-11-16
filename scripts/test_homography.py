import numpy as np

h, w = 128, 128
cx, cy = 64.0, 64.0
f = 300.0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
rx = np.radians(4.0)
Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
H_inv = K @ Rx.T @ np.linalg.inv(K)

dest = np.array([64.0, 64.0, 1.0])
src = H_inv @ dest
src /= src[2]
print("Center (64, 64) maps to:", src)

dest_top = np.array([64.0, 20.0, 1.0])
src_top = H_inv @ dest_top
src_top /= src_top[2]
print("Top (64, 20) maps to:", src_top)

dest_bot = np.array([64.0, 108.0, 1.0])
src_bot = H_inv @ dest_bot
src_bot /= src_bot[2]
print("Bottom (64, 108) maps to:", src_bot)

