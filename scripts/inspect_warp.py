import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.generators.procedural_palm import CleanRoomPalmGenerator

gen = CleanRoomPalmGenerator(base_seed=42)

# Check what generate_surface_creases returns
rng1 = np.random.RandomState(100)
c1 = gen.generate_surface_creases(rng1)

rng2 = np.random.RandomState(200)
c2 = gen.generate_surface_creases(rng2)

print("Crease 1 mean:", c1.mean(), "min:", c1.min(), "max:", c1.max())
print("Crease 2 mean:", c2.mean(), "min:", c2.min(), "max:", c2.max())
print("Crease corr between two random calls:", np.corrcoef(c1.flatten(), c2.flatten())[0, 1])

# Check apply_3d_pose_warp!
w1 = gen.apply_3d_pose_warp(c1[:, :, None], pitch_deg=3.0, roll_deg=2.0, yaw_deg=0.0)[:, :, 0]
w2 = gen.apply_3d_pose_warp(c1[:, :, None], pitch_deg=-4.0, roll_deg=1.0, yaw_deg=0.0)[:, :, 0]
print("apply_3d_pose_warp corr on SAME crease c1 (pitch=3 vs pitch=-4):", np.corrcoef(w1.flatten(), w2.flatten())[0, 1])
print("Zero pixels in w1:", np.count_nonzero(w1 == 0), "in w2:", np.count_nonzero(w2 == 0))

