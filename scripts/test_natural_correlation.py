import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from common.generators.procedural_palm import CleanRoomPalmGenerator

gen = CleanRoomPalmGenerator(base_seed=42)

# Identity 0
id0_seed = 42000
rng_id0 = np.random.RandomState(id0_seed)
melanin0 = float(rng_id0.uniform(0.20, 0.45))

# Presentation 1 of ID 0 (subtle tilt, minor lighting)
p1 = gen.generate_pair(identity_seed=id0_seed, presentation_seed=101, melanin_fraction=melanin0, pitch_deg=3.0, roll_deg=2.0)
# Presentation 2 of ID 0 (another subtle tilt)
p2 = gen.generate_pair(identity_seed=id0_seed, presentation_seed=102, melanin_fraction=melanin0, pitch_deg=-4.0, roll_deg=1.0)

# Identity 1
id1_seed = 42001
rng_id1 = np.random.RandomState(id1_seed)
melanin1 = float(rng_id1.uniform(0.20, 0.45))
p_other = gen.generate_pair(identity_seed=id1_seed, presentation_seed=201, melanin_fraction=melanin1, pitch_deg=2.0, roll_deg=1.0)

s1 = p1["rgb"]
s2 = p2["rgb"]
s_other = p_other["rgb"]

corr_same = np.corrcoef(s1.flatten(), s2.flatten())[0, 1]
corr_diff = np.corrcoef(s1.flatten(), s_other.flatten())[0, 1]

print(f"Realistic Natural Presentation:")
print(f"  Pixel Correlation Same Identity:      {corr_same:.4f}")
print(f"  Pixel Correlation Different Identity: {corr_diff:.4f}")
