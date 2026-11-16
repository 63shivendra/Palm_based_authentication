import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indipalm_edge.training.train_edge_model import MultiSpectralIdentityDataset

ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=5, seed=42)

# Sample 0 and Sample 1 are from Identity 0
s0 = ds[0]["rgb"].numpy() # (3, 128, 128)
s1 = ds[1]["rgb"].numpy() # (3, 128, 128)

# Sample 5 is from Identity 1
s5 = ds[5]["rgb"].numpy() # (3, 128, 128)

diff_same = np.mean(np.abs(s0 - s1))
diff_diff = np.mean(np.abs(s0 - s5))

print(f"Mean Abs Pixel Diff - Same Identity (variant 0 vs 1): {diff_same:.4f}")
print(f"Mean Abs Pixel Diff - Different Identity (ID 0 vs ID 1): {diff_diff:.4f}")

# Check correlation of RGB flattened
corr_same = np.corrcoef(s0.flatten(), s1.flatten())[0, 1]
corr_diff = np.corrcoef(s0.flatten(), s5.flatten())[0, 1]
print(f"Pixel Pearson Corr - Same Identity: {corr_same:.4f}")
print(f"Pixel Pearson Corr - Different Identity: {corr_diff:.4f}")

