import os
import glob
import re
from collections import defaultdict

files = glob.glob("data/real_palm_dataset/**/*.jpg", recursive=True)
print(f"Total files in real_palm_dataset: {len(files)}")

subjects = defaultdict(list)
for f in files:
    fname = os.path.basename(f)
    # Match sXXX_(...)_L or R
    m = re.match(r"(s\d+)_(\d+)_([LR])_(S\d+)\.jpg", fname)
    if m:
        sub, idx, hand, sess = m.groups()
        class_id = f"{sub}_{hand}"
        subjects[class_id].append((sess, idx, f))
    else:
        print("Unmatched:", fname)

print(f"Total unique palm classes (Subject + Hand): {len(subjects)}")
print(f"Sample classes and image counts:")
for k in list(subjects.keys())[:10]:
    print(f"  Class {k}: {len(subjects[k])} images, sessions: {[x[0] for x in subjects[k]]}")

