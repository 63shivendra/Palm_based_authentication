import os
import glob
from collections import Counter

print("Exploring data/PalmMatchDB:")
for root, dirs, files in os.walk("data/PalmMatchDB"):
    if files:
        exts = Counter([os.path.splitext(f)[1].lower() for f in files])
        print(f"{root}: {len(files)} files, exts: {dict(exts)}")
        print(f"  sample files: {files[:3]}")
        break

total_files = 0
all_exts = Counter()
for root, dirs, files in os.walk("data/PalmMatchDB"):
    total_files += len(files)
    for f in files:
        all_exts[os.path.splitext(f)[1].lower()] += 1

print(f"Total files in data/PalmMatchDB: {total_files}")
print(f"Extension counts: {dict(all_exts)}")

