import os
import glob
import re

def check_splits():
    train_files = glob.glob("data/real_palm_dataset/PalmDataset/Palm/Train/**/*.jpg", recursive=True)
    test_files = glob.glob("data/real_palm_dataset/PalmDataset/Palm/Test/**/*.jpg", recursive=True)
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files:  {len(test_files)}")
    
    pattern = re.compile(r"s(\d+)_(\d+)_([LR])_S([12])", re.IGNORECASE)
    
    train_palms = set()
    for f in train_files:
        m = pattern.search(os.path.basename(f))
        if m:
            sub = int(m.group(1))
            hand = m.group(3).upper()
            train_palms.add((sub, hand))
            
    test_palms = set()
    for f in test_files:
        m = pattern.search(os.path.basename(f))
        if m:
            sub = int(m.group(1))
            hand = m.group(3).upper()
            test_palms.add((sub, hand))
            
    print(f"Train unique palms: {len(train_palms)}")
    print(f"Test unique palms:  {len(test_palms)}")
    overlap = train_palms.intersection(test_palms)
    print(f"Overlap between Train and Test: {len(overlap)}")
    if overlap:
        print(f"Sample overlapping palms: {list(overlap)[:5]}")

check_splits()

