import os
import glob
import re
from collections import defaultdict

base_dir = "data/PalmMatchDB"
all_images = glob.glob(f"{base_dir}/**/*.png", recursive=True)
print(f"Total PNG images: {len(all_images):,}")

# Parse individual palms (Subject ID + Hand)
palms = defaultdict(list)
for p in all_images:
    rel = os.path.relpath(p, base_dir).replace("\\", "/")
    parts = rel.split("/")
    # Structure:
    # PalmMatchDB / <GENDER_FOLDER> / <AGE_GROUP> / <SUBJECT_ID> / <SESSION> / <HAND> / <FILENAME>
    # e.g.: PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F101(15-24)/F12MB/left/ei_...png
    
    # Extract subject ID
    sub_id = None
    hand = None
    for part in parts:
        # Match F101 or M101 or m101 or similar
        m_sub = re.match(r"^([FMfm]\d+)", part)
        if m_sub and sub_id is None:
            sub_id = m_sub.group(1).upper()
        if part.lower() in ["left", "right"]:
            hand = part.lower()
            
    if sub_id is None:
        # Fallback to parent directory heuristics
        for part in parts:
            if "age" in part.lower() and any(c.isdigit() for c in part):
                sub_id = part.strip()
                break
                
    if hand is None:
        if "left" in rel.lower():
            hand = "left"
        elif "right" in rel.lower():
            hand = "right"
        else:
            hand = "unknown"
            
    palm_key = f"{sub_id}_{hand}"
    palms[palm_key].append(p)

print(f"Total unique palm classes parsed: {len(palms)}")
sample_palms = list(palms.keys())[:10]
for pk in sample_palms:
    print(f"  Palm ID '{pk}': {len(palms[pk])} presentations")

