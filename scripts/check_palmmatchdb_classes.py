import os
import glob
from collections import defaultdict

def check_palmmatchdb():
    files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
    print(f"Total PNGs in PalmMatchDB: {len(files)}")
    
    classes = defaultdict(list)
    distances = defaultdict(list)
    hands = defaultdict(list)
    
    for f in files:
        parts = f.replace('\\', '/').split('/')
        # Look for subject id, distance, hand
        # data/PalmMatchDB/PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F101(15-24)/F12MB/left/...
        try:
            sub = [p for p in parts if ('F' in p or 'M' in p) and '(' in p][0]
            dist = [p for p in parts if '5M' in p.upper() or '12M' in p.upper()][0]
            hand = [p for p in parts if 'left' in p.lower() or 'right' in p.lower()][0]
            class_id = f"{sub}_{hand}"
            classes[class_id].append(f)
            distances[dist].append(f)
            hands[hand].append(f)
        except Exception as e:
            pass
            
    print(f"Total parsed classes in PalmMatchDB: {len(classes)}")
    print(f"Distances distribution: { {k: len(v) for k, v in distances.items()} }")
    print(f"Hands distribution: { {k: len(v) for k, v in hands.items()} }")
    print(f"Sample classes and image counts:")
    for k in list(classes.keys())[:10]:
        print(f"  {k}: {len(classes[k])} images")

check_palmmatchdb()

