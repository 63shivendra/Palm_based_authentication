import glob
import re

files = glob.glob("data/real_palm_dataset/**/*.jpg", recursive=True)
sessions = set()
for f in files:
    m = re.search(r"S\d+", f)
    if m:
        sessions.add(m.group(0))
print("Sessions found in real_palm_dataset:", sessions)

