import os
import zipfile

zip_path = "data/PalmMatchDB.zip"
extract_dir = "data/PalmMatchDB"
os.makedirs(extract_dir, exist_ok=True)

print(f"[*] Opening {zip_path}...")
with zipfile.ZipFile(zip_path, "r") as zf:
    namelist = zf.namelist()
    print(f"[*] Total files in archive: {len(namelist):,}")
    print("[*] Sample files in archive:")
    for name in namelist[:20]:
        print(" ", name)
        
    print(f"[*] Extracting to {extract_dir}...")
    zf.extractall(extract_dir)

print("[+] Extraction complete!")

