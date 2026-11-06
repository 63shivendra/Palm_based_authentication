import os
import urllib.request
import ssl
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://huggingface.co/datasets/aspmirlab/PalmMatchDB/resolve/main/PalmMatchDB.zip"
dest_dir = "data"
dest_path = os.path.join(dest_dir, "PalmMatchDB.zip")
os.makedirs(dest_dir, exist_ok=True)

print(f"[*] Starting download of PalmMatchDB (1.17 GB) to {dest_path}...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

# Check resume if partially downloaded
initial_size = 0
mode = "wb"
if os.path.exists(dest_path):
    initial_size = os.path.getsize(dest_path)
    if initial_size > 0:
        req.headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        print(f"[*] Resuming from byte {initial_size} ({initial_size / (1024*1024):.1f} MB)...")

start_t = time.time()
with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(dest_path, mode) as f:
    total_size = int(resp.headers.get("Content-Length", 0)) + initial_size
    downloaded = initial_size
    chunk_size = 1024 * 1024 * 4 # 4MB chunks
    
    last_print = time.time()
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        
        if time.time() - last_print > 5.0 or downloaded >= total_size:
            elapsed = time.time() - start_t
            speed_mb = (downloaded - initial_size) / (elapsed * 1024 * 1024) if elapsed > 0 else 0
            pct = (downloaded / total_size) * 100 if total_size > 0 else 0
            print(f"    Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s", flush=True)
            last_print = time.time()

print(f"[+] PalmMatchDB download complete! Saved to {dest_path}")

