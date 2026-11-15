#!/usr/bin/env python3
"""
scripts/fetch_datasets.py - download the palm datasets used by benchmarks/run_benchmark.py, with live progress.

  python scripts/fetch_datasets.py --list            show every dataset, its licence and whether it is on disk
  python scripts/fetch_datasets.py all               download everything that is missing
  python scripts/fetch_datasets.py palmmatchdb gabor download only these

Everything goes to data/<Dataset>/ (git-ignored). data/MANIFEST.json records the source URL, licence,
archive SHA-256, file count and download date of each dataset, for the dataset assessment.

Licence column ("use"):
  permissive  licence allows research, development and redistribution (what the NPCI brief asks for)
  eval-only   no licence, research-only or unclear provenance: used here ONLY to measure accuracy of a
              matcher that was never trained on it; never for training, never redistributed
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(REPO, ".bench_env")
if os.path.isdir(ENV):
    sys.path.insert(0, ENV)

from rich.console import Console  # noqa: E402
from rich.progress import (BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn,  # noqa: E402
                           TransferSpeedColumn, MofNCompleteColumn)
from rich.table import Table  # noqa: E402

DATA = os.path.join(REPO, "data")
MANIFEST = os.path.join(DATA, "MANIFEST.json")
DL = os.path.join(DATA, "_downloads")
con = Console()
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DATASETS = {
    "palmmatchdb": dict(
        name="PalmMatchDB", dir="PalmMatchDB", use="permissive", licence="Apache-2.0 (Hugging Face dataset tag)",
        source="https://huggingface.co/datasets/aspmirlab/PalmMatchDB",
        content="phone photos of whole hands, 5/8/12 MP cameras; many duplicate files and unreliable labels",
        note="Consent terms of the subjects are not stated on the dataset page."),
    "gabor": dict(
        name="Palmprint Image Dataset with Gabor Filter (raw images)", dir="GaborPalm", use="permissive",
        licence="CC BY 4.0", source="https://data.mendeley.com/datasets/r8hxykxnk5/2",
        content="15 people, left hand, 5-8 raw palm crops each (the Gabor-filtered copies are not used)",
        note="Universitas Udayana. Very small: a smoke test, not an accuracy measurement."),
    "tongji": dict(
        name="Tongji Contactless Palmprint (ROI images)", dir="Tongji", use="eval-only",
        licence="none stated on the official page", source="https://cslinzhang.github.io/ContactlessPalm/",
        content="600 palms, 2 sessions about 61 days apart, 10 images per palm per session, 128x128 ROI",
        note="The standard benchmark in the literature, so it is needed for a like-for-like comparison."),
    "hewelxx": dict(
        name="Zenodo palmprint bundle: XINHUA, BJTU v2", dir="HewelXX", use="eval-only",
        licence="CC BY 4.0 on the Zenodo record, but the images are re-hosted copies of other databases",
        source="https://zenodo.org/records/15473268",
        content="XINHUA (100 palms, 20 photos each), BJTU palm v2 (small contactless set)",
        note="Original owners and consent terms are not documented; the CC BY tag of a re-upload cannot be relied on. "
             "The GitHub copy also holds PolyU II, whose terms forbid redistribution; it is not used."),
    "11khands": dict(
        name="11k Hands (palmar side)", dir="11kHands", use="eval-only",
        licence="free for reasonable academic fair use (project page)", source="https://sites.google.com/view/11khands",
        content="11,076 photos of 190 people, 1600x1200, dorsal and palmar; only palmar photos are used",
        note="Whole-hand photos on a plain background: tests hand detection and alignment end to end."),
}


# ------------------------------------------------------------------------------------------------ helpers
def progress():
    return Progress(TextColumn("[bold]{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                    TimeRemainingColumn(), console=con)


def http_download(url, dest, label):
    """Resumable HTTP download with a live progress bar. Returns the SHA-256 of the file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "palm-benchmark/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                         # nothing left to fetch: already complete
            con.print(f"[green]✓[/] {label} already downloaded")
            return sha256(dest)
        raise
    with r:
        if have and r.status != 206:                       # server ignored the range: start again
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        with progress() as p, open(dest, "ab" if have else "wb") as f:
            t = p.add_task(label, total=total or None, completed=have)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                p.advance(t, len(chunk))
    return sha256(dest)


def gdrive_download(file_id, dest, label):
    import gdown
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        con.print(f"[bold]{label}[/]: Google Drive file {file_id}")
        out = gdown.download(id=file_id, output=dest + ".part", quiet=False, resume=True)
        if not out:
            raise RuntimeError(f"Google Drive refused the download of {file_id} (quota or permission). "
                               f"Download it in a browser and save it as {dest}")
        os.replace(dest + ".part", dest)
    return sha256(dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def extract(archive, dest, label):
    os.makedirs(dest, exist_ok=True)
    if open(archive, "rb").read(4) == b"Rar!":              # Tongji's ".zip" is a RAR: Windows' tar (libarchive) reads it
        con.print(f"[bold]unpack {label}[/] (RAR, with tar) ...")
        subprocess.run(["tar", "-xf", archive, "-C", dest], check=True)
        return
    if archive.lower().endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(dest)
        return
    with zipfile.ZipFile(archive) as z, Progress(TextColumn("[bold]{task.description}"), BarColumn(),
                                                 MofNCompleteColumn(), TimeRemainingColumn(), console=con) as p:
        names = z.infolist()
        t = p.add_task(f"unzip {label}", total=len(names))
        for i in names:
            z.extract(i, dest)
            p.advance(t)


def count_images(d):
    return sum(f.lower().endswith(IMG) for _, _, fs in os.walk(d) for f in fs)


# ------------------------------------------------------------------------------------------------ fetchers
def fetch_palmmatchdb(d):
    url = "https://huggingface.co/datasets/aspmirlab/PalmMatchDB/resolve/main/PalmMatchDB.zip"
    z = os.path.join(DL, "PalmMatchDB.zip")
    h = http_download(url, z, "PalmMatchDB.zip")
    extract(z, d, "PalmMatchDB")
    return dict(archive_sha256=h, url=url)


def fetch_gabor(d):
    """Mendeley Data: file-by-file download of the raw/ folder, each file checked against its SHA-256."""
    base = "https://data.mendeley.com/public-api/datasets/r8hxykxnk5"
    get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "palm-benchmark/1.0"}), timeout=60))
    folders = get(base + "/folders/2?version=2")
    byid = {f["id"]: f for f in folders}
    raw = [f for f in folders if f.get("parent_id") in byid and byid[f["parent_id"]]["name"] == "raw"]
    files = [(f["name"], x) for f in raw for x in get(base + f"/files?folder_id={f['id']}&version=2")]
    bad = 0
    with Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), console=con) as p:
        t = p.add_task("Gabor raw images", total=len(files))
        for person, x in files:
            out = os.path.join(d, "raw", person, x["filename"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if not os.path.exists(out):                     # Mendeley refuses Python's default user agent (403)
                req = urllib.request.Request(x["content_details"]["download_url"], headers={"User-Agent": "palm-benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                    f.write(r.read())
            bad += sha256(out) != x["content_details"]["sha256_hash"]
            p.advance(t)
    if bad:
        raise RuntimeError(f"{bad} Gabor files failed the SHA-256 check")
    return dict(url=base, files_verified_sha256=len(files))


def fetch_tongji(d):
    z = os.path.join(DL, "Tongji_ROI.zip")
    h = gdrive_download("1KZCXi6zAk5mZ1nQHFdeYHboAII3DOjls", z, "Tongji ROI images")
    tmp = d + "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    extract(z, tmp, "Tongji")
    for root, dirs, _ in os.walk(tmp):                      # move session1/ and session2/ to data/Tongji/
        if "session1" in dirs and "session2" in dirs:
            os.makedirs(d, exist_ok=True)
            for s in ("session1", "session2"):
                shutil.move(os.path.join(root, s), os.path.join(d, s))
            break
    else:
        raise RuntimeError("Tongji archive has no session1/ and session2/ folders")
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(archive_sha256=h, url="https://drive.google.com/file/d/1KZCXi6zAk5mZ1nQHFdeYHboAII3DOjls")


def fetch_hewelxx(d):
    url = "https://zenodo.org/api/records/15473268/files/HewelXX/Dataset-v1.0.0.zip/content"
    z = os.path.join(DL, "HewelXX_Dataset-v1.0.0.zip")
    h = http_download(url, z, "Zenodo palmprint bundle")
    tmp = d + "_tmp"
    if not os.path.isdir(tmp):
        extract(z, tmp, "bundle")
    top = os.path.dirname(next(os.path.join(r, x) for r, ds, _ in os.walk(tmp) for x in ds if x == "XINHUA"))
    os.makedirs(d, exist_ok=True)
    for sub in ("PolyUII", "XINHUA"):                       # the Zenodo snapshot holds XINHUA only (no PolyU II)
        if os.path.isdir(os.path.join(top, sub)):
            shutil.move(os.path.join(top, sub), os.path.join(d, sub))
    bj = os.path.join(top, "BJTU_V2", "bjtu_palm_v2.zip")
    if os.path.exists(bj):
        extract(bj, os.path.join(d, "BJTU_V2"), "BJTU v2")
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(archive_sha256=h, url=url)


def fetch_11khands(d):
    z = os.path.join(DL, "11kHands_Hands.zip")
    h = gdrive_download("1KcMYcNJgtK1zZvfl_9sTqnyBUTri2aP2", z, "11k Hands images (about 600 MB)")
    extract(z, d, "11k Hands")
    csv = os.path.join(d, "HandInfo.csv")
    if not os.path.exists(csv):                             # metadata (palmar/dorsal, left/right) is a separate file
        for fid in ("1RC86-rVOR8c93XAfM9b9R45L7C2B0FdA", "1nZN5cjLE_J8AFtY_8j9KO_Y7IkqYIWQ-",
                    "1UGyXf7NqvYfqz7Rq2wPg9ATw1cTKP_D9", "1W19htDugIJU0EmkKP-MgGIsiCX9aq3yQ"):
            try:
                tmp = os.path.join(DL, f"11k_{fid}")
                gdrive_download(fid, tmp, "11k Hands metadata")
                head = open(tmp, "rb").read(200)
                if b"aspectOfHand" in head:
                    shutil.copy(tmp, csv)
                    break
            except Exception as e:  # wrong file or refused: try the next link from the project page
                con.print(f"  [dim]{fid}: {e}[/]")
    if not os.path.exists(csv):
        raise RuntimeError("could not get HandInfo.csv from the 11k Hands project page")
    return dict(archive_sha256=h, url="https://drive.google.com/open?id=1KcMYcNJgtK1zZvfl_9sTqnyBUTri2aP2")


FETCH = dict(palmmatchdb=fetch_palmmatchdb, gabor=fetch_gabor, tongji=fetch_tongji, hewelxx=fetch_hewelxx,
             **{"11khands": fetch_11khands})


# ------------------------------------------------------------------------------------------------ main
def load_manifest():
    return json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else {}


def show(manifest):
    t = Table(title="Palm datasets", show_lines=False)
    for c in ("key", "dataset", "use", "licence", "on disk"):
        t.add_column(c)
    for k, ds in DATASETS.items():
        m = manifest.get(k)
        here = f"[green]{m['images']:,} images[/]" if m else "[dim]not downloaded[/]"
        t.add_row(k, ds["name"], "[green]permissive[/]" if ds["use"] == "permissive" else "[yellow]eval-only[/]",
                  ds["licence"], here)
    con.print(t)


def main():
    ap = argparse.ArgumentParser(description="download palm datasets for the benchmark")
    ap.add_argument("names", nargs="*", help="dataset keys, or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="download again even if already present")
    a = ap.parse_args()
    manifest = load_manifest()
    if a.list or not a.names:
        show(manifest)
        return
    names = list(DATASETS) if a.names == ["all"] else a.names
    failed = []
    for k in names:
        if k not in DATASETS:
            con.print(f"[red]unknown dataset {k}[/]; choose from {', '.join(DATASETS)}")
            continue
        ds, d = DATASETS[k], os.path.join(DATA, DATASETS[k]["dir"])
        if k in manifest and os.path.isdir(d) and not a.force:
            con.print(f"[green]✓[/] {ds['name']} already present ({manifest[k]['images']:,} images)")
            continue
        con.rule(f"{ds['name']}  [{ds['use']}: {ds['licence']}]")
        t0 = time.time()
        try:
            info = FETCH[k](d)
        except Exception as e:
            con.print(f"[red]✗ {ds['name']}: {e}[/]")
            failed.append(k)
            continue
        n = count_images(d)
        manifest[k] = dict(ds, **info, images=n, path=os.path.relpath(d, REPO),
                           downloaded=datetime.date.today().isoformat(), seconds=round(time.time() - t0))
        os.makedirs(DATA, exist_ok=True)
        json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2)
        con.print(f"[green]✓[/] {ds['name']}: {n:,} images in {os.path.relpath(d, REPO)}")
    show(manifest)
    if failed:
        con.print(f"[red]failed: {', '.join(failed)}[/] - run again to retry (downloads resume)")
        sys.exit(1)


if __name__ == "__main__":
    main()
