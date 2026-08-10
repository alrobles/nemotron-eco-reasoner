#!/usr/bin/env python3
"""Fast concurrent download of Copernicus GLO-30 DEM tiles from S3 using requests."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

URLS_FILE = "/home/a474r867/beegfs/glo30_dem/conus_urls.txt"
OUTDIR = "/home/a474r867/beegfs/glo30_dem/tiles"
MAX_WORKERS = 32
TIMEOUT = 120
RETRIES = 3

session = requests.Session()

def download_one(url):
    basename = os.path.basename(url)
    out_path = os.path.join(OUTDIR, basename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, "exists", 0.0
    for attempt in range(RETRIES):
        try:
            t0 = time.time()
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            elapsed = time.time() - t0
            return out_path, "downloaded", elapsed
        except Exception as e:
            if attempt == RETRIES - 1:
                return out_path, f"error: {e}", 0.0
            time.sleep(2 ** attempt)
    return out_path, "error: max retries", 0.0

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]
    total = len(urls)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Downloading {total} tiles to {OUTDIR} with {MAX_WORKERS} workers")
    done = 0
    errors = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, url): url for url in urls}
        for future in as_completed(futures):
            path, status, elapsed = future.result()
            done += 1
            if status.startswith("error"):
                errors += 1
            if done % 50 == 0 or status.startswith("error"):
                mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
                print(f"[{done}/{total}] {os.path.basename(path)} {status} ({elapsed:.1f}s, {mb:.1f}MB) errors={errors}")
    elapsed_total = time.time() - t_start
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done. {done} tiles, {errors} errors in {elapsed_total/60:.1f} min")

if __name__ == "__main__":
    main()
