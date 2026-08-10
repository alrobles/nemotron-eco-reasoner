#!/usr/bin/env python3
"""Compute annual mean tas from monthly ERA5-Land GeoTIFFs."""
import glob
import os
import numpy as np
import rasterio

YEAR = 2015
MONTHLY_DIR = f"/home/a474r867/scratch/era5/era5-land/era5_bioclim/monthly/{YEAR}"
OUT_DIR = f"/home/a474r867/scratch/era5/era5-land/era5_bioclim/annual"
OUT_FILE = os.path.join(OUT_DIR, f"tas_{YEAR}.tif")

def main():
    files = sorted(glob.glob(os.path.join(MONTHLY_DIR, "tas_*.tif")))
    if len(files) != 12:
        raise ValueError(f"Expected 12 monthly files, found {len(files)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read first to get profile
    with rasterio.open(files[0]) as src:
        profile = src.profile
        shape = src.shape
        nodata = src.nodata

    # Stack and compute mean in float32, ignoring NaN
    stack = np.empty((12, *shape), dtype=np.float32)
    for i, fp in enumerate(files):
        with rasterio.open(fp) as src:
            band = src.read(1)
            if band.dtype != np.float32:
                band = band.astype(np.float32)
            stack[i] = band

    if nodata is not None and np.isnan(nodata):
        mean = np.nanmean(stack, axis=0)
    else:
        # Assume NaN for missing
        mean = np.nanmean(stack, axis=0)

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='deflate',
        nodata=np.nan,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(OUT_FILE, 'w', **profile) as dst:
        dst.write(mean.astype(np.float32), 1)

    print(f"Wrote {OUT_FILE}, shape={mean.shape}, min={np.nanmin(mean):.2f}, max={np.nanmax(mean):.2f}")

if __name__ == "__main__":
    main()
