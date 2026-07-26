#!/usr/bin/env python3
"""Filter Copernicus DEM GLO-30 tileList for CONUS and emit S3 HTTPS URLs."""
import re
import sys

# CONUS bounding box (lower-left inclusive)
MIN_LAT = 24
MAX_LAT = 49   # lower-left lat must be <= MAX_LAT-1
MIN_LON = -125
MAX_LON = -66  # lower-left lon must be <= MAX_LON-1

def parse_tile(name):
    m = re.match(r'Copernicus_DSM_COG_10_([NS])(\d+)_00_([EW])(\d+)_00_DEM', name)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lat = int(lat)
    lon = int(lon)
    if ns == 'S':
        lat = -lat
    if ew == 'W':
        lon = -lon
    return lat, lon

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parsed = parse_tile(line)
    if parsed is None:
        continue
    lat, lon = parsed
    if MIN_LAT <= lat <= (MAX_LAT - 1) and MIN_LON <= lon <= (MAX_LON - 1):
        print(f"https://copernicus-dem-30m.s3.amazonaws.com/{line}/{line}.tif")
