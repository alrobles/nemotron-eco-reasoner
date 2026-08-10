#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --account=kbs
#SBATCH --job-name=prepare_stage1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a474r867/beegfs/glo30_dem/scripts/prepare_stage1_%j.out
#SBATCH --error=/home/a474r867/beegfs/glo30_dem/scripts/prepare_stage1_%j.err

set -euo pipefail
SIF="/home/a474r867/geospatial-rserver/geospatial_latest.sif"
OUTDIR="/home/a474r867/beegfs/stage1_conus_2015"
DEMDIR="/home/a474r867/beegfs/glo30_dem"
mkdir -p "$OUTDIR"

# Target CONUS grid at 1/120 deg
RES120="0.0083333333"
TE="-125 24 -66 50"

echo "[$(date)] Build DEM VRT"
ls "$DEMDIR"/tiles/*.tif > "$OUTDIR/dem_tiles.txt"

apptainer exec --cleanenv \
  --bind "$DEMDIR:$DEMDIR" \
  --bind "$OUTDIR:$OUTDIR" \
  "$SIF" \
  gdalbuildvrt -input_file_list "$OUTDIR/dem_tiles.txt" "$OUTDIR/dem_conus.vrt"

echo "[$(date)] Resample DEM to 1/120 deg"
apptainer exec --cleanenv \
  --bind "$OUTDIR:$OUTDIR" \
  "$SIF" \
  gdalwarp -t_srs EPSG:4326 -tr "$RES120" "$RES120" -te $TE -r average \
    -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co ZLEVEL=6 \
    "$OUTDIR/dem_conus.vrt" "$OUTDIR/dem_1-120.tif"

echo "[$(date)] Reproject ERA5 tas 2015 to 1/120 deg"
apptainer exec --cleanenv \
  --bind "/home/a474r867/scratch/era5:/home/a474r867/scratch/era5" \
  --bind "$OUTDIR:$OUTDIR" \
  "$SIF" \
  gdalwarp -t_srs EPSG:4326 -tr "$RES120" "$RES120" -te $TE -r bilinear \
    -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co ZLEVEL=6 \
    "/home/a474r867/scratch/era5/era5-land/era5_bioclim/annual/tas_2015.tif" \
    "$OUTDIR/era5_tas_2015_1-120.tif"

echo "[$(date)] Reproject CHELSA tas 2015 to CONUS 1/120 deg"
apptainer exec --cleanenv \
  --bind "/home/a474r867/work/chelsa_legacy:/home/a474r867/work/chelsa_legacy" \
  --bind "$OUTDIR:$OUTDIR" \
  "$SIF" \
  gdalwarp -t_srs EPSG:4326 -tr "$RES120" "$RES120" -te $TE -r bilinear \
    -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co ZLEVEL=6 \
    "/home/a474r867/work/chelsa_legacy/chelsa_NorthAmerica/2015/CHELSA_tas_2015_V.2.1.tif" \
    "$OUTDIR/chelsa_tas_2015_1-120.tif"

echo "[$(date)] Done"
ls -lh "$OUTDIR"
