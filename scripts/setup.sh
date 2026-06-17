#!/bin/bash
set -euo pipefail

LOG_FILE="setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

sudo apt-get update
python -m pip install --upgrade pip
sudo apt-get install -y build-essential cmake ninja-build git unzip libglm-dev colmap imagemagick ffmpeg
python -m pip install "setuptools<82" wheel ninja

python -m pip install -r requirements.txt

# Patch diff-gaussian-rasterization for newer compiler / Python environment
cd ../gaussian-splatting/submodules/diff-gaussian-rasterization

python - <<'PY'
from pathlib import Path

path = Path("cuda_rasterizer/rasterizer_impl.h")
text = path.read_text()

if "#include <cstdint>" not in text:
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip() == "#pragma once":
            insert_at = i + 1
            break
    lines.insert(insert_at, "#include <cstdint>")
    path.write_text("\n".join(lines) + "\n")

print("Patched:", path)
PY

rm -rf build
python -m pip install --no-build-isolation .

cd ../../../scripts
python -m pip install --no-build-isolation ../gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation ../gaussian-splatting/submodules/fused-ssim

cd ../Turtle
NO_CUDA_EXT=1 python -m pip install -e . --no-build-isolation --no-deps -v

# Download only selected Turtle pretrained models
python -m pip install gdown
mkdir -p ../Turtle/basicsr/trained_models

python - <<'PY'
import json, subprocess, sys
from pathlib import Path

url = "https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA"
out = Path("../Turtle/basicsr/trained_models")
keep = {"GoPro_Deblur.pth", "Desnow.pth", "RainDrop.pth"}

items = json.loads(subprocess.check_output(
    [sys.executable, "-m", "gdown", url, "--folder", "--json", "--quiet"],
    text=True
))

for item in items:
    name = Path(item["path"]).name
    if name in keep:
        subprocess.check_call([
            sys.executable, "-m", "gdown",
            item["url"],
            "-O", str(out / name),
            "--continue"
        ])
PY

# Download datasets
mkdir -p ../datasets/raw
gdown "https://drive.google.com/file/d/1S3fOnl-SEgiapFPm2s0VtUDeVYwdAnL_/view" \
  -O ../datasets/raw/final_scenes.zip

# Preprocess datasets
#cd ../datasets/raw
# unzip the datasets if needed
# The dataset consist of images in this structure:
# └── raw/
#     ├── final_scenes/
#     └── factory_rain/
#         ├── gt
#         │   ├── xxxx.png
#              ...
#         │   ├── xxxx.png
#         ├── images
#         │   ├── xxxx.png
#              ...
#         │   ├── xxxx.png
#         ├── masks
#         ├── sparse
#         ├── weather_images
#     └── #other scenes with same structure as factory rain/

# For each scene, copy gt and images folders to the WeatherGS folder with this structure
# The "images" folder becomes the "degraded" folder, and the "gt" folder becomes the "gt" folder in the WeatherGS dataset.
# └── WeatherGS/
#         ├── degraded
#            ├── factory_rain/
#            │   ├── xxxx.png
#            │   ....
#            └── Other_scenes/
#            │   ├── xxxx.png
#            │   ....
#         └── gt
#            ├── factory_rain/
#            │   ├── xxxx.png
#            │   ....
#            └── Other_scenes/
#            │   ├── xxxx.png
#            │   ....

# Preprocess WeatherGS-Snow and WeatherGS-Rain datasets
RAW_DIR="../datasets/raw"
SCENE_ROOT="$RAW_DIR/final_scenes"

SNOW_DIR="../datasets/WeatherGS-Snow"
RAIN_DIR="../datasets/WeatherGS-Rain"

echo "Extracting final_scenes.zip..."
unzip -o -q "$RAW_DIR/final_scenes.zip" -d "$RAW_DIR"

echo "Creating WeatherGS-Snow and WeatherGS-Rain dataset structures..."

rm -rf "$SNOW_DIR/degraded" "$SNOW_DIR/gt"
rm -rf "$RAIN_DIR/degraded" "$RAIN_DIR/gt"

mkdir -p "$SNOW_DIR/degraded" "$SNOW_DIR/gt"
mkdir -p "$RAIN_DIR/degraded" "$RAIN_DIR/gt"

snow_count=0
rain_count=0
skip_count=0

for scene_dir in "$SCENE_ROOT"/*; do
    [ -d "$scene_dir" ] || continue

    scene_name="$(basename "$scene_dir")"
    scene_name_lower="${scene_name,,}"

    src_images="$scene_dir/images"
    src_gt="$scene_dir/gt"

    if [[ ! -d "$src_images" || ! -d "$src_gt" ]]; then
        echo "Skipping $scene_name because images/ or gt/ is missing"
        skip_count=$((skip_count + 1))
        continue
    fi

    if [[ "$scene_name_lower" == *"snow"* ]]; then
        target_dir="$SNOW_DIR"
        snow_count=$((snow_count + 1))
        echo "Processing snow scene: $scene_name"

    elif [[ "$scene_name_lower" == *"rain"* ]]; then
        target_dir="$RAIN_DIR"
        rain_count=$((rain_count + 1))
        echo "Processing rain scene: $scene_name"

    else
        echo "Skipping $scene_name because scene name does not contain snow or rain"
        skip_count=$((skip_count + 1))
        continue
    fi

    mkdir -p "$target_dir/degraded/$scene_name"
    mkdir -p "$target_dir/gt/$scene_name"

    cp -a "$src_images"/. "$target_dir/degraded/$scene_name"/
    cp -a "$src_gt"/. "$target_dir/gt/$scene_name"/
done

echo "WeatherGS preprocessing finished."
echo "Snow scenes:    $snow_count -> $SNOW_DIR"
echo "Rain scenes:    $rain_count -> $RAIN_DIR"
echo "Skipped scenes: $skip_count"