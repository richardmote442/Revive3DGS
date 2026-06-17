#!/bin/bash
set -euo pipefail

LOG_FILE="setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------

sudo apt-get update

# Remove CPU-only COLMAP if installed from apt
sudo apt-get remove -y colmap || true

sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    unzip \
    libglm-dev \
    imagemagick \
    ffmpeg \
    xvfb \
    xauth \
    libboost-all-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libsuitesparse-dev \
    libceres-dev \
    libsqlite3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libcgal-dev \
    libopencv-dev \
    libmetis-dev

python -m pip install --upgrade pip
python -m pip install "setuptools<82" wheel ninja
python -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Build COLMAP from source with CUDA
# -----------------------------------------------------------------------------

cd "$ROOT_DIR"

if [ ! -d "colmap" ]; then
    git clone https://github.com/colmap/colmap.git
fi

cd colmap
rm -rf build
mkdir build
cd build

cmake .. \
    -GNinja \
    -DCUDA_ENABLED=ON \
    -DGUI_ENABLED=OFF \
    -DOPENGL_ENABLED=OFF \
    -DCMAKE_BUILD_TYPE=Release

ninja -j"$(nproc)"
sudo ninja install
sudo ldconfig

echo "Checking COLMAP CUDA status:"
colmap -h | grep -i cuda || true

# -----------------------------------------------------------------------------
# Patch GraphDeco convert.py for newer COLMAP option names
# -----------------------------------------------------------------------------

cd "$ROOT_DIR"

sed -i 's/SiftExtraction.use_gpu/FeatureExtraction.use_gpu/g' gaussian-splatting/convert.py
sed -i 's/SiftMatching.use_gpu/FeatureMatching.use_gpu/g' gaussian-splatting/convert.py

# -----------------------------------------------------------------------------
# Install Gaussian Splatting submodules
# -----------------------------------------------------------------------------

cd "$ROOT_DIR/gaussian-splatting/submodules/diff-gaussian-rasterization"

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

cd "$SCRIPT_DIR"

python -m pip install --no-build-isolation "$ROOT_DIR/gaussian-splatting/submodules/simple-knn"
python -m pip install --no-build-isolation "$ROOT_DIR/gaussian-splatting/submodules/fused-ssim"

# -----------------------------------------------------------------------------
# Install Turtle
# -----------------------------------------------------------------------------

cd "$ROOT_DIR/Turtle"
NO_CUDA_EXT=1 python -m pip install -e . --no-build-isolation --no-deps -v

# -----------------------------------------------------------------------------
# Download selected Turtle pretrained models
# -----------------------------------------------------------------------------

cd "$SCRIPT_DIR"

python -m pip install gdown
mkdir -p "$ROOT_DIR/Turtle/basicsr/trained_models"

python - <<'PY'
import json
import subprocess
import sys
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

    if name not in keep:
        continue

    dst = out / name

    if dst.exists() and dst.stat().st_size > 0:
        print(f"Already exists, skipping: {dst}")
        continue

    subprocess.check_call([
        sys.executable, "-m", "gdown",
        item["url"],
        "-O", str(dst),
        "--continue"
    ])
PY

# -----------------------------------------------------------------------------
# Download datasets
# -----------------------------------------------------------------------------

mkdir -p "$ROOT_DIR/datasets/raw"

gdown "https://drive.google.com/file/d/1S3fOnl-SEgiapFPm2s0VtUDeVYwdAnL_/view" \
    -O "$ROOT_DIR/datasets/raw/final_scenes.zip"

# Optional RainGS download
# gdown "https://drive.google.com/file/d/1SIX8D_j0t9l6qmGOt-VsTOQYQfh_tNUl/view?usp=sharing" \
#     -O "$ROOT_DIR/datasets/raw/rain_streak.tar"

# -----------------------------------------------------------------------------
# Preprocess WeatherGS-Snow and WeatherGS-Rain datasets
# -----------------------------------------------------------------------------

RAW_DIR="$ROOT_DIR/datasets/raw"
SCENE_ROOT="$RAW_DIR/final_scenes"

SNOW_DIR="$ROOT_DIR/datasets/WeatherGS-Snow"
RAIN_DIR="$ROOT_DIR/datasets/WeatherGS-Rain"

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