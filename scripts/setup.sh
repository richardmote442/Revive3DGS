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
mkdir -p ../Turtle/trained_models

python - <<'PY'
import json, subprocess, sys
from pathlib import Path

url = "https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA"
out = Path("../Turtle/trained_models")
keep = {"GoPro_Deblur.pth", "Desnow.pth", "Raindrop.pth"}

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
gdown --folder "https://drive.google.com/drive/folders/1h21xh9JVhb_gmet8Vj8gxZy1_PpKJr8L" \
  -O ../datasets/raw 

# Preprocess datasets
#cd ../datasets/raw
# unzip the datasets if needed
# Check if the datasets consist of videos or images
# If the datasets consist of videos, move it to ../degraded/video
# If the dataset consists of images, move it to ../degraded/images
# The images structure must follow this structure
# └── images/
#     ├── train/
#     └── test/
#         ├── blur
#            ├── video_1
#            │   ├── Fame1
#            │   ....
#            └── video_n
#            │   ├── Fame1
#            │   ....
#         └── gt
#            ├── video_1
#            │   ├── Fame1
#            │   ....
#            └── video_n
#            │   ├── Fame1
#            │   ....

# Preprocess datasets
RAW_DIR="../datasets/raw"
DEG_DIR="../datasets/degraded"
VIDEO_DIR="$DEG_DIR/video"
IMAGE_DIR="$DEG_DIR/images"

mkdir -p "$RAW_DIR" "$VIDEO_DIR"
mkdir -p "$IMAGE_DIR"/{train,test}/{blur,gt}

echo "Extracting archives if needed..."

find "$RAW_DIR" -type f \( \
  -iname "*.zip" -o \
  -iname "*.tar" -o \
  -iname "*.tar.gz" -o \
  -iname "*.tgz" \
\) -print0 | while IFS= read -r -d '' archive; do
    lower="${archive,,}"

    if [[ "$lower" == *.tar.gz || "$lower" == *.tgz ]]; then
        out_dir="${archive%.*.*}"
    else
        out_dir="${archive%.*}"
    fi

    mkdir -p "$out_dir"

    case "$lower" in
        *.zip)
            unzip -o -q "$archive" -d "$out_dir"
            ;;
        *.tar|*.tar.gz|*.tgz)
            tar -xf "$archive" -C "$out_dir"
            ;;
    esac
done

echo "Moving video datasets..."

find "$RAW_DIR" -type f \( \
  -iname "*.mp4" -o \
  -iname "*.avi" -o \
  -iname "*.mov" -o \
  -iname "*.mkv" -o \
  -iname "*.webm" \
\) -print0 | while IFS= read -r -d '' video; do
    rel="${video#$RAW_DIR/}"
    dest="$VIDEO_DIR/$(dirname "$rel")"
    mkdir -p "$dest"
    mv -n "$video" "$dest/"
done

echo "Moving image datasets..."

declare -A group_ids
group_counter=0

while IFS= read -r -d '' img; do
    lower="/${img,,}/"

    # Detect train/test from path name
    if [[ "$lower" == *"/train/"* ]]; then
        split="train"
    else
        split="test"
    fi

    # Detect blur/gt from path name
    if [[ "$lower" == *"/gt/"* || "$lower" == *"/ground_truth/"* || "$lower" == *"/sharp/"* || "$lower" == *"/clear/"* ]]; then
        kind="gt"
    else
        kind="blur"
    fi

    rel="${img#$RAW_DIR/}"
    parent="$(dirname "$rel")"
    key="$split/$kind/$parent"

    # Each original folder becomes video_1, video_2, ...
    if [[ -z "${group_ids[$key]+x}" ]]; then
        group_counter=$((group_counter + 1))
        group_ids[$key]="video_${group_counter}"
    fi

    dest="$IMAGE_DIR/$split/$kind/${group_ids[$key]}"
    mkdir -p "$dest"
    mv -n "$img" "$dest/"

done < <(find "$RAW_DIR" -type f \( \
  -iname "*.jpg" -o \
  -iname "*.jpeg" -o \
  -iname "*.png" -o \
  -iname "*.bmp" -o \
  -iname "*.tif" -o \
  -iname "*.tiff" -o \
  -iname "*.webp" \
\) -print0)

echo "Dataset preprocessing finished."
echo "Videos are in: $VIDEO_DIR"
echo "Images are in: $IMAGE_DIR"