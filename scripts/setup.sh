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


