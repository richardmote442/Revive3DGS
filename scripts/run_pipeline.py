import os
import subprocess
import shutil
from pathlib import Path
import re


# Editable variables
DATASET_NAME = "RainGS"
SCENE_NAME = "garden"
TRAIN_ITERATIONS = 7000


ROOT_DIR = Path(__file__).resolve().parents[1]
TURTLE_DIR = ROOT_DIR / "Turtle"
BASICSR_DIR = TURTLE_DIR / "basicsr"
INFERENCE_SCRIPT = BASICSR_DIR / "inference.py"

GAUSSIAN_SPLATTING_DIR = ROOT_DIR / "gaussian-splatting"

RESTORED_SCENE_DIR = ROOT_DIR / "datasets" / DATASET_NAME / "restored" / SCENE_NAME

COLMAP_SCENE_DIR = ROOT_DIR / "datasets" / DATASET_NAME / "colmap" / SCENE_NAME
COLMAP_INPUT_DIR = COLMAP_SCENE_DIR / "input"

GS_OUTPUT_DIR = ROOT_DIR / "datasets"/ DATASET_NAME / "3DGS_output" / SCENE_NAME


def run_step(name, cmd, cwd):
    print("=" * 80)
    print(f"Running step: {name}")
    print("=" * 80)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        env=env
    )

def natural_key(path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.name)]

def prepare_colmap_input():
    print("=" * 80)
    print("Preparing COLMAP input images")
    print("=" * 80)

    if not RESTORED_SCENE_DIR.exists():
        raise FileNotFoundError(f"Restored scene folder not found: {RESTORED_SCENE_DIR}")

    COLMAP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old input images
    for old_file in COLMAP_INPUT_DIR.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    pred_images = sorted([
        p for p in RESTORED_SCENE_DIR.glob("*")
        if p.is_file()
        and "pred" in p.stem.lower()
        and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ], key=natural_key)

    if len(pred_images) == 0:
        raise FileNotFoundError(f"No restored images containing 'pred' found in: {RESTORED_SCENE_DIR}")

    for i, src in enumerate(pred_images, start=1):
        dst = COLMAP_INPUT_DIR / f"{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    print(f"Copied {len(pred_images)} restored images to: {COLMAP_INPUT_DIR}")


def main():
    # run_step(
    #     name="Turtle weather restoration",
    #     cmd=["python", "inference.py", "--scene_name", SCENE_NAME],
    #     cwd=BASICSR_DIR
    # )

    prepare_colmap_input()

    run_step(
        name="COLMAP conversion for 3DGS",
        cmd=["xvfb-run", "-a", "python", "convert.py", "-s", str(COLMAP_SCENE_DIR)],
        cwd=GAUSSIAN_SPLATTING_DIR
    )

    run_step(
        name="3D Gaussian Splatting training",
        cmd=[
            "python", "train.py",
            "-s", str(COLMAP_SCENE_DIR),
            "-m", str(GS_OUTPUT_DIR),
            "--iterations", str(TRAIN_ITERATIONS)
        ],
        cwd=GAUSSIAN_SPLATTING_DIR
    )


if __name__ == "__main__":
    main()