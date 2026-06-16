#!/usr/bin/env python3
"""Run the WeatherGS-based degraded-scene reconstruction pipeline.

The heavy lifting stays in the upstream WeatherGS repository:

* AEF: image cleanup
* LED: occlusion/weather mask detection
* confidence: soft weather confidence map generation
* 3DGS: confidence-aware mask-guided Gaussian Splatting training/render/metrics

This script only orchestrates those entrypoints with reproducible paths and
multi-environment support.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


STEPS = ("aef", "led", "refine", "confidence", "residual", "consensus", "train", "render", "metrics")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_weathergs_root() -> Path:
    return repo_root() / "external" / "WeatherGS"


def default_gs_python() -> Path:
    preferred = Path("/mnt/afs_e/miniconda/envs/3DGSqzj/bin/python")
    return preferred if preferred.exists() else Path("/mnt/afs_e/miniconda/envs/3dgs/bin/python")


def parse_extra(value: str) -> list[str]:
    return shlex.split(value) if value else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeatherGS rain/snow reconstruction pipeline")
    parser.add_argument("--scene", required=True, type=Path, help="Scene directory with COLMAP sparse/ and images/.")
    parser.add_argument("--weathergs-root", type=Path, default=default_weathergs_root())
    parser.add_argument("--task", choices=("derain", "desnow"), default=None)
    parser.add_argument("--images", default="images", help="Image subdirectory inside the scene.")
    parser.add_argument("--processed-images", default="processed_images")
    parser.add_argument("--masks", default="masks")
    parser.add_argument("--confidences", default="confidences_soft")
    parser.add_argument("--residuals", default="residuals")
    parser.add_argument("--consensuses", default="consensus_maps")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value for child commands.")
    parser.add_argument("--start-at", choices=STEPS, default="aef")
    parser.add_argument("--stop-after", choices=STEPS, default="metrics")
    parser.add_argument("--skip", nargs="*", choices=STEPS, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-processed", action="store_true")

    parser.add_argument("--aef-python", type=Path, default=None)
    parser.add_argument("--led-python", type=Path, default=None)
    parser.add_argument("--gs-python", type=Path, default=default_gs_python())

    parser.add_argument("--aef-extra", default="")
    parser.add_argument("--led-extra", default="")
    parser.add_argument("--confidence-extra", default="")
    parser.add_argument("--residual-extra", default="")
    parser.add_argument("--consensus-extra", default="")
    parser.add_argument("--train-extra", default="")
    parser.add_argument("--render-extra", default="")
    parser.add_argument("--metrics-extra", default="")

    parser.add_argument("--mask-min-thresh", type=float, default=0.5)
    parser.add_argument("--mask-max-thresh", type=float, default=0.7)
    parser.add_argument("--mask-kernel-size", type=int, default=3)
    parser.add_argument("--mask-iterations", type=int, default=3)
    parser.add_argument("--skip-aef", action="store_true", help="Use existing processed_images or original images.")
    parser.add_argument("--skip-led", action="store_true", help="Use existing vis/masks.npy or masks.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


def step_enabled(args: argparse.Namespace, step: str) -> bool:
    first = STEPS.index(args.start_at)
    last = STEPS.index(args.stop_after)
    return first <= STEPS.index(step) <= last and step not in set(args.skip)


def child_env(gpus: str, weathergs_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONUNBUFFERED", "1")

    python_paths = [
        str(weathergs_root / "3DGS"),
        str(weathergs_root / "submodules" / "simple-knn"),
        str(weathergs_root / "submodules" / "diff-gaussian-rasterization"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(python_paths + ([existing] if existing else []))
    return env


def run(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n[{cwd}] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def require_python(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"{label} python is required for this step. Pass --{label.lower()}-python.")
    if not path.exists():
        raise SystemExit(f"{label} python not found: {path}")
    return path


def list_images(directory: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(directory.glob(pattern)))
    return images


def ensure_png_link_dir(scene: Path, source_dir_name: str, target_dir_name: str, overwrite: bool) -> str:
    source_dir = scene / source_dir_name
    target_dir = scene / target_dir_name
    if not source_dir.exists():
        raise SystemExit(f"Source image directory not found: {source_dir}")
    if target_dir.exists():
        if not overwrite:
            return target_dir.name
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(source_dir)
    if not images:
        raise SystemExit(f"No images found in {source_dir}")
    for image_path in images:
        link_path = target_dir / f"{image_path.stem}.png"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(image_path)
    return target_dir.name


def copy_processed_to_images(scene: Path, processed_images: str, overwrite: bool) -> str:
    src = scene / processed_images
    dst = scene / "images_cleaned"
    if not src.exists():
        raise SystemExit(f"Processed image directory not found: {src}")
    if dst.exists():
        if not overwrite:
            return dst.name
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst.name


def choose_train_images(args: argparse.Namespace, scene: Path, ran_aef: bool) -> str:
    processed_dir = scene / args.processed_images
    if ran_aef or (args.skip_aef and processed_dir.exists()):
        return copy_processed_to_images(scene, args.processed_images, args.overwrite_processed)
    return args.images


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    wg_root = args.weathergs_root.resolve()
    env = child_env(args.gpus, wg_root)
    model_path = (args.model_path or (repo_root() / "outputs" / scene.name)).resolve()

    if not wg_root.exists():
        raise SystemExit(f"WeatherGS root not found: {wg_root}")
    if not scene.exists():
        raise SystemExit(f"Scene not found: {scene}")

    aef = step_enabled(args, "aef") and not args.skip_aef
    led = step_enabled(args, "led") and not args.skip_led
    refine = step_enabled(args, "refine") and not args.skip_led
    confidence = step_enabled(args, "confidence")
    residual = step_enabled(args, "residual")
    consensus = step_enabled(args, "consensus")
    train = step_enabled(args, "train")
    render = step_enabled(args, "render") and not args.skip_render
    metrics = step_enabled(args, "metrics") and not args.skip_metrics

    if aef:
        aef_py = require_python(args.aef_python, "AEF")
        aef_input_dir = scene / args.images
        if any(path.suffix.lower() != ".png" for path in list_images(aef_input_dir)):
            aef_input_name = ensure_png_link_dir(scene, args.images, f"{args.images}_aef_png", args.overwrite_processed)
        else:
            aef_input_name = args.images
        cmd = [
            str(aef_py),
            "infer.py",
            "--image_path",
            str(scene / aef_input_name),
        ]
        if args.task:
            cmd.extend(["--task", args.task])
        cmd.extend(parse_extra(args.aef_extra))
        run(cmd, wg_root / "AEF", env, args.dry_run)

    if led:
        led_py = require_python(args.led_python, "LED")
        cmd = [
            str(led_py),
            "detect_occlusion.py",
            "--image_path",
            str(scene / args.processed_images),
            "--weights_path",
            str((wg_root / "LED" / "ckpt" / "derain_gan.ckpt-100000").resolve()),
        ]
        cmd.extend(parse_extra(args.led_extra))
        run(cmd, wg_root / "LED", env, args.dry_run)

    if refine:
        cmd = [
            str(args.gs_python),
            str(repo_root() / "scripts" / "refine_masks_batch.py"),
            "--scene-path",
            str(scene),
            "--image-dir",
            args.processed_images,
            "--output-dir",
            args.masks,
            "--min-thresh",
            str(args.mask_min_thresh),
            "--max-thresh",
            str(args.mask_max_thresh),
            "--kernel-size",
            str(args.mask_kernel_size),
            "--iterations",
            str(args.mask_iterations),
        ]
        run(cmd, repo_root(), env, args.dry_run)

    if confidence:
        cmd = [
            str(args.gs_python),
            str(repo_root() / "scripts" / "build_soft_confidence.py"),
            "--scene",
            str(scene),
            "--mask-dir",
            args.masks,
            "--output-dir",
            args.confidences,
        ]
        cmd.extend(parse_extra(args.confidence_extra))
        run(cmd, repo_root(), env, args.dry_run)

    train_images = args.images
    if train:
        train_images = choose_train_images(args, scene, aef)

    if residual:
        cmd = [
            str(args.gs_python),
            str(repo_root() / "scripts" / "build_weather_residual.py"),
            "--scene",
            str(scene),
            "--weather-dir",
            train_images,
            "--clean-dir",
            "images",
            "--output-dir",
            args.residuals,
        ]
        cmd.extend(parse_extra(args.residual_extra))
        run(cmd, repo_root(), env, args.dry_run)

    if consensus:
        cmd = [
            str(args.gs_python),
            str(repo_root() / "scripts" / "build_cross_view_consensus.py"),
            "--scene",
            str(scene),
            "--weather-dir",
            train_images,
            "--renders-dir",
            str(model_path / "train" / f"ours_{args.render_extra.split()[-1] if args.render_extra else 500}" / "renders"),
            "--camera-json",
            str(model_path / "cameras.json"),
            "--output-dir",
            args.consensuses,
        ]
        cmd.extend(parse_extra(args.consensus_extra))
        run(cmd, repo_root(), env, args.dry_run)

    if train:
        cmd = [
            str(args.gs_python),
            "train.py",
            "-s",
            str(scene),
            "--images",
            train_images,
            "--model_path",
            str(model_path),
            "--confidences",
            str(scene / args.confidences),
            "--residuals",
            str(scene / args.residuals),
            "--consensuses",
            str(scene / args.consensuses),
        ]
        masks_dir = scene / args.masks
        if masks_dir.exists():
            cmd.extend(["--masks", str(masks_dir)])
        cmd.extend(parse_extra(args.train_extra))
        run(cmd, wg_root / "3DGS", env, args.dry_run)

    if render:
        cmd = [
            str(args.gs_python),
            "render.py",
            "-s",
            str(scene),
            "--images",
            train_images,
            "--model_path",
            str(model_path),
        ]
        cmd.extend(parse_extra(args.render_extra))
        run(cmd, wg_root / "3DGS", env, args.dry_run)

    if metrics:
        cmd = [str(args.gs_python), "metrics.py", "-m", str(model_path)]
        cmd.extend(parse_extra(args.metrics_extra))
        run(cmd, wg_root / "3DGS", env, args.dry_run)

    print(f"\nPipeline done. Model/output path: {model_path}")


if __name__ == "__main__":
    main()
