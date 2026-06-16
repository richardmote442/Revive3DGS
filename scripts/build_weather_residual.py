#!/usr/bin/env python3
"""Build weather residual prior maps from degraded-clean image pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate residual prior maps for weather scenes.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--weather-dir", default="weather_images")
    parser.add_argument("--clean-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--mask-weight", type=float, default=0.35)
    parser.add_argument("--mode", choices=("rgb", "gray", "edge", "mixed"), default="gray")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    return sorted(p for p in directory.glob("*.png"))


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def compute_residual(weather: np.ndarray, clean: np.ndarray, mode: str) -> np.ndarray:
    diff = np.abs(weather - clean)
    gray_weather = cv2.cvtColor((weather * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_clean = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_diff = np.abs(gray_weather - gray_clean)
    edge_weather = cv2.Canny((gray_weather * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    edge_clean = cv2.Canny((gray_clean * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    edge_diff = np.abs(edge_weather - edge_clean)

    if mode == "rgb":
        residual = diff.mean(axis=2)
    elif mode == "gray":
        residual = gray_diff
    elif mode == "edge":
        residual = edge_diff
    else:
        residual = 0.55 * gray_diff + 0.45 * edge_diff
    return residual.astype(np.float32)


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    weather_dir = scene / args.weather_dir
    clean_dir = scene / args.clean_dir
    mask_dir = scene / args.mask_dir
    output_dir = scene / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_paths = image_paths(weather_dir)
    if not weather_paths:
        raise FileNotFoundError(f"No PNG weather images found in {weather_dir}")

    for weather_path in weather_paths:
        clean_path = clean_dir / weather_path.name
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        weather = load_rgb(weather_path)
        clean = load_rgb(clean_path)
        residual = compute_residual(weather, clean, args.mode)
        if args.sigma > 0:
            residual = cv2.GaussianBlur(residual, (0, 0), sigmaX=args.sigma, sigmaY=args.sigma)
        mask_path = mask_dir / f"{weather_path.name}.png"
        if mask_path.exists():
            mask = load_gray(mask_path)
            residual = (1.0 - args.mask_weight) * residual + args.mask_weight * (1.0 - mask)
        if residual.max() > 0:
            residual = residual / residual.max()
        Image.fromarray((residual * 255.0).astype(np.uint8), mode="L").save(output_dir / f"{weather_path.name}.png")

    print(f"Saved {len(weather_paths)} residual maps to {output_dir}")


if __name__ == "__main__":
    main()
