#!/usr/bin/env python3
"""Build soft weather confidence maps from binary masks and optional heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate soft weather confidence maps.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--output-dir", default="confidences")
    parser.add_argument("--mask-npy", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=9.0)
    parser.add_argument("--heatmap-weight", type=float, default=0.5)
    return parser.parse_args()


def image_paths(mask_dir: Path) -> list[Path]:
    return sorted(p for p in mask_dir.glob("*.png"))


def load_heatmap(mask_npy: Path | None) -> np.ndarray | None:
    if mask_npy is None or not mask_npy.exists():
        return None
    heat = np.load(mask_npy)
    if heat.ndim == 3:
        heat = np.squeeze(heat)
    if heat.ndim != 2:
        raise ValueError(f"Unexpected heatmap shape: {heat.shape}")
    heat = heat.astype(np.float32)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def build_confidence(mask_path: Path, heatmap: np.ndarray | None, sigma: float, heatmap_weight: float) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    occluded = (mask == 0).astype(np.float32)
    blurred = cv2.GaussianBlur(occluded, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    confidence = blurred
    if heatmap is not None:
        resized_heat = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        confidence = (1.0 - heatmap_weight) * confidence + heatmap_weight * resized_heat
    confidence = np.clip(confidence, 0.0, 1.0)
    return (confidence * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    mask_dir = scene / args.mask_dir
    out_dir = scene / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = load_heatmap(args.mask_npy or (scene / "vis" / "masks.npy"))

    paths = image_paths(mask_dir)
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    for path in paths:
        confidence = build_confidence(path, heatmap, args.sigma, args.heatmap_weight)
        Image.fromarray(confidence, mode="L").save(out_dir / path.name)

    print(f"Saved {len(paths)} confidence maps to {out_dir}")


if __name__ == "__main__":
    main()
