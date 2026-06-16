#!/usr/bin/env python3
"""Build cross-view transient consensus maps from baseline render errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-view transient consensus maps.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--weather-dir", default="weather_images")
    parser.add_argument("--renders-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output-dir", default="consensus_maps")
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=3.0)
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def make_error_map(render_path: Path, weather_path: Path) -> np.ndarray:
    render = load_gray(render_path)
    weather = load_gray(weather_path)
    if render.shape != weather.shape:
        weather = cv2.resize(weather, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(render - weather)
    return diff.astype(np.float32)


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    weather_dir = scene / args.weather_dir
    renders_dir = args.renders_dir.resolve()
    camera_data = json.loads(args.camera_json.read_text(encoding="utf-8"))
    render_paths = sorted(renders_dir.glob("*.png"))
    stems = [Path(item["img_name"]).stem for item in camera_data[: len(render_paths)]]
    error_maps = []
    for render_path, stem in zip(render_paths, stems):
        weather_path = weather_dir / f"{stem}.png"
        if not weather_path.exists():
            raise FileNotFoundError(weather_path)
        error_maps.append(make_error_map(render_path, weather_path))

    output_dir = scene / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (stem, current_error) in enumerate(zip(stems, error_maps)):
        neighbors = []
        for offset in range(-args.neighbor_window, args.neighbor_window + 1):
            if offset == 0:
                continue
            j = idx + offset
            if 0 <= j < len(error_maps):
                neighbors.append(error_maps[j])
        if neighbors:
            neighbor_mean = np.mean(np.stack(neighbors, axis=0), axis=0)
            consensus = np.clip(current_error - neighbor_mean, 0.0, 1.0)
        else:
            consensus = current_error.copy()
        if args.sigma > 0:
            consensus = cv2.GaussianBlur(consensus, (0, 0), sigmaX=args.sigma, sigmaY=args.sigma)
        if consensus.max() > 0:
            consensus = consensus / consensus.max()
        Image.fromarray((consensus * 255.0).astype(np.uint8), mode="L").save(output_dir / f"{stem}.png")

    print(f"Saved {len(stems)} consensus maps to {output_dir}")


if __name__ == "__main__":
    main()
