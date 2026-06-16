#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rendered views against clean target images.")
    parser.add_argument("--renders-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-lpips", action="store_true")
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0).cuda()


def safe_lpips(render: torch.Tensor, clean: torch.Tensor, enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        return float(lpips(render, clean, net_type="vgg").item())
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}")
        return None


def main() -> None:
    args = parse_args()
    camera_data = json.loads(args.camera_json.read_text(encoding="utf-8"))
    render_paths = sorted(args.renders_dir.glob("*.png"))
    if len(render_paths) > len(camera_data):
        raise ValueError(f"Render count {len(render_paths)} exceeds camera count {len(camera_data)}")
    image_names = [Path(item["img_name"]).stem for item in camera_data[: len(render_paths)]]

    per_view = {}
    psnrs = []
    ssims = []
    lpipss = []
    lpips_enabled = args.use_lpips

    for render_path, image_stem in zip(render_paths, image_names):
        clean_path = args.clean_dir / f"{image_stem}.png"
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        render = load_rgb(render_path)
        clean = load_rgb(clean_path)
        if render.shape[-2:] != clean.shape[-2:]:
            clean = torch.nn.functional.interpolate(clean, size=render.shape[-2:], mode="bilinear", align_corners=False)
        psnr_value = float(psnr(render, clean).mean().item())
        ssim_value = float(ssim(render, clean).item())
        lpips_value = safe_lpips(render, clean, lpips_enabled)
        if lpips_enabled and lpips_value is None:
            lpips_enabled = False
            lpipss = []
        elif lpips_value is not None:
            lpipss.append(lpips_value)

        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        entry = {"PSNR": psnr_value, "SSIM": ssim_value}
        if lpips_value is not None:
            entry["LPIPS"] = lpips_value
        per_view[image_stem] = entry

    summary = {
        "PSNR": sum(psnrs) / len(psnrs),
        "SSIM": sum(ssims) / len(ssims),
        "per_view": per_view,
    }
    if lpipss:
        summary["LPIPS"] = sum(lpipss) / len(lpipss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_view'}, indent=2))


if __name__ == "__main__":
    main()
