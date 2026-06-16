#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and confidence-aware evaluations.")
    parser.add_argument("--pairs", nargs="+", required=True, help="scene=baseline_eval.json,confidence_eval.json")
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data.items() if k != "per_view"}


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.pairs:
        name, pair = item.split("=", 1)
        baseline_path, confidence_path = pair.split(",", 1)
        baseline = load_metrics(Path(baseline_path))
        confidence = load_metrics(Path(confidence_path))
        row = {
            "scene": name,
            "baseline": baseline,
            "confidence": confidence,
            "delta": {
                "PSNR": confidence["PSNR"] - baseline["PSNR"],
                "SSIM": confidence["SSIM"] - baseline["SSIM"],
            },
        }
        rows.append(row)

    avg_delta_psnr = sum(row["delta"]["PSNR"] for row in rows) / len(rows)
    avg_delta_ssim = sum(row["delta"]["SSIM"] for row in rows) / len(rows)
    summary = {"rows": rows, "average_delta": {"PSNR": avg_delta_psnr, "SSIM": avg_delta_ssim}}

    for row in rows:
        print(f"[{row['scene']}] baseline PSNR={row['baseline']['PSNR']:.4f} SSIM={row['baseline']['SSIM']:.4f}")
        print(f"[{row['scene']}] confidence PSNR={row['confidence']['PSNR']:.4f} SSIM={row['confidence']['SSIM']:.4f}")
        print(f"[{row['scene']}] delta PSNR={row['delta']['PSNR']:+.4f} SSIM={row['delta']['SSIM']:+.4f}\n")
    print(f"Average delta: PSNR={avg_delta_psnr:+.4f}, SSIM={avg_delta_ssim:+.4f}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
