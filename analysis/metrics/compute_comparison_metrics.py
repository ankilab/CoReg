#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Recompute registration-quality metrics across multiple methods for the same
clip, using ONE shared border-crop margin per clip so scores are directly
comparable -- even when the methods were run as separate jobs (e.g. cli.py for
CoTracker and cli_baseline.py for NoRMCorre/pyflowreg/ANTs, run independently,
possibly on different machines/queues, at different times).

Expects the on-disk layout produced by cli.py / cli_baseline.py:
    <root>/<folder>/<stem>/<stem>_<method>.h5             (required: warped video)
    <root>/<folder>/<stem>/<stem>_<method>_shift.npy       (optional: (T, 2) per-frame shift)
    <root>/<folder>/<stem>/<stem>_<method>_points.npy      (optional: (N, 2) seed points)

Pass one or more --root directories; clips are matched by (folder, stem)
across all of them, so a CoTracker run's output dir and a baseline run's
output dir can be combined even though they were produced separately.

Usage
-----
python compute_comparison_metrics.py \\
    --root /path/to/cotracker_out /path/to/baseline_out \\
    --output benchmark_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from metrics_common import (
    PERFRAME_FIELDS,
    SUMMARY_FIELDS,
    VIDEO_EXTENSIONS,
    _load_video,
    _method_name,
    find_clip_dirs,
)
from utils.utils import shared_crop_drift, video_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute registration metrics across methods for the same clip, "
            "using one shared border-crop margin per clip."
        )
    )
    parser.add_argument(
        "--root",
        nargs="+",
        required=True,
        help="One or more output directories produced by cli.py / cli_baseline.py "
        "(each searched as <root>/<folder>/<stem>/).",
    )
    parser.add_argument(
        "--output",
        default="benchmark_compare.csv",
        help="Path for the output summary CSV (default: benchmark_compare.csv). "
        "The per-frame CSV is written alongside with a _perframe suffix.",
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.output)
    perframe_path = summary_path.parent / (summary_path.stem + "_perframe.csv")

    summary_rows: list[dict] = []
    perframe_rows: list[dict] = []

    # Merge clip directories from all roots: (folder, stem) -> list of (clip_dir, video_paths)
    clips: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    for root_str in args.root:
        root = Path(root_str)
        for folder_name, stem, clip_dir, video_paths in find_clip_dirs(root):
            key = (folder_name, stem)
            clips.setdefault(key, [])
            for vp in video_paths:
                clips[key].append((clip_dir, vp))

    for (folder_name, stem), entries in sorted(clips.items()):
        methods_data: dict[str, dict] = {}
        for clip_dir, video_path in entries:
            method = _method_name(video_path, stem)
            try:
                warped = _load_video(video_path)
            except Exception as e:
                print(f"Warning: could not load {video_path}: {e}", file=sys.stderr)
                continue

            shift_path = clip_dir / f"{stem}_{method}_shift.npy"
            shift = np.load(shift_path) if shift_path.is_file() else None

            points_path = clip_dir / f"{stem}_{method}_points.npy"
            n_points = int(np.load(points_path).shape[0]) if points_path.is_file() else 0

            methods_data[method] = {
                "warped": warped,
                "shift": shift,
                "n_points": n_points,
            }

        if not methods_data:
            continue

        T = max(v["warped"].shape[0] for v in methods_data.values())
        shared_drift = shared_crop_drift(
            [v["shift"] for v in methods_data.values()], T
        )

        for method, data in sorted(methods_data.items()):
            warped = data["warped"]
            print(f"  {folder_name}/{stem} [{method}] ...", end="", flush=True)
            try:
                metrics = video_metrics(
                    warped, mode="ref_first", return_per_frame=True, drift=shared_drift
                )
                print(f" ssim={metrics['ssim']:.4f}")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            summary_rows.append(
                {
                    "folder": folder_name,
                    "file": stem,
                    "preset": method,
                    "n_frames": int(warped.shape[0]),
                    "n_points": data["n_points"],
                    "ssim": metrics["ssim"],
                    "psnr": metrics["psnr"],
                    "mse": metrics["mse"],
                    "zncc": metrics["zncc"],
                    "mi": metrics["mi"],
                    "crispness_mean": metrics["crispness_mean"],
                    "corr_mean": metrics["corr_mean"],
                    "residual_flow": metrics["residual_flow"],
                }
            )

            pf = metrics["per_frame"]
            for t, (ssim_t, psnr_t, mse_t, zncc_t, mi_t, corr_t, flow_t) in enumerate(
                zip(
                    pf["ssim"],
                    pf["psnr"],
                    pf["mse"],
                    pf["zncc"],
                    pf["mi"],
                    pf["corr_mean"],
                    pf["residual_flow"],
                )
            ):
                perframe_rows.append(
                    {
                        "folder": folder_name,
                        "file": stem,
                        "preset": method,
                        "frame": t,
                        "ssim": ssim_t,
                        "psnr": psnr_t,
                        "mse": mse_t,
                        "zncc": zncc_t,
                        "mi": mi_t,
                        "corr_mean": corr_t,
                        "residual_flow": flow_t,
                    }
                )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    with perframe_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERFRAME_FIELDS)
        w.writeheader()
        w.writerows(perframe_rows)

    print(f"\nWrote {len(summary_rows)} rows  -> {summary_path}")
    print(f"Wrote {len(perframe_rows)} rows -> {perframe_path}")


if __name__ == "__main__":
    main()
