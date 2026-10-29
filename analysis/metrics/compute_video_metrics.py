#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Recompute registration-quality metrics directly from already-saved
motion-corrected videos, with no rerun of registration and no cross-method
comparison -- each video is scored independently against its own
_shift.npy sidecar, exactly like cli.py / cli_baseline.py score it during a
live run. Use this to (re)build a benchmark CSV when the videos already
exist on disk but their metrics are missing, incomplete, or need
recomputing with an updated metric function.

Expects the on-disk layout produced by cli.py / cli_baseline.py:
    <root>/<folder>/<stem>/<stem>_<method>.h5             (required: warped video)
    <root>/<folder>/<stem>/<stem>_<method>_shift.npy       (optional: (T, 2) per-frame shift)
    <root>/<folder>/<stem>/<stem>_<method>_points.npy      (optional: (N, 2) seed points)

Pass one or more --root directories; every <method> video found under each
is scored on its own.

Usage
-----
python compute_video_metrics.py \\
    --root /path/to/output_dir \\
    --output benchmark_recomputed.csv
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
from utils.utils import video_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute registration metrics directly from saved motion-corrected "
            "videos, each scored independently (no cross-method shared margin)."
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
        default="benchmark_recomputed.csv",
        help="Path for the output summary CSV (default: benchmark_recomputed.csv). "
        "The per-frame CSV is written alongside with a _perframe suffix.",
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.output)
    perframe_path = summary_path.parent / (summary_path.stem + "_perframe.csv")

    summary_rows: list[dict] = []
    perframe_rows: list[dict] = []

    for root_str in args.root:
        root = Path(root_str)
        for folder_name, stem, clip_dir, video_paths in find_clip_dirs(root):
            for video_path in video_paths:
                method = _method_name(video_path, stem)
                print(f"  {folder_name}/{stem} [{method}] ...", end="", flush=True)
                try:
                    warped = _load_video(video_path)
                except Exception as e:
                    print(f" could not load: {e}", file=sys.stderr)
                    continue

                shift_path = clip_dir / f"{stem}_{method}_shift.npy"
                shift = np.load(shift_path) if shift_path.is_file() else None

                points_path = clip_dir / f"{stem}_{method}_points.npy"
                n_points = int(np.load(points_path).shape[0]) if points_path.is_file() else 0

                try:
                    metrics = video_metrics(
                        warped, mode="ref_first", return_per_frame=True, drift=shift
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
                        "n_points": n_points,
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
