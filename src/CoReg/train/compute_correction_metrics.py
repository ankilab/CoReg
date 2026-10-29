#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Generates the motion-corrected (TPS-warped) video for baseline, corr_mlp
and updateformer tracks on every soba_fullsequence video, then computes the
real registration-quality metrics (SSIM, PSNR, ZNCC, MI, crispness, mean
frame-to-mean correlation, residual flow) on each -- not the tracking-only
proxies (jitter/displacement) used earlier, but the actual output quality of
the thing this project exists to produce: a motion-corrected video.

Reuses utils.mc_cotracker.warping() (the same TPS warp the GUI's "CoReg"
view computes) driven by each model's already-tracked points
(train/compare_checkpoints.py's saved tracks/*.npz, regular_grid preset),
and utils.utils.video_metrics() (via metrics_common.compute_metrics) for the
metrics themselves.

Output matches cli.py's own convention exactly, not a bespoke layout, so any
existing tooling built around it (metrics_common.find_clip_dirs(),
compute_video_metrics.py, compute_comparison_metrics.py) can read this
directly:
  <out-dir>/corrected/<folder>/<stem>/<stem>_<preset>.h5   -- save_video_hdf5()
  <out-dir>/corrected/<folder>/<stem>/<stem>_<preset>_shift.npy  -- drift sidecar
  <out-dir>/corrected/<folder>/<stem>/<stem>_<preset>_tracks.h5  -- cli._save_tracks_overlay_video()
  <out-dir>/correction_metrics.csv  -- metrics_common.SUMMARY_FIELDS columns

Here "preset" (the CLI's own field name -- see cli.py's writer.writerow) is
overloaded to mean which CoTracker checkpoint produced the tracks
(baseline / corr_mlp_500steps / updateformer_2000steps), the same way the
CLI's own multi-method runs reuse "preset" for whichever registration method
produced that row, not the query-point preset.

Usage
-----
python -m train.compute_correction_metrics \\
    --input-dir /path/to/videos --tracks-dir /path/to/finetuning/tracks \\
    --out-dir /path/to/finetuning
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from cli import _save_tracks_overlay_video
from metrics_common import compute_metrics
from train.compare_checkpoints import _discover_videos, _load_video
from utils.mc_cotracker import warping
from utils.utils import save_video_hdf5

MODELS = ["baseline", "corr_mlp_500steps", "updateformer_2000steps"]
METRIC_KEYS = ["ssim", "psnr", "zncc", "mi", "crispness_mean", "corr_mean", "residual_flow"]
# Matches cli.py's own inline fieldnames list exactly (metrics_common.SUMMARY_FIELDS is the same minus elapsed_s).
FIELDNAMES = ["folder", "file", "preset", "n_frames", "n_points", "elapsed_s", *METRIC_KEYS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Root folder with low_movement/strong_movement subfolders of videos.")
    ap.add_argument("--tracks-dir", required=True, help="compare_checkpoints.py's saved tracks/*.npz directory.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--smoothing", type=float, default=5.0, help="TPS Gaussian smoothing sigma (matches the GUI's default df_sigma).")
    ap.add_argument("--save-videos", action="store_true", help="Also write each corrected video to disk as gzip-compressed HDF5, matching cli.py's own save_video_hdf5() output exactly.")
    ap.add_argument("--videos", nargs="*", default=None, help="Optional subset of video stems (for testing).")
    ap.add_argument("--models", nargs="*", default=None, help="Optional subset of model names (for testing).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = args.models or MODELS
    tracks_dir = Path(args.tracks_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    corrected_dir = out_root / "corrected"

    videos = _discover_videos(Path(args.input_dir))
    if args.videos:
        wanted = set(args.videos)
        videos = [(p, c) for p, c in videos if p.stem in wanted]
    print(f"{len(videos)} videos, {len(models)} models")

    rows = []
    for vpath, category in videos:
        print(f"\n=== {category}/{vpath.name} ===")
        video = _load_video(vpath, args.size, None).to(device)  # (T,3,H,W) [0,255]
        T = video.shape[0]
        print(f"  {T} frames")

        for model_name in models:
            npz_path = tracks_dir / f"{vpath.stem}_regular_grid_{model_name}.npz"
            if not npz_path.exists():
                print(f"  missing {npz_path.name}, skipping")
                continue
            d = np.load(npz_path)
            n_pts = d["tracks"].shape[1]
            tracks_t = torch.from_numpy(d["tracks"]).float().unsqueeze(0).to(device)

            t0 = time.perf_counter()
            warped, flows, drift = warping(tracks_t, video, smoothing=args.smoothing)
            warped_np = warped.detach().cpu().numpy()
            drift_np = drift.detach().cpu().numpy()
            elapsed_s = time.perf_counter() - t0

            m = compute_metrics(warped_np, drift=drift_np)
            rows.append({
                "folder": category, "file": vpath.stem, "preset": model_name,
                "n_frames": T, "n_points": n_pts, "elapsed_s": elapsed_s,
                **{k: m[k] for k in METRIC_KEYS},
            })
            print(f"  {model_name}: " + " ".join(f"{k}={m[k]:.4f}" for k in METRIC_KEYS))

            if args.save_videos:
                clip_dir = corrected_dir / category / vpath.stem
                clip_dir.mkdir(parents=True, exist_ok=True)
                save_video_hdf5(warped_np, str(clip_dir / f"{vpath.stem}_{model_name}.h5"))
                np.save(str(clip_dir / f"{vpath.stem}_{model_name}_shift.npy"), drift_np)
                _save_tracks_overlay_video(
                    video.detach().cpu(), d["tracks"],
                    clip_dir / f"{vpath.stem}_{model_name}_tracks.h5",
                )

            if device == "cuda":
                del warped, flows, drift, tracks_t
                torch.cuda.empty_cache()

    out_csv = out_root / "correction_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
