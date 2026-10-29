#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Shared building blocks for the registration-metrics scripts.

Used by compute_video_metrics.py, compute_comparison_metrics.py (on-disk
clip/video discovery and CSV field layout) and cli.py / cli_baseline.py
(the shared metrics wrapper around utils.utils.video_metrics).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile

from utils.utils import video_metrics

SUMMARY_FIELDS = [
    "folder",
    "file",
    "preset",
    "n_frames",
    "n_points",
    "ssim",
    "psnr",
    "mse",
    "zncc",
    "mi",
    "crispness_mean",
    "corr_mean",
    "residual_flow",
]
PERFRAME_FIELDS = [
    "folder",
    "file",
    "preset",
    "frame",
    "ssim",
    "psnr",
    "mse",
    "zncc",
    "mi",
    "corr_mean",
    "residual_flow",
]

VIDEO_EXTENSIONS = (".h5", ".hdf5", ".tif", ".tiff")


def compute_metrics(warped, drift=None) -> dict[str, float]:
    """Compute the shared registration metrics for a registered video."""
    return video_metrics(warped, mode="ref_first", return_per_frame=True, drift=drift)


def _load_video(path: Path) -> np.ndarray:
    """Load a saved warped-video file back to a float32 (T, H, W) array."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(str(path)).astype(np.float32)
    elif ext in (".h5", ".hdf5"):
        with h5py.File(str(path), "r") as f:
            key = "data" if "data" in f else next(iter(f.keys()))
            arr = np.array(f[key]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    if arr.ndim == 4:
        arr = arr.mean(axis=-1) if arr.shape[-1] in (1, 3) else arr[:, 0]
    return arr


def _method_name(video_path: Path, stem: str) -> str:
    """<stem>_<method>.<ext> -> <method>"""
    name = video_path.stem
    prefix = f"{stem}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def find_clip_dirs(root: Path):
    """Yield (folder_name, stem, clip_dir, video_paths) for every
    <root>/<folder>/<stem>/ directory holding at least one warped-video file."""
    if not root.is_dir():
        return
    for folder_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in folder_dir.iterdir() if p.is_dir()):
            stem = clip_dir.name
            videos = [
                p
                for ext in VIDEO_EXTENSIONS
                for p in clip_dir.glob(f"{stem}_*{ext}")
                if "_tracks" not in p.stem
            ]
            if videos:
                yield folder_dir.name, stem, clip_dir, sorted(set(videos))
