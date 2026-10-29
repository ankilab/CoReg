#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Discover multiple short frame-range subsets from one or more TIFF
videos, for CoTracker adaptation (train/run_adapt.py) -- rather than
test-time-adapting to just the first N frames of a single clip, cycle
gradient steps across subsets drawn from (potentially many) real videos
so the adaptation generalizes beyond one clip's own noise/contrast quirks.

load_tiff() in utils/utils.py only loads the first N frames of a stack
(no start offset), so it can't pull an arbitrary window -- _load_tiff_window
below reads a specific (start, length) slice directly via tifffile.memmap,
reusing to_uint8_contrast for the same percentile-stretch convention
load_tiff() itself uses, for consistency with the rest of the project.

Query points for each subset are a combination of a regular grid (uniform
spatial coverage, including background) and segmentation centroids
(preset_segmentation's Otsu-thresholded connected components -- lands
points directly on actual structures, e.g. cell bodies, that a grid alone
can land between). See _grid_plus_segmentation_points().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from scipy.spatial import cKDTree

from utils.presets import clip_border, preset_regular_grid, preset_segmentation
from utils.utils import memmap_or_load_tiff, to_uint8_contrast


def _grid_plus_segmentation_points(
    gray_u8: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    seg_points: int,
    min_sep: float = 6.0,
) -> tuple[np.ndarray, int]:
    """Union of a regular grid and segmentation centroids. Grid points that
    fall within *min_sep* px of a segmentation centroid are dropped, so a
    structure doesn't end up with two near-duplicate query points on it.
    Returns (points, n_segmentation_points) -- the count is kept separate
    so callers can log the grid/segmentation split.
    """
    H, W = gray_u8.shape[:2]
    grid_pts = preset_regular_grid(H, W, grid_rows, grid_cols)
    seg_pts = preset_segmentation(gray_u8, num_points=seg_points)
    if len(seg_pts) == 0:
        return grid_pts, 0
    if len(grid_pts) == 0:
        return seg_pts, len(seg_pts)
    tree = cKDTree(seg_pts)
    dist, _ = tree.query(grid_pts, k=1)
    keep_grid = grid_pts[dist >= min_sep]
    return np.concatenate([keep_grid, seg_pts], axis=0).astype(np.float32), len(seg_pts)


@dataclass
class ClipSubset:
    path: Path
    start_frame: int
    n_frames: int
    video: torch.Tensor       # (1, T, 3, H, W) float32 in [0, 255]
    queries_xy: torch.Tensor  # (N, 2)

    @property
    def label(self) -> str:
        return f"{self.path.stem}[{self.start_frame}:{self.start_frame + self.n_frames}]"


def _load_tiff_window(path: Path, start: int, length: int, size_wh: tuple[int, int]) -> tuple[torch.Tensor, np.ndarray]:
    """Returns (video, projection_u8).
    video: (1, T, 3, H, W) float32 in [0, 255] for frames [start, start+length).
    projection_u8: (H, W) uint8 max-intensity projection across the window,
    used for segmentation-based query points -- a single frame can miss a
    cell that's dim or out of focus at that instant; the max-projection
    surfaces anything visible anywhere in the window.
    """
    mm = memmap_or_load_tiff(str(path))
    total = mm.shape[0] if mm.ndim == 3 else 1
    end = min(start + length, total)

    out_w, out_h = size_wh
    frames = np.empty((end - start, out_h, out_w), dtype=np.float32)
    for i, src_i in enumerate(range(start, end)):
        frame = mm[src_i] if mm.ndim == 3 else mm
        if not frame.dtype.isnative:
            frame = frame.astype(frame.dtype.type)
        if frame.dtype != np.uint8:
            frame = to_uint8_contrast(frame)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames[i] = frame.astype(np.float32)

    projection_u8 = frames.max(axis=0).astype(np.uint8)
    video = torch.from_numpy(frames).unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, H, W), already [0, 255]
    return video.unsqueeze(0), projection_u8  # (1, T, 3, H, W)


def discover_subsets(
    root_dir: str | None = None,
    *,
    paths: list[Path] | None = None,
    subset_frames: int = 50,
    subsets_per_video: int = 2,
    size_wh: tuple[int, int] = (256, 256),
    grid_rows: int = 10,
    grid_cols: int = 10,
    seg_points: int = 50,
    seed: int = 0,
    log_fn=print,
) -> list[ClipSubset]:
    """Find every .tif/.tiff under root_dir (or use the given *paths*
    directly -- a single explicit file works too, single-clip mode is just
    the one-video case of this), and for each one carve out
    *subsets_per_video* non-overlapping *subset_frames*-length windows
    (evenly spaced across the video, not just the first subset_frames
    frames every time, so later parts of each clip get covered too).
    """
    if paths is None:
        if root_dir is None:
            raise ValueError("discover_subsets needs either root_dir or paths")
        paths = sorted(
            p for p in Path(root_dir).rglob("*") if p.suffix.lower() in (".tif", ".tiff")
        )
    else:
        paths = sorted(Path(p) for p in paths)
    if not paths:
        raise ValueError(f"No .tif/.tiff files found" + (f" under {root_dir}" if root_dir else ""))

    rng = np.random.default_rng(seed)
    subsets: list[ClipSubset] = []

    for path in paths:
        with tifffile.TiffFile(str(path)) as tf:
            total = len(tf.pages)
        if total < subset_frames:
            log_fn(f"[discover] skipping {path.name}: only {total} frames (< subset_frames={subset_frames})")
            continue

        # Evenly spaced, non-overlapping start points across the video.
        max_start = total - subset_frames
        if subsets_per_video == 1:
            starts = [0]
        else:
            starts = [int(round(i * max_start / (subsets_per_video - 1))) for i in range(subsets_per_video)]
            starts = sorted(set(starts))  # dedupe if the video is short relative to subsets_per_video

        for start in starts:
            video, projection_u8 = _load_tiff_window(path, start, subset_frames, size_wh)
            H, W = video.shape[-2], video.shape[-1]
            pts, n_seg = _grid_plus_segmentation_points(projection_u8, grid_rows, grid_cols, seg_points)
            pts = clip_border(pts, H, W, border=4)
            queries_xy = torch.from_numpy(pts).float()
            subsets.append(ClipSubset(path=path, start_frame=start, n_frames=subset_frames,
                                       video=video, queries_xy=queries_xy))
            log_fn(f"[discover] {path.name}[{start}:{start + subset_frames}]  "
                   f"{len(queries_xy)} query points ({len(queries_xy) - n_seg} grid, {n_seg} segmentation)")

    rng.shuffle(subsets)  # mix videos/subsets together, not grouped by source video
    return subsets
