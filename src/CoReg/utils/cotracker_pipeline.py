#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Pure-Python CoTracker + warping pipeline with no Qt dependency.

Used by the CLI so it can run without PyQt6 on headless systems (e.g. HPC).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from utils.mc_cotracker import cotrack_model, warping
from utils.utils import _to_numpy

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def run_cotracker_pipeline(
    video_in: torch.Tensor,
    queries: torch.Tensor | None,
    df_sigma: float = 5.0,
    device: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
    offline: bool = True,
    online_chunk_size: int = 150,
    video_out: torch.Tensor | None = None,
    grid_size: int = 0,
    weights_path: str | None = None,
    compute_warp: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Run CoTracker tracking (+ dense warping, unless skipped) without any
    Qt dependency.

    Args:
        video_in:    (T, C, H, W) float32 tensor.
        queries:     (N, 2) or (1, N, 2) float32 tensor of (x, y) seed points.
        df_sigma:    Gaussian smoothing sigma for the displacement field.
        device:      'cuda', 'cpu', or None (auto-detect).
        progress_cb: Optional callable(message: str) for progress reporting.
        compute_warp: If False, skip the dense per-pixel warp entirely and
                      return None for warped_np/drift_np -- callers that only
                      want the tracked points (e.g. cell_annotator.py, which
                      never uses warped_np) avoid moving a second full-video
                      tensor onto the GPU and running the TPS flow-field
                      computation over it, both scaling with clip length and
                      otherwise wasted. Default True keeps existing callers
                      (e.g. the CLI, which does use warped_np) unchanged.

    Returns:
        (warped_np, tracks_np, drift_np) where warped_np is (T, C, H, W)
        float32 (or None if compute_warp=False), tracks_np is (T, N, 2)
        float32, and drift_np is (T, 2) float32 per-frame (dx, dy) (or None
        if compute_warp=False) — see mc_cotracker.warping().
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    if device is None:
        device = DEFAULT_DEVICE

    video = video_in.to(device) if isinstance(video_in, torch.Tensor) else video_in

    # Ensure 3-channel input for CoTracker
    if isinstance(video, torch.Tensor) and video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1)

    # CoTracker expects float in [0, 255]; normalize arbitrary-range input (e.g. 16-bit TIFFs)
    if isinstance(video, torch.Tensor) and video.dtype != torch.uint8:
        v_min = video.amin()
        v_max = video.amax()
        if v_max > v_min:
            video = (video - v_min) / (v_max - v_min) * 255.0

    # Normalise queries to (1, N, 3): [t=0, x, y]  — None is allowed when grid_size > 0
    if queries is not None:
        if isinstance(queries, np.ndarray):
            queries = torch.from_numpy(queries)
        if queries.ndim == 2:
            queries = queries.unsqueeze(0)   # (1, N, 2)
        queries = queries.to(device).float()
        if queries.shape[-1] == 2:
            b, n, _ = queries.shape
            t0 = torch.zeros((b, n, 1), device=device, dtype=queries.dtype)
            queries = torch.cat([t0, queries], dim=-1)  # (1, N, 3)

    _log("Tracking running on " + device.upper())

    def _tracker_cb(p: float, stage: str, **meta):
        stage_text = {
            "start": "Preparing tracking...",
            "fmaps": "Computing features...",
            "iters": "Refining tracks...",
            "done":  "Tracking done",
        }.get(stage, f"Tracking: {stage}")
        if stage == "fmaps":
            chunk = int(meta.get("chunk", 0))
            num_chunks = int(meta.get("num_chunks", 0))
            if num_chunks > 0:
                stage_text = f"Computing features (chunk {chunk}/{num_chunks})..."
        elif stage == "iters":
            it = int(meta.get("it", 0))
            iters = int(meta.get("iters", 0))
            if iters > 0:
                stage_text = f"Refining tracks (iter {it}/{iters})..."
        _log(stage_text)

    pred_tracks, _ = cotrack_model(
        video,
        queries=queries,
        device=device,
        progress_cb=_tracker_cb,
        offline=offline,
        online_chunk_size=online_chunk_size,
        grid_size=grid_size,
        weights_path=weights_path,
    )

    tracks_raw = pred_tracks.detach().cpu().numpy()
    tracks_gui = tracks_raw[0] if tracks_raw.ndim == 4 else tracks_raw

    if not compute_warp:
        # Free GPU memory explicitly -- without it, repeated calls (e.g. tracking cells one after another)
        # accumulate cached-but-unused GPU memory until a later one fails to allocate.
        del video, pred_tracks
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log("Done")
        return None, tracks_gui, None

    tracks_for_warp = tracks_gui[None, ...]  # (1, T, N, 2)

    _log("Warping / Displacement optimization ...")

    tracks_tensor = torch.from_numpy(tracks_for_warp).to(device).float()
    frames_to_warp = video_out if video_out is not None else video_in
    frames_to_warp = frames_to_warp.to(device) if isinstance(frames_to_warp, torch.Tensor) else frames_to_warp

    warped_video, _, drift = warping(tracks_tensor, frames_to_warp, smoothing=df_sigma)

    warped_np = (
        warped_video.detach().cpu().numpy()
        if isinstance(warped_video, torch.Tensor)
        else np.asarray(warped_video)
    )
    if warped_np.ndim >= 4 and warped_np.shape[0] == 1:
        warped_np = warped_np[0]

    drift_np = _to_numpy(drift)

    _log("Done")
    return warped_np, tracks_gui, drift_np
