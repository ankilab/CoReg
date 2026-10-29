#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""napari widget wrapping CoReg's headless CoTracker pipeline.

This module only imports the Qt-free pieces of the main CoReg package
(``utils.cotracker_pipeline``, ``utils.presets``) -- never ``gui``,
``controller``, or anything else that pulls in PyQt6. That keeps this
plugin installable (and its own Qt backend, whichever one napari picks)
fully independent of the standalone CoReg GUI's PyQt6 dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari

try:
    from utils.cotracker_pipeline import run_cotracker_pipeline
    from utils.presets import (
        preset_features,
        preset_greedy_motion,
        preset_matchanything,
        preset_motion,
        preset_regular_grid,
        preset_regular_grid_spacing,
        preset_segmentation,
    )
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - surfaced in the widget UI instead
    run_cotracker_pipeline = None
    preset_features = preset_greedy_motion = preset_matchanything = preset_motion = None
    preset_regular_grid = preset_regular_grid_spacing = preset_segmentation = None
    _IMPORT_ERROR = exc

# CoReg's 7 user-facing point-seeding presets ("random" is excluded: it's only
# an internal empty-result fallback in utils/presets.py, not a real preset).
# "matchanything" soft-falls-back to shi_tomasi if `transformers` is missing.
PRESET_CHOICES = [
    "regular_grid",
    "regular_grid_spacing",
    "shi_tomasi",
    "motion",
    "greedy_motion",
    "segmentation",
    "matchanything",
]


def _video_to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """First frame, as the uint8 grayscale image preset_features/shi-tomasi expect."""
    f = frame[..., :3].mean(axis=-1) if frame.ndim == 3 else frame
    f = f.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi > lo:
        f = (f - lo) / (hi - lo) * 255.0
    return f.astype(np.uint8)


def _video_to_gray_stack(data: np.ndarray) -> np.ndarray:
    """Full (T,H,W) stack as uint8 grayscale, for the two presets that seed
    points from inter-frame motion (preset_motion/preset_greedy_motion) --
    unlike the single-frame presets, these need every frame, not just the
    first."""
    arr = data[..., :3].mean(axis=-1) if data.ndim == 4 else data
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def _video_to_tensor(data: np.ndarray):
    """napari Image layer data, (T,H,W) or (T,H,W,3), to a (T,C,H,W) float32 tensor."""
    import torch

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:  # (T, H, W) -> (T, 1, H, W)
        arr = arr[:, None, :, :]
    elif arr.ndim == 4:  # (T, H, W, C) -> (T, C, H, W)
        arr = np.moveaxis(arr, -1, 1)
    else:
        raise ValueError(f"Expected a (T,H,W) or (T,H,W,C) stack, got shape {arr.shape}")
    return torch.from_numpy(arr)


def _tracks_to_napari_tracks(tracks: np.ndarray) -> np.ndarray:
    """(T, N, 2) xy tracks -> napari Tracks layer data: (T*N, 4) rows of
    (track_id, t, y, x)."""
    n_frames, n_points, _ = tracks.shape
    t_idx, p_idx = np.meshgrid(np.arange(n_frames), np.arange(n_points), indexing="ij")
    x = tracks[..., 0]
    y = tracks[..., 1]
    return np.stack([p_idx.ravel(), t_idx.ravel(), y.ravel(), x.ravel()], axis=1)


def _run_coreg(
    viewer: "napari.viewer.Viewer",
    video: "napari.layers.Image",
    preset: str = "regular_grid",
    grid_rows: int = 16,
    grid_cols: int = 16,
    spacing_px: int = 20,
    num_points: int = 200,
    min_dist: float = 10.0,
    df_sigma: float = 5.0,
    device: str = "auto",
    offline: bool = True,
) -> None:
    """Track *video* with CoTracker and add the corrected result (+ tracked
    points) back into the viewer as new layers."""
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "napari-coreg could not import CoReg's utils.cotracker_pipeline / "
            "utils.presets. Install the main 'coreg' package first (pip install "
            "-e <path-to-registration_gui>), then restart napari."
        ) from _IMPORT_ERROR

    data = np.asarray(video.data)
    if data.ndim not in (3, 4):
        raise ValueError(f"'{video.name}' must be a (T,H,W) or (T,H,W,C) stack, got {data.shape}")

    first_gray = _video_to_gray_u8(data[0])
    H, W = first_gray.shape

    # Mirrors controller.py's preset dispatch, same utils/presets.py functions, called directly instead of via Qt.
    if preset == "regular_grid":
        queries = preset_regular_grid(H, W, grid_rows, grid_cols)
    elif preset == "regular_grid_spacing":
        queries = preset_regular_grid_spacing(H, W, spacing_px)
    elif preset == "shi_tomasi":
        queries = preset_features(first_gray, num_points=num_points, min_dist=min_dist)
    elif preset == "motion":
        queries = preset_motion(_video_to_gray_stack(data), num_points=num_points)
    elif preset == "greedy_motion":
        queries = preset_greedy_motion(_video_to_gray_stack(data), num_points=num_points)
    elif preset == "segmentation":
        queries = preset_segmentation(first_gray, num_points=num_points)
    elif preset == "matchanything":
        queries = preset_matchanything(first_gray, num_points=num_points, min_dist=min_dist)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if queries is None or len(queries) == 0:
        raise RuntimeError(f"Preset '{preset}' found no query points for '{video.name}'.")

    video_tensor = _video_to_tensor(data)
    dev = None if device == "auto" else device

    warped_np, tracks_np, _drift_np = run_cotracker_pipeline(
        video_tensor,
        queries,
        df_sigma=df_sigma,
        device=dev,
        offline=offline,
        progress_cb=lambda msg: print(f"[napari-coreg] {msg}"),
    )

    if warped_np.ndim == 4:  # (T, C, H, W) -> squeeze back to the layer's own ndim
        if data.ndim == 3:
            corrected = warped_np.mean(axis=1)
        else:
            corrected = np.moveaxis(warped_np, 1, -1)
    else:
        corrected = warped_np

    viewer.add_image(corrected, name=f"{video.name} [CoReg corrected]")
    viewer.add_tracks(_tracks_to_napari_tracks(tracks_np), name=f"{video.name} [CoReg tracks]")


make_coreg_widget = magic_factory(
    _run_coreg,
    preset={"choices": PRESET_CHOICES},
    device={"choices": ["auto", "cuda", "cpu"]},
    call_button="Run CoReg",
)
