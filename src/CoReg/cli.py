#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import presets as _presets
from utils.cotracker_pipeline import run_cotracker_pipeline
from utils.utils import (
    load_hdf5,
    load_tiff,
    load_video,
    memmap_or_load_tiff,
    resize_video_tensor_to_wh,
    save_video_hdf5,
)
from metrics_common import compute_metrics
from cli_baseline import BASELINE_METHODS, run_baseline
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SUPPORTED_VIDEO_EXT = [".avi", ".mp4"]
SUPPORTED_TIFF_EXT = [".tiff", ".tif"]
SUPPORTED_HDF5_EXT = [".h5", ".hdf5"]
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXT + SUPPORTED_TIFF_EXT + SUPPORTED_HDF5_EXT

COTRACKER_PRESETS = sorted(
    {
        "none",
        "regular_grid",
        "shi_tomasi",
        "motion",
        "segmentation",
        "matchanything",
        "greedy_motion",
    }
)

VALID_FILTERS = ("none", "gaussian", "sobel", "canny", "rgb_stack")


def get_total_frames(path: str) -> int:
    """Return total frame count for a file without loading all data into memory."""
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_TIFF_EXT:
        mm = memmap_or_load_tiff(path)
        return int(mm.shape[0]) if mm.ndim == 3 else 1
    if ext in SUPPORTED_HDF5_EXT:
        import h5py

        with h5py.File(path, "r") as f:
            for key in ("/data", "data", "/video", "video"):
                if key in f:
                    return int(f[key].shape[0])
        return 0
    if ext in SUPPORTED_VIDEO_EXT:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    return 0


def load_clip(
    path: str,
    num_frames: int | None = None,
    frame_start: int = 0,
    target_w: int | None = None,
    target_h: int | None = None,
) -> torch.Tensor:
    """Load a single file and return a (T, C, H, W) float32 tensor.

    *frame_start* sets the first frame to load (0-indexed).
    *num_frames* caps the number of frames loaded starting from *frame_start*.
    *target_w* / *target_h* resize every frame if both are provided.
    """
    ext = Path(path).suffix.lower()

    if ext in SUPPORTED_VIDEO_EXT:
        video = load_video(path, resize=False, max_frames=None)
        end = (frame_start + num_frames) if num_frames is not None else None
        video = video[frame_start:end]

    elif ext in SUPPORTED_TIFF_EXT:
        mm = memmap_or_load_tiff(path)
        end = (frame_start + num_frames) if num_frames is not None else None
        chunk = mm[frame_start:end] if mm.ndim == 3 else mm[np.newaxis]
        video = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        if video.ndim == 3:
            video = video.unsqueeze(1)

    elif ext in SUPPORTED_HDF5_EXT:
        import h5py

        with h5py.File(path, "r") as f:
            ds = None
            for key in ("/data", "data", "/video", "video"):
                if key in f:
                    ds = f[key]
                    break
            if ds is None:
                raise KeyError(f"No suitable dataset found in {path}")
            end = (frame_start + num_frames) if num_frames is not None else ds.shape[0]
            chunk = ds[frame_start:end]
        video = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        if video.ndim == 3:
            video = video.unsqueeze(1)

    else:
        raise ValueError(f"Unsupported extension: '{ext}'")

    if not isinstance(video, torch.Tensor):
        video = torch.from_numpy(np.asarray(video))
    if video.ndim == 3:
        video = video.unsqueeze(1)
    if video.ndim == 4 and video.shape[1] not in (1, 3) and video.shape[-1] in (1, 3):
        video = video.permute(0, 3, 1, 2)
    video = video.float()

    if target_w is not None and target_h is not None:
        video = resize_video_tensor_to_wh(video, target_w=target_w, target_h=target_h)

    return video


def iter_folder(
    folder: str,
    num_frames: int | None = None,
    chunk_frames: int | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    file_filter: list[str] | None = None,
    file_nth: int = 1,
) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Yield (folder_name, chunk_name, clip_tensor) for every supported file in *folder*.

    When *chunk_frames* is set each file is split into consecutive chunks of that
    size and yielded separately as '<stem>_chunk000.<ext>', '<stem>_chunk001.<ext>', ...
    *num_frames* limits the clip length (or chunk size when chunking is off).
    *file_filter* restricts processing to filenames in that list (stem or full name match).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if file_filter is not None:
        filter_set = {f.lower() for f in file_filter}
        files = [
            p
            for p in files
            if p.name.lower() in filter_set or p.stem.lower() in filter_set
        ]
    if file_nth > 1:
        files = files[::file_nth]
    if not files:
        raise ValueError(
            f"No supported files found in {folder}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    for fpath in files:
        try:
            if chunk_frames is not None:
                total = get_total_frames(str(fpath))
                starts = range(0, max(1, total), chunk_frames)
                for i, start in enumerate(starts):
                    chunk_name = f"{fpath.stem}_chunk{i:03d}{fpath.suffix}"
                    try:
                        clip = load_clip(
                            str(fpath),
                            num_frames=chunk_frames,
                            frame_start=start,
                            target_w=target_w,
                            target_h=target_h,
                        )
                        if int(clip.shape[0]) < chunk_frames:
                            print(
                                f"Skipping {chunk_name}: incomplete chunk "
                                f"({int(clip.shape[0])}/{chunk_frames} frames).",
                                file=sys.stderr,
                            )
                            continue
                        yield folder_path.name, chunk_name, clip
                    except Exception as e:
                        print(
                            f"Warning: could not load chunk {i} of {fpath.name}: {e}",
                            file=sys.stderr,
                        )
            else:
                clip = load_clip(
                    str(fpath),
                    num_frames=num_frames,
                    target_w=target_w,
                    target_h=target_h,
                )
                if num_frames is not None and int(clip.shape[0]) != num_frames:
                    print(
                        f"Skipping {fpath.name}: expected {num_frames} frames, "
                        f"got {int(clip.shape[0])}",
                        file=sys.stderr,
                    )
                    continue
                yield folder_path.name, fpath.name, clip
        except Exception as e:
            print(f"Warning: could not load {fpath.name}: {e}", file=sys.stderr)


"""Generation of Points"""


def _clip_to_gray_u8(clip: torch.Tensor) -> np.ndarray:
    """First frame of (T, C, H, W) -> uint8 grayscale."""
    frame = clip[0]
    gray = (
        (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()
        if frame.shape[0] == 3
        else frame[0].numpy()
    )
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        return ((gray - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


def _clip_to_gray_video(clip: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) -> (T, H, W) float32 grayscale."""
    gray = (
        0.299 * clip[:, 0] + 0.587 * clip[:, 1] + 0.114 * clip[:, 2]
        if clip.shape[1] == 3
        else clip[:, 0]
    )
    return gray.numpy().astype(np.float32)


def _save_points_overlay(
    clip: torch.Tensor,
    pts: np.ndarray,
    out_path: Path,
) -> None:
    """Save the first frame of *clip* with *pts* overlaid as a PNG."""
    frame = clip[0]  # (C, H, W)
    gray = (
        (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()
        if frame.shape[0] == 3
        else frame[0].numpy()
    )
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        gray = (gray - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1], s=12, c="lime", linewidths=0.5, edgecolors="black"
        )
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tracks_overlay_video(
    clip: torch.Tensor,
    tracks: np.ndarray,
    out_path: Path,
) -> None:
    """Save the original clip with full track paths overlaid as an HDF5 file.

    At frame t the full path from frame 0
    up to t is drawn as a line, with a filled dot at the current position.

    Args:
        clip:    (T, C, H, W) float tensor.
        tracks:  (T, N, 2) array of (x, y) coordinates.
        out_path: Destination .h5 path.
    """
    T, C, H, W = clip.shape
    N = tracks.shape[1]
    color = (0, 255, 0)

    # Accumulate only the new trail segment per frame -- old approach redrew all prior segments, O(T^2*N).
    # trail_mask tracks touched pixels so the trail composites onto each frame's own background.
    trail_color = np.zeros((H, W, 3), dtype=np.uint8)
    trail_mask = np.zeros((H, W), dtype=np.uint8)

    frames_out = []
    for t in range(T):
        # Normalize frame to uint8 RGB
        frame = clip[t]
        gray = (
            (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()
            if C == 3
            else frame[0].numpy()
        )
        mn, mx = float(gray.min()), float(gray.max())
        if mx > mn:
            gray = (gray - mn) / (mx - mn)
        img = (
            (np.stack([gray, gray, gray], axis=-1) * 255)
            .clip(0, 255)
            .astype(np.uint8)
            .copy()
        )

        if t > 0:
            for n in range(N):
                p1 = (int(round(tracks[t - 1, n, 0])), int(round(tracks[t - 1, n, 1])))
                p2 = (int(round(tracks[t, n, 0])), int(round(tracks[t, n, 1])))
                cv2.line(trail_color, p1, p2, color, 1)
                cv2.line(trail_mask, p1, p2, 255, 1)

        touched = trail_mask > 0
        img[touched] = trail_color[touched]

        for n in range(N):
            # Current position dot
            cx = int(round(tracks[t, n, 0]))
            cy = int(round(tracks[t, n, 1]))
            cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)

        frames_out.append(img)

    arr = np.stack(frames_out, axis=0)
    save_video_hdf5(arr, str(out_path))


def generate_points(
    clip: torch.Tensor,
    preset: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    num_points: int | None = None,
) -> np.ndarray:
    """Return (N, 2) seed points for *preset*.

    Preset can be a single name or two names joined with '+' to combine both
    point sets (e.g. 'regular_grid+shi_tomasi').
    Falls back to a regular grid when the preset produces no points.
    """
    # Combined preset: generate each sub-preset and concatenate
    if "+" in preset:
        parts = [
            generate_points(
                clip,
                p.strip(),
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                num_points=num_points,
            )
            for p in preset.split("+")
        ]
        combined = np.concatenate([p for p in parts if len(p) > 0], axis=0)
        if len(combined) == 0:
            print(
                f"Combined preset '{preset}' produced no points — falling back to random points.",
                file=sys.stderr,
            )
            _, _, H, W = clip.shape
            n = num_points if (num_points and num_points > 0) else grid_rows * grid_cols
            return _presets.preset_random(H, W, n)
        return combined.astype(np.float32)

    _, _, H, W = clip.shape
    n = num_points if (num_points and num_points > 0) else grid_rows * grid_cols
    min_dist = max(1.0, min(H, W) / (2.0 * max(grid_rows, grid_cols)))

    if preset == "regular_grid":
        return _presets.preset_regular_grid(H, W, grid_rows, grid_cols)

    gray_u8 = _clip_to_gray_u8(clip)

    if preset == "shi_tomasi":
        pts = _presets.preset_features(gray_u8, num_points=n, min_dist=min_dist)
    elif preset == "motion":
        pts = _presets.preset_motion(_clip_to_gray_video(clip), num_points=n)
    elif preset == "greedy_motion":
        pts = _presets.preset_greedy_motion(_clip_to_gray_video(clip), num_points=n)
    elif preset == "segmentation":
        pts = _presets.preset_segmentation(gray_u8, num_points=n)
    elif preset == "matchanything":
        pts = _presets.preset_matchanything(
            gray_u8, num_points=n, min_dist=min_dist * 2
        )
    else:
        pts = np.zeros((0, 2), dtype=np.float32)

    if len(pts) == 0:
        print(
            f"Preset '{preset}' produced no points — falling back to random points.",
            file=sys.stderr,
        )
        pts = _presets.preset_random(H, W, n)

    return pts.astype(np.float32)


"""Input filter"""


def apply_filter_to_video(clip: torch.Tensor, filter_name: str) -> torch.Tensor:
    """Return a filtered (T, 3, H, W) float32 tensor for use as tracking input.

    The original *clip* is left unchanged and should be used for warping.
    Filters match the GUI: Gaussian (sigma=1.2), Sobel magnitude, Canny edges.
    'rgb_stack' stacks grayscale / Sobel / Canny as the three channels so
    CoTracker receives complementary structural information in each channel.
    """
    filt = (filter_name or "none").lower()
    if filt == "none":
        return clip

    # Work on (T, H, W) in [0, 1]
    frames = clip[:, 0].detach().cpu().numpy()  # (T, H, W)

    def _to_u8(frame):
        mn, mx = float(frame.min()), float(frame.max())
        return (
            (np.clip((frame - mn) / (mx - mn + 1e-8), 0, 1) * 255).astype(np.uint8)
            if mx > mn
            else np.zeros_like(frame, dtype=np.uint8)
        )

    def _sobel_u8(gray_u8):
        sx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.normalize(
            cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    out = []
    for frame in frames:
        gray_u8 = _to_u8(frame)

        if filt == "gaussian":
            filtered = cv2.GaussianBlur(gray_u8, (0, 0), 1.2)
            out.append(filtered[:, :, None])
        elif filt == "sobel":
            out.append(_sobel_u8(gray_u8)[:, :, None])
        elif filt == "canny":
            out.append(cv2.Canny(gray_u8, 50, 150)[:, :, None])
        elif filt == "rgb_stack":
            # 3 channels: grayscale | Sobel | Canny
            ch0 = gray_u8
            ch1 = _sobel_u8(gray_u8)
            ch2 = cv2.Canny(gray_u8, 50, 150)
            out.append(np.stack([ch0, ch1, ch2], axis=-1))  # (H, W, 3)
        else:
            out.append(gray_u8[:, :, None])

    arr = np.stack(out, axis=0).astype(np.float32) / 255.0  # (T, H, W, 1or3)
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, 1or3, H, W)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


"""Registration and metrics"""


def run_cotracker(
    clip: torch.Tensor,
    preset: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    df_sigma: float = 5.0,
    num_points: int | None = None,
    force_cpu: bool = False,
    offline: bool = True,
    online_chunk_size: int = 150,
    input_filter: str = "none",
    cotracker_grid_size: int = 0,
    weights_path: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Run CoTracker on *clip* with *preset*.

    *input_filter* is applied to the tracking input only; warping always uses
    the original *clip* so the saved output is never the filtered version.

    When *cotracker_grid_size* > 0 CoTracker adds an N×N dense grid on top of
    any preset query points.  Pass preset="none" to use the grid exclusively.

    Returns:
        (warped, tracks, drift) where warped is the registered video as a
        numpy array, tracks is the (T, N, 2) track array (x, y), and drift is
        the (T, 2) per-frame (dx, dy) array from mc_cotracker.warping().
    """
    if preset == "none":
        queries = None
    else:
        pts = generate_points(
            clip,
            preset,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_points=num_points,
        )
        queries = torch.from_numpy(pts) if len(pts) > 0 else None
    device = "cpu" if force_cpu else None
    video_in = apply_filter_to_video(clip, input_filter)
    warped, tracks, drift = run_cotracker_pipeline(
        video_in=video_in,
        queries=queries,
        df_sigma=float(df_sigma),
        device=device,
        progress_cb=lambda msg: print(msg, file=sys.stderr) if msg else None,
        offline=offline,
        online_chunk_size=online_chunk_size,
        video_out=clip,
        grid_size=cotracker_grid_size,
        weights_path=weights_path,
    )
    return warped, tracks, drift


# Evaluation loop and CLI
def run_evaluation(
    folders: list[str],
    presets: list[str],
    *,
    baseline_methods: list[str] | None = None,
    run_name: str | None = None,
    num_frames: int | None = None,
    chunk_frames: int | None = None,
    file_filter: list[str] | None = None,
    grid_rows: int = 8,
    grid_cols: int = 8,
    df_sigma: float = 5.0,
    num_points: int | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    output_dir: str | None = None,
    output_csv: str | None = None,
    save_videos: bool = True,
    save_nth: int = 1,
    overwrite: bool = False,
    force_cpu: bool = False,
    offline: bool = True,
    online_chunk_size: int = 150,
    input_filter: str = "none",
    cotracker_grid_size: int = 0,
    weights_path: str | None = None,
) -> None:
    """Run CoTracker with each preset (and optionally NoRMCorre/pyflowreg/ANTs/LDDMM
    baselines) on every clip in *folders* and write metrics to a CSV.

    Each method's metrics use only its own drift/shift (if any) to size the
    video_metrics border-crop margin — methods are not compared against a
    shared margin here. When save_videos is on, each method's per-frame
    shift/drift is also saved as a <stem>_<method>_shift.npy sidecar, so a
    separate later pass (e.g. comparing CoTracker and baseline runs done as
    independent jobs) can compute a shared margin across them without
    re-running anything.

    Warped videos are saved as <stem>_<preset>.h5 (gzip-compressed HDF5,
    see save_video_hdf5()) under *output_dir* (defaults to
    <first_folder>/registered).
    CSV columns: folder, file, preset, n_frames, n_points, elapsed_s, ssim, psnr, mse, zncc, mi, crispness_mean, corr_mean, residual_flow
    """
    baseline_methods = baseline_methods or []
    effective_n = (
        num_points if (num_points and num_points > 0) else grid_rows * grid_cols
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = run_name if run_name else timestamp
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(f"registered_{effective_n}") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": folder_name,
        "timestamp": timestamp,
        "folders": folders,
        "presets": presets,
        "baseline_methods": baseline_methods,
        "num_frames": num_frames,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "df_sigma": df_sigma,
        "num_points": num_points,
        "target_w": target_w,
        "target_h": target_h,
        "save_videos": save_videos,
        "save_nth": save_nth,
        "cotracker_mode": "offline" if offline else "online",
        "input_filter": input_filter,
        "checkpoint": weights_path,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to: {out_dir / 'config.json'}")

    csv_path = Path(output_csv) if output_csv else out_dir / "benchmark_cotracker.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not overwrite:
        raise SystemExit(
            f"Output CSV already exists: {csv_path}\nUse --overwrite to replace it."
        )

    invalid = [
        p for p in presets if not all(s in COTRACKER_PRESETS for s in p.split("+"))
    ]
    if invalid:
        raise SystemExit(
            f"Unknown preset(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(COTRACKER_PRESETS)}. "
            f"Combine with '+', e.g. regular_grid+shi_tomasi."
        )

    invalid_baseline = [m for m in baseline_methods if m not in BASELINE_METHODS]
    if invalid_baseline:
        raise SystemExit(
            f"Unknown baseline method(s): {', '.join(invalid_baseline)}. "
            f"Available: {', '.join(BASELINE_METHODS)}."
        )

    all_methods = list(presets) + list(baseline_methods)

    clips: list[tuple[str, str, torch.Tensor]] = []
    for folder in folders:
        clips.extend(
            iter_folder(
                folder,
                num_frames=num_frames,
                chunk_frames=chunk_frames,
                target_w=target_w,
                target_h=target_h,
                file_filter=file_filter,
            )
        )
    total = len(clips) * len(all_methods)

    fieldnames = [
        "folder",
        "file",
        "preset",
        "n_frames",
        "n_points",
        "elapsed_s",
        "ssim",
        "psnr",
        "mse",
        "zncc",
        "mi",
        "crispness_mean",
        "corr_mean",
        "residual_flow",
    ]
    frame_fieldnames = [
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
    frame_csv_path = csv_path.parent / (csv_path.stem + "_perframe.csv")

    csv_file = csv_path.open("w", newline="")
    frame_csv_file = frame_csv_path.open("w", newline="")
    try:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        frame_writer = csv.DictWriter(frame_csv_file, fieldnames=frame_fieldnames)
        writer.writeheader()
        frame_writer.writeheader()
        csv_file.flush()
        frame_csv_file.flush()

        with tqdm(total=total, desc="Benchmark", unit="run", file=sys.stderr) as pbar:
            for file_idx, (folder_name, filename, clip) in enumerate(clips):
                T = int(clip.shape[0])
                stem = Path(filename).stem
                video_dir = out_dir / folder_name / stem
                video_dir.mkdir(parents=True, exist_ok=True)
                saved_warped: list[tuple[Path, str]] = []
                saved_tracks: list[tuple[Path, str]] = []
                for method in all_methods:
                    pbar.set_postfix(file=filename, preset=method)
                    warped = tracks = shift = None
                    elapsed_s = float("nan")
                    n_pts = 0
                    try:
                        t0 = time.perf_counter()
                        if method in presets:
                            warped, tracks, shift = run_cotracker(
                                clip,
                                method,
                                grid_rows=grid_rows,
                                grid_cols=grid_cols,
                                df_sigma=df_sigma,
                                num_points=num_points,
                                force_cpu=force_cpu,
                                offline=offline,
                                online_chunk_size=online_chunk_size,
                                input_filter=input_filter,
                                cotracker_grid_size=cotracker_grid_size,
                                weights_path=weights_path,
                            )
                        else:
                            warped, shift = run_baseline(clip, method)
                        elapsed_s = time.perf_counter() - t0
                        if warped is None:
                            raise RuntimeError("No output produced.")

                        metrics = compute_metrics(warped, drift=shift)

                        if method in presets:
                            pts = generate_points(
                                clip,
                                method,
                                grid_rows=grid_rows,
                                grid_cols=grid_cols,
                                num_points=num_points,
                            )
                            n_pts = len(pts)
                            # Save initial frame + points overlay
                            pts_path = video_dir / f"{stem}_{method}_points.png"
                            if not pts_path.exists() or overwrite:
                                _save_points_overlay(clip, pts, pts_path)
                            # Save points as npy
                            npy_path = video_dir / f"{stem}_{method}_points.npy"
                            if not npy_path.exists() or overwrite:
                                np.save(str(npy_path), pts)
                            # Save track overlay video (only in single-pass mode)
                            if tracks is not None:
                                track_path = video_dir / f"{stem}_{method}_tracks.h5"
                                if not track_path.exists() or overwrite:
                                    _save_tracks_overlay_video(clip, tracks, track_path)
                                saved_tracks.append((track_path, method))
                                # Raw (T, N, 2) coordinates, separate from the overlay video -- lets a later pass
                                # reuse the exact trajectory without re-running CoTracker.
                                tracks_npy_path = video_dir / f"{stem}_{method}_tracks.npy"
                                if not tracks_npy_path.exists() or overwrite:
                                    np.save(str(tracks_npy_path), tracks)

                        if save_videos and file_idx % save_nth == 0:
                            out_path = video_dir / f"{stem}_{method}.h5"
                            if not out_path.exists() or overwrite:
                                save_video_hdf5(np.asarray(warped), str(out_path))
                                print(f"Saved: {out_path}")
                            saved_warped.append((out_path, method))
                            # Shift/drift sidecar lets a later, separate comparison pass size a shared
                            # border-crop margin without re-running anything.
                            if shift is not None:
                                shift_path = video_dir / f"{stem}_{method}_shift.npy"
                                if not shift_path.exists() or overwrite:
                                    np.save(str(shift_path), np.asarray(shift))
                    except Exception as e:
                        print(
                            f"Error on {folder_name}/{filename} / {method}: {e}",
                            file=sys.stderr,
                        )
                        metrics = {
                            "ssim": float("nan"),
                            "zncc": float("nan"),
                            "mi": float("nan"),
                            "psnr": float("nan"),
                            "mse": float("nan"),
                            "crispness_mean": float("nan"),
                            "corr_mean": float("nan"),
                            "residual_flow": float("nan"),
                            "per_frame": {
                                "ssim": [],
                                "psnr": [],
                                "mse": [],
                                "zncc": [],
                                "mi": [],
                                "corr_mean": [],
                                "residual_flow": [],
                            },
                        }
                        n_pts = 0

                    writer.writerow(
                        {
                            "folder": folder_name,
                            "file": filename,
                            "preset": method,
                            "n_frames": T,
                            "n_points": n_pts,
                            "elapsed_s": elapsed_s,
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
                    csv_file.flush()

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
                        ),
                        start=0,
                    ):
                        frame_writer.writerow(
                            {
                                "folder": folder_name,
                                "file": filename,
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
                    frame_csv_file.flush()

                    del warped, tracks, shift
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pbar.update(1)

                del clip
    finally:
        csv_file.close()
        frame_csv_file.close()

    # Deduplicate summary CSV: keep only the max n_frames row per (file, preset)
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["folder"], row["file"], row["preset"])
        try:
            n = int(row.get("n_frames", 0))
        except ValueError:
            n = 0
        if key not in best or n > int(best[key].get("n_frames", 0)):
            best[key] = row
    deduped = sorted(best.values(), key=lambda r: (r["folder"], r["file"], r["preset"]))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Saved benchmark CSV to: {csv_path}")
    print(f"Saved per-frame CSV to: {frame_csv_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CoReg benchmark: run CoTracker with multiple point presets on a folder of videos."
    )
    parser.add_argument(
        "--folder",
        nargs="+",
        help="One or more folders containing video/tiff/hdf5 files to process.",
    )
    parser.add_argument(
        "--presets",
        default=",".join(COTRACKER_PRESETS),
        help=(
            f"Comma-separated presets to run. "
            f"Available: {', '.join(COTRACKER_PRESETS)}. "
            f"Default: all presets."
        ),
    )
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=[],
        help=(
            "Optional NoRMCorre/pyflowreg/ANTs/LDDMM baseline methods to run "
            f"alongside the CoTracker presets, in the same per-clip loop. "
            f"Available: {', '.join(BASELINE_METHODS)}. Default: none. "
            "Each method's metrics still use only its own drift/shift, not a "
            "margin shared across methods."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Only process these filenames (stem or full name) within the given folders.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to load per file (default: all).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help=(
            "Split each file into consecutive chunks of this many frames and "
            "process each chunk separately. Output names get _chunk000, _chunk001, ... suffix."
        ),
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=8,
        help="Grid rows for the regular_grid preset (default: 8).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=8,
        help="Grid columns for the regular_grid preset (default: 8).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=64,
        help=(
            "Max seed points for non-grid presets. "
            "Defaults to grid-rows x grid-cols."
        ),
    )
    parser.add_argument(
        "--df-sigma",
        type=float,
        default=5.0,
        help="Gaussian smoothing sigma for the displacement field (default: 5.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize width in pixels before processing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize height in pixels before processing.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run (used as subdirectory under --output-dir). "
        "Defaults to a timestamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--output-dir",
        default="out/",
        help="Base directory for output. Results go into <output-dir>/<run-name>/.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the output CSV (default: <output-dir>/benchmark_cotracker.csv).",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        default=False,
        help="Save registered videos as gzip-compressed HDF5 files (default: off).",
    )
    parser.add_argument(
        "--save-nth",
        type=int,
        default=1,
        metavar="N",
        help="When saving videos, keep only every Nth frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CoTracker to run on CPU regardless of GPU availability.",
    )
    parser.add_argument(
        "--cotracker-mode",
        choices=["offline", "online"],
        default="offline",
        help="CoTracker inference mode: 'offline' processes the full sequence at once (default), "
        "'online' uses a sliding window and works with arbitrary-length sequences.",
    )
    parser.add_argument(
        "--online-chunk-size",
        type=int,
        default=150,
        help="Number of frames per GPU chunk in online mode (default: 150). "
        "Reduce if you hit CUDA OOM; increase for better track continuity.",
    )
    parser.add_argument(
        "--filter",
        choices=VALID_FILTERS,
        default="none",
        dest="input_filter",
        help="Input filter applied to the video before tracking. "
        "Warping always uses the original unfiltered frames. "
        f"Choices: {', '.join(VALID_FILTERS)} (default: none).",
    )
    parser.add_argument(
        "--cotracker-grid-size",
        type=int,
        default=0,
        help="Side length N of CoTracker's built-in N×N dense point grid (default: 0 = disabled). "
        "When > 0, CoTracker tracks N² uniformly spaced points in addition to any preset "
        "query points.  Use --presets none to track the grid exclusively.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a local CoTracker weights file (.pt/.pth). "
        "When provided, skips the HuggingFace download and loads these weights instead.",
    )
    args = parser.parse_args(argv)

    if (args.width is None) != (args.height is None):
        raise SystemExit("Provide both --width and --height, or neither.")

    presets = [p.strip().lower() for p in args.presets.split(",") if p.strip()]

    run_evaluation(
        folders=args.folder,
        presets=presets,
        baseline_methods=args.baseline_methods,
        run_name=args.run_name,
        num_frames=args.num_frames,
        chunk_frames=args.chunk_frames,
        file_filter=args.files,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        df_sigma=args.df_sigma,
        num_points=args.num_points,
        target_w=args.width,
        target_h=args.height,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        save_videos=args.save_videos,
        save_nth=args.save_nth,
        overwrite=args.overwrite,
        force_cpu=args.force_cpu,
        offline=args.cotracker_mode == "offline",
        online_chunk_size=args.online_chunk_size,
        input_filter=args.input_filter,
        cotracker_grid_size=args.cotracker_grid_size,
        weights_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
