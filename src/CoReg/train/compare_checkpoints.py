#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Tracks every available soba_fullsequence video (low_movement + strong_movement)
with every finetuned CoTracker checkpoint produced in this session, on two
query-point presets (regular grid, Shi-Tomasi features), and writes side-by-side
comparison figures plus a quantitative summary.

There is no ground truth for these real microscopy videos, so "the result" here
is not tracking error against labels -- it's (1) a visual, one-panel-per-model
comparison figure per video/preset, and (2) self-consistency proxies computed
from the tracks themselves: start->end displacement, total path length, a
jitter ratio (path length / displacement -- high means a noisy/wobbly track
even if it ends up near where it started), and the fraction of frames each
point stayed visible. These don't replace the held-out cycle-consistency
numbers from training (adapt.py) -- they're a full-video, real-usage sanity
check on top of that, using the model exactly as the GUI/CLI would.

Usage
-----
python -m train.compare_checkpoints \\
    --input-dir /path/to/videos --out-dir /path/to/output \\
    --checkpoints corr_mlp_500steps=/path/to/train_multi_corrmlp/adapted.pt \\
                  updateformer_2000steps=/path/to/train_multi_updateformer/adapted.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.mc_cotracker import cotrack_model
from utils.presets import preset_features, preset_regular_grid
from utils.utils import load_tiff

# Unmodified HuggingFace CoTracker is always compared against; finetuned checkpoints come from --checkpoints.
CHECKPOINTS_BASELINE = [("baseline", None)]

# One fixed color per model, shared between the static overlay figure and the overlay video.
MODEL_COLORS = {
    "baseline": "#7f7f7f",
    "corr_mlp_200steps": "#1f77b4",
    "corr_mlp_500steps": "#17becf",
    "updateformer_500steps": "#ff7f0e",
    "updateformer_2000steps": "#d62728",
}


def _discover_videos(root: Path) -> list[tuple[Path, str]]:
    out = []
    for category_dir in ("low_movement", "strong_movement"):
        d = root / category_dir
        if not d.is_dir():
            continue
        for p in sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff"))):
            out.append((p, category_dir))
    return out


def _load_video(path: Path, size: int, max_frames: int | None) -> torch.Tensor:
    """(T, 3, H, W) float32 in [0, 255]."""
    video = load_tiff(str(path), frames=max_frames, resize=True, size_wh=(size, size))  # (T,1,H,W) in [0,1]
    video = (video * 255.0).repeat(1, 3, 1, 1)
    return video


def _track_full_video_chunked(video: torch.Tensor, queries_xy: np.ndarray, device: str,
                               weights_path: str | None, window: int = 450) -> tuple[np.ndarray, np.ndarray]:
    """Tracks the *entire* video even though the offline CoTracker model has
    no internal temporal chunking and a hard OOM cliff around 600-700 frames
    at any resolution (profiled directly -- not a smooth memory/frame-count
    curve, confirmed independent of spatial size). Splits the video into
    consecutive, non-overlapping windows of at most *window* frames; each
    window is tracked independently with the offline model (so every window
    individually stays under the cliff), and each window after the first is
    re-queried at its own frame 0 using the tracked xy positions from the
    previous window's last frame -- carrying the trajectory across the
    boundary with a 1-frame lag, the same approximation CoTracker's own
    online/chunked mode makes internally. Returns full-length (T,N,2) tracks
    and (T,N) visibility, concatenated across windows.
    """
    T = video.shape[0]
    N = len(queries_xy)
    cur_xy = np.asarray(queries_xy, dtype=np.float32)
    tracks_chunks, vis_chunks = [], []

    for start in range(0, T, window):
        end = min(start + window, T)
        q = torch.from_numpy(cur_xy).float()
        t_col = torch.zeros((N, 1), dtype=q.dtype)
        queries = torch.cat([t_col, q], dim=-1).unsqueeze(0)  # (1,N,3), all at local frame 0

        pred_tracks, pred_vis = cotrack_model(
            video[start:end], queries=queries, device=device, grid_size=0, weights_path=weights_path,
        )
        tracks_np = pred_tracks[0].detach().cpu().numpy()  # (end-start,N,2)
        vis_np = pred_vis[0].detach().cpu().numpy()
        if vis_np.ndim == 3:
            vis_np = vis_np[..., 0]
        vis_np = vis_np > 0.5

        tracks_chunks.append(tracks_np)
        vis_chunks.append(vis_np)
        cur_xy = tracks_np[-1]  # seed the next window with where points ended up

        if device == "cuda":
            del pred_tracks, pred_vis
            torch.cuda.empty_cache()

    return np.concatenate(tracks_chunks, axis=0), np.concatenate(vis_chunks, axis=0)


def _query_points(preset: str, gray_u8: np.ndarray, rows: int, cols: int, num_features: int) -> np.ndarray:
    H, W = gray_u8.shape[:2]
    if preset == "regular_grid":
        return preset_regular_grid(H, W, rows, cols)
    if preset == "features":
        return preset_features(gray_u8, num_points=num_features, min_dist=8)
    raise ValueError(preset)


def _track_metrics(tracks: np.ndarray, visible: np.ndarray) -> dict:
    """tracks: (T,N,2). visible: (T,N) bool."""
    start = tracks[0]
    end = tracks[-1]
    displacement = np.linalg.norm(end - start, axis=-1)
    step_delta = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)  # (T-1, N)
    path_length = step_delta.sum(axis=0)
    jitter_ratio = path_length / np.maximum(displacement, 1.0)
    visible_fraction = visible.mean(axis=0)
    return dict(
        mean_displacement_px=float(np.mean(displacement)),
        mean_path_length_px=float(np.mean(path_length)),
        mean_jitter_ratio=float(np.mean(jitter_ratio)),
        mean_visible_fraction=float(np.mean(visible_fraction)),
    )


def _plot_comparison(frame0: np.ndarray, results: list[tuple[str, np.ndarray, np.ndarray, dict]],
                      title: str, out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8))
    if n == 1:
        axes = [axes]
    for ax, (name, tracks, visible, metrics) in zip(axes, results):
        ax.imshow(frame0, cmap="gray")
        N = tracks.shape[1]
        cmap = plt.get_cmap("plasma")
        for i in range(N):
            vis_i = visible[:, i]
            if vis_i.sum() < 2:
                continue
            xy = tracks[:, i]
            ax.plot(xy[vis_i, 0], xy[vis_i, 1], color=cmap(i / max(N - 1, 1)), lw=0.5, alpha=0.5)
        ax.scatter(tracks[0, :, 0], tracks[0, :, 1], marker="x", c="lime", s=14, label="start")
        end_vis = visible[-1]
        ax.scatter(tracks[-1, end_vis, 0], tracks[-1, end_vis, 1], marker="o",
                   facecolors="none", edgecolors="red", s=14, label="end")
        ax.set_title(
            f"{name}\ndisp={metrics['mean_displacement_px']:.2f}px  jitter={metrics['mean_jitter_ratio']:.2f}\n"
            f"vis={metrics['mean_visible_fraction']*100:.0f}%",
            fontsize=9,
        )
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_overlay(frame0: np.ndarray, results: list[tuple[str, np.ndarray, np.ndarray, dict]],
                   title: str, out_path: Path) -> None:
    """All models on one panel, one fixed color per model (MODEL_COLORS) instead
    of per-point -- lets you see how far models' predictions for the same point
    diverge from each other, which the side-by-side panels can't show directly.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(frame0, cmap="gray")
    for name, tracks, visible, _ in results:
        color = MODEL_COLORS.get(name, "white")
        N = tracks.shape[1]
        for i in range(N):
            vis_i = visible[:, i]
            if vis_i.sum() < 2:
                continue
            xy = tracks[:, i]
            ax.plot(xy[vis_i, 0], xy[vis_i, 1], color=color, lw=0.4, alpha=0.35)
        end_vis = visible[-1]
        ax.scatter(tracks[-1, end_vis, 0], tracks[-1, end_vis, 1], color=color, s=6, label=name)
    ax.scatter(results[0][1][0, :, 0], results[0][1][0, :, 1], marker="x", c="black", s=10, label="start", zorder=5)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Root folder with low_movement/strong_movement subfolders of videos.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[], metavar="NAME=PATH",
                     help="Finetuned checkpoints to compare against baseline, as name=path (repeatable). "
                          "'baseline' (unmodified HuggingFace CoTracker) is always included first.")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=None,
                     help="Truncate videos to this many frames (default: full video -- long videos are "
                          "handled by tracking in --window-sized chunks, see _track_full_video_chunked).")
    ap.add_argument("--window", type=int, default=450,
                     help="Chunk length for tracking (frames). The offline model has a hard OOM cliff "
                          "around 600-700 frames at any resolution, so long videos are tracked in "
                          "consecutive chunks of this size, each re-queried from the previous chunk's "
                          "last tracked position.")
    ap.add_argument("--grid-rows", type=int, default=10)
    ap.add_argument("--grid-cols", type=int, default=10)
    ap.add_argument("--num-features", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--videos", nargs="*", default=None, help="Optional subset of video stems (for testing).")
    ap.add_argument("--models", nargs="*", default=None, help="Optional subset of checkpoint names (for testing).")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints = CHECKPOINTS_BASELINE + [tuple(spec.split("=", 1)) for spec in args.checkpoints]
    if args.models:
        wanted = set(args.models)
        checkpoints = [(n, c) for n, c in checkpoints if n in wanted]

    out_root = Path(args.out_dir)
    fig_dir_grid = out_root / "figures" / "regular_grid"
    fig_dir_feat = out_root / "figures" / "features"
    overlay_dir_grid = out_root / "figures" / "regular_grid" / "overlay"
    overlay_dir_feat = out_root / "figures" / "features" / "overlay"
    tracks_dir = out_root / "tracks"
    for d in (fig_dir_grid, fig_dir_feat, overlay_dir_grid, overlay_dir_feat, tracks_dir):
        d.mkdir(parents=True, exist_ok=True)

    videos = _discover_videos(Path(args.input_dir))
    if args.videos:
        wanted = set(args.videos)
        videos = [(p, c) for p, c in videos if p.stem in wanted]
    print(f"{len(videos)} videos found, {len(checkpoints)} checkpoints each")

    summary_rows = []

    for vpath, category in videos:
        print(f"\n=== {category}/{vpath.name} ===")
        video = _load_video(vpath, args.size, args.max_frames)  # (T,3,H,W) [0,255]
        T, _, H, W = video.shape
        print(f"  {T} frames tracked, {H}x{W}")
        frame0_gray = video[0, 0].numpy().astype(np.uint8)

        for preset in ("regular_grid", "features"):
            pts = _query_points(preset, frame0_gray, args.grid_rows, args.grid_cols, args.num_features)
            if len(pts) == 0:
                print(f"  [{preset}] no points found, skipping")
                continue
            N = len(pts)

            panel_results = []
            for model_name, ckpt in checkpoints:
                tracks_np, vis_np = _track_full_video_chunked(
                    video, pts, device, ckpt, window=args.window,
                )
                metrics = _track_metrics(tracks_np, vis_np)
                panel_results.append((model_name, tracks_np, vis_np, metrics))
                summary_rows.append(dict(
                    video=vpath.stem, movement_category=category, preset=preset, model=model_name,
                    n_points=N, n_frames=T, **metrics,
                ))
                np.savez_compressed(
                    tracks_dir / f"{vpath.stem}_{preset}_{model_name}.npz",
                    tracks=tracks_np, visible=vis_np,
                )
                print(f"  [{preset}] {model_name}: disp={metrics['mean_displacement_px']:.2f}px "
                      f"jitter={metrics['mean_jitter_ratio']:.2f} vis={metrics['mean_visible_fraction']*100:.0f}%")

            out_fig = (fig_dir_grid if preset == "regular_grid" else fig_dir_feat) / f"{vpath.stem}.png"
            _plot_comparison(frame0_gray, panel_results, f"{category}/{vpath.stem}  ({preset}, N={N})", out_fig)

            overlay_dir = overlay_dir_grid if preset == "regular_grid" else overlay_dir_feat
            _plot_overlay(frame0_gray, panel_results, f"{category}/{vpath.stem}  ({preset}, N={N})",
                          overlay_dir / f"{vpath.stem}.png")

    summary_path = out_root / "summary.csv"
    fieldnames = ["video", "movement_category", "preset", "model", "n_points", "n_frames",
                  "mean_displacement_px", "mean_path_length_px", "mean_jitter_ratio", "mean_visible_fraction"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
