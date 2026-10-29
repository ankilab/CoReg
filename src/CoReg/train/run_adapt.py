#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Test-time-adapts CoTracker to real video via the self-supervised
cycle-consistency loss (cycle_consistency.py/adapt.py) -- no manual
point-track labels needed. Cycles gradient steps across short subsets
(train/multi_clip.py) so the adaptation generalizes beyond one clip's own
noise/contrast quirks, rather than overfitting a single window.

Two modes, same script:
    --input <file.tif>      single-clip mode: subsets are windows of that
                             one video.
    --input-dir <dir>       multi-clip mode: subsets are drawn from every
                             .tif/.tiff found under that directory.

Either way, held-out evaluation is by WHOLE CLIP, not by point: a fixed
number of discovered subsets are set aside and never appear in the
training loop at all -- a stronger separation than holding out individual
points within a trained-on clip, and it works the same whether there's
one source video or many.

Query points per subset are a combination of a regular grid and
segmentation centroids (see multi_clip.py's _grid_plus_segmentation_points).

Usage
-----
python -m train.run_adapt \\
    --input-dir /path/to/videos --out-dir /path/to/output/train_multi \\
    --subset-frames 50 --subsets-per-video 2 --grid-rows 10 --grid-cols 10 \\
    --num-steps 60 --held-out-clips 6

python -m train.run_adapt \\
    --input /path/to/videos/clip.tif --out-dir /path/to/output/train_single

Writes, under --out-dir:
    training_curves.png/.pdf   per-step training loss (colored by clip, with a rolling
                           mean) plus held-out cycle error measured periodically
                           during training (see --eval-every), not just before/after
    held_out_errors.csv   per-held-out-clip before/after cycle error
    before_after_<clip>.png   round-trip overlay for a few representative held-out clips
    adapted.pt             adapted weights, loadable via --checkpoint
    summary.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Keep text as real text (not outline paths) in vector exports, matching figures/metric_boxplot*.py.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import torch
from tqdm import tqdm

from utils.mc_cotracker import _get_cotracker

from .adapt import freeze_all_but, VERIFIED_MODULES
from .cycle_consistency import cycle_consistency_loss, round_trip_tracks
from .multi_clip import discover_subsets


def _clip_cycle_error(model, subset, device) -> tuple[float, int, int]:
    """Moves this one subset's video/queries to *device* for the duration
    of the call only -- subsets are kept on CPU in the master list (discover_
    subsets() loads all of them, ~40MB each at 50 frames/256px, fine for
    CPU RAM) and only the clip actually being used at any moment lives on
    the GPU. Moving every discovered subset's video to the GPU up front
    (an earlier version of this function did exactly that) is what caused
    a real CUDA OOM on the very first training step of a 48-subset run --
    confirmed by it crashing immediately after an all-clips-on-GPU-at-once
    run, while an isolated single-subset timing test at the identical
    frame/size/point scale ran repeated steps with no issue."""
    video = subset.video.to(device)
    queries_xy = subset.queries_xy.to(device)
    with torch.no_grad():
        rt = round_trip_tracks(model, video, queries_xy)
        _, per_point_error = cycle_consistency_loss(queries_xy, rt)
        mask = rt.visible_mask[0]
        n_visible = int(mask.sum().item())
        n_total = len(queries_xy)
        if n_visible == 0:
            return float("nan"), 0, n_total
        return float(per_point_error[mask].mean().item()), n_visible, n_total


def _evaluate_held_out(model, held_out_subsets, device, log_fn=print) -> list[dict]:
    rows = []
    for subset in held_out_subsets:
        err, n_vis, n_total = _clip_cycle_error(model, subset, device)
        rows.append({"clip": subset.label, "error_px": err, "n_visible": n_vis, "n_total": n_total})
        log_fn(f"  {subset.label}: {err:.3f}px ({n_vis}/{n_total} visible)")
    return rows


def _plot_training_curves(step_losses, step_clip_labels, held_out_curve, mean_before: float, out_path: Path) -> None:
    """Two panels instead of just raw per-step loss:
    top -- training loss per step (still colored by clip, since which clip
    a step came from explains a lot of the point-to-point noise), with a
    rolling mean overlaid so the actual trend is visible through that noise;
    bottom -- held-out cycle error measured periodically DURING training
    (not just before/after), which is the metric that actually answers
    "is this still improving or has it plateaued/started overfitting" --
    the single most useful thing the old loss-only plot couldn't show.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    unique_clips = sorted(set(step_clip_labels))
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(unique_clips)}
    for c in unique_clips:
        idx = [i for i, lbl in enumerate(step_clip_labels) if lbl == c]
        xs = [i + 1 for i in idx]
        ys = [step_losses[i] for i in idx]
        ax1.scatter(xs, ys, s=14, color=colors[c], alpha=0.6, label=c if len(unique_clips) <= 12 else None)

    window = max(3, len(step_losses) // 25)
    if len(step_losses) >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(step_losses, kernel, mode="valid")
        ax1.plot(range(window, len(step_losses) + 1), rolling, color="black", lw=1.8,
                  label=f"rolling mean (window={window})")

    ax1.set_ylabel("training loss (px)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=6, ncol=2, loc="upper right")

    ho_steps = [s for s, _ in held_out_curve]
    ho_errs = [e for _, e in held_out_curve]
    ax2.axhline(mean_before, color="gray", ls="--", lw=1, label=f"before training ({mean_before:.3f}px)")
    ax2.plot(ho_steps, ho_errs, color="crimson", marker="o", ms=4, lw=1.5, label="mean held-out error")
    ax2.set_xlabel("step")
    ax2.set_ylabel("held-out cycle error (px)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_clip_before_after(subset, before_state, after_state, model, device, out_path: Path) -> None:
    frame0 = subset.video[0, 0, 0].numpy()  # subset.video stays on CPU
    q_np = subset.queries_xy.numpy()

    video = subset.video.to(device)
    queries_xy = subset.queries_xy.to(device)

    def _rt():
        with torch.no_grad():
            r = round_trip_tracks(model, video, queries_xy)
            return r.roundtrip_xy[0].detach().cpu().numpy(), r.visible_mask[0].detach().cpu().numpy()

    model.model.load_state_dict(before_state)
    model.model.eval()
    rt_before, vis_before = _rt()
    err_before = float(np.linalg.norm(rt_before[vis_before] - q_np[vis_before], axis=-1).mean()) if vis_before.any() else float("nan")

    model.model.load_state_dict(after_state)
    model.model.eval()
    rt_after, vis_after = _rt()
    err_after = float(np.linalg.norm(rt_after[vis_after] - q_np[vis_after], axis=-1).mean()) if vis_after.any() else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, rt, vis, title, err in (
        (axes[0], rt_before, vis_before, "before", err_before),
        (axes[1], rt_after, vis_after, "after", err_after),
    ):
        ax.imshow(frame0, cmap="gray")
        ax.scatter(q_np[:, 0], q_np[:, 1], marker="x", c="lime", s=25, label="query")
        ax.scatter(rt[vis, 0], rt[vis, 1], marker="o", facecolors="none", edgecolors="red", s=25, label="round-trip")
        for i in range(len(q_np)):
            if vis[i]:
                ax.plot([q_np[i, 0], rt[i, 0]], [q_np[i, 1], rt[i, 1]], c="red", lw=0.6, alpha=0.5)
        ax.set_title(f"{title}: {err:.3f}px ({int(vis.sum())}/{len(vis)} visible)")
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(subset.label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", help="Root directory to search for .tif/.tiff videos (multi-clip mode).")
    src.add_argument("--input", help="A single TIFF stack (single-clip mode).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--subset-frames", type=int, default=50)
    ap.add_argument("--subsets-per-video", type=int, default=None,
                     help="Windows carved out of each video (default: 2 for --input-dir, 6 for --input).")
    ap.add_argument("--size", type=int, default=256,
                     help="Square resize (legacy default). Ignored if --width/--height are given.")
    ap.add_argument("--width", type=int, default=None, help="Resize width -- use with --height for a non-square (e.g. native) aspect ratio.")
    ap.add_argument("--height", type=int, default=None, help="Resize height -- use with --width for a non-square (e.g. native) aspect ratio.")
    ap.add_argument("--grid-rows", type=int, default=10)
    ap.add_argument("--grid-cols", type=int, default=10)
    ap.add_argument("--seg-points", type=int, default=50,
                     help="Max segmentation-centroid query points per subset, added to the regular grid.")
    ap.add_argument("--modules", nargs="+", default=["corr_mlp"], choices=list(VERIFIED_MODULES))
    ap.add_argument("--num-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--held-out-clips", type=int, default=None,
                     help="Number of discovered subsets reserved for evaluation only, never trained on "
                          "(default: 6 for --input-dir, 2 for --input).")
    ap.add_argument("--plot-n-clips", type=int, default=4, help="How many held-out clips to render before/after plots for.")
    ap.add_argument("--eval-every", type=int, default=None,
                     help="Re-measure held-out cycle error every this many steps during training, "
                          "not just before/after (default: ~20 points spread across the run).")
    ap.add_argument("--max-train-points", type=int, default=None,
                     help="Randomly subsample each training step's query points down to this many "
                          "(held-out evaluation still uses all of them) -- the forward+backward round "
                          "trip keeps a full autograd graph over every point x frame, so this is the "
                          "most direct way to cut a training step's GPU memory use if you hit an OOM.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    subsets_per_video = args.subsets_per_video if args.subsets_per_video is not None else (2 if args.input_dir else 6)
    held_out_clips = args.held_out_clips if args.held_out_clips is not None else (6 if args.input_dir else 2)

    size_wh = (args.width, args.height) if args.width and args.height else (args.size, args.size)
    print(f"Resize target (W,H): {size_wh}")
    discover_kwargs = dict(
        subset_frames=args.subset_frames, subsets_per_video=subsets_per_video,
        size_wh=size_wh, grid_rows=args.grid_rows, grid_cols=args.grid_cols,
        seg_points=args.seg_points, seed=args.seed,
    )
    if args.input_dir:
        source_desc = args.input_dir
        print(f"Discovering subsets under {args.input_dir} ...")
        subsets = discover_subsets(args.input_dir, **discover_kwargs)
    else:
        source_desc = args.input
        print(f"Discovering subsets from {args.input} ...")
        subsets = discover_subsets(paths=[Path(args.input)], **discover_kwargs)
    if len(subsets) <= held_out_clips:
        raise ValueError(f"Only {len(subsets)} subsets discovered, need more than --held-out-clips={held_out_clips}.")

    held_out_subsets = subsets[:held_out_clips]
    train_subsets = subsets[held_out_clips:]
    # Subsets stay on CPU, moved to GPU one at a time on demand -- moving all at once OOM'd a 48-subset run.
    print(f"{len(subsets)} subsets total: {len(train_subsets)} for training, {len(held_out_subsets)} held out (whole clips, never trained on)")

    print(f"Loading CoTracker ({'local: ' + args.checkpoint if args.checkpoint else 'HuggingFace default'})...")
    model = _get_cotracker(device, offline=True, weights_path=args.checkpoint)

    trainable = freeze_all_but(model, tuple(args.modules))
    print(f"Trainable: {args.modules} ({len(trainable)} parameter tensors, "
          f"{sum(model.model.get_parameter(n).numel() for n in trainable):,} values)")

    before_state = {k: v.clone() for k, v in model.model.state_dict().items()}

    print("\nBefore training -- held-out clip errors:")
    held_out_before = _evaluate_held_out(model, held_out_subsets, device)
    if device == "cuda":
        torch.cuda.empty_cache()  # release cached-but-unused memory from the eval pass before training

    optimizer = torch.optim.AdamW([model.model.get_parameter(n) for n in trainable], lr=args.lr)
    model.model.train()

    step_losses: list[float] = []
    step_clip_labels: list[str] = []
    skipped = 0

    mean_before = float(np.nanmean([r["error_px"] for r in held_out_before]))
    held_out_curve: list[tuple[int, float]] = [(0, mean_before)]
    eval_every = args.eval_every if args.eval_every is not None else max(1, args.num_steps // 20)

    train_rng = torch.Generator().manual_seed(args.seed)

    pbar = tqdm(range(args.num_steps), desc="multi-clip adapt", unit="step")
    for step in pbar:
        subset = train_subsets[step % len(train_subsets)]
        video = subset.video.to(device)
        queries_xy = subset.queries_xy.to(device)
        if args.max_train_points and len(queries_xy) > args.max_train_points:
            keep = torch.randperm(len(queries_xy), generator=train_rng)[: args.max_train_points]
            queries_xy = queries_xy[keep]
        optimizer.zero_grad()
        rt = round_trip_tracks(model, video, queries_xy)
        loss, _ = cycle_consistency_loss(queries_xy, rt)
        if loss is None:
            skipped += 1
            pbar.set_postfix(clip=subset.path.stem, status="skipped")
            continue
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.item()))
        step_clip_labels.append(subset.path.stem)
        pbar.set_postfix(clip=subset.path.stem, loss=f"{loss.item():.4f}")

        if (step + 1) % eval_every == 0 or step == args.num_steps - 1:
            model.model.eval()
            rows = _evaluate_held_out(model, held_out_subsets, device, log_fn=lambda *a, **k: None)
            held_out_curve.append((step + 1, float(np.nanmean([r["error_px"] for r in rows]))))
            model.model.train()
            if device == "cuda":
                torch.cuda.empty_cache()
        if device == "cuda":
            del video, queries_xy, rt, loss
            torch.cuda.empty_cache()

    model.model.eval()
    after_state = {k: v.clone() for k, v in model.model.state_dict().items()}

    print(f"\n{len(step_losses)}/{args.num_steps} steps applied, {skipped} skipped (too few visible points)")
    print("\nAfter training -- held-out clip errors:")
    held_out_after = _evaluate_held_out(model, held_out_subsets, device)

    mean_after = float(np.nanmean([r["error_px"] for r in held_out_after]))
    improved = mean_after < mean_before
    if held_out_curve[-1][0] == args.num_steps:
        held_out_curve[-1] = (args.num_steps, mean_after)
    else:
        held_out_curve.append((args.num_steps, mean_after))

    print("\n=== SUMMARY ===")
    print(f"input: {source_desc}")
    print(f"subsets: {len(subsets)} total ({len(train_subsets)} train / {len(held_out_subsets)} held-out), "
          f"{args.subset_frames} frames each, up to {args.grid_rows}x{args.grid_cols} grid + {args.seg_points} segmentation points")
    print(f"modules adapted: {args.modules}")
    print(f"steps: {args.num_steps} requested, {len(step_losses)} applied, {skipped} skipped")
    print(f"mean held-out cycle error: before={mean_before:.3f}px  after={mean_after:.3f}px  "
          f"({'improved' if improved else 'did NOT improve'}, {(mean_after - mean_before) / mean_before * 100:+.1f}%)")

    with open(out_dir / "held_out_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "error_before_px", "n_visible_before", "error_after_px", "n_visible_after", "n_total"])
        w.writeheader()
        for b, a in zip(held_out_before, held_out_after):
            w.writerow({
                "clip": b["clip"], "error_before_px": b["error_px"], "n_visible_before": b["n_visible"],
                "error_after_px": a["error_px"], "n_visible_after": a["n_visible"], "n_total": b["n_total"],
            })
    print(f"Saved: {out_dir / 'held_out_errors.csv'}")

    summary = (
        f"input: {source_desc}\n"
        f"subsets: {len(subsets)} total ({len(train_subsets)} train / {len(held_out_subsets)} held-out), "
        f"{args.subset_frames} frames each, up to {args.grid_rows}x{args.grid_cols} grid + {args.seg_points} segmentation points\n"
        f"modules adapted: {args.modules}\n"
        f"steps: {args.num_steps} requested, {len(step_losses)} applied, {skipped} skipped\n"
        f"mean held-out cycle error: before={mean_before:.3f}px  after={mean_after:.3f}px  "
        f"({'improved' if improved else 'did NOT improve'}, {(mean_after - mean_before) / mean_before * 100:+.1f}%)\n"
    )
    (out_dir / "summary.txt").write_text(summary)

    _plot_training_curves(step_losses, step_clip_labels, held_out_curve, mean_before, out_dir / "training_curves.png")

    # Raw numbers behind training_curves.png -- lets it be replotted later without rerunning training.
    with open(out_dir / "training_loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "clip", "loss_px"])
        w.writeheader()
        w.writerows({"step": i + 1, "clip": clip, "loss_px": loss}
                    for i, (clip, loss) in enumerate(zip(step_clip_labels, step_losses)))
    with open(out_dir / "held_out_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "mean_held_out_error_px"])
        w.writeheader()
        w.writerows({"step": s, "mean_held_out_error_px": e} for s, e in held_out_curve)
    print(f"Saved: {out_dir / 'training_loss.csv'}")
    print(f"Saved: {out_dir / 'held_out_curve.csv'}")

    n_plot = min(args.plot_n_clips, len(held_out_subsets))
    print(f"\nRendering before/after plots for {n_plot} held-out clips...")
    for subset in held_out_subsets[:n_plot]:
        safe_name = subset.label.replace("[", "_").replace("]", "").replace(":", "-")
        _plot_clip_before_after(subset, before_state, after_state, model, device, out_dir / f"before_after_{safe_name}.png")

    model.model.load_state_dict(after_state)
    ckpt_path = out_dir / "adapted.pt"
    torch.save(after_state, ckpt_path)
    print(f"Saved: {ckpt_path}  (use with --checkpoint {ckpt_path})")

    if not improved:
        print("\nNOTE: mean held-out cycle error did not improve -- a real possible "
              "outcome, not necessarily a bug. Check held_out_errors.csv and the "
              "before/after plots per clip before concluding either way.", file=sys.stderr)


if __name__ == "__main__":
    main()
