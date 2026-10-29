#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Test-time adaptation of CoTracker via the cycle-consistency loss in
cycle_consistency.py. No manual point-track labels needed -- the loss is
computed purely from the video being tracked.

VERIFIED_MODULES was read directly off the loaded model's own
named_children() (`for n, _ in model.model.named_children(): print(n)`),
not guessed:
fnet (2.6M params, the CNN feature extractor), updateformer (21.7M params,
the transformer doing iterative track refinement -- the bulk of the
model), corr_mlp (1.0M params, maps correlation features for the
updateformer). Default adaptation target is corr_mlp alone: small enough
for a handful of gradient steps to be fast and low-risk of catastrophically
forgetting general tracking ability, and it's the part most plausibly
sensitive to this project's specific image statistics (calcium-imaging
photon noise/contrast vs. the natural-video data CoTracker was trained on).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cycle_consistency import cycle_consistency_loss, round_trip_tracks

VERIFIED_MODULES = ("fnet", "updateformer", "corr_mlp")


def freeze_all_but(model, module_names: tuple[str, ...]) -> list[str]:
    """Set requires_grad=False on every parameter of model.model except
    those belonging to the named top-level submodules. Returns the names
    of parameters left trainable, for the caller to print/verify -- an
    empty list here means a typo'd module name silently adapted nothing,
    which is the one way this function could fail invisibly otherwise."""
    unknown = [n for n in module_names if n not in VERIFIED_MODULES]
    if unknown:
        raise ValueError(
            f"Unrecognized module name(s) {unknown}; verified top-level "
            f"children are {VERIFIED_MODULES}. Re-check with "
            f"`for n, _ in model.model.named_children(): print(n)` if the "
            f"CoTracker version/architecture has changed."
        )

    trainable = []
    for name, param in model.model.named_parameters():
        top = name.split(".", 1)[0]
        param.requires_grad_(top in module_names)
        if top in module_names:
            trainable.append(name)
    return trainable


@dataclass
class AdaptResult:
    step_losses: list[float] = field(default_factory=list)
    step_grad_norms: list[float] = field(default_factory=list)
    step_n_visible: list[int] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)  # step indices with too few visible points
    held_out_error_before: float | None = None
    held_out_error_after: float | None = None
    held_out_n_visible_before: int = 0
    held_out_n_visible_after: int = 0
    mean_track_displacement_before: float = 0.0
    mean_track_displacement_after: float = 0.0
    trainable_param_names: list[str] = field(default_factory=list)


def _mean_track_displacement(model, video, queries_xy) -> float:
    """Mean per-point distance travelled frame-0 -> last-frame, forward
    only. A collapse toward ~0 here (relative to its value before
    adaptation) is the signature of the trivial cycle-consistent-but-
    useless solution (every point mapped to one fixed location) --
    reported so that failure mode is visible, not silently accepted."""
    with torch.no_grad():
        N = queries_xy.shape[0]
        device = video.device
        queries0 = torch.cat(
            [torch.zeros((N, 1), device=device), queries_xy], dim=1
        ).unsqueeze(0)
        tracks, vis = model._compute_sparse_tracks(
            video, queries0, grid_size=0, add_support_grid=True,
            grid_query_frame=0, backward_tracking=False,
        )
        disp = torch.linalg.vector_norm(tracks[0, -1] - tracks[0, 0], dim=-1)
        visible = vis[0, -1] & vis[0, 0]
        if visible.sum() == 0:
            return 0.0
        return float(disp[visible].mean().item())


def _held_out_cycle_error(model, video, queries_xy) -> tuple[float, int]:
    with torch.no_grad():
        result = round_trip_tracks(model, video, queries_xy)
        _, per_point_error = cycle_consistency_loss(queries_xy, result)
        mask = result.visible_mask[0]
        n_visible = int(mask.sum().item())
        if n_visible == 0:
            return float("nan"), 0
        return float(per_point_error[mask].mean().item()), n_visible


def test_time_adapt(
    model,
    video: torch.Tensor,
    queries_xy: torch.Tensor,
    module_names: tuple[str, ...] = ("corr_mlp",),
    num_steps: int = 10,
    lr: float = 1e-6,
    held_out_frac: float = 0.3,
    seed: int = 0,
    log_fn=print,
) -> AdaptResult:
    """Adapt *model* (a CoTrackerPredictor) to *video* using the
    cycle-consistency loss on a subset of *queries_xy*, holding the rest
    out purely for before/after evaluation (never used in the loss) so
    "the loss went down" and "tracking actually got better" aren't
    conflated -- a model can drive the training loss to ~0 by overfitting
    exactly those points without generalizing at all.
    """
    result = AdaptResult()

    g = torch.Generator().manual_seed(seed)
    N = queries_xy.shape[0]
    n_held_out = max(1, int(round(N * held_out_frac)))
    perm = torch.randperm(N, generator=g)
    held_out_idx = perm[:n_held_out]
    train_idx = perm[n_held_out:]
    if len(train_idx) == 0:
        raise ValueError(
            f"held_out_frac={held_out_frac} leaves no training points out of N={N}; "
            f"lower held_out_frac or provide more query points."
        )

    train_xy = queries_xy[train_idx]
    held_out_xy = queries_xy[held_out_idx]

    log_fn(f"[adapt] {N} query points: {len(train_idx)} for training, {len(held_out_idx)} held out for evaluation only")

    result.trainable_param_names = freeze_all_but(model, module_names)
    log_fn(f"[adapt] trainable: {module_names} ({len(result.trainable_param_names)} parameter tensors, "
           f"{sum(model.model.get_parameter(n).numel() for n in result.trainable_param_names):,} values)")
    if not result.trainable_param_names:
        raise RuntimeError("freeze_all_but() left zero trainable parameters -- adaptation would be a no-op.")

    model.model.train()

    log_fn("[adapt] before: measuring held-out cycle error and track displacement...")
    result.held_out_error_before, result.held_out_n_visible_before = _held_out_cycle_error(model, video, held_out_xy)
    result.mean_track_displacement_before = _mean_track_displacement(model, video, queries_xy)
    log_fn(f"[adapt] before: held-out cycle error = {result.held_out_error_before:.3f}px "
           f"({result.held_out_n_visible_before}/{n_held_out} visible), "
           f"mean track displacement = {result.mean_track_displacement_before:.3f}px")

    trainable_params = [model.model.get_parameter(n) for n in result.trainable_param_names]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        rt = round_trip_tracks(model, video, train_xy)
        loss, _ = cycle_consistency_loss(train_xy, rt)
        n_visible = int(rt.visible_mask.sum().item())
        result.step_n_visible.append(n_visible)

        if loss is None:
            result.skipped_steps.append(step)
            log_fn(f"[adapt] step {step+1}/{num_steps}: skipped (only {n_visible} points visible both ways)")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float("inf"))
        optimizer.step()

        result.step_losses.append(float(loss.item()))
        result.step_grad_norms.append(float(grad_norm.item()))
        log_fn(f"[adapt] step {step+1}/{num_steps}: loss={loss.item():.4f}px  "
               f"grad_norm={grad_norm.item():.6g}  visible={n_visible}/{len(train_idx)}")

    model.model.eval()

    log_fn("[adapt] after: measuring held-out cycle error and track displacement...")
    result.held_out_error_after, result.held_out_n_visible_after = _held_out_cycle_error(model, video, held_out_xy)
    result.mean_track_displacement_after = _mean_track_displacement(model, video, queries_xy)
    log_fn(f"[adapt] after:  held-out cycle error = {result.held_out_error_after:.3f}px "
           f"({result.held_out_n_visible_after}/{n_held_out} visible), "
           f"mean track displacement = {result.mean_track_displacement_after:.3f}px")

    if result.mean_track_displacement_before > 1e-6:
        ratio = result.mean_track_displacement_after / result.mean_track_displacement_before
        if ratio < 0.2:
            log_fn(f"[adapt] WARNING: mean track displacement dropped to {ratio:.0%} of its "
                   f"pre-adaptation value -- possible degenerate collapse (points barely "
                   f"moving is a cheap way to look cycle-consistent). Inspect the before/after "
                   f"overlay before trusting this checkpoint.")

    return result
