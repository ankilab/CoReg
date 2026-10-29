#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Forward-backward (cycle-consistency) tracking for self-supervised
CoTracker adaptation -- needs no manual point-track labels: the only signal
is whether tracking a point forward to the last frame and then backward to
frame 0 again lands back where it started.

Must call CoTrackerPredictor._compute_sparse_tracks directly, not the
public .forward()/__call__ -- forward() is hard-decorated with
torch.no_grad() in cotracker/predictor.py, so gradients can never flow
through it no matter what context manager wraps the call. _compute_sparse_
tracks has no such decorator and does the identical pre/post-processing
(resize to the model's interp_shape, rescale query coords, add+strip the
support grid, threshold+correct visibility at the query frame) -- verified
directly against predictor.py's source rather than assumed, and confirmed
empirically that tracks.requires_grad is True and a real gradient reaches
model.model.corr_mlp after calling it this way.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoundTripResult:
    roundtrip_xy: torch.Tensor      # (1, N, 2) -- frame-0 position after forward+backward
    visible_mask: torch.Tensor      # (1, N) bool -- visible at both the turnaround and the return
    tracks_fwd: torch.Tensor        # (1, T, N, 2) -- full forward trajectory, for diagnostics/plotting
    tracks_bwd: torch.Tensor        # (1, T, N, 2) -- full backward trajectory (time already un-flipped)


def round_trip_tracks(
    model,
    video: torch.Tensor,
    queries_xy: torch.Tensor,
    grid_query_frame: int = 0,
) -> RoundTripResult:
    """Track *queries_xy* (N, 2) forward from frame *grid_query_frame* to the
    last frame of *video* (1, T, 3, H, W), then backward from there to frame
    *grid_query_frame* again. Returns the round-trip frame-0 position and
    the visibility at both turnaround points.
    """
    device = video.device
    N = queries_xy.shape[0]

    queries0 = torch.cat(
        [torch.full((N, 1), float(grid_query_frame), device=device), queries_xy], dim=1
    ).unsqueeze(0)  # (1, N, 3): (t, x, y)

    tracks_fwd, vis_fwd = model._compute_sparse_tracks(
        video, queries0, grid_size=0, add_support_grid=True,
        grid_query_frame=grid_query_frame, backward_tracking=False,
    )  # (1, T, N, 2), (1, T, N) bool

    end_xy = tracks_fwd[0, -1]     # (N, 2) position at the last frame
    end_vis = vis_fwd[0, -1]       # (N,) bool

    # Track backward: reverse the video in time, query from its frame 0 (= original's last frame) at the forward endpoint.
    inv_video = video.flip(1)
    queries_end = torch.cat(
        [torch.zeros((N, 1), device=device), end_xy], dim=1
    ).unsqueeze(0)  # (1, N, 3), t=0 in the reversed video == T-1 in the original

    tracks_bwd_rev, vis_bwd_rev = model._compute_sparse_tracks(
        inv_video, queries_end, grid_size=0, add_support_grid=True,
        grid_query_frame=0, backward_tracking=False,
    )  # (1, T, N, 2) indexed in REVERSED time

    tracks_bwd = tracks_bwd_rev.flip(1)  # un-reverse so index 0 == original frame 0
    vis_bwd = vis_bwd_rev.flip(1)

    roundtrip_xy = tracks_bwd[:, grid_query_frame]       # (1, N, 2)
    return_vis = vis_bwd[0, grid_query_frame]            # (N,)

    visible_mask = (end_vis & return_vis).unsqueeze(0)   # (1, N)

    return RoundTripResult(
        roundtrip_xy=roundtrip_xy,
        visible_mask=visible_mask,
        tracks_fwd=tracks_fwd,
        tracks_bwd=tracks_bwd,
    )


def cycle_consistency_loss(
    queries_xy: torch.Tensor,
    result: RoundTripResult,
    min_visible_points: int = 4,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Masked Huber loss between the original query positions and their
    round-trip (forward then backward) predicted positions.

    Returns (loss, per_point_error). loss is None if fewer than
    *min_visible_points* points were visible at both turnaround points --
    too few points makes the mean error noisy/meaningless and risks
    optimizing toward a handful of outliers; the caller should skip the
    gradient step for that case rather than take a degenerate step.
    per_point_error is always returned (unmasked) for diagnostics.
    """
    target = queries_xy.unsqueeze(0)  # (1, N, 2)
    per_point_error = torch.linalg.vector_norm(result.roundtrip_xy - target, dim=-1)[0]  # (N,)

    mask = result.visible_mask[0]
    n_visible = int(mask.sum().item())
    if n_visible < min_visible_points:
        return None, per_point_error

    loss = torch.nn.functional.huber_loss(
        result.roundtrip_xy[0][mask], target[0][mask], reduction="mean"
    )
    return loss, per_point_error
