#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Standalone point-preset functions shared by the GUI controller and the CLI.

All functions operate on plain NumPy arrays and have no Qt dependency.
"""

import sys

import cv2
import numpy as np
import torch

try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None

try:
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
except ImportError:
    AutoImageProcessor = None
    AutoModelForKeypointMatching = None
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

# ---------------------------------------------------------------------------
# Module-level MatchAnything model cache (lazy-loaded on first use)
# ---------------------------------------------------------------------------
_matchanything_model = None
_matchanything_processor = None
_matchanything_available = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clip_border(pts: np.ndarray, H: int, W: int, border: int = 0) -> np.ndarray:
    """Remove points that fall within *border* pixels of any edge."""
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    good = (
        (pts[:, 0] >= border)
        & (pts[:, 0] < (W - border))
        & (pts[:, 1] >= border)
        & (pts[:, 1] < (H - border))
    )
    return pts[good]


# ---------------------------------------------------------------------------
# Preset functions
# ---------------------------------------------------------------------------

def preset_regular_grid(
    H: int,
    W: int,
    rows: int,
    cols: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """Equally spaced grid of (rows × cols) points spanning the full image.

    Points are placed at cell centers so the grid covers the entire image extent.
    *offset_x* / *offset_y* shift all points by that many pixels.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    cell_w = float(W) / cols
    cell_h = float(H) / rows
    xs = np.linspace(cell_w / 2.0, W - cell_w / 2.0, cols, dtype=np.float32) + float(offset_x)
    ys = np.linspace(cell_h / 2.0, H - cell_h / 2.0, rows, dtype=np.float32) + float(offset_y)
    xs = np.clip(xs, 0.0, float(W) - 1.0).astype(np.float32)
    ys = np.clip(ys, 0.0, float(H) - 1.0).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


def preset_random(H: int, W: int, n: int, border: int = 4, seed: int | None = None) -> np.ndarray:
    """*n* uniformly random (x, y) points, kept *border* px away from every
    edge -- used as a fallback when a content-driven preset (segmentation,
    features, ...) finds nothing to seed on, so tracking still gets *some*
    query points instead of silently degrading to zero."""
    rng = np.random.default_rng(seed)
    n = max(0, int(n))
    xs = rng.uniform(border, max(border + 1.0, W - border), size=n).astype(np.float32)
    ys = rng.uniform(border, max(border + 1.0, H - border), size=n).astype(np.float32)
    return np.stack([xs, ys], axis=-1)


def preset_regular_grid_spacing(H: int, W: int, spacing_px: int) -> np.ndarray:
    """Centered rectangular grid with a fixed pixel spacing between points."""
    s = max(1.0, float(spacing_px))
    margin_x = max(5.0, s / 2.0)
    margin_y = max(5.0, s / 2.0)

    usable_w = max(0.0, float(W) - 2.0 * margin_x)
    usable_h = max(0.0, float(H) - 2.0 * margin_y)
    cols = max(2, int(np.floor(usable_w / s)) + 1)
    rows = max(2, int(np.floor(usable_h / s)) + 1)

    total_w = (cols - 1) * s
    total_h = (rows - 1) * s
    start_x = max(margin_x, (float(W) - total_w) / 2.0)
    start_y = max(margin_y, (float(H) - total_h) / 2.0)
    end_x = min(float(W) - margin_x, start_x + total_w)
    end_y = min(float(H) - margin_y, start_y + total_h)

    total_w = end_x - start_x
    total_h = end_y - start_y
    s_x = total_w / (cols - 1) if cols > 1 else s
    s_y = total_h / (rows - 1) if rows > 1 else s

    xs = (start_x + np.arange(cols, dtype=np.float32) * s_x).astype(np.float32)
    ys = (start_y + np.arange(rows, dtype=np.float32) * s_y).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


def preset_features(
    gray_u8: np.ndarray,
    num_points: int = 50,
    min_dist: float = 10,
) -> np.ndarray:
    """Shi-Tomasi feature detection."""
    H, W = gray_u8.shape[:2]
    _, m = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_OTSU)
    m = cv2.medianBlur(m, 5)
    pts = cv2.goodFeaturesToTrack(
        gray_u8,
        maxCorners=int(num_points),
        qualityLevel=0.01,
        minDistance=float(min_dist),
        blockSize=7,
        useHarrisDetector=False,
        mask=m,
    )
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = pts.reshape(-1, 2).astype(np.float32)
    return clip_border(pts, H, W, border=0)


def preset_motion(
    gray_video: np.ndarray,
    num_points: int = 200,
) -> np.ndarray:
    """Points in regions of high inter-frame intensity change."""
    if gray_video.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    H, W = gray_video.shape[1], gray_video.shape[2]
    diffs = np.abs(gray_video[1:].astype(np.float32) - gray_video[:-1].astype(np.float32))
    sum_diff = diffs.sum(axis=0)

    act = sum_diff
    motion_percentile = np.percentile(act, 70)
    m = (act >= motion_percentile).astype(np.uint8) * 255

    # Morphological opening to remove small noise, closing to fill tiny holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

    # compute total motion per component and drop weak ones (< 10% of strongest)
    component_motion = np.zeros(n_labels)
    for label in range(1, n_labels):
        component_motion[label] = act[labels == label].sum()
    component_motion_fg = component_motion[1:]
    max_motion = component_motion_fg.max() if len(component_motion_fg) > 0 else 0.0
    valid_labels = [
        label for label in range(1, n_labels)
        if component_motion[label] >= 0.1 * max_motion
    ]

    # --- Phase 1: one point per valid connected component (guaranteed coverage) ---
    phase1_points = []
    for label in valid_labels:
        component_mask = (labels == label)
        local_act = np.where(component_mask, act, -1)
        r, c = np.unravel_index(np.argmax(local_act), local_act.shape)
        phase1_points.append((c, r))  # store as (x, y)

    n_components = len(phase1_points)
    remaining_points = max(0, num_points - n_components)

    # --- Phase 2: distribute remaining points proportional to motion activity ---
    component_motion_fg = component_motion[
        [label for label in range(1, n_labels) if label in set(valid_labels)]
    ]
    total_motion = component_motion_fg.sum()

    # proportional allocation
    motion_weights = component_motion_fg / total_motion
    extra_per_component = (motion_weights * remaining_points).astype(int)

    # fix rounding: assign leftover points to highest-motion components
    leftover = remaining_points - extra_per_component.sum()
    top_indices = np.argsort(motion_weights)[::-1][:leftover]
    extra_per_component[top_indices] += 1

    all_points = phase1_points.copy()
    occupied = set(map(tuple, phase1_points)) 

    for i, label in enumerate(valid_labels):
        n_extra = extra_per_component[i]
        if n_extra == 0:
            continue

        component_mask = (labels == label)
        local_act = np.where(component_mask, act, 0)

        # get all possible peaks as candidates
        coords = peak_local_max(
            local_act,
            num_peaks=n_extra + len(occupied),
            min_distance=5,
            labels=component_mask,
        )

        for r, c in coords:
            if len(all_points) >= num_points:
                break
            pt = (c, r)
            if pt not in occupied:
                all_points.append(pt)
                occupied.add(pt)

    if not all_points:
        return np.zeros((0, 2), dtype=np.float32)
    return clip_border(np.array(all_points[:num_points], dtype=np.float32), H, W, border=0)


def preset_segmentation(gray_u8: np.ndarray, num_points: int = 200) -> np.ndarray:
    """Centroids of the largest connected components after Otsu thresholding."""
    H, W = gray_u8.shape[:2]
    _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        return np.zeros((0, 2), dtype=np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    cent = centroids[1:].astype(np.float32)
    order = np.argsort(-areas)
    cent = cent[order][: int(num_points)]
    return clip_border(cent, H, W, border=0)


def preset_matchanything(
    gray_u8_a: np.ndarray,
    gray_u8_b: np.ndarray | None = None,
    num_points: int = 200,
    min_score: float = 0.5,
    min_dist: float = 10,
    status_cb=None,
) -> np.ndarray:
    """MatchAnything keypoint matching.

    *gray_u8_a* / *gray_u8_b* are two uint8 grayscale images to match.
    When *gray_u8_b* is None, *gray_u8_a* is matched against itself.
    Returns points in the coordinate space of *gray_u8_b* (the reference).
    Falls back to Shi-Tomasi features when the model is unavailable.
    *status_cb*, if provided, is called with a status string on warnings/errors.
    """
    global _matchanything_model, _matchanything_processor, _matchanything_available

    if gray_u8_b is None:
        gray_u8_b = gray_u8_a

    def _status(msg: str):
        if status_cb is not None:
            status_cb(msg)
        else:
            print(msg, file=sys.stderr)

    def _fallback():
        return preset_features(gray_u8_b, num_points=num_points, min_dist=min_dist)

    if (
        _PIL_Image is None
        or AutoImageProcessor is None
        or AutoModelForKeypointMatching is None
        or not _matchanything_available
    ):
        return _fallback()

    def _to_pil(arr: np.ndarray):
        a = np.asarray(arr, dtype=np.float32)
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if mx > mn:
            a = (a - mn) / (mx - mn) * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        return _PIL_Image.fromarray(a)

    if _matchanything_model is None:
        try:
            model_name = "zju-community/matchanything_eloftr"
            _matchanything_processor = AutoImageProcessor.from_pretrained(model_name)
            _matchanything_model = AutoModelForKeypointMatching.from_pretrained(model_name)
            _matchanything_model.eval().to("cpu")
        except Exception as e:
            _status(f"MatchAnything model not available: {e}. Using fallback points.")
            _matchanything_available = False
            return _fallback()

    try:
        img0 = _to_pil(gray_u8_a)
        img1 = _to_pil(gray_u8_b)
        inputs = _matchanything_processor([img0, img1], return_tensors="pt")
        with torch.inference_mode():
            outputs = _matchanything_model(**inputs)
        result = _matchanything_processor.post_process_keypoint_matching(
            outputs,
            [[(img0.height, img0.width), (img1.height, img1.width)]],
            threshold=float(min_score),
        )[0]
        points1 = result["keypoints1"].detach().cpu().numpy().astype(np.float32)
        scores = result["matching_scores"].detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        _status(f"MatchAnything inference failed: {e}. Using fallback points.")
        _matchanything_available = False
        return _fallback()

    if points1 is None or len(points1) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    keep = scores >= float(min_score)
    points1, scores = points1[keep], scores[keep]
    if len(points1) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # keep only the top 5% by score within the filtered set
    score_thr = float(np.percentile(scores, 95))
    keep2 = scores >= score_thr
    points1, scores = points1[keep2], scores[keep2]
    if len(points1) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    order = np.argsort(-scores)
    points1 = points1[order]

    if min_dist > 0:
        selected: list[np.ndarray] = []
        thr2 = min_dist * min_dist
        for p in points1:
            if not selected:
                selected.append(p)
            else:
                q = np.asarray(selected, dtype=np.float32)
                if float(np.min(np.sum((q - p) ** 2, axis=1))) >= thr2:
                    selected.append(p)
        points1 = (
            np.asarray(selected, dtype=np.float32) if selected else np.zeros((0, 2), dtype=np.float32)
        )

    H, W = gray_u8_b.shape[:2]
    return clip_border(points1, H, W, border=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Greedy-motion preset helpers (private)
# ---------------------------------------------------------------------------

def _gm_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    img -= img.min()
    return img / (img.max() + eps)


def _gm_sampling_density(
    diff_img: np.ndarray,
    sigma: float = 1.8,
    gamma: float = 0.5,
    background_weight: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (density, smooth) from a 2-D absolute-diff image."""
    smooth = _gm_normalize(gaussian_filter(_gm_normalize(diff_img), sigma=sigma))
    density = smooth ** gamma + background_weight
    density /= density.sum()
    return density, smooth


def _gm_peak_points(
    smooth: np.ndarray,
    max_peaks: int,
    min_distance: int,
    threshold_rel: float,
) -> np.ndarray:
    """Return (N, 2) xy peak coordinates sorted by descending intensity."""
    peaks_yx = peak_local_max(
        smooth,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False,
        num_peaks=max_peaks,
    )
    if len(peaks_yx) == 0:
        return np.empty((0, 2), dtype=np.float32)
    peaks_xy = peaks_yx[:, ::-1].astype(np.float32)
    order = np.argsort(smooth[peaks_yx[:, 0], peaks_yx[:, 1]])[::-1]
    return peaks_xy[order]


def _gm_sample_candidates(
    density: np.ndarray,
    num_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw (num_candidates, 2) xy points biased by density with sub-pixel jitter."""
    H, W = density.shape
    indices = rng.choice(H * W, size=num_candidates, replace=True, p=density.ravel())
    y, x = np.unravel_index(indices, (H, W))
    x = np.clip(x + rng.uniform(-0.5, 0.5, num_candidates), 0, W - 1)
    y = np.clip(y + rng.uniform(-0.5, 0.5, num_candidates), 0, H - 1)
    return np.stack([x, y], axis=1).astype(np.float32)


def _gm_greedy_select(
    candidates: np.ndarray,
    density: np.ndarray,
    initial: np.ndarray | None,
    num_points: int,
    min_distance: float,
    relaxation: float,
    min_density_fraction: float = 0.01,
) -> np.ndarray:
    """Greedy density-aware spatially-balanced selection."""
    H, W = density.shape
    selected: list[np.ndarray] = [] if initial is None else list(initial[:num_points])

    if len(selected) >= num_points:
        return np.array(selected, dtype=np.float32)

    cx = np.clip(np.round(candidates[:, 0]).astype(int), 0, W - 1)
    cy = np.clip(np.round(candidates[:, 1]).astype(int), 0, H - 1)
    order = np.argsort(density[cy, cx])[::-1]
    candidates = candidates[order]

    current_dist = float(min_distance)
    while len(selected) < num_points and current_dist >= 1.0:
        sel_arr = np.array(selected, dtype=np.float32) if selected else np.empty((0, 2), dtype=np.float32)
        tree = cKDTree(sel_arr) if len(sel_arr) > 0 else None

        density_cutoff = min_density_fraction * density.max()
        for p in candidates:
            if len(selected) >= num_points:
                break
            cx = int(np.clip(round(p[0]), 0, density.shape[1] - 1))
            cy = int(np.clip(round(p[1]), 0, density.shape[0] - 1))
            if density[cy, cx] < density_cutoff:
                continue
            if tree is not None:
                dist, _ = tree.query(p, k=1)
                if dist < current_dist:
                    continue
            selected.append(p)
            if len(selected) % 25 == 0:
                sel_arr = np.array(selected, dtype=np.float32)
                tree = cKDTree(sel_arr)

        current_dist *= relaxation

    return np.array(selected, dtype=np.float32) if selected else np.zeros((0, 2), dtype=np.float32)



def preset_greedy_motion(
    gray_video: np.ndarray,
    num_points: int = 200,
    sigma: float = 1.8,
    gamma: float = 0.5,
    background_weight: float = 0.001,
    peak_min_distance: int = 10,
    peak_threshold_rel: float = 0.7,
    max_peak_fraction: float = 0.8,
    candidate_factor: int = 5,
    sample_min_distance: float = 8.0,
    random_seed: int = 0,
) -> np.ndarray:
    """Point placement from a summed absolute-difference motion map: local
    maxima as guaranteed seeds, then density-weighted candidates added
    greedily (min-distance relaxed until the budget is filled)."""
    if gray_video.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    H, W = gray_video.shape[1], gray_video.shape[2]

    diff_img = np.abs(
        gray_video[1:].astype(np.float32) - gray_video[:-1].astype(np.float32)
    ).sum(axis=0)

    density, smooth = _gm_sampling_density(diff_img, sigma=sigma, gamma=gamma,
                                           background_weight=background_weight)

    peaks = _gm_peak_points(smooth, max_peaks=num_points,
                            min_distance=peak_min_distance,
                            threshold_rel=peak_threshold_rel)

    # Peaks can be sparse on near-static clips -- <3 control points makes the downstream TPS solve singular.
    # Top up with density-weighted candidates only when under MIN_POINTS.
    MIN_POINTS = 5
    if len(peaks) < MIN_POINTS:
        rng = np.random.default_rng(random_seed)
        candidates = _gm_sample_candidates(density, candidate_factor * num_points, rng)
        selected = _gm_greedy_select(
            candidates, density,
            initial=peaks if len(peaks) > 0 else None,
            num_points=MIN_POINTS,
            min_distance=sample_min_distance,
            relaxation=0.7,
        )
    else:
        selected = peaks

    if len(selected) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return clip_border(selected, H, W, border=0)
