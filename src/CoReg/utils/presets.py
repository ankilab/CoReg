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
    spacing_px: int = 0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """Centered rectangular grid of (rows × cols) points.

    *spacing_px* shifts points apart (positive) or closer together (negative)
    relative to the default dynamic margin.  *offset_x* / *offset_y* shift the
    grid center in pixels.
    """
    rows = max(2, int(rows))
    cols = max(2, int(cols))

    base_mx = max(5.0, float(W) / float(cols + 1))
    base_my = max(5.0, float(H) / float(rows + 1))
    extra = float(spacing_px)
    margin_x = max(1.0, base_mx - extra)
    margin_y = max(1.0, base_my - extra)
    margin_x = min(margin_x, max(1.0, float(W) / 2.0 - 1.0))
    margin_y = min(margin_y, max(1.0, float(H) / 2.0 - 1.0))

    start_x = margin_x
    end_x = float(W) - margin_x
    start_y = margin_y
    end_y = float(H) - margin_y

    if end_x <= start_x:
        start_x, end_x = 1.0, float(W) - 1.0
    if end_y <= start_y:
        start_y, end_y = 1.0, float(H) - 1.0

    xs = np.linspace(start_x, end_x, cols, dtype=np.float32) + float(offset_x)
    ys = np.linspace(start_y, end_y, rows, dtype=np.float32) + float(offset_y)
    xs = np.clip(xs, 1.0, float(W) - 1.0).astype(np.float32)
    ys = np.clip(ys, 1.0, float(H) - 1.0).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


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


def preset_activity(
    gray_video: np.ndarray,
    num_points: int = 200,
    min_dist: float = 15,
) -> np.ndarray:
    """Points in regions of high inter-frame intensity change."""
    if gray_video.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    H, W = gray_video.shape[1], gray_video.shape[2]
    diffs = np.abs(gray_video[1:] - gray_video[:-1])
    act = diffs.mean(axis=0)
    act_u8 = cv2.normalize(act, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    act_u8 = cv2.GaussianBlur(act_u8, (0, 0), 1.0)
    _, m = cv2.threshold(act_u8, 0, 255, cv2.THRESH_OTSU)
    m = cv2.medianBlur(m, 5)
    pts = cv2.goodFeaturesToTrack(
        act_u8,
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
    min_score: float = 0.2,
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
            if len(selected) >= int(num_points):
                break
        points1 = (
            np.asarray(selected, dtype=np.float32) if selected else np.zeros((0, 2), dtype=np.float32)
        )
    else:
        points1 = points1[: int(num_points)]

    H, W = gray_u8_b.shape[:2]
    return clip_border(points1, H, W, border=0).astype(np.float32)
