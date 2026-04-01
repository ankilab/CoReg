from __future__ import annotations
from typing import Callable
import numpy as np
import torch

try:
    from jnormcorre.motion_correction import (
        register_frames_to_template_pwrigid,
        register_frames_to_template_rigid,
    )
except ImportError as e:
    raise ImportError(
        "jnormcorre is not installed. Install it with `pip install jnormcorre` "
        "or from the GitHub repo after installing the appropriate JAX build."
    ) from e


def _to_thwc(video: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input video to numpy THWC format."""
    if isinstance(video, torch.Tensor):
        v = video.detach().cpu().numpy()
    else:
        v = np.asarray(video)

    if v.ndim == 4 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
        # TCHW -> THWC
        v = np.transpose(v, (0, 2, 3, 1))
    elif v.ndim == 3:
        # THW -> THWC
        v = v[..., None]
    elif v.ndim != 4:
        raise ValueError(f"Unsupported video shape: {v.shape}")

    return np.asarray(v)


def _to_gray_thw(video_thwc: np.ndarray) -> np.ndarray:
    """Convert THWC to grayscale THW float32."""
    if video_thwc.shape[-1] == 1:
        return video_thwc[..., 0].astype(np.float32)

    if video_thwc.shape[-1] >= 3:
        rgb = video_thwc[..., :3].astype(np.float32)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        return gray.astype(np.float32)

    raise ValueError(f"Unsupported channel count: {video_thwc.shape[-1]}")


def register_video(
    video: np.ndarray | torch.Tensor,
    reference_index: int = 0,
    callback: Callable[[float], None] | None = None,
    *,
    pw_rigid: bool = True,
    max_shifts: tuple[int, int] = (6, 6),
    strides: tuple[int, int] = (96, 96),
    overlaps: tuple[int, int] = (32, 32),
    upsample_factor_fft: int = 4,
    max_deviation_rigid: int = 3,
    add_to_movie: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Standardize Input
    v = _to_thwc(video).astype(np.float32)
    t, h, w, c = v.shape
    ref_idx = int(np.clip(reference_index, 0, max(0, t - 1)))

    # 2. Prepare Template (Grayscale)
    gray = _to_gray_thw(v)
    template = gray[ref_idx]

    if add_to_movie is None:
        min_val = float(np.min(gray))
        add_to_movie = -min_val if min_val < 0 else 0.0

    if callback:
        callback(0.1)

    # 3. Core NoRMCorre Logic
    # Apply registration to the GRAYSCALE video to get the shifts.
    # Match the installed jnormcorre signatures exactly to avoid JAX vmap errors.
    if pw_rigid:
        strides_0, strides_1 = int(strides[0]), int(strides[1])
        overlaps_0, overlaps_1 = int(overlaps[0]), int(overlaps[1])
        max_shifts_arr = np.asarray(max_shifts, dtype=gray.dtype)
        up_fft = int(upsample_factor_fft)
        max_dev = int(max_deviation_rigid)
        add_val = np.asarray(add_to_movie, dtype=gray.dtype)

        warped_gray, shifts = register_frames_to_template_pwrigid(
            gray,
            template,
            strides_0,
            strides_1,
            overlaps_0,
            overlaps_1,
            max_shifts_arr,
            up_fft,
            max_dev,
            add_val,
        )
    else:
        max_shifts_arr = np.asarray(max_shifts, dtype=gray.dtype)
        add_val = np.asarray(add_to_movie, dtype=gray.dtype)
        warped_gray, shifts = register_frames_to_template_rigid(
            gray,
            template,
            max_shifts_arr,
            add_val,
        )

    if callback:
        callback(0.9)

    # 4. Handle Multi-Channel
    # jnormcorre operates on single-channel data. For multi-channel input we
    # reuse the grayscale warp for all channels.
    warped_gray = np.asarray(warped_gray, dtype=np.float32)
    if warped_gray.shape != (t, h, w):
        raise RuntimeError(
            f"Unexpected corrected shape from NoRMCorre: {warped_gray.shape}, expected {(t, h, w)}"
        )

    warped = warped_gray[..., None]
    if c > 1:
        warped = np.repeat(warped, c, axis=-1)

    if callback:
        callback(1.0)
    return warped, np.asarray(shifts)
