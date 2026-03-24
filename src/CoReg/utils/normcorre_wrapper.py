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


def _broadcast_rigid_shifts_to_dense(
    shifts: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert rigid shifts (T, 2) into dense displacement field (T, H, W, 2).
    """
    shifts = np.asarray(shifts, dtype=np.float32)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError(f"Expected rigid shifts of shape (T, 2), got {shifts.shape}")

    t = shifts.shape[0]
    disp = np.zeros((t, height, width, 2), dtype=np.float32)
    disp[..., 0] = shifts[:, None, None, 0]
    disp[..., 1] = shifts[:, None, None, 1]
    return disp


def _compute_patch_grid(
    height: int,
    width: int,
    strides: tuple[int, int],
    overlaps: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Reconstruct the pw-rigid patch grid as rectangles (r0, r1, c0, c1).

    Patch size is stride + overlap, and the final patch in each dimension is
    forced to end at the image boundary.
    """
    sy, sx = int(strides[0]), int(strides[1])
    oy, ox = int(overlaps[0]), int(overlaps[1])

    py = sy + oy
    px = sx + ox

    if py <= 0 or px <= 0:
        raise ValueError(
            f"Invalid patch size from strides={strides}, overlaps={overlaps}"
        )

    def _starts(size: int, stride: int, patch: int) -> list[int]:
        if size <= patch:
            return [0]

        starts = list(range(0, size - patch + 1, stride))
        last = size - patch
        if starts[-1] != last:
            starts.append(last)
        return starts

    y_starts = _starts(height, sy, py)
    x_starts = _starts(width, sx, px)

    patches: list[tuple[int, int, int, int]] = []
    for r0 in y_starts:
        r1 = min(r0 + py, height)
        for c0 in x_starts:
            c1 = min(c0 + px, width)
            patches.append((r0, r1, c0, c1))

    return patches


def _patch_shifts_to_dense(
    patch_shifts: np.ndarray,
    height: int,
    width: int,
    strides: tuple[int, int],
    overlaps: tuple[int, int],
) -> np.ndarray:
    """
    Convert pw-rigid patch shifts (T, num_patches, 2) to an approximate
    dense displacement field (T, H, W, 2) by averaging overlapping patches.
    """
    patch_shifts = np.asarray(patch_shifts, dtype=np.float32)
    if patch_shifts.ndim != 3 or patch_shifts.shape[-1] != 2:
        raise ValueError(
            f"Expected pw-rigid shifts of shape (T, num_patches, 2), got {patch_shifts.shape}"
        )

    patches = _compute_patch_grid(
        height=height,
        width=width,
        strides=strides,
        overlaps=overlaps,
    )

    num_patches = len(patches)
    if patch_shifts.shape[1] != num_patches:
        raise RuntimeError(
            f"Patch shift count mismatch: got {patch_shifts.shape[1]} shifts but "
            f"expected {num_patches} patches from strides={strides}, overlaps={overlaps}, "
            f"image size={(height, width)}"
        )

    t = patch_shifts.shape[0]
    disp_sum = np.zeros((t, height, width, 2), dtype=np.float32)
    disp_count = np.zeros((height, width, 1), dtype=np.float32)

    for patch_idx, (r0, r1, c0, c1) in enumerate(patches):
        disp_sum[:, r0:r1, c0:c1, 0] += patch_shifts[:, patch_idx, 0][:, None, None]
        disp_sum[:, r0:r1, c0:c1, 1] += patch_shifts[:, patch_idx, 1][:, None, None]
        disp_count[r0:r1, c0:c1, 0] += 1.0

    disp_count = np.maximum(disp_count, 1.0)
    return disp_sum / disp_count[None, ...]


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
    """
    Thin in-memory wrapper around jnormcorre.

    Parameters
    ----------
    video : np.ndarray | torch.Tensor
        Input video in THW, THWC, or TCHW format.
    reference_index : int
        Frame used as the registration template.
    callback : callable | None
        Optional progress callback receiving values in [0, 1].
    pw_rigid : bool
        If True, use piecewise-rigid registration; otherwise rigid only.
    max_shifts : tuple[int, int]
        Max allowed shifts in (row, col).
    strides : tuple[int, int]
        Patch strides for pw-rigid mode.
    overlaps : tuple[int, int]
        Patch overlaps for pw-rigid mode.
    upsample_factor_fft : int
        Subpixel resolution / shift upsampling factor.
    max_deviation_rigid : int
        Max patch deviation from rigid shift in pw-rigid mode.
    add_to_movie : float | None
        Constant offset added before registration. If None, chosen automatically
        so the working movie is non-negative.

    Returns
    -------
    warped : np.ndarray
        Registered video as THWC float32 numpy array.
    disp : np.ndarray
        Dense displacement field as THW2 float32 numpy array.

    Notes
    -----
    - This wrapper uses a single reference frame as template.
    - For pw-rigid mode, `disp` is an approximate dense field reconstructed
      from the returned per-patch shifts by averaging overlaps.
    """

    v = _to_thwc(video).astype(np.float32)
    t, h, w, c = v.shape
    ref_idx = int(np.clip(reference_index, 0, max(0, t - 1)))

    gray = _to_gray_thw(v).astype(np.float32)  # (T, H, W)
    template = np.asarray(gray[ref_idx], dtype=np.float32)

    if add_to_movie is None:
        min_val = float(np.min(gray))
        add_to_movie = -min_val if min_val < 0 else 0.0

    if callback is not None:
        callback(0.05)

    if pw_rigid:
        warped_gray, patch_shifts = register_frames_to_template_pwrigid(
            gray,
            template,
            int(strides[0]),
            int(strides[1]),
            int(overlaps[0]),
            int(overlaps[1]),
            np.asarray(max_shifts),
            int(upsample_factor_fft),
            int(max_deviation_rigid),
            np.asarray(add_to_movie, dtype=np.float32),
        )
    else:
        warped_gray, rigid_shifts = register_frames_to_template_rigid(
            gray,
            template,
            np.asarray(max_shifts),
            np.asarray(add_to_movie, dtype=np.float32),
        )

    if callback is not None:
        callback(0.80)

    warped_gray = np.asarray(warped_gray, dtype=np.float32)
    if warped_gray.shape != (t, h, w):
        raise RuntimeError(
            f"Unexpected corrected shape: {warped_gray.shape}, expected {(t, h, w)}"
        )

    warped = warped_gray[..., None]
    if c > 1:
        warped = np.repeat(warped, c, axis=-1)

    if pw_rigid:
        disp = _patch_shifts_to_dense(
            patch_shifts=np.asarray(patch_shifts, dtype=np.float32),
            height=h,
            width=w,
            strides=strides,
            overlaps=overlaps,
        )
    else:
        disp = _broadcast_rigid_shifts_to_dense(
            shifts=np.asarray(rigid_shifts, dtype=np.float32),
            height=h,
            width=w,
        )

    if callback is not None:
        callback(1.0)

    return warped.astype(np.float32), disp.astype(np.float32)
