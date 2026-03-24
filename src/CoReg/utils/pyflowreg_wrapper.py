from __future__ import annotations

from pathlib import Path
from typing import Callable
import shutil
import tempfile

import h5py
import numpy as np
import tifffile
import torch
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_recording import (
    RegistrationConfig,
    compensate_recording,
)



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


def _find_first_dataset(group: h5py.Group) -> np.ndarray:
    """Return the first dataset found recursively in an HDF5 group."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            return item[()]
        if isinstance(item, h5py.Group):
            out = _find_first_dataset(item)
            if out is not None:
                return out
    return None


def register_video(
    video: np.ndarray | torch.Tensor,
    reference_index: int = 0,
    callback: Callable[[float], None] | None = None,
    *,
    alpha: float = 1.5,
    min_level: int = 4,
    buffer_size: int = 100,
    save_w: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thin file-based wrapper around pyflowreg.

    Parameters
    ----------
    video : np.ndarray | torch.Tensor
        Input video in THW, THWC, or TCHW format.
    reference_index : int
        Frame used as the registration template.
    callback : callable | None
        Optional progress callback receiving values in [0, 1].
    alpha : float
        Optical flow regularization strength. Lower values = more flexible.
    min_level : int
        Minimum pyramid level (coarser levels catch larger jumps).
    buffer_size : int
        Processing buffer size (shorter segments avoid discontinuities).
    save_w : bool
        Whether to save displacement fields.

    Returns
    -------
    warped : np.ndarray
        Registered video as THWC float32 numpy array.
    disp : np.ndarray
        Displacement field as THW2 float32 numpy array.

    Notes
    -----
    This wrapper uses pyflowreg's file-based API.
    Currently, displacement fields are returned as zeros unless you set
    save_w=True and add code to read the saved displacement output.
    """

    v = _to_thwc(video).astype(np.float32)
    t, h, w, c = v.shape
    ref_idx = int(np.clip(reference_index, 0, max(0, t - 1)))

    # pyflowreg is primarily intended for single-channel microscopy data
    gray = _to_gray_thw(v).astype(np.float32)  # (T, H, W)

    if callback is not None:
        callback(0.05)

    tmpdir = Path(tempfile.mkdtemp(prefix="pyflowreg_"))
    try:
        input_file = tmpdir / "input.tif"
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write temporary input stack
        tifffile.imwrite(str(input_file), gray)

        if callback is not None:
            callback(0.15)

        options = OFOptions(
            input_file=str(input_file),
            output_path=str(output_dir),
            output_format="HDF5",
            alpha=alpha,
            min_level=min_level,
            bin_size=1,
            buffer_size=min(buffer_size, t),
            reference_frames=[ref_idx],
            save_meta_info=True,
            save_w=save_w,
        )

        config = RegistrationConfig(
            n_jobs=-1,
            verbose=False,
        )

        compensate_recording(options, config=config)

        if callback is not None:
            callback(0.85)

        compensated_file = output_dir / "compensated.HDF5"
        if not compensated_file.exists():
            raise RuntimeError(
                f"pyflowreg completed, but output file was not found: {compensated_file}"
            )

        with h5py.File(compensated_file, "r") as f:
            # Try a few likely dataset names first
            dataset = None
            for key in ("data", "video", "images", "stack", "compensated"):
                if key in f and isinstance(f[key], h5py.Dataset):
                    dataset = f[key][:]
                    break

            # Fallback: first dataset found recursively
            if dataset is None:
                dataset = _find_first_dataset(f)

            if dataset is None:
                raise RuntimeError(f"No readable dataset found in {compensated_file}")

        warped_gray = np.asarray(dataset, dtype=np.float32)

        if warped_gray.shape != (t, h, w):
            raise RuntimeError(
                f"Unexpected compensated shape: {warped_gray.shape}, expected {(t, h, w)}"
            )

        warped = warped_gray[..., None]
        if c > 1:
            warped = np.repeat(warped, c, axis=-1)

        # Placeholder: pyflowreg displacement fields are not read here
        disp = np.zeros((t, h, w, 2), dtype=np.float32)

        if callback is not None:
            callback(1.0)

        return warped.astype(np.float32), disp

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
