#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Callable
import pickle
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import tifffile
import torch
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

from utils.utils import _to_thwc, _to_gray_thw


_CHILD_SCRIPT = (
    "import pickle, sys\n"
    "from pyflowreg.motion_correction.compensate_recording import compensate_recording\n"
    "with open(sys.argv[1], 'rb') as fh:\n"
    "    options, config = pickle.load(fh)\n"
    "compensate_recording(options, config=config)\n"
)


def _compensate_in_clean_process(options, config, workdir: Path) -> None:
    """Run pyflowreg's compensate_recording in a fresh Python process.

    pyflowreg parallelises with a fork()ed process pool. Forking the GUI
    itself is unsafe once JAX (NoRMCorre) and torch have started their own
    threads -- Python warns "os.fork() is incompatible with multithreaded
    code ... will likely lead to a deadlock" and can hang. A freshly started
    interpreter has no such threads (pyflowreg itself doesn't load JAX), so
    it can fork normally and keeps pyflowreg's full multi-process speed;
    threading/sequential/spawn backends were 5-6x slower or paid a large
    per-worker start-up cost. The wrapper is file-based already, so nothing
    but the pickled options has to cross the process boundary."""
    args_file = workdir / "pyflowreg_args.pkl"
    with open(args_file, "wb") as fh:
        pickle.dump((options, config), fh)

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, str(args_file)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pyflowreg failed (exit code {proc.returncode}):\n{proc.stderr[-2000:]}"
        )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Dense displacement field as THW2 float32 numpy array (real per-pixel
        (u, v) from pyflowreg's saved w.h5 when save_w is True, else zeros).
    shift : np.ndarray
        Compact (T, 2) per-frame (dx, dy) summary: mean absolute (u, v) over
        the valid (non-OOB) region of each frame. Meant for sizing a shared
        border-crop margin (see utils.utils.shared_crop_drift) — analogous to
        NoRMCorre's shifts and CoTracker's drift — not as a metric itself.

    Notes
    -----
    This wrapper uses pyflowreg's file-based API.

    Out-of-bounds pixels revealed by the warp are zero-filled. pyflowreg's own
    imregister_wrapper uses cv2.BORDER_REPLICATE internally (edge pixels are
    repeated, not zero or reference-frame content, despite what its docstring
    claims), so save_valid_idx=True is used here to get the exact per-pixel
    valid mask pyflowreg itself computes, and everything outside it is zeroed
    after the fact — no need to patch the installed package.
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
            save_valid_idx=True,
        )

        config = RegistrationConfig(
            n_jobs=-1,
            verbose=False,
        )

        _compensate_in_clean_process(options, config, tmpdir)

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

        # Zero-fill out-of-bounds pixels via pyflowreg's valid mask, replacing its default edge-replicate border.
        valid_mask = None
        idx_file = output_dir / "idx.hdf"
        if idx_file.exists():
            with h5py.File(idx_file, "r") as f:
                mask = np.asarray(_find_first_dataset(f))
            if mask.shape == (t, h, w):
                valid_mask = mask != 0
                warped_gray[~valid_mask] = 0.0

        warped = warped_gray[..., None]
        if c > 1:
            warped = np.repeat(warped, c, axis=-1)

        # Dense displacement field from pyflowreg's saved w.h5, plus a compact per-frame shift summary.
        disp = np.zeros((t, h, w, 2), dtype=np.float32)
        shift = np.zeros((t, 2), dtype=np.float32)
        w_file = output_dir / "w.h5"
        if w_file.exists():
            with h5py.File(w_file, "r") as f:
                if "u" in f and "v" in f:
                    u = np.asarray(f["u"], dtype=np.float32)
                    v = np.asarray(f["v"], dtype=np.float32)
                    if u.shape == (t, h, w) and v.shape == (t, h, w):
                        disp = np.stack([u, v], axis=-1)
                        region = valid_mask if valid_mask is not None else np.ones(
                            (t, h, w), dtype=bool
                        )
                        for i in range(t):
                            m = region[i]
                            if m.any():
                                shift[i, 0] = float(np.abs(u[i][m]).mean())
                                shift[i, 1] = float(np.abs(v[i][m]).mean())

        if callback is not None:
            callback(1.0)

        return warped.astype(np.float32), disp, shift

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
