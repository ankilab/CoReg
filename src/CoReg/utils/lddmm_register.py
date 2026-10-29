#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

import gc

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import utils.lddmm_wrapper as lddmm_wrapper


def integrate_field(v, num_steps=10):
    """
    Forward-integrate a velocity field v(x,t) over [0,1].
    :param v: (2, H, W) velocity field.
    :param num_steps: Number of integration steps.
    :return: (2, H, W) deformation field phi.
    """
    # Basic coordinate grid
    _, H, W = v.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    # Initialize phi
    phi_x = grid_x.astype(np.float32)
    phi_y = grid_y.astype(np.float32)

    dt = 1.0 / num_steps
    for _ in range(num_steps):
        # Euler step: phi_{t+dt} = phi_t + dt * v(phi_t)
        vx = map_coordinates(v[0], [phi_y, phi_x], order=1, mode="nearest")
        vy = map_coordinates(v[1], [phi_y, phi_x], order=1, mode="nearest")
        phi_x += dt * vx
        phi_y += dt * vy

    return np.stack([phi_x, phi_y], axis=0)


def _normalize_pair(fixed: np.ndarray, moving: np.ndarray):
    """Jointly normalise two frames to [0, 1] float64."""
    lo    = float(min(fixed.min(), moving.min()))
    hi    = float(max(fixed.max(), moving.max()))
    scale = hi - lo if hi > lo else 1.0
    return (
        ((fixed  - lo) / scale).astype(np.float64),
        ((moving - lo) / scale).astype(np.float64),
        lo, scale,
    )


def register_lddmm(fixed, moving):
    """
    Compute LDDMM registration between two frames.
    :param fixed: Reference/target frame (any intensity range).
    :param moving: Source/template frame (any intensity range).
    :return: Warped numpy array in the original intensity range.

    Images are jointly normalised to [0, 1] before registration so that
    sigma/sigmaR are scale-independent.  The result is mapped back to the
    original intensity range before returning.
    """
    fixed_n, moving_n, lo, scale = _normalize_pair(
        np.asarray(fixed, dtype=np.float64),
        np.asarray(moving, dtype=np.float64),
    )

    lddmm = lddmm_wrapper.LDDMM(
        template=[moving_n],
        target=[fixed_n],
        do_affine=1,
        do_lddmm=1,
        a=40,
        nt=5,
        niter=50,
        epsilon=0.001,
        sigma=0.1,
        sigmaR=1.0,
        optimizer="gdr",
        gpu_number=0 if torch.cuda.is_available() else None,  # None -> CPU
        verbose=False,
        dx=None,
    )

    try:
        lddmm.run()
        warped_n = np.asarray(lddmm.outputDeformedTemplate()[0], dtype=np.float64)
    finally:
        # delete() used to leave self.I/J/M (GPU tensors) referenced until fixed in lddmm_wrapper.py --
        # del + gc.collect() here removes any remaining dependency on GC timing.
        lddmm.delete()
        del lddmm
        gc.collect()

    return (warped_n * scale + lo).astype(np.float32)


def lddmm_registration(frames):
    """
    Perform LDDMM registration on a sequence of frames, using frame 0 as reference.
    :param frames: List or array of frames (HW or HWC).
    :return: List of warped numpy arrays (same shape as input frames).
    """
    # LDDMM wrapper expects 2-D (HW) arrays; squeeze channel dim if present
    def _to_hw(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return arr[..., 0] if arr.ndim == 3 else arr

    ref_hw = _to_hw(frames[0])
    total = len(frames)
    lddmm_registrations = [frames[0]]
    for i in tqdm(range(1, total), desc="LDDMM frames", unit="frame"):
        mov_hw = _to_hw(frames[i])
        reg_hw = register_lddmm(ref_hw, mov_hw)  # returns HW array
        # restore channel dim to match input shape
        if np.asarray(frames[i]).ndim == 3:
            reg_hw = reg_hw[..., np.newaxis]
        lddmm_registrations.append(reg_hw)

    return lddmm_registrations


