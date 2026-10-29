#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

import gc

import ants
import numpy as np
import torch

# Memory cleanup to prevent OOM errors
torch.cuda.empty_cache()
gc.collect()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def ants_registration(frames):
    """
    Perform SyN registration on a sequence of frames, using frame 0 as reference.
    :param frames: List or array of (H, W, C) frames.
    :return: List of warped numpy arrays.
    """
    results = []
    # Use frame 0 as the fixed reference
    fixed_image = ants.from_numpy(frames[0][..., 0])

    for i in range(1, len(frames)):
        moving_arr = frames[i][..., 0]
        if np.asarray(moving_arr).var() < 1e-6:
            # Blank/constant frame -> SyN's solver degenerates to a singular matrix. Pass through unchanged instead.
            results.append(np.asarray(moving_arr, dtype=np.float32))
            continue
        moving_image = ants.from_numpy(moving_arr)
        reg = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyNOnly")
        results.append(reg["warpedmovout"].numpy())

    return results

