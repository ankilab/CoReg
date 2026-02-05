import gc
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

# Memory cleanup to prevent OOM errors
torch.cuda.empty_cache()
gc.collect()




def cotrack_model(video, queries, device, grid_size=0):
    """
    Run the CoTracker model on the provided video and grid points.

    :param video: (T, C, H, W) video tensor.
    :param queries: (1, N, 2) query points or None.
    :param device: torch device.
    :param grid_size: Grid size for automatic point generation (if queries is None).
    :return: (pred_tracks, pred_visibility)
    """

    import os
    os.environ["TORCH_HOME"] = "/data/users/va11moje/cache/cotracker"

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    video = video.to(device)
    model.model.model_resolution = video.shape[2:]

    
    pred_tracks, pred_visibility = model(
        video.unsqueeze(0).to(device),
        queries=queries.to(device),
        grid_size=grid_size,
        grid_query_frame=0,      # ignored when queries is provided, but fine to keep
        backward_tracking=False
    )

    return pred_tracks, pred_visibility

def _crop_or_pad_phi_to_hw(phi_2hw: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize a deformation field phi_2hw (2, h, w) to target size (2, H, W).
    - If larger: crop to (H, W).
    - If smaller: pad with edge replication.
    """
    phi = np.asarray(phi_2hw, dtype=np.float32)
    if phi.ndim != 3 or phi.shape[0] != 2:
        raise ValueError(f"phi must be (2,h,w), got {phi.shape}")

    h, w = phi.shape[1], phi.shape[2]

    # crop falls nötig
    phi = phi[:, :min(h, H), :min(w, W)]
    h2, w2 = phi.shape[1], phi.shape[2]

    # pad falls nötig (unten/rechts), mit edge-replication
    pad_h = max(0, H - h2)
    pad_w = max(0, W - w2)
    if pad_h > 0 or pad_w > 0:
        phi = np.pad(phi, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

    return phi


def gaussian_smooth(u, sigma=2.0):
    """Apply Gaussian smoothing to a 2D array."""
    return cv2.GaussianBlur(u.astype(np.float32, copy=False), (0, 0), float(sigma))
    

def warping(predicted_tracks, frames, smoothing=3, callback=None):
    """
    Apply motion correction by warping frames according to predicted tracks.

    :param predicted_tracks: (B, T, G, 2) array of point tracks.
    :param frames: (T, C, H, W) or (T, H, W, C) video data.
    :param smoothing: Sigma for Gaussian smoothing of the displacement field.
    :param callback: Optional function(float) for progress reporting.
    :return: (warped_frames, displacement_fields)
    """

    # ---- to numpy ----
    if torch.is_tensor(predicted_tracks):
        predicted_tracks = predicted_tracks.detach().cpu().numpy()
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()

    tracks = np.asarray(predicted_tracks)
    frames = np.asarray(frames)

    # ---- normalize frames shape ----
    if frames.ndim == 4 and frames.shape[1] in (1, 3):   # (T,C,H,W)
        T_orig, C, H, W = frames.shape
        frame0_gray = frames[0, 0].astype(np.float32)
        get_frame_gray = lambda i: frames[i, 0].astype(np.float32)
    elif frames.ndim == 4 and frames.shape[-1] in (1, 3): # (T,H,W,C)
        T_orig, H, W, C = frames.shape
        frame0_gray = frames[0, ..., 0].astype(np.float32)
        get_frame_gray = lambda i: frames[i, ..., 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported frames shape: {frames.shape}")

    # ---- input range (gray channel 0) ----
    if frames.ndim == 4 and frames.shape[1] in (1, 3):          # (T,C,H,W)
        gray_stack = frames[:, 0].astype(np.float32, copy=False)
    else:                                                       # (T,H,W,C)
        gray_stack = frames[..., 0].astype(np.float32, copy=False)

    in_min = float(np.nanmin(gray_stack))
    in_max = float(np.nanmax(gray_stack))
    do_clip = np.isfinite(in_min) and np.isfinite(in_max) and (in_max >= in_min)

    # clip the reference frame too (optional but keeps everything consistent)
    if do_clip:
        frame0_gray = np.clip(frame0_gray, in_min, in_max).astype(np.float32, copy=False)

    B, T, G, D = tracks.shape
    if D != 2:
        raise ValueError(f"predicted_tracks last dim must be 2, got {D}")
    grid_size = int(math.sqrt(G))

    if T_orig > T:
        raise ValueError(f"frames has {T_orig} frames but predicted_tracks has only {T} timesteps")

    sigma = float(smoothing) if smoothing is not None else 0.0

    # Tracks for frame 0 (reference)
    sparse_x = tracks[0, :, :, 0]  # (T, G)
    sparse_y = tracks[0, :, :, 1]  # (T, G)
    disp_sparse = tracks[0] - tracks[0, 0]  # (T, G, 2)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))  # (H,W)
    dx_dense = np.zeros((T, H, W), dtype=np.float32)
    dy_dense = np.zeros((T, H, W), dtype=np.float32)

    # Linear interpolation der sparse displacements auf dense grid
    pts0 = (sparse_x[0], sparse_y[0])

    for t in range(T):
        dx_lin = griddata(
            pts0,
            disp_sparse[t, :, 0],
            (X, Y),
            method="linear", #cubic
            fill_value=np.nan
        )

        dy_lin = griddata(
            pts0,
            disp_sparse[t, :, 1],
            (X, Y),
            method="linear", #linear
            fill_value=np.nan
        )

        # NaNs mit nearest auffüllen (pro t!)
        dx_nn = griddata(pts0, disp_sparse[t, :, 0], (X, Y), method="nearest")
        dy_nn = griddata(pts0, disp_sparse[t, :, 1], (X, Y), method="nearest")

        dx_dense[t] = np.where(np.isnan(dx_lin), dx_nn, dx_lin).astype(np.float32)
        dy_dense[t] = np.where(np.isnan(dy_lin), dy_nn, dy_lin).astype(np.float32)



    vp = np.stack((dx_dense, dy_dense), axis=1)  # (T,2,H,W)

    grid_x0, grid_y0 = np.meshgrid(np.arange(W, dtype=np.float32),
                                   np.arange(H, dtype=np.float32))

    warpeds = [frame0_gray]
    phis = [np.zeros((H, W, 2), dtype=np.float32)]

    for i in range(1, T_orig):
        
        if callback is not None:
             callback(i / T_orig)

        phi = vp[i]  # (2,h,w) evtl nicht exakt H,W
        phi = _crop_or_pad_phi_to_hw(phi, H, W)  # -> (2,H,W), pad mit Randwert
        phi[0] = gaussian_smooth(phi[0], smoothing)
        phi[1] = gaussian_smooth(phi[1], smoothing)

        phis.append(np.stack([phi[0], phi[1]], axis=-1))  # (H,W,2) (smoothed)

        sample_x = grid_x0 + phi[0]
        sample_y = grid_y0 + phi[1]

        src = get_frame_gray(i).astype(np.float32, copy=False)


        warped = map_coordinates(
            src,
            [sample_y, sample_x],
            order=2,
            mode="nearest"   
        ).astype(np.float32)

        if do_clip:
            warped = np.clip(warped, in_min, in_max, out=warped)

        warpeds.append(warped)

    return np.asarray(warpeds, dtype=np.float32), np.asarray(phis, dtype=np.float32)







