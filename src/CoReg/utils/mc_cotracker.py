#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

import gc
import math
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

_COTRACKER_CACHE: dict[tuple[str, bool], object] = {}

def _default_torch_home() -> str:
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]
    return str(Path.home() / ".cache" / "torch")

_WEIGHTS_URL = {
    True:  "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
    False: "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth",
}

def _get_cotracker(device: str, offline: bool = True, weights_path: str | None = None):
    key = (str(device), offline, str(weights_path) if weights_path else "")
    if key not in _COTRACKER_CACHE:
        torch_home = _default_torch_home()
        os.environ["TORCH_HOME"] = torch_home
        Path(torch_home).mkdir(parents=True, exist_ok=True)

        from cotracker.predictor import CoTrackerPredictor

        window_len = 60 if offline else 16
        model = CoTrackerPredictor(checkpoint=None, offline=offline, window_len=window_len, v2=False)

        if weights_path is not None:
            print(f"Loading local CoTracker weights: {weights_path}", flush=True)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                _WEIGHTS_URL[offline], map_location="cpu"
            )

        # handle both raw state dicts and checkpoint dicts with a nested key
        if isinstance(state_dict, dict):
            for key_ in ("model_state", "model", "state_dict", "model_state_dict"):
                if key_ in state_dict:
                    state_dict = state_dict[key_]
                    break

        # strip "model." prefix if checkpoint was saved from the CoTrackerPredictor wrapper
        if isinstance(state_dict, dict) and all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

        model.model.load_state_dict(state_dict)
        model = model.to(device).eval()

        if not offline:
            # CoTrackerThreeOnline.forward() does not accept progress_cb but
            # CoTrackerPredictor always forwards it — patch the inner model to drop it.
            _orig_fwd = model.model.forward
            def _fwd_no_cb(*args, **kwargs):
                kwargs.pop("progress_cb", None)
                return _orig_fwd(*args, **kwargs)
            model.model.forward = _fwd_no_cb

        _COTRACKER_CACHE[key] = model

    return _COTRACKER_CACHE[key]


def release_gpu_memory():
    """Drop the cached CoTracker models and free all GPU memory from PyTorch and JAX."""
    _COTRACKER_CACHE.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # JAX cleanup (only if JAX has already been imported — avoid triggering it)
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            jax.clear_caches()
        except Exception:
            pass
        try:
            for backend in ("gpu", "cuda"):
                try:
                    client = jax.lib.xla_bridge.get_backend(backend)
                    for buf in client.live_buffers():
                        buf.delete()
                except Exception:
                    pass
        except Exception:
            pass
        gc.collect()


def _online_chunked(model, video_cpu, queries_gpu, grid_size, device, chunk_size, amp_ctx):
    """
    Process online CoTracker in temporal chunks to avoid moving the full video to GPU.
    Each chunk's queries are seeded from the last-frame positions of the previous chunk.

    video_cpu:   (T, 3, H, W) on CPU
    queries_gpu: (1, N, 3) on device
    Returns: (1, T, N, 2), (1, T, N, ?)  on CPU
    """
    T = video_cpu.shape[0]
    all_tracks: list[torch.Tensor] = []
    all_vis:    list[torch.Tensor] = []

    start = 0
    while start < T:
        end   = min(start + chunk_size, T)
        chunk = video_cpu[start:end].unsqueeze(0).to(device)   # (1, L, 3, H, W)

        call_kw: dict = dict(queries=queries_gpu, grid_size=grid_size, grid_query_frame=0)
        with amp_ctx:
            t_chunk, v_chunk = model(chunk, **call_kw)          # (1, L, N, 2/?)

        all_tracks.append(t_chunk.cpu())
        all_vis.append(v_chunk.cpu())

        # seed next chunk: query positions = last-frame positions of this chunk
        last_pos = t_chunk[:, -1]                               # (1, N, 2)
        b, n, _ = last_pos.shape
        t0 = torch.zeros((b, n, 1), device=device, dtype=last_pos.dtype)
        queries_gpu = torch.cat([t0, last_pos], dim=-1)         # (1, N, 3)
        grid_size = 0                                           # no new grid points

        del chunk, t_chunk, v_chunk
        torch.cuda.empty_cache()

        start = end

    return torch.cat(all_tracks, dim=1), torch.cat(all_vis, dim=1)


@torch.inference_mode()
def cotrack_model(
    video,
    queries,
    device,
    grid_size=0,
    progress_cb=None,
    offline: bool = True,
    online_chunk_size: int = 150,
    weights_path: str | None = None,
):
    """
    NON-BATCHED version:
      video:   [T,3,H,W]
      queries: [1,N,3] (t,x,y)
    returns:
      pred_tracks:      [1,T,N,2]
      pred_visibility:  [1,T,N,1]  (depending on model)

    For online mode the video is processed in temporal chunks of `online_chunk_size`
    frames so that only one chunk is on the GPU at a time, avoiding OOM on long sequences.
    """
    model = _get_cotracker(device, offline=offline, weights_path=weights_path)

    # mixed-precision inference on CUDA halves feature-map memory
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if str(device).startswith("cuda")
        else torch.amp.autocast("cpu", dtype=torch.bfloat16)
    )

    # set resolution (if supported)
    try:
        model.model.model_resolution = video.shape[-2:]
    except Exception:
        pass

    if not offline:
        # Online mode: keep video on CPU, process one chunk at a time to avoid OOM.
        video_cpu = video.contiguous()                          # stays on CPU
        queries_gpu = queries.to(device).contiguous()
        T = video_cpu.shape[0]

        if T <= online_chunk_size:
            # Short enough to fit in one shot
            video_b = video_cpu.unsqueeze(0).to(device)
            call_kw: dict = dict(queries=queries_gpu, grid_size=grid_size, grid_query_frame=0)
            with amp_ctx:
                pred_tracks, pred_visibility = model(video_b, **call_kw)
        else:
            pred_tracks, pred_visibility = _online_chunked(
                model, video_cpu, queries_gpu, grid_size, device, online_chunk_size, amp_ctx
            )
            pred_tracks    = pred_tracks.to(device)
            pred_visibility = pred_visibility.to(device)

        return pred_tracks.float(), pred_visibility.float()

    # --- offline mode: full video to GPU (unchanged) ---
    video = video.to(device).contiguous()
    queries = queries.to(device).contiguous()
    video_b = video.unsqueeze(0)

    call_kwargs: dict = dict(queries=queries, grid_size=grid_size, grid_query_frame=0)
    if progress_cb is not None:
        call_kwargs["progress_cb"] = progress_cb

    try:
        with amp_ctx:
            pred_tracks, pred_visibility = model(video_b, **call_kwargs)
    except TypeError:
        call_kwargs.pop("progress_cb", None)
        with amp_ctx:
            pred_tracks, pred_visibility = model(video_b, **call_kwargs)

    return pred_tracks.float(), pred_visibility.float()


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

    phi = phi[:, : min(h, H), : min(w, W)]
    h2, w2 = phi.shape[1], phi.shape[2]

    pad_h = max(0, H - h2)
    pad_w = max(0, W - w2)
    if pad_h > 0 or pad_w > 0:
        phi = np.pad(phi, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

    return phi


def gaussian_smooth(u, sigma=2.0):
    """Apply Gaussian smoothing to a 2D array."""
    return cv2.GaussianBlur(u.astype(np.float32, copy=False), (0, 0), float(sigma))


def tps_solve(src_xy: torch.Tensor, disp_g2: torch.Tensor, lam: float = 1e-3):
    """
    Solve TPS weights for scattered control points.

    src_xy:   (G, 2)  control point locations in any coordinate system
    disp_g2:  (G, 2)  displacement at each control point
    lam:      regularisation (ridge on kernel block); 0 = exact interpolation

    Returns
      w:  (G, 2)   kernel weights
      a:  (3, 2)   affine weights  [bias, x, y]
    """
    G = src_xy.shape[0]
    device, dtype = src_xy.device, src_xy.dtype

    # --- kernel matrix K (G×G) ---
    # U(r) = r² log r²
    diff = src_xy.unsqueeze(0) - src_xy.unsqueeze(1)   # (G, G, 2)
    r2   = (diff ** 2).sum(-1).clamp_min(0.0)           # (G, G)
    K    = r2 * torch.log(r2 + 1e-10)                   # (G, G)

    if lam > 0:
        K = K + lam * torch.eye(G, device=device, dtype=dtype)

    # --- build (G+3) × (G+3) system ---
    ones = torch.ones(G, 1, device=device, dtype=dtype)
    P    = torch.cat([ones, src_xy], dim=1)              # (G, 3)

    top    = torch.cat([K,                  P           ], dim=1)  # (G,   G+3)
    bottom = torch.cat([P.T, torch.zeros(3, 3, device=device, dtype=dtype)], dim=1)  # (3, G+3)
    L      = torch.cat([top, bottom], dim=0)             # (G+3, G+3)

    rhs = torch.cat([disp_g2,
                     torch.zeros(3, 2, device=device, dtype=dtype)], dim=0)  # (G+3, 2)

    params = torch.linalg.solve(L, rhs)                 # (G+3, 2)
    return params[:G], params[G:]                        # w (G,2), a (3,2)


def tps_dense_flow(
    pts_xy:  torch.Tensor,   # (G, 2) pixels (x, y)
    disp_g2: torch.Tensor,   # (G, 2)
    H: int,
    W: int,
    lam:   float = 1e-3,
    chunk: int  = 65536,
):
    """
    Dense TPS interpolation of a sparse displacement field.
    Returns (2, H, W) flow in pixels.
    """
    if pts_xy.ndim != 2 or pts_xy.shape[-1] != 2:
        raise ValueError(f"pts_xy must be (G,2), got {tuple(pts_xy.shape)}")
    if disp_g2.ndim != 2 or disp_g2.shape[-1] != 2:
        raise ValueError(f"disp_g2 must be (G,2), got {tuple(disp_g2.shape)}")
    G = pts_xy.shape[0]
    if G == 0:
        raise ValueError("No sparse points (G==0). Cannot interpolate flow.")
    if disp_g2.shape[0] != G:
        raise ValueError(f"pts_xy has G={G} but disp_g2 has {disp_g2.shape[0]}")

    device, dtype = pts_xy.device, pts_xy.dtype

    if G < 3:
        # <3 points -> TPS solve is singular; fall back to uniform mean-displacement flow.
        mean_disp = disp_g2.mean(dim=0)  # (2,)
        flow_hw2 = mean_disp.view(1, 1, 2).expand(H, W, 2).contiguous()
        return flow_hw2.permute(2, 0, 1).contiguous()

    # normalise control points to [-1, 1] for numerical stability
    scale   = torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
    src_n   = (pts_xy / scale) * 2.0 - 1.0              # (G, 2)
    disp_n  = disp_g2 / scale * 2.0                     # (G, 2)  same scaling

    w, a = tps_solve(src_n, disp_n, lam=lam)            # (G,2), (3,2)

    # build dense query grid in normalised coords
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    q = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N, 2)

    out = torch.empty((q.shape[0], 2), device=device, dtype=dtype)

    for s in range(0, q.shape[0], chunk):
        qe = q[s : s + chunk]                           # (M, 2)

        # kernel evaluation  U(r) = r² log r²
        diff = qe.unsqueeze(1) - src_n.unsqueeze(0)     # (M, G, 2)
        r2   = (diff ** 2).sum(-1).clamp_min(0.0)       # (M, G)
        K    = r2 * torch.log(r2 + 1e-10)               # (M, G)

        # affine basis
        ones = torch.ones(qe.shape[0], 1, device=device, dtype=dtype)
        P    = torch.cat([ones, qe], dim=1)              # (M, 3)

        out[s : s + chunk] = K @ w + P @ a              # (M, 2)

    # de-normalise back to pixel displacements
    out = out * (scale / 2.0)

    flow_hw2 = out.view(H, W, 2)
    return flow_hw2.permute(2, 0, 1).contiguous()       # (2, H, W)



def warp_like_numpy(
    src_hw: torch.Tensor, flow_2hw: torch.Tensor, mode: str = "bilinear"
):
    H, W = src_hw.shape
    device, dtype = src_hw.device, src_hw.dtype

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    x = xx + flow_2hw[0]
    y = yy + flow_2hw[1]

    x_norm = 2.0 * (x / (W - 1)) - 1.0
    y_norm = 2.0 * (y / (H - 1)) - 1.0
    grid   = torch.stack([x_norm, y_norm], dim=-1)[None]

    src = src_hw[None, None]
    out = F.grid_sample(src, grid, mode=mode,
                        padding_mode="zeros", align_corners=True)
    return out[0, 0]


def gaussian_blur_2hw(flow_2hw: torch.Tensor, sigma: float):
    if sigma is None or sigma <= 0:
        return flow_2hw
    radius = int(max(1, round(3 * sigma)))
    k = 2 * radius + 1
    device, dtype = flow_2hw.device, flow_2hw.dtype

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()

    flow = flow_2hw[None]
    gh = g.view(1, 1, 1, k).repeat(2, 1, 1, 1)
    flow = F.pad(flow, (radius, radius, 0, 0), mode="replicate")
    flow = F.conv2d(flow, gh, groups=2)
    gv = g.view(1, 1, k, 1).repeat(2, 1, 1, 1)
    flow = F.pad(flow, (0, 0, radius, radius), mode="replicate")
    flow = F.conv2d(flow, gv, groups=2)
    return flow[0]


@torch.inference_mode()
def warping(
    predicted_tracks,
    frames,
    smoothing: float = 5.0,
    lam:       float = 1e-3,    # TPS regularisation  (replaces k + power)
    mode:      str   = "bilinear",
    callback         = None,
):
    """
    predicted_tracks: (B, T, G, 2) pixel coords (x, y)
    frames:           (T, C, H, W) or (T, H, W, C)
    lam:              TPS regularisation; 0 = exact interpolation,
                      larger = smoother field (analogous to IDW smoothing)
    Returns
      warped: (T, H, W)
      flows:  (T, H, W, 2)
      drift:  (T, 2) mean (dx, dy) of the flow field per frame, averaged over
              the central region with a 10% border margin excluded (the TPS
              field is least reliable outside the tracked points' convex hull,
              which sits near the image edge). Frame 0 is (0, 0).
    """
    tracks   = predicted_tracks.float()
    frames_t = frames.float()

    if frames_t.ndim == 4 and frames_t.shape[1] in (1, 3):
        T_orig, C, H, W = frames_t.shape
        get_gray = lambda i: frames_t[i, 0]
    elif frames_t.ndim == 4 and frames_t.shape[-1] in (1, 3):
        T_orig, H, W, C = frames_t.shape
        frames_t = frames_t.permute(0, 3, 1, 2).contiguous()
        get_gray = lambda i: frames_t[i, 0]
    else:
        raise ValueError(f"Unsupported frames shape: {frames_t.shape}")

    B, T, G, D = tracks.shape
    if D != 2:
        raise ValueError(f"tracks last dim must be 2, got {D}")
    if T_orig > T:
        raise ValueError(f"frames has {T_orig} frames but tracks has {T}")

    device = tracks.device

    gray_flat   = frames_t[:, 0].reshape(-1)
    finite_mask = torch.isfinite(gray_flat)
    if finite_mask.any():
        in_min = gray_flat[finite_mask].min()
        in_max = gray_flat[finite_mask].max()
        do_clip = in_max >= in_min
    else:
        do_clip = False

    flows  = []
    warped = []
    drift  = []

    pts0 = tracks[0, 0]               # (G, 2)  reference positions
    disp = tracks[0] - tracks[0, 0]   # (T, G, 2)  relative to frame 0

    frame0 = get_gray(0)
    if do_clip:
        frame0 = frame0.clamp(in_min, in_max)
    warped.append(frame0)
    flows.append(torch.zeros((H, W, 2), device=device, dtype=tracks.dtype))
    drift.append(torch.zeros(2, device=device, dtype=tracks.dtype))

    # Precompute pixel-coordinate grids for OOB detection (reused each frame)
    _yy, _xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=tracks.dtype),
        torch.arange(W, device=device, dtype=tracks.dtype),
        indexing="ij",
    )

    # Interior region for the drift summary: excludes a 5% border margin,
    # matching the crop used elsewhere for registration-quality metrics.
    bh = max(1, int(0.05 * H))
    bw = max(1, int(0.05 * W))

    if callback is not None:
        callback(0.0)

    for i in range(1, T_orig):
        flow_2hw = tps_dense_flow(pts0, disp[i], H, W, lam=lam)
        flow_2hw = gaussian_blur_2hw(flow_2hw, float(smoothing) if smoothing else 0.0)

        # Pixels whose source coordinates fall outside the frame boundary
        src_x = _xx + flow_2hw[0]
        src_y = _yy + flow_2hw[1]
        oob = (src_x < 0) | (src_x > W - 1) | (src_y < 0) | (src_y > H - 1)

        out = warp_like_numpy(get_gray(i), flow_2hw, mode=mode)
        if do_clip:
            out = out.clamp(in_min, in_max)
        out[oob] = 0  # fill undefined regions with zeros

        warped.append(out)
        flows.append(flow_2hw.permute(1, 2, 0).contiguous())

        interior = flow_2hw[:, bh:-bh, bw:-bw]
        drift.append(interior.reshape(2, -1).mean(dim=1))

        if callback is not None:
            callback(float(i) / float(max(1, T_orig - 1)))

    return torch.stack(warped, 0), torch.stack(flows, 0), torch.stack(drift, 0)

