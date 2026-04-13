import gc
import math
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

_COTRACKER = None
_COTRACKER_DEVICE = None

def _default_torch_home() -> str:
    # Respect existing TORCH_HOME if the user already set it
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]

    # Otherwise use a cache dir in the user's home directory
    return str(Path.home() / ".cache" / "torch")

def _get_cotracker(device: str):
    global _COTRACKER, _COTRACKER_DEVICE

    if _COTRACKER is None or _COTRACKER_DEVICE != str(device):
        torch_home = _default_torch_home()
        os.environ["TORCH_HOME"] = torch_home
        Path(torch_home).mkdir(parents=True, exist_ok=True)

        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline"
        )
        model = model.to(device).eval()

        _COTRACKER = model
        _COTRACKER_DEVICE = str(device)

    return _COTRACKER


@torch.inference_mode()
def cotrack_model(video, queries, device, grid_size=0, progress_cb=None):
    """
    NON-BATCHED version:
      video:   [T,3,H,W]
      queries: [1,N,3] (t,x,y)
    returns:
      pred_tracks:      [1,T,N,2]
      pred_visibility:  [1,T,N,1]  (depending on model)
    """
    model = _get_cotracker(device)

    video = video.to(device).contiguous()
    queries = queries.to(device).contiguous()

    # model expects [1,T,3,H,W]
    video_b = video.unsqueeze(0)

    # set resolution (if supported)
    try:
        model.model.model_resolution = video.shape[-2:]
    except Exception:
        pass

    try:
        pred_tracks, pred_visibility = model(
            video_b,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=0,
            progress_cb=progress_cb,
        )
    except TypeError:
        # Fallback for models that don't accept progress_cb
        pred_tracks, pred_visibility = model(
            video_b,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=0,
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
    phi = phi[:, : min(h, H), : min(w, W)]
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


def warp_like_numpy(
    src_hw: torch.Tensor, flow_2hw: torch.Tensor, mode: str = "bilinear"
):
    """
    src_hw: (H,W)
    flow_2hw: (2,H,W) where flow[0]=dx, flow[1]=dy in *pixels*
    """
    H, W = src_hw.shape
    device, dtype = src_hw.device, src_hw.dtype

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    x = xx + flow_2hw[0]
    y = yy + flow_2hw[1]

    # normalize for grid_sample, align_corners=True matches pixel indexing
    x_norm = 2.0 * (x / (W - 1)) - 1.0
    y_norm = 2.0 * (y / (H - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)[None]  # (1,H,W,2)

    src = src_hw[None, None]  # (1,1,H,W)
    out = F.grid_sample(
        src,
        grid,
        mode=mode,  # 'bilinear' or 'bicubic'
        padding_mode="border",  # closest to SciPy "nearest" border extension
        align_corners=True,
    )
    return out[0, 0]


def idw_dense_flow(
    pts_xy: torch.Tensor,  # (G,2) in pixels (x,y)
    disp_g2: torch.Tensor,  # (G,2)
    H: int,
    W: int,
    k: int = 16,
    power: float = 2.0,
    eps: float = 1e-6,
    chunk: int = 65536,
):
    """
    Dense IDW interpolation of a sparse displacement field.
    Returns (2,H,W) flow in pixels.
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

    # clamp k
    k_eff = int(min(max(1, k), G))

    device, dtype = pts_xy.device, pts_xy.dtype

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    q = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N,2)
    out = torch.empty((q.shape[0], 2), device=device, dtype=dtype)

    for s in range(0, q.shape[0], chunk):
        qe = q[s : s + chunk]  # (M,2)

        # distances (M,G)
        d = torch.cdist(qe, pts_xy).clamp_min(eps)

        if k_eff == 1:
            # nearest neighbor special-case
            idx = d.argmin(dim=1)
            out[s : s + chunk] = disp_g2[idx]
            continue

        knn_d, knn_idx = torch.topk(d, k=k_eff, dim=1, largest=False)  # (M,k)
        w = 1.0 / (knn_d**power)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)

        disp_knn = disp_g2[knn_idx]  # (M,k,2)
        out[s : s + chunk] = (disp_knn * w.unsqueeze(-1)).sum(dim=1)

    flow_hw2 = out.view(H, W, 2)
    return flow_hw2.permute(2, 0, 1).contiguous()  # (2,H,W)


def gaussian_blur_2hw(flow_2hw: torch.Tensor, sigma: float):
    if sigma is None or sigma <= 0:
        return flow_2hw
    # simple kernel size rule
    radius = int(max(1, round(3 * sigma)))
    k = 2 * radius + 1
    device, dtype = flow_2hw.device, flow_2hw.dtype

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()

    # separable conv
    flow = flow_2hw[None]  # (1,2,H,W)
    # horizontal
    gh = g.view(1, 1, 1, k).repeat(2, 1, 1, 1)  # (2,1,1,k)
    flow = F.pad(flow, (radius, radius, 0, 0), mode="replicate")
    flow = F.conv2d(flow, gh, groups=2)
    # vertical
    gv = g.view(1, 1, k, 1).repeat(2, 1, 1, 1)  # (2,1,k,1)
    flow = F.pad(flow, (0, 0, radius, radius), mode="replicate")
    flow = F.conv2d(flow, gv, groups=2)

    return flow[0]


def warping(
    predicted_tracks,
    frames,
    smoothing=5.0,
    k=16,
    power=1.5,
    mode="bilinear",
    callback=None,
):
    """
    predicted_tracks: (B,T,G,2) in pixel coords (x,y)
    frames: (T,C,H,W) or (T,H,W,C)
    Returns:
      warped: (T,H,W) grayscale
      flows:  (T,H,W,2) dense flow (dx,dy) in pixels
    """
    tracks = predicted_tracks.float()
    frames_t = frames.float()

    # normalize to (T,C,H,W)
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

    # determine input gray-channel range (like NumPy warping())
    gray_stack = frames_t[:, 0]  # (T_orig,H,W)
    gray_flat = gray_stack.reshape(-1)
    finite_mask = torch.isfinite(gray_flat)
    if finite_mask.any():
        in_min = gray_flat[finite_mask].min()
        in_max = gray_flat[finite_mask].max()
        do_clip = in_max >= in_min
    else:
        do_clip = False
    flows = []
    warped = []

    # reference sparse points (frame 0)
    pts0 = tracks[0, 0]  # (G,2)
    disp = tracks[0] - tracks[0, 0]  # (T,G,2), disp[t] relative to frame 0

    # exactly like your NumPy: frame0 unchanged, flow0 zeros
    frame0 = get_gray(0)
    if do_clip:
        frame0 = frame0.clamp(in_min, in_max)
    warped.append(frame0)
    flows.append(torch.zeros((H, W, 2), device=device, dtype=tracks.dtype))

    if callback is not None:
        callback(0.0)

    for i in range(1, T_orig):

        # # clamp once for all frames to remove outliers (optional but keeps everything consistent)
        # disp_i = disp[i]  # (G,2)
        # # ---- (2) robust clamp outlier displacements (per-frame) ----
        # mag = disp_i.norm(dim=-1)                 # (G,)
        # thr = mag.median() * 3.0                  # tune: 2.5–4.0
        # scale = (mag.clamp_max(thr) / (mag + 1e-6)).unsqueeze(-1)  # (G,1)
        # disp_i = disp_i * scale                   # (G,2)
        # # ------------------------------------------------------------

        # dense flow from sparse displacement
        flow_2hw = idw_dense_flow(pts0, disp[i], H, W, k=k, power=power)  # (2,H,W)

        # spatial smoothing (optional)
        flow_2hw = gaussian_blur_2hw(flow_2hw, float(smoothing) if smoothing else 0.0)

        # warp
        out = warp_like_numpy(get_gray(i), flow_2hw, mode=mode)

        if do_clip:
            out = out.clamp(in_min, in_max)

        warped.append(out)
        flows.append(flow_2hw.permute(1, 2, 0).contiguous())  # (H,W,2)

        if callback is not None:
            callback(float(i) / float(max(1, T_orig - 1)))

    return torch.stack(warped, 0), torch.stack(flows, 0)
