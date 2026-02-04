import os
from pathlib import Path
import cv2
import h5py
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path


# -----------------------------
# Resizing / conversion helpers
# -----------------------------
def resize_video_tensor_to_wh(video_t: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    Resize a video tensor to exact (target_w, target_h).

    Args:
        video_t: torch Tensor shaped (T,C,H,W) or (T,H,W)
        target_w: target width
        target_h: target height

    Returns:
        torch Tensor shaped (T,C,target_h,target_w) with dtype preserved when reasonable.
    """
    if video_t is None:
        return video_t

    if not isinstance(video_t, torch.Tensor):
        video_t = torch.as_tensor(video_t)

    if video_t.ndim == 3:  # (T,H,W) -> (T,1,H,W)
        video_t = video_t.unsqueeze(1)

    if video_t.ndim != 4:
        raise ValueError(f"Expected (T,C,H,W) or (T,H,W), got shape {tuple(video_t.shape)}")

    T, C, H, W = video_t.shape
    target_w = int(target_w)
    target_h = int(target_h)

    if target_w <= 0 or target_h <= 0:
        return video_t
    if target_w == W and target_h == H:
        return video_t

    orig_dtype = video_t.dtype
    vt = video_t.float()
    vt = F.interpolate(vt, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # preserve integer-ish types if input was integer-like
    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        vt = vt.clamp(0, 255).to(orig_dtype)
    else:
        vt = vt.to(orig_dtype)

    return vt


def _ensure_thwc_rgb_uint8(video_np: np.ndarray) -> np.ndarray:
    """
    Ensure video is (T,H,W,3) uint8.

    Accepts:
      - (H,W)
      - (T,H,W)
      - (H,W,3)
      - (T,H,W,3)
      - (T,H,W,4) (drops alpha)

    Returns:
      (T,H,W,3) uint8
    """
    v = np.asarray(video_np)

    if v.ndim == 2:  # (H,W)
        v = v[None, ..., None]  # (1,H,W,1)

    if v.ndim == 3:
        # could be (T,H,W) or (H,W,C)
        if v.shape[-1] in (3, 4):  # (H,W,C)
            v = v[None, ...]  # (1,H,W,C)
        else:
            v = v[..., None]  # (T,H,W,1)

    if v.ndim != 4:
        raise ValueError(f"Unsupported video shape: {v.shape}")

    # Drop alpha
    if v.shape[-1] == 4:
        v = v[..., :3]

    # Gray -> RGB
    if v.shape[-1] == 1:
        v = np.repeat(v, 3, axis=-1)

    if v.shape[-1] != 3:
        raise ValueError(f"Unsupported channel count: {v.shape}")

    return v.astype(np.uint8, copy=False)


def _resize_video_thwc(video_thwc: np.ndarray, resize: bool, target_size_wh=(256, 256)) -> np.ndarray:
    """
    Resize THWC video using OpenCV.

    Args:
        video_thwc: (T,H,W,C)
        resize: bool
        target_size_wh: (W,H)

    Returns:
        (T,H,W,C)
    """
    if not resize:
        return np.ascontiguousarray(video_thwc)

    target_w, target_h = int(target_size_wh[0]), int(target_size_wh[1])
    out = []
    for fr in video_thwc:
        out.append(cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_AREA))
    return np.ascontiguousarray(np.stack(out, axis=0))


def _to_torch_tchw(video_thwc_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert (T,H,W,3) uint8 -> torch float (T,3,H,W)
    """
    return torch.from_numpy(video_thwc_uint8).permute(0, 3, 1, 2).contiguous().float()


def to_uint8_contrast(video_any, p_low=1.0, p_high=99.8) -> np.ndarray:
    """
    Percentile contrast stretch to uint8.
    Works for (T,H,W) or (H,W).
    """
    v = np.asarray(video_any).astype(np.float32)
    lo, hi = np.percentile(v, (p_low, p_high))
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)


# -----------------------------
# Loading
# -----------------------------
def load_hdf5(h5_path: str, dataset_path: str, frames: int | None = None, resize=True, size_wh=(256, 256)) -> torch.Tensor:
    """
    Load HDF5 dataset into torch (T,3,H,W) float.

    Expects dataset to be (T,H,W) or (T,H,W,C). Non-uint8 is contrast-stretched.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"Dataset '{dataset_path}' not found in file.")
        ds = h5[dataset_path]
        video = np.array(ds[:frames] if frames is not None else ds[:])

    if video.ndim == 2:  # (H,W)
        video = video[None, ...]

    # If not uint8, stretch to uint8
    if video.dtype != np.uint8:
        video = to_uint8_contrast(video)

    v_thwc = _ensure_thwc_rgb_uint8(video)          # (T,H,W,3) uint8
    v_thwc = _resize_video_thwc(v_thwc, resize, size_wh)
    return _to_torch_tchw(v_thwc)


def load_tiff(path: str, frames: int | None = None, resize=True, size_wh=(256, 256)) -> torch.Tensor:
    """
    Load TIFF stack into torch (T,3,H,W) float.
    """
    mm = tifffile.memmap(path)

    if mm.ndim == 3:
        T = int(mm.shape[0]) if frames is None else min(int(frames), int(mm.shape[0]))
        video = np.ascontiguousarray(mm[:T])
    elif mm.ndim == 2:
        video = np.ascontiguousarray(mm[None, ...])
    else:
        raise ValueError(f"Unsupported TIFF ndim={mm.ndim} shape={mm.shape}")

    if not video.dtype.isnative:
        video = video.astype(video.dtype.type)

    v_u8 = to_uint8_contrast(video)                 # (T,H,W) uint8
    v_thwc = _ensure_thwc_rgb_uint8(v_u8)           # (T,H,W,3)
    v_thwc = _resize_video_thwc(v_thwc, resize, size_wh)
    return _to_torch_tchw(v_thwc)


def load_video(file: str, resize=True, size_wh=(256, 256), max_frames: int | None = None) -> torch.Tensor:
    """
    Load a video file into torch (T,3,H,W) float.

    read_video_from_path typically returns frames as (H,W,C) or (H,W).
    """
    video = read_video_from_path(file)
    if video is None or len(video) == 0:
        raise ValueError(f"No frames read from: {file}")

    if max_frames is not None:
        video = video[: int(max_frames)]

    frames = []
    for frame in video:
        f = np.asarray(frame)
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        elif f.ndim == 3 and f.shape[2] == 4:
            f = f[..., :3]

        if resize:
            target_w, target_h = int(size_wh[0]), int(size_wh[1])
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)

        frames.append(f)

    v_thwc = np.ascontiguousarray(np.stack(frames, axis=0)).astype(np.uint8, copy=False)  # (T,H,W,3)
    return _to_torch_tchw(v_thwc)


# -----------------------------
# Saving
# -----------------------------
def save_and_display_video(array: np.ndarray, filename="output.mp4", fps=30) -> None:
    """
    Save grayscale or RGB video to MP4.

    Accepts:
      - (T,H,W) grayscale
      - (T,H,W,3) RGB
    """
    v = np.asarray(array)

    if v.ndim == 3:
        # (T,H,W) -> (T,H,W,3)
        if v.dtype != np.uint8:
            vmin, vmax = float(np.min(v)), float(np.max(v))
            v = 255.0 * (v - vmin) / (vmax - vmin + 1e-8)
            v = np.clip(v, 0, 255).astype(np.uint8)
        v = np.repeat(v[..., None], 3, axis=-1)

    elif v.ndim == 4 and v.shape[-1] == 3:
        if v.dtype != np.uint8:
            v = np.clip(v, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported array shape for save_and_display_video: {v.shape}")

    T, H, W, _ = v.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)

    for i in range(T):
        # v is RGB -> convert to BGR for OpenCV
        out.write(cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved to {filename}")


def save_video_tiff(video, filename: str) -> None:
    """
    Save video as TIFF stack.
    Accepts:
      - (T,H,W,3) or (T,H,W)
      - (T,C,H,W) torch/numpy
    """
    v = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)

    if v.ndim == 4 and v.shape[1] in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
        v = np.transpose(v, (0, 2, 3, 1))

    if v.ndim == 4 and v.shape[-1] == 1:
        v = v[..., 0]  # (T,H,W)

    if v.dtype != np.uint8:
        if np.max(v) <= 1.0:
            v = (v * 255.0).astype(np.uint8)
        else:
            v = np.clip(v, 0, 255).astype(np.uint8)

    photometric = "rgb" if (v.ndim == 4 and v.shape[-1] == 3) else "minisblack"
    tifffile.imwrite(filename, v, photometric=photometric)
    print(f"TIFF saved to {filename}")


def save_video_hdf5(video, filename: str) -> None:
    """
    Save video as HDF5 dataset '/data' with gzip compression.
    Accepts:
      - (T,H,W,3) or (T,H,W)
      - (T,C,H,W) torch/numpy
    """
    v = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)

    if v.ndim == 4 and v.shape[1] in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
        v = np.transpose(v, (0, 2, 3, 1))

    if v.dtype != np.uint8:
        if np.max(v) <= 1.0:
            v = (v * 255.0).astype(np.uint8)
        else:
            v = np.clip(v, 0, 255).astype(np.uint8)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=v, compression="gzip", compression_opts=9)
    print(f"HDF5 saved to {filename}")


# -----------------------------
# SSIM
# -----------------------------
def _to_gray_float(video) -> np.ndarray:
    """
    Convert video to grayscale float32 array shaped (T,H,W).
    Accepts (T,C,H,W), (T,H,W,C), (T,H,W).
    Uses first channel if multi-channel.
    """
    v = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)

    if v.ndim == 4 and v.shape[1] in (1, 3):          # (T,C,H,W)
        gray = v[:, 0].astype(np.float32)
    elif v.ndim == 4 and v.shape[-1] in (3, 4):       # (T,H,W,C)
        gray = v[..., 0].astype(np.float32)
    elif v.ndim == 3:                                 # (T,H,W)
        gray = v.astype(np.float32)
    else:
        raise ValueError(f"Unsupported video shape for gray conversion: {v.shape}")

    return gray


def video_ssim_score(video, mode="ref_first") -> float:
    """
    Compute mean SSIM over a video.

    mode:
      - "ref_first": SSIM(frame[t], frame[0]) averaged over t=1..T-1
      - "consecutive": SSIM(frame[t], frame[t-1]) averaged over t=1..T-1

    Returns:
        float in [0,1] (typically), higher is more similar.
    """
    gray = _to_gray_float(video)
    T = int(gray.shape[0])
    if T < 2:
        return 1.0

    # global scaling for stable SSIM comparisons
    finite = np.isfinite(gray)
    lo = float(np.percentile(gray[finite], 1)) if finite.any() else 0.0
    hi = float(np.percentile(gray[finite], 99)) if finite.any() else 1.0
    if hi <= lo:
        hi = lo + 1.0

    frames_u8 = np.clip((gray - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        # fallback: 1 - normalized L1 difference (crude but stable)
        if mode == "ref_first":
            ref = frames_u8[0].astype(np.float32)
            diffs = [np.mean(np.abs(frames_u8[t].astype(np.float32) - ref)) for t in range(1, T)]
        else:
            diffs = [np.mean(np.abs(frames_u8[t].astype(np.float32) - frames_u8[t - 1].astype(np.float32))) for t in range(1, T)]
        d = float(np.mean(diffs)) if diffs else 0.0
        return float(np.clip(1.0 - d / 255.0, 0.0, 1.0))

    scores = []
    if mode == "ref_first":
        ref = frames_u8[0]
        for t in range(1, T):
            scores.append(ssim(ref, frames_u8[t], data_range=255))
    else:
        for t in range(1, T):
            scores.append(ssim(frames_u8[t - 1], frames_u8[t], data_range=255))

    return float(np.mean(scores)) if scores else 1.0


# -----------------------------
# Media probe
# -----------------------------
def probe_media_info(path: str):
    """
    Try to get (frames, width, height) without loading full data.
    Supports:
      - mp4/avi/mov/mkv via cv2
      - tiff via tifffile
      - h5/hdf5 via h5py (prefers /data)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".avi", ".mp4", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None, None
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        frames = int(frames) if frames and frames > 0 else None
        w = int(w) if w and w > 0 else None
        h = int(h) if h and h > 0 else None
        return frames, w, h

    if ext in [".tif", ".tiff"]:
        try:
            with tifffile.TiffFile(path) as tf:
                pages = len(tf.pages)
                shp = tf.pages[0].shape
                h = int(shp[0]) if len(shp) >= 2 else None
                w = int(shp[1]) if len(shp) >= 2 else None
            return int(pages) if pages else None, w, h
        except Exception:
            return None, None, None

    if ext in [".h5", ".hdf5"]:
        try:
            with h5py.File(path, "r") as f:
                if "/data" in f:
                    dset = f["/data"]
                else:
                    # find first dataset
                    def _first_dataset(g):
                        for k in g.keys():
                            obj = g[k]
                            if isinstance(obj, h5py.Dataset):
                                return obj
                            if isinstance(obj, h5py.Group):
                                r = _first_dataset(obj)
                                if r is not None:
                                    return r
                        return None

                    dset = _first_dataset(f)

                if dset is None:
                    return None, None, None

                shp = dset.shape
                frames = int(shp[0]) if len(shp) >= 1 else None
                if len(shp) == 3:
                    h, w = int(shp[1]), int(shp[2])
                elif len(shp) >= 4:
                    h, w = int(shp[-2]), int(shp[-1])
                else:
                    h, w = None, None

            return frames, w, h
        except Exception:
            return None, None, None

    return None, None, None
