import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path
import tifffile
import h5py
from pathlib import Path
import hdf5plugin
import imageio.v3 as io


def resize_video_tensor_to_wh( video_t: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    Resize (T,C,H,W) tensor to exact target_w/target_h.
    Arguments order is (target_w, target_h).
    """
    if video_t is None:
        return video_t

    if not isinstance(video_t, torch.Tensor):
        video_t = torch.as_tensor(video_t)

    # normalize to (T,C,H,W) if someone passes (T,H,W)
    if video_t.ndim == 3:
        video_t = video_t.unsqueeze(1)

    T, C, H, W = video_t.shape
    target_w = int(target_w)
    target_h = int(target_h)

    if target_w <= 0 or target_h <= 0:
        return video_t
    if target_w == W and target_h == H:
        return video_t

    orig_dtype = video_t.dtype
    vt = video_t.float()

    # interpolate expects size=(H, W)
    vt = F.interpolate(vt, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # keep integer types if you want
    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        vt = vt.clamp(0, 255).to(orig_dtype)

    return vt


def _ensure_3ch_uint8(video_np):
    """
    video_np: (T,H,W) or (T,H,W,3) or (H,W) etc.
    Returns: (T,H,W,3) uint8
    """
    v = np.asarray(video_np)
    if v.ndim == 2:  # (H,W)
        v = v[None, ...]  # (1,H,W)

    if v.ndim == 3:  # (T,H,W)
        # grayscale -> 3ch
        v3 = np.stack([v, v, v], axis=-1)  # (T,H,W,3)
        return v3.astype(np.uint8)

    if v.ndim == 4 and v.shape[-1] == 3:
        return v.astype(np.uint8)

    if v.ndim == 4 and v.shape[-1] == 4:
        return v[..., :3].astype(np.uint8)

    raise ValueError(f"Unsupported video shape for 3ch: {v.shape}")


def _maybe_resize_video_THWC(video_thwc, resize=True, size=(256, 256)):
    """
    video_thwc: (T,H,W,C)
    size: (W,H) for cv2
    """
    if not resize:
        return video_thwc
    out = []
    for fr in video_thwc:
        out.append(cv2.resize(fr, size, interpolation=cv2.INTER_AREA))
    return np.ascontiguousarray(np.stack(out, axis=0))


def _to_torch_TCHW(video_thwc_uint8):
    """(T,H,W,3) uint8 -> torch float (T,3,H,W)"""
    video_t = torch.from_numpy(video_thwc_uint8).permute(0, 3, 1, 2).contiguous().float()
    return video_t

def to_uint8_contrast(video16, p_low=1, p_high=99.8):
    v = video16.astype(np.float32)
    lo, hi = np.percentile(v, (p_low, p_high))
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)

def load_hdf5(h5_path, dataset_path, frames: int = None, resize=True, size=(256, 256)):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"Dataset '{dataset_path}' not found in file.")
        ds = h5[dataset_path]
        if frames is not None:
            video = np.array(ds[:frames])
        else:
            video = np.array(ds[:])

    # video should be (T,H,W) typically
    if video.ndim == 2:
        video = video[None, ...]

    # if uint16 etc -> contrast to uint8
    if video.dtype != np.uint8:
        video = to_uint8_contrast(video)

    v_thwc = _ensure_3ch_uint8(video)
    v_thwc = _maybe_resize_video_THWC(v_thwc, resize=resize, size=size)
    return _to_torch_TCHW(v_thwc)

def load_tiff(path, frames: int = None, resize: bool = True, size=(256, 256)):
    mm = tifffile.memmap(path)

    if mm.ndim == 3:
        T = min(int(frames), mm.shape[0]) if frames is not None else mm.shape[0]
        video = np.ascontiguousarray(mm[:T])
    elif mm.ndim == 2:
        video = np.ascontiguousarray(mm[None, ...])
    else:
        raise ValueError(f"Unsupported TIFF ndim={mm.ndim} with shape={mm.shape}")

    if not video.dtype.isnative:
        video = video.astype(video.dtype.type)

    # contrast to uint8 (T,H,W)
    v_u8 = to_uint8_contrast(video)

    # -> (T,H,W,3)
    v_thwc = _ensure_3ch_uint8(v_u8)

    # optional resize
    v_thwc = _maybe_resize_video_THWC(v_thwc, resize=resize, size=size)

    # Returns already the correct shape...
    return _to_torch_TCHW(v_thwc)

def load_video(file: str, resize: bool = True, size=(256, 256), max_frames: int = None):
    """
    Load a video from the given file path and return a torch tensor (T, C, H, W).

    Args:
        file: Path to the video file.
        resize: If True, resize frames to `size`.
        size: (W, H) target size for cv2.resize.
        max_frames: Limit number of frames loaded/returned.

    Returns:
        video: torch.FloatTensor of shape (T, C, H, W)
    """
    video = read_video_from_path(file)  # expected iterable/list/array of frames (H,W,C) or (H,W)
    if video is None or len(video) == 0:
        raise ValueError(f"No frames read from: {file}")

    # optionally cap early to avoid unnecessary work
    if max_frames is not None:
        video = video[:max_frames]

    frames = []
    for frame in video:
        f = np.asarray(frame)

        # ensure 3 channels (H,W,3)
        if f.ndim == 2:  # grayscale
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        elif f.ndim == 3 and f.shape[2] == 4:  # RGBA -> RGB
            f = f[..., :3]

        if resize:
            f = cv2.resize(f, size)  # size is (W,H) in OpenCV

        frames.append(f)

    video_np = np.ascontiguousarray(np.stack(frames, axis=0))  # (T,H,W,C)
    video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()  # (T,C,H,W)

    return video_t

def save_and_display_video(array, filename='output.mp4', fps=30):
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min = array.min()
        array_max = array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = array[i]
        # Check if frame is already 3-channel (RGB) or needs conversion from grayscale
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3:
            if frame.shape[2] == 3:
                # RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Fallback or other channel counts
                frame_bgr = frame
        else:
            frame_bgr = frame
            
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {filename}")


def save_video_tiff(video, filename):
    """
    Save video as TIFF stack.
    video: (T, H, W, 3) or (T, C, H, W)
    """
    v = np.asarray(video)
    
    # Handle tensor
    if isinstance(video, torch.Tensor):
        v = video.detach().cpu().numpy()
        
    # Standardize to (T, H, W, C)
    if v.ndim == 4 and v.shape[1] in [1, 3]:  # (T, C, H, W)
        v = np.transpose(v, (0, 2, 3, 1))
        
    # Handle single channel
    if v.ndim == 4 and v.shape[-1] == 1:
        v = v.squeeze(-1)
        
    # Ensure uint8
    if v.dtype != np.uint8:
        if v.max() <= 1.0:
            v = (v * 255).astype(np.uint8)
        else:
            v = v.astype(np.uint8)
            
    try:
        tifffile.imwrite(filename, v, photometric='rgb' if v.ndim==4 and v.shape[-1]==3 else 'minisblack')
        print(f"TIFF saved to {filename}")
    except Exception as e:
        print(f"Error saving TIFF: {e}")
        raise e


def save_video_hdf5(video, filename):
    """
    Save video as HDF5 file.
    video: (T, H, W, 3) or (T, C, H, W)
    """
    v = np.asarray(video)
    
    # Handle tensor
    if isinstance(video, torch.Tensor):
        v = video.detach().cpu().numpy()
        
    # Standardize to (T, H, W, C)
    if v.ndim == 4 and v.shape[1] in [1, 3]:  # (T, C, H, W)
        v = np.transpose(v, (0, 2, 3, 1))
        
    # Ensure uint8
    if v.dtype != np.uint8:
        if v.max() <= 1.0:
            v = (v * 255).astype(np.uint8)
        else:
            v = v.astype(np.uint8)
            
    try:
        with h5py.File(filename, 'w') as f:
            # Create dataset with compression
            f.create_dataset('data', data=v, compression='gzip', compression_opts=9)
        print(f"HDF5 saved to {filename}")
    except Exception as e:
        print(f"Error saving HDF5: {e}")
        raise e

def _to_gray_u8_frame(frame):
    """
    frame: torch or np, can be (C,H,W), (H,W,C), (H,W).
    Returns uint8 (H,W) with robust scaling using percentiles of that frame.
    """
    a = frame
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)

    if a.ndim == 3 and a.shape[0] in (1,3):  # (C,H,W)
        a = a[0]
    elif a.ndim == 3 and a.shape[-1] in (3,4):  # (H,W,C)
        a = a[..., 0]
    elif a.ndim != 2:
        raise ValueError(f"Unsupported frame shape for gray: {a.shape}")

    a = a.astype(np.float32)
    finite = np.isfinite(a)
    if finite.any():
        lo = float(np.percentile(a[finite], 1))
        hi = float(np.percentile(a[finite], 99))
        if hi <= lo:
            hi = lo + 1.0
        a = (a - lo) / (hi - lo) * 255.0
    else:
        a[:] = 0.0
    return np.clip(a, 0, 255).astype(np.uint8)


# def ssim2d_u8(img1_u8, img2_u8, sigma=1.5):
#     x = img1_u8.astype(np.float32)
#     y = img2_u8.astype(np.float32)

#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     mu_x = cv2.GaussianBlur(x, (0, 0), sigma)
#     mu_y = cv2.GaussianBlur(y, (0, 0), sigma)

#     mu_x2 = mu_x * mu_x
#     mu_y2 = mu_y * mu_y
#     mu_xy = mu_x * mu_y

#     sig_x2 = cv2.GaussianBlur(x * x, (0, 0), sigma) - mu_x2
#     sig_y2 = cv2.GaussianBlur(y * y, (0, 0), sigma) - mu_y2
#     sig_xy = cv2.GaussianBlur(x * y, (0, 0), sigma) - mu_xy

#     num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
#     den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
#     ssim_map = num / (den + 1e-12)
#     return float(np.mean(ssim_map)), ssim_map


def video_ssim_score(video, mode="ref_first"):
    v = video
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)

    # Convert to grayscale float32 frames WITHOUT per-frame normalization
    if v.ndim == 4 and v.shape[1] in (1, 3):       # (T,C,H,W)
        gray = v[:, 0].astype(np.float32) if v.shape[1] == 1 else v[:, 0].astype(np.float32)  # or proper RGB->gray if needed
    elif v.ndim == 4 and v.shape[-1] in (3, 4):    # (T,H,W,C)
        gray = v[..., 0].astype(np.float32)        # or proper RGB->gray
    elif v.ndim == 3:                               # (T,H,W)
        gray = v.astype(np.float32)
    else:
        raise ValueError(f"Unsupported video shape for SSIM: {v.shape}")

    T = gray.shape[0]
    if T < 2:
        return 0.0

    # Global scaling (same for all frames)
    finite = np.isfinite(gray)
    lo = float(np.percentile(gray[finite], 1)) if finite.any() else 0.0
    hi = float(np.percentile(gray[finite], 99)) if finite.any() else 1.0
    if hi <= lo:
        hi = lo + 1.0

    frames_u8 = np.clip((gray - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if mode == "ref_first":
        ref = frames_u8[0]
        scores = [(frames_u8[t]- ref) for t in range(1, T)]
        return float(np.mean(scores)) if scores else 0.0

    # consecutive
    scores = [(frames_u8[t] - frames_u8[t - 1]) for t in range(1, T)]
    return float(np.mean(scores)) if scores else 0.0


def probe_media_info(path: str):
    """
    Try to get (frames, width, height) without loading full data.
    Works for: mp4/avi via cv2, tiff via tifffile (if available), h5 via h5py (if available).
    Returns (frames, w, h) possibly None for unknown fields.
    """
    ext = os.path.splitext(path)[1].lower()

    # video container
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

    # tiff stack
    if ext in [".tif", ".tiff"]:
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                pages = len(tf.pages)
                # first page shape (H,W) or (H,W,C)
                shp = tf.pages[0].shape
                if len(shp) >= 2:
                    h, w = int(shp[0]), int(shp[1])
                else:
                    h, w = None, None
            return int(pages) if pages else None, w, h
        except Exception:
            return None, None, None

    # hdf5
    if ext in [".h5", ".hdf5"]:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                dset = None
                # prefer /data if present
                if "/data" in f:
                    dset = f["/data"]
                else:
                    # pick first dataset found
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
                # common layouts:
                # (T,H,W) or (T,C,H,W)
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