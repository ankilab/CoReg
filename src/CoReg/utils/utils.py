import os
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path


def resize_video_tensor_to_wh(
    video_t: torch.Tensor, target_w: int, target_h: int
) -> torch.Tensor:
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
        raise ValueError(
            f"Expected (T, C, H, W) or (T, H, W), got shape {tuple(video_t.shape)}"
        )

    T, C, H, W = video_t.shape
    target_w = int(target_w)
    target_h = int(target_h)

    if target_w <= 0 or target_h <= 0:
        return video_t
    if target_w == W and target_h == H:
        return video_t

    orig_dtype = video_t.dtype
    vt = video_t.float()
    vt = F.interpolate(
        vt, size=(target_h, target_w), mode="bilinear", align_corners=False
    )

    # preserve integer-ish types if input was integer-like
    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        vt = vt.clamp(0, 255).to(orig_dtype)
    else:
        vt = vt.to(orig_dtype)

    return vt


def _ensure_thwc_gray_uint8(video_np: np.ndarray) -> np.ndarray:
    """
    Ensure video is (T,H,W,1) uint8.
    If input is color (RGB/RGBA), it is converted to grayscale.

    Accepts:
      - (H,W)
      - (T,H,W)
      - (H,W,3)
      - (T,H,W,3)
      - (T,H,W,4)

    Returns:
      (T,H,W,1) uint8
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

    # Convert color to gray if present
    if v.shape[-1] in (3, 4):
        gray_frames = []
        for i in range(v.shape[0]):
            code = cv2.COLOR_RGB2GRAY if v.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY
            gray_frames.append(cv2.cvtColor(v[i], code))
        v = np.stack(gray_frames, axis=0)[..., None]  # (T,H,W,1)

    return v.astype(np.uint8, copy=False)


def _resize_video_thwc(
    video_thwc: np.ndarray, resize: bool, target_size_wh=(256, 256)
) -> np.ndarray:
    """
    Resize THWC video using OpenCV. Ensures C dimension is preserved.
    """
    if not resize:
        return np.ascontiguousarray(video_thwc)

    target_w, target_h = int(target_size_wh[0]), int(target_size_wh[1])
    T, H, W, C = video_thwc.shape
    out = []
    for fr in video_thwc:
        res = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if res.ndim == 2 and C == 1:
            res = res[..., None]
        out.append(res)
    return np.ascontiguousarray(np.stack(out, axis=0))


def _to_torch_tchw(video_thwc_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert (T, H, W, C) uint8 -> torch float (T, C, H, W).
    """
    v = np.asarray(video_thwc_uint8)
    if v.ndim == 3:
        # (T, H, W) -> (T, 1, H, W)
        return torch.from_numpy(v).unsqueeze(1).contiguous().float()

    if v.ndim == 4:
        # (T, H, W, C) -> (T, C, H, W)
        return torch.from_numpy(v).permute(0, 3, 1, 2).contiguous().float()

    raise ValueError(f"Unsupported dimensionality for torch conversion: {v.ndim}")


def to_uint8_contrast(video_any, p_low=1.0, p_high=99.8, bounds=None) -> np.ndarray:
    """
    Percentile contrast stretch to uint8.
    If 'bounds' (lo, hi) is provided, it skips computation.
    """
    if bounds is not None:
        lo, hi = bounds
    else:
        v_sample = np.asarray(video_any).astype(np.float32)
        lo, hi = np.percentile(v_sample, (p_low, p_high))

    v = np.asarray(video_any).astype(np.float32)
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)


def _calc_sampled_bounds(
    data_handle, frames_to_load, p_low=1.0, p_high=99.8, sample_n=20
):
    """
    Estimate contrast bounds by sampling a subset of frames.
    """
    total = len(data_handle)
    n = min(total, frames_to_load)
    indices = np.linspace(0, n - 1, sample_n, dtype=int)

    samples = []
    for idx in indices:
        samples.append(np.asarray(data_handle[idx]).astype(np.float32))

    all_samples = np.concatenate([s.ravel() for s in samples])
    return np.percentile(all_samples, (p_low, p_high))


def load_hdf5(
    h5_path: str,
    dataset_path: str | None = None,
    frames: int | None = None,
    resize=True,
    size_wh=(256, 256),
) -> torch.Tensor:
    """
    Load HDF5 dataset into torch (T, 1, H, W) float.

    Expects dataset to be (T, H, W) or (T, H, W, C).
    Non-uint8 data is automatically contrast-stretched.
    If dataset_path is None or not found, tries 'data' or 'video' defaults.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        ds = None
        # 1. Try provided path
        if dataset_path and dataset_path in h5:
            ds = h5[dataset_path]

        # 2. Try common defaults if not found
        if ds is None:
            for candidate in [
                "/data",
                "data",
                "/video",
                "video",
                "/dataset",
                "dataset",
            ]:
                if candidate in h5:
                    ds = h5[candidate]
                    break

        # 3. Last resort: pick the first dataset found
        if ds is None:

            def find_datasets(group):
                for key, val in group.items():
                    if isinstance(val, h5py.Dataset):
                        return val
                    if isinstance(val, h5py.Group):
                        res = find_datasets(val)
                        if res is not None:
                            return res
                return None

            ds = find_datasets(h5)

        if ds is None:
            raise KeyError(f"Could not find any suitable dataset in {h5_path}")

        # Streaming parameters
        T_total = int(ds.shape[0]) if ds.ndim >= 2 else 1
        T = int(frames) if frames is not None else T_total
        T = min(T, T_total)

        # Sampling for contrast
        bounds = None
        if ds.dtype != np.uint8:
            bounds = _calc_sampled_bounds(ds, T)

        # Output dims
        if ds.ndim == 2:
            orig_h, orig_w = ds.shape
            C = 1
        elif ds.ndim == 3:
            orig_h, orig_w = ds.shape[1], ds.shape[2]
            C = 1
        else:  # (T, H, W, C)
            orig_h, orig_w = ds.shape[1], ds.shape[2]
            C = ds.shape[3]

        out_w, out_h = size_wh if resize else (orig_w, orig_h)
        final_video = torch.zeros((T, 1, out_h, out_w), dtype=torch.float32)

        try:
            for i in range(T):
                frame = ds[i] if ds.ndim >= 3 else ds[:]

                # Convert to uint8 with bounds
                if bounds is not None:
                    frame = to_uint8_contrast(frame, bounds=bounds)
                elif frame.dtype != np.uint8:
                    frame = to_uint8_contrast(frame)

                # Resize
                if resize:
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )

                # Grayscale conversion and store
                if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                    code = (
                        cv2.COLOR_RGB2GRAY
                        if frame.shape[-1] == 3
                        else cv2.COLOR_RGBA2GRAY
                    )
                    frame = cv2.cvtColor(frame, code)

                if frame.ndim == 2:
                    final_video[i, 0] = torch.from_numpy(
                        frame.astype(np.float32) / 255.0
                    )
                else:
                    final_video[i, 0] = torch.from_numpy(
                        frame[..., 0].astype(np.float32) / 255.0
                    )

        except Exception as e:
            if "filter" in str(e).lower() or "read" in str(e).lower():
                raise RuntimeError(
                    f"Error reading HDF5 dataset. This often happens if the data is compressed "
                    f"with a format requiring 'hdf5plugin'. Please ensure 'hdf5plugin' is installed.\n"
                    f"Original error: {e}"
                ) from e
            raise e

    return final_video


def load_tiff(
    path: str, frames: int | None = None, resize=True, size_wh=(256, 256)
) -> torch.Tensor:
    """
    Load TIFF stack into torch (T, 1, H, W) float.
    Uses memmap and streaming processing for speed.
    """
    mm = tifffile.memmap(path)

    if mm.ndim == 3:
        T_total = int(mm.shape[0])
    elif mm.ndim == 2:
        T_total = 1
    else:
        raise ValueError(f"Unsupported TIFF ndim={mm.ndim} shape={mm.shape}")

    T = int(frames) if frames is not None else T_total
    T = min(T, T_total)

    # Calculate bounds once via sampling if not uint8
    bounds = None
    if mm.dtype != np.uint8:
        bounds = _calc_sampled_bounds(mm, T)

    # Result dims
    orig_h, orig_w = (mm.shape[1], mm.shape[2]) if mm.ndim == 3 else mm.shape
    out_w, out_h = size_wh if resize else (orig_w, orig_h)

    final_video = torch.zeros((T, 1, out_h, out_w), dtype=torch.float32)

    for i in range(T):
        frame = mm[i] if mm.ndim == 3 else mm

        # Handle endianness
        if not frame.dtype.isnative:
            frame = frame.astype(frame.dtype.type)

        # Contrast stretch
        if bounds is not None:
            frame = to_uint8_contrast(frame, bounds=bounds)
        elif frame.dtype != np.uint8:
            frame = to_uint8_contrast(frame)

        # Resize
        if resize:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # To grayscale if needed (though TIFF usually is)
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            code = cv2.COLOR_RGB2GRAY if frame.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY
            frame = cv2.cvtColor(frame, code)

        if frame.ndim == 2:
            final_video[i, 0] = torch.from_numpy(frame.astype(np.float32) / 255.0)
        else:
            final_video[i, 0] = torch.from_numpy(
                frame[..., 0].astype(np.float32) / 255.0
            )

    return final_video


def load_video(
    file: str, resize=True, size_wh=(256, 256), max_frames: int | None = None
) -> torch.Tensor:
    """
    Load a video file into torch (T, 1, H, W) float.
    """
    video = read_video_from_path(file)
    if video is None or len(video) == 0:
        raise ValueError(f"No frames read from: {file}")

    if max_frames is not None:
        video = video[: int(max_frames)]

    processed_frames = []
    for frame in video:
        f = np.asarray(frame)
        # Convert color to gray if needed
        if f.ndim == 3:
            if f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_RGBA2GRAY)
            elif f.shape[2] == 3:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

        if resize:
            target_w, target_h = int(size_wh[0]), int(size_wh[1])
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if f.ndim == 2:
            f = f[..., None]  # Ensure (H,W,1)

        processed_frames.append(f)

    v_thwc = np.stack(processed_frames, axis=0).astype(np.uint8)  # (T,H,W,1)
    return _to_torch_tchw(v_thwc)


def save_and_display_video(array: np.ndarray, filename="output.mp4", fps=30) -> None:
    """
    Save grayscale or RGB video to MP4.

    Accepts:
      - (T, H, W) grayscale
      - (T, H, W, 3) RGB
    """
    v = np.asarray(array)

    if v.ndim == 3:
        # (T, H, W) -> (T, H, W, 3)
        if v.dtype != np.uint8:
            vmin, vmax = float(np.min(v)), float(np.max(v))
            v = 255.0 * (v - vmin) / (vmax - vmin + 1e-8)
            v = np.clip(v, 0, 255).astype(np.uint8)
        v = np.repeat(v[..., None], 3, axis=-1)

    elif v.ndim == 4 and v.shape[-1] == 3:
        if v.dtype != np.uint8:
            v = np.clip(v, 0, 255).astype(np.uint8)
    else:
        raise ValueError(
            f"Unsupported array shape for save_and_display_video: {v.shape}"
        )

    t_size, h_size, w_size, _ = v.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (w_size, h_size), isColor=True)

    for i in range(t_size):
        # v is RGB -> convert to BGR for OpenCV
        out.write(cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved to {filename}")


def save_video_tiff(video, filename: str) -> None:
    """
    Save video as TIFF stack.

    Supports:
      - (T, H, W, 3) or (T, H, W) numpy/torch
      - (T, C, H, W) torch/numpy
    """
    v = (
        video.detach().cpu().numpy()
        if isinstance(video, torch.Tensor)
        else np.asarray(video)
    )

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
      - (T, H, W, 3) or (T, H, W)
      - (T, C, H, W) torch/numpy
    """
    v = (
        video.detach().cpu().numpy()
        if isinstance(video, torch.Tensor)
        else np.asarray(video)
    )

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


def _to_gray_float(video) -> np.ndarray:
    """
    Convert video to grayscale float32 array shaped (T, H, W).
    Accepts (T, C, H, W), (T, H, W, C), (T, H, W).
    """
    v = (
        video.detach().cpu().numpy()
        if isinstance(video, torch.Tensor)
        else np.asarray(video)
    )

    # (T, C, H, W) layout
    if v.ndim == 4 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
        gray = v[:, 0].astype(np.float32)

    # (T, H, W, C) layout with 1, 3 or 4 channels
    elif v.ndim == 4 and v.shape[-1] in (1, 3, 4):
        gray = v[..., 0].astype(np.float32)

    # (T, H, W) already grayscale
    elif v.ndim == 3:
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
            diffs = [
                np.mean(np.abs(frames_u8[t].astype(np.float32) - ref))
                for t in range(1, T)
            ]
        else:
            diffs = [
                np.mean(
                    np.abs(
                        frames_u8[t].astype(np.float32)
                        - frames_u8[t - 1].astype(np.float32)
                    )
                )
                for t in range(1, T)
            ]
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


def probe_media_info(path: str):
    """
    Try to get (frames, width, height) without loading full data.

    Supports:
      - mp4/avi/mov/mkv via OpenCV.
      - tiff via tifffile.
      - h5/hdf5 via h5py (prefers /data).
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
