#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

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

try:
    from skimage.metrics import structural_similarity as _ssim

    _HAVE_SSIM = True
except ImportError:
    _ssim = None
    _HAVE_SSIM = False

try:
    from skimage.registration import optical_flow_tvl1 as _optical_flow_tvl1

    _HAVE_TVL1 = True
except ImportError:
    _optical_flow_tvl1 = None
    _HAVE_TVL1 = False


class LazyVideo:
    """
    Lazy frame accessor for TIFF stacks and video files (AVI/MP4).

    Frames are decoded one at a time for display; the full tensor is
    only materialized when needed for processing.  Backed by a
    tifffile memmap (TIFF) or a cv2.VideoCapture (video).
    """

    def __init__(
        self,
        *,
        source_type: str,
        memmap=None,
        path: str | None = None,
        T: int,
        H: int,
        W: int,
        bounds=None,
        h5file=None,
        dataset=None,
        frame_offset: int = 0,
        owns_h5file: bool = True,
    ):
        self._source_type = source_type  # "tiff", "video", or "hdf5"
        self._memmap = memmap
        self._path = path
        self._T = T
        self._H = H
        self._W = W
        self._bounds = bounds  # (lo, hi) for TIFF/HDF5 contrast stretch
        self._tensor: torch.Tensor | None = None
        self._cap = None  # persistent cv2.VideoCapture for display
        self._cap_pos = -1  # next frame the cap will deliver
        self._h5file = h5file  # kept open so `dataset` stays readable
        self._dataset = dataset  # h5py.Dataset, read one frame at a time
        self._frame_offset = frame_offset  # index into memmap/dataset for frame 0
        # False for a slice view sharing another LazyVideo's h5 handle —
        # only the owner closes it (see __del__).
        self._owns_h5file = owns_h5file

    # ---- shape / length --------------------------------------------------

    @property
    def shape(self):
        return (self._T, 1, self._H, self._W)

    @property
    def ndim(self):
        return 4

    def __len__(self):
        return self._T

    # ---- lazy single-frame access ----------------------------------------

    def get_frame_hwc(self, idx: int) -> np.ndarray:
        """Decode frame *idx* and return it as (H, W, 1) float32 in [0, 1]."""
        idx = int(np.clip(idx, 0, self._T - 1)) + self._frame_offset

        if self._source_type == "tiff":
            mm = self._memmap
            frame = np.asarray(mm[idx] if mm.ndim == 3 else mm)
            if not frame.dtype.isnative:
                frame = frame.astype(frame.dtype.type)
            if self._bounds is not None:
                frame = to_uint8_contrast(frame, bounds=self._bounds)
            elif frame.dtype != np.uint8:
                frame = to_uint8_contrast(frame)
            if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                code = (
                    cv2.COLOR_RGB2GRAY if frame.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY
                )
                frame = cv2.cvtColor(frame, code)
            return (frame.astype(np.float32) / 255.0)[..., None]

        elif self._source_type == "hdf5":
            ds = self._dataset
            frame = np.asarray(ds[idx] if ds.ndim >= 3 else ds[:])
            if not frame.dtype.isnative:
                frame = frame.astype(frame.dtype.type)
            if self._bounds is not None:
                frame = to_uint8_contrast(frame, bounds=self._bounds)
            elif frame.dtype != np.uint8:
                frame = to_uint8_contrast(frame)
            if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                code = (
                    cv2.COLOR_RGB2GRAY if frame.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY
                )
                frame = cv2.cvtColor(frame, code)
            return (frame.astype(np.float32) / 255.0)[..., None]

        else:  # video
            if self._cap is None:
                self._cap = cv2.VideoCapture(self._path)
                self._cap_pos = 0
            if self._cap_pos != idx:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._cap_pos = idx
            ok, frame = self._cap.read()
            self._cap_pos = idx + 1
            if not ok:
                return np.zeros((self._H, self._W, 1), dtype=np.float32)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            return (gray.astype(np.float32) / 255.0)[..., None]

    # ---- full materialization --------------------------------------------

    def materialize(self) -> torch.Tensor:
        """Build and cache the full (T, 1, H, W) float32 tensor."""
        if self._tensor is not None:
            return self._tensor

        print(f"Loading {self._source_type} ({self._T} frames) into memory…")

        if self._source_type in ("tiff", "hdf5"):
            frames = []
            for i in range(self._T):
                hwc = self.get_frame_hwc(i)
                frames.append(torch.from_numpy(hwc).permute(2, 0, 1))  # (1,H,W)
            self._tensor = torch.stack(frames, 0).float()

        else:  # video — use a fresh cap so the display cap is undisturbed
            cap = cv2.VideoCapture(self._path)
            frames = []
            for _ in range(self._T):
                ok, frame = cap.read()
                if not ok:
                    frames.append(torch.zeros(1, self._H, self._W))
                    continue
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3
                    else frame
                )
                frames.append(
                    torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
                )
            cap.release()
            self._tensor = torch.stack(frames, 0).float()

        return self._tensor

    # ---- tensor-like proxy -----------------------------------------------

    def permute(self, *args):
        return self.materialize().permute(*args)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._T)
            if step == 1:
                # Lazy sub-range view: shares the underlying memmap/dataset rather than materializing to trim it.
                return LazyVideo(
                    source_type=self._source_type,
                    memmap=self._memmap,
                    path=self._path,
                    T=max(0, stop - start),
                    H=self._H,
                    W=self._W,
                    bounds=self._bounds,
                    h5file=self._h5file,
                    dataset=self._dataset,
                    frame_offset=self._frame_offset + start,
                    owns_h5file=False,
                )
        return self.materialize()[key]

    def __del__(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        if self._h5file is not None and self._owns_h5file:
            try:
                self._h5file.close()
            except Exception:
                pass


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


def _imagej_saved_display_range(path: str) -> tuple[float, float] | None:
    """ImageJ stores the contrast window the user had set (Image > Adjust >
    Brightness/Contrast) in the file's own ImageJ metadata ('min'/'max'
    tags), independent of the actual pixel value range -- reopening the
    file in ImageJ reproduces that exact window rather than recomputing one
    from scratch. Our own percentile-of-actual-values stretch can pick a
    wildly different window for data whose meaningful dynamic range is a
    narrow slice of its raw value range (e.g. a bright, high-valued
    background with a sparse, low-valued signal) -- using ImageJ's own
    saved window instead matches what the acquisition software/user judged
    to be the meaningful contrast range for this specific file.

    Returns None for non-TIFFs, files with no ImageJ metadata, or any read
    error -- callers fall back to their own percentile-based bounds."""
    if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            meta = tf.imagej_metadata
    except Exception:
        return None
    if not meta or "min" not in meta or "max" not in meta:
        return None
    lo, hi = float(meta["min"]), float(meta["max"])
    if hi <= lo:
        return None
    return lo, hi


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
) -> "torch.Tensor | LazyVideo":
    """
    Load HDF5 dataset into torch (T, 1, H, W) float, or lazily via LazyVideo.

    Expects dataset to be (T, H, W) or (T, H, W, C).
    Non-uint8 data is automatically contrast-stretched.
    If dataset_path is None or not found, tries 'data' or 'video' defaults.

    When *resize* is False (the common path from load_data) a
    :class:`LazyVideo` is returned so frames are decoded on demand and the
    file handle is kept open for its lifetime — the full recording is never
    materialized in RAM at once, which otherwise gets large native-resolution
    H5 recordings OOM-killed. When *resize* is True the full tensor is built
    eagerly (each frame is still decoded one at a time from disk, but the
    downsized result is small).
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    h5 = h5py.File(h5_path, "r")
    lazy_result = None
    try:
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
        elif ds.ndim == 3:
            orig_h, orig_w = ds.shape[1], ds.shape[2]
        else:  # (T, H, W, C)
            orig_h, orig_w = ds.shape[1], ds.shape[2]

        if not resize:
            # Ownership of `h5` transfers to the LazyVideo, which closes it on GC (see LazyVideo.__del__).
            lazy_result = LazyVideo(
                source_type="hdf5",
                h5file=h5,
                dataset=ds,
                T=T,
                H=orig_h,
                W=orig_w,
                bounds=bounds,
            )
            return lazy_result

        # --- eager path (resize=True) ----------------------------------
        out_w, out_h = size_wh
        final_video = torch.zeros((T, 1, out_h, out_w), dtype=torch.float32)

        try:
            for i in range(T):
                frame = ds[i] if ds.ndim >= 3 else ds[:]

                # Convert to uint8 with bounds
                if bounds is not None:
                    frame = to_uint8_contrast(frame, bounds=bounds)
                elif frame.dtype != np.uint8:
                    frame = to_uint8_contrast(frame)

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
    finally:
        if lazy_result is None:
            h5.close()


def memmap_or_load_tiff(path: str) -> np.ndarray:
    """tifffile.memmap(path), falling back to a full eager tifffile.imread()
    when the file isn't memory-mappable -- tifffile can only memmap
    uncompressed, contiguously-stored data, so any LZW/Deflate/JPEG-
    compressed TIFF (a common export setting in ImageJ/Fiji and other
    microscopy software) raises "image data are not memory-mappable"
    instead of just working slower. The eager array supports the exact
    same indexing every caller here already uses -- it just loses
    memmap's lazy-paging memory benefit for this one file.

    tifffile.memmap() sizes its memmap from the file's series/metadata
    frame count, which can overclaim relative to what's actually walkable
    in the real page chain (see _reliable_series_frame_count) -- e.g. a
    non-ImageJ BigTIFF whose axes='TYX' series.shape said 50000 frames
    while only 1250 real IFDs existed on disk. tifffile.imread() handles
    that case correctly (confirmed: returns exactly the 1250 real frames),
    so pre-check for the mismatch and route straight to imread() instead
    of letting memmap() attempt an oversized mapping."""
    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            computed = 1
            for ax, n in zip(series.axes, series.shape):
                if ax not in ("Y", "X"):
                    computed *= int(n)
            if _reliable_series_frame_count(tf, computed) != computed:
                print(
                    f"Note: {path}'s metadata claims more frames than are "
                    f"actually present on disk -- loading it eagerly to get "
                    f"the real, fully-present frames instead of memmap'ing "
                    f"a mapping sized from the unreliable count."
                )
                return tifffile.imread(path)
    except Exception:
        pass  # any issue here just falls through to the normal memmap attempt below

    try:
        return tifffile.memmap(path)
    except ValueError as e:
        if "memory-mappable" not in str(e):
            raise
        print(f"Note: {path} is not memory-mappable (likely compressed) -- loading it fully into memory instead.")
        return tifffile.imread(path)


def load_tiff(
    path: str,
    frames: int | None = None,
    resize=True,
    size_wh=(256, 256),
    axes_order: list[str] | None = None,
) -> "torch.Tensor | LazyVideo":
    """
    Load a TIFF stack.

    When *resize* is False (the common path from load_data) a :class:`LazyVideo`
    is returned so frames are decoded on demand.  When *resize* is True the full
    tensor is built eagerly so spatial interpolation can be applied.

    *axes_order*: for a genuine multi-axis hyperstack (e.g. Z-and-T), the
    non-spatial axis letters (e.g. ``["T", "Z"]``) in the order the caller
    wants them flattened, outermost first -- overrides the file's own
    metadata order (see get_tiff_axes_info()). Ignored for 3D/2D data.
    """
    mm = memmap_or_load_tiff(path)

    if mm.ndim > 3:
        # ndim>3 means a genuine multi-axis hyperstack -- tifffile already returns the true logical shape from
        # series/ImageJ/OME metadata, not raw page count, so every axis here is real.
        if axes_order:
            try:
                with tifffile.TiffFile(path) as tf:
                    native_axes = tf.series[0].axes
            except Exception:
                native_axes = None
            non_spatial_native = (
                [a for a in native_axes if a not in ("Y", "X")] if native_axes else None
            )
            if (
                non_spatial_native
                and len(native_axes) == mm.ndim
                and sorted(axes_order) == sorted(non_spatial_native)
                and list(axes_order) != non_spatial_native
            ):
                spatial_idx = [i for i, a in enumerate(native_axes) if a in ("Y", "X")]
                perm = [native_axes.index(a) for a in axes_order] + spatial_idx
                # Non-identity permutation of a memmap is a strided, non-contiguous view -- forces a full eager
                # copy into RAM, same tradeoff memmap_or_load_tiff() already accepts for compressed files.
                mm = np.transpose(mm, perm)
        # Flatten every leading axis into one frame axis -- zero-copy when *mm* is still C-contiguous, else a copy.
        mm = mm.reshape(-1, mm.shape[-2], mm.shape[-1])

    # The read above degenerates to a single 2D frame when the page chain is broken past frame 0 (incomplete
    # export, not mislabeled axes) -- rebuild the frames actually present instead of reporting just 1.
    series_frames = int(mm.shape[0]) if mm.ndim == 3 else (1 if mm.ndim == 2 else None)
    trunc = _detect_truncated_imagej_stack(path, series_frames)
    if trunc is not None:
        print(
            f"Note: {path} appears truncated -- its ImageJ header declares "
            f"{trunc['declared_frames']} frames but only "
            f"{trunc['actual_frames']} are fully present on disk; "
            f"loading the {trunc['actual_frames']} usable frames only."
        )
        mm = _memmap_truncated_imagej_stack(path, trunc)

    if mm.ndim == 3:
        T_total = int(mm.shape[0])
        orig_h, orig_w = int(mm.shape[1]), int(mm.shape[2])
    elif mm.ndim == 2:
        T_total = 1
        orig_h, orig_w = int(mm.shape[0]), int(mm.shape[1])
    else:
        raise ValueError(f"Unsupported TIFF ndim={mm.ndim} shape={mm.shape}")

    T = min(int(frames), T_total) if frames is not None else T_total

    bounds = None
    if mm.dtype != np.uint8:
        bounds = _imagej_saved_display_range(path) or _calc_sampled_bounds(mm, T)

    if not resize:
        return LazyVideo(
            source_type="tiff",
            memmap=mm,
            T=T,
            H=orig_h,
            W=orig_w,
            bounds=bounds,
        )

    # --- eager path (resize=True) ----------------------------------------
    out_w, out_h = size_wh
    final_video = torch.zeros((T, 1, out_h, out_w), dtype=torch.float32)

    for i in range(T):
        frame = mm[i] if mm.ndim == 3 else mm

        if not frame.dtype.isnative:
            frame = frame.astype(frame.dtype.type)

        if bounds is not None:
            frame = to_uint8_contrast(frame, bounds=bounds)
        elif frame.dtype != np.uint8:
            frame = to_uint8_contrast(frame)

        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

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
) -> "torch.Tensor | LazyVideo":
    """
    Load a video file.

    When *resize* is False (the common path from load_data) a :class:`LazyVideo`
    is returned backed by cv2.VideoCapture so frames are decoded on demand.
    When *resize* is True the full tensor is built eagerly.
    """
    cap = cv2.VideoCapture(file)
    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if T_total <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"Could not read video metadata from: {file}")

    T = min(T_total, int(max_frames)) if max_frames is not None else T_total

    if not resize:
        return LazyVideo(source_type="video", path=file, T=T, H=H, W=W)

    # --- eager path (resize=True) ----------------------------------------
    video = read_video_from_path(file)
    if video is None or len(video) == 0:
        raise ValueError(f"No frames read from: {file}")

    if max_frames is not None:
        video = video[: int(max_frames)]

    target_w, target_h = int(size_wh[0]), int(size_wh[1])
    processed_frames = []
    for frame in video:
        f = np.asarray(frame)
        if f.ndim == 3:
            if f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_RGBA2GRAY)
            elif f.shape[2] == 3:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if f.ndim == 2:
            f = f[..., None]
        processed_frames.append(f)

    v_thwc = np.stack(processed_frames, axis=0).astype(np.uint8)
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

    # squeeze trailing singleton channel: (T,H,W,1) -> (T,H,W) for ImageJ compat
    if v.ndim == 4 and v.shape[-1] == 1:
        v = v[..., 0]

    if v.dtype != np.uint8:
        if np.max(v) <= 1.0:
            v = (v * 255.0).astype(np.uint8)
        elif np.max(v) <= 255.0:
            v = np.clip(v, 0, 255).astype(np.uint8)
        else:
            # high bit-depth (e.g. 16-bit TIFF): percentile-normalize to [0, 255]
            lo = float(np.percentile(v, 1))
            hi = float(np.percentile(v, 99))
            if hi > lo:
                v = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            else:
                v = np.zeros_like(v, dtype=np.uint8)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=v, compression="gzip", compression_opts=9)
    print(f"HDF5 saved to {filename}")


def _to_numpy(x) -> np.ndarray:
    """Detach+move a torch.Tensor to a numpy array; pass anything else through np.asarray."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _to_thwc(video) -> np.ndarray:
    """Convert input video to numpy THWC format."""
    v = (
        video.detach().cpu().numpy()
        if isinstance(video, torch.Tensor)
        else np.asarray(video)
    )

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
    """Convert THWC to grayscale THW float32 via luminance-weighted RGB."""
    if video_thwc.shape[-1] == 1:
        return video_thwc[..., 0].astype(np.float32)

    if video_thwc.shape[-1] >= 3:
        rgb = video_thwc[..., :3].astype(np.float32)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        return gray.astype(np.float32)

    raise ValueError(f"Unsupported channel count: {video_thwc.shape[-1]}")


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


def video_metrics(
    video,
    mode="ref_first",
    *,
    return_per_frame: bool = False,
    rof_max_frames: int = 10,
    drift=None,
) -> dict[str, float]:
    """
    Compute registration-quality metrics for a video.

    mode:
      - "ref_first":   compare each frame against frame 0
      - "consecutive": compare each frame against the previous frame

    Intensities are stretched by the 1st/99th percentiles so SSIM and PSNR
    share the same normalized scale. MSE (mean squared error, on this same
    normalized [0, 1] scale) is also returned -- it's what PSNR is derived
    from (psnr = 10*log10(1/mse)), so mse = 10**(-psnr/10) exactly recovers
    it from psnr alone wherever only psnr was saved historically.

    *drift* is an optional (T, 2) array of per-frame (dx, dy) shift, e.g. from
    mc_cotracker.warping(). When given, the border margin excluded from every
    metric below is widened to cover the largest observed shift, so pixels
    revealed by warp-induced zero-fill (out-of-bounds regions) don't bias the
    result. This is a translational approximation: it assumes the invalid
    region is a uniform border strip, which holds for global drift but can
    under- or over-exclude for strongly non-rigid warps.

        Additional summary metrics:
            - crispness_mean: mean-image gradient RMS (higher = sharper)
            - corr_mean: mean Pearson correlation of each frame with the mean image
            - residual_flow: leftover motion estimated directly from the corrected
              pixels. The temporal mean of the (registered) stack is used as a
              template, and TV-L1 optical flow (skimage.registration.optical_flow_tvl1)
              is run between the template and up to *rof_max_frames* frames sampled
              evenly across the stack. Each sampled frame's value is the median
              magnitude of its dense displacement field; unsampled frames are NaN.

    If *return_per_frame* is True, the result includes a ``per_frame`` block
    containing frame 0 self-comparison followed by frames 1..T-1.
    """
    gray = _to_gray_float(video).astype(np.float32, copy=False)
    T = int(gray.shape[0])

    finite = np.isfinite(gray)
    if finite.any():
        lo = float(np.percentile(gray[finite], 1))
        hi = float(np.percentile(gray[finite], 99))
        gray_safe = np.where(finite, gray, lo)
    else:
        lo, hi = 0.0, 1.0
        gray_safe = np.zeros_like(gray, dtype=np.float32)

    if hi <= lo:
        hi = lo + 1.0

    gray_norm = np.clip((gray_safe - lo) / (hi - lo), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    frames_u8 = np.clip(gray_norm * 255.0, 0, 255).astype(np.uint8)
    mean_img = gray_norm.mean(axis=0)

    _, H, W = gray_norm.shape
    bh = max(1, int(0.10 * H))
    bw = max(1, int(0.10 * W))
    if drift is not None:
        drift_arr = np.asarray(drift, dtype=np.float32)
        if drift_arr.size:
            max_abs_dx = float(np.nanmax(np.abs(drift_arr[..., 0])))
            max_abs_dy = float(np.nanmax(np.abs(drift_arr[..., 1])))
            if np.isfinite(max_abs_dx):
                bw = max(bw, int(np.ceil(max_abs_dx)) + 1)
            if np.isfinite(max_abs_dy):
                bh = max(bh, int(np.ceil(max_abs_dy)) + 1)
            # leave at least a 10%-wide valid interior even for extreme drift
            bw = min(bw, max(1, W // 2 - 1))
            bh = min(bh, max(1, H // 2 - 1))
    frames_u8 = frames_u8[:, bh:-bh, bw:-bw]
    gray_norm = gray_norm[:, bh:-bh, bw:-bw]
    mean_img = mean_img[bh:-bh, bw:-bw]

    def _ref_and_cur(t):
        if mode == "ref_first":
            return 0, t
        return t - 1, t

    def _entropy(p):
        p = np.asarray(p, dtype=np.float64)
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p))) if p.size else 0.0

    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12
        return float(np.dot(a, b) / denom)

    gy, gx = np.gradient(mean_img.astype(np.float32, copy=False))
    crispness_mean = float(np.sqrt(np.mean(gx**2 + gy**2)))
    mean_corr_vals = [_pearson_corr(gray_norm[t], mean_img) for t in range(T)]

    residual_flow_vals = [float("nan")] * T
    if _HAVE_TVL1 and T > 0:
        if T <= rof_max_frames:
            rof_indices = range(T)
        else:
            rof_indices = sorted(
                set(np.linspace(0, T - 1, rof_max_frames).round().astype(int).tolist())
            )
        template = mean_img.astype(np.float64, copy=False)
        for t in rof_indices:
            try:
                flow = _optical_flow_tvl1(
                    template, gray_norm[t].astype(np.float64, copy=False)
                )
                mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
                residual_flow_vals[t] = float(np.median(mag))
            except Exception:
                residual_flow_vals[t] = float("nan")

    ssim_vals = [1.0]
    zncc_vals = [1.0]
    mi_vals = [0.0]
    psnr_vals = [float("inf")]
    mse_vals = []

    if T >= 1:
        ref0 = gray_norm[0].ravel().astype(np.float64)
        try:
            hist_ref, _ = np.histogram(ref0, bins=64, density=False)
            p_ref = hist_ref.astype(np.float64)
            total_ref = p_ref.sum()
            if total_ref > 0:
                p_ref /= total_ref
            mi_vals[0] = _entropy(p_ref)
        except Exception:
            mi_vals[0] = float("nan")

    for t in range(1, T):
        ri, ci = _ref_and_cur(t)

        if _HAVE_SSIM:
            try:
                ssim_vals.append(
                    float(_ssim(frames_u8[ri], frames_u8[ci], data_range=255))
                )
            except Exception:
                ssim_vals.append(float("nan"))
        else:
            d = float(
                np.mean(
                    np.abs(
                        frames_u8[ci].astype(np.float32)
                        - frames_u8[ri].astype(np.float32)
                    )
                )
            )
            ssim_vals.append(float(np.clip(1.0 - d / 255.0, 0.0, 1.0)))

        ref = gray_norm[ri].ravel().astype(np.float64)
        cur = gray_norm[ci].ravel().astype(np.float64)

        mse_t = float(np.mean((cur - ref) ** 2))
        mse_vals.append(mse_t)
        psnr_vals.append(float("inf") if mse_t <= 0 else 10.0 * np.log10(1.0 / mse_t))

        ref_c = ref - ref.mean()
        cur_c = cur - cur.mean()
        denom = np.sqrt(np.dot(ref_c, ref_c) * np.dot(cur_c, cur_c)) + 1e-12
        zncc_vals.append(float(np.dot(ref_c, cur_c) / denom))

        try:
            hist2d, _, _ = np.histogram2d(ref, cur, bins=64, density=False)
            pxy = hist2d.astype(np.float64)
            total = pxy.sum()
            if total > 0:
                pxy /= total
                px, py = pxy.sum(axis=1), pxy.sum(axis=0)
                mi_vals.append(_entropy(px) + _entropy(py) - _entropy(pxy.ravel()))
            else:
                mi_vals.append(float("nan"))
        except Exception:
            mi_vals.append(float("nan"))

    def _mean(vals, default):
        finite_vals = [v for v in vals[1:] if np.isfinite(v)]
        return float(np.mean(finite_vals)) if finite_vals else float(default)

    summary = {
        "ssim": _mean(ssim_vals, 1.0),
        "psnr": (
            float("inf")
            if mse_vals and float(np.mean(mse_vals)) <= 0
            else (
                10.0 * np.log10(1.0 / float(np.mean(mse_vals)))
                if mse_vals
                else float("nan")
            )
        ),
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "zncc": _mean(zncc_vals, 1.0),
        "mi": _mean(mi_vals, 0.0),
        "crispness_mean": crispness_mean,
        "corr_mean": (
            float(np.mean([v for v in mean_corr_vals if np.isfinite(v)]))
            if any(np.isfinite(v) for v in mean_corr_vals)
            else float("nan")
        ),
        "residual_flow": (
            float(np.mean([v for v in residual_flow_vals if np.isfinite(v)]))
            if any(np.isfinite(v) for v in residual_flow_vals)
            else float("nan")
        ),
    }

    if return_per_frame:
        summary["per_frame"] = {
            "ssim": ssim_vals,
            "psnr": psnr_vals,
            "mse": [0.0] + mse_vals,  # frame-0 placeholder (self-comparison) to match the other T-length lists
            "zncc": zncc_vals,
            "mi": mi_vals,
            "corr_mean": mean_corr_vals,
            "residual_flow": residual_flow_vals,
        }

    return summary


def shared_crop_drift(shift_arrays, T: int) -> np.ndarray:
    """
    Combine several per-frame (dx, dy)-like arrays (e.g. CoTracker drift, NoRMCorre
    shifts — one array per registration method run on the same clip, or None where
    unavailable) into a single synthetic (T, 2) drift array reflecting the largest
    shift seen across all of them.

    Passing this to video_metrics(..., drift=...) for every method being compared
    on that clip forces them all to use the same border-crop margin, so metrics are
    computed over an identical region regardless of which method produced the widest
    out-of-bounds artifact.
    """
    max_dx = 0.0
    max_dy = 0.0
    for arr in shift_arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        finite_dx = a[..., 0][np.isfinite(a[..., 0])]
        finite_dy = a[..., 1][np.isfinite(a[..., 1])]
        if finite_dx.size:
            max_dx = max(max_dx, float(np.max(np.abs(finite_dx))))
        if finite_dy.size:
            max_dy = max(max_dy, float(np.max(np.abs(finite_dy))))
    return np.tile(np.array([max_dx, max_dy], dtype=np.float32), (max(T, 1), 1))


def _reliable_series_frame_count(tf: "tifffile.TiffFile", computed_frames: int) -> int:
    """series.shape-derived frame counts can overclaim even with no ImageJ
    metadata involved at all (seen: axes='TYX', series.shape said 50000,
    but tifffile's own page-chain only actually walks 1250 real IFDs --
    some non-ImageJ metadata field the series parser trusts was simply
    wrong/stale; the file itself is complete and consistent with holding
    exactly 1250 frames). _detect_truncated_imagej_stack() only covers the
    ImageJ-metadata-specific variant of this, so this is a second, more
    general cross-check.

    A low len(tf.pages) alone can't be used as a blanket frame count
    though: a normal, complete ImageJ-style "single IFD + raw contiguous
    dump" hyperstack legitimately has just ONE real physical page
    representing many logical frames, so that case must NOT be clamped.
    Only trust the page count over the series computation when there's
    more than one actual walkable page (i.e. this file really does use
    one real IFD per frame, not the virtual-stack convention) AND that
    count is smaller than what the series claims."""
    n_pages = len(tf.pages)
    if 1 < n_pages < computed_frames:
        return n_pages
    return computed_frames


def _detect_truncated_imagej_stack(path: str, series_frames: int | None) -> dict | None:
    """Detect an ImageJ hyperstack whose own header/description claims more
    frames than the file can physically hold -- an incomplete export/copy
    (write cut short, not merely mislabeled axes), which makes tifffile's
    own page-chain and series parsing give up entirely (warnings like
    "invalid page offset"/"corrupted file") and degenerate to reporting a
    single frame.

    Only called when *series_frames* already looks degenerate (None or 1) --
    a file whose series/page parsing resolved normally never needs this.

    Recovery is done by computing frame byte layout directly from the one
    page tifffile COULD read (first_offset, frame_nbytes), rather than
    trusting the broken page chain -- only safe for the single-IFD,
    uncompressed, single-contiguous-strip layout ImageJ normally uses for
    big raw hyperstacks, so this returns None rather than guess for any
    other layout (e.g. compressed or tiled data).

    Returns None when the file isn't in this situation, else a dict with
    'actual_frames' (frames fully present on disk), 'declared_frames' (what
    the header claims), 'first_offset', 'frame_nbytes', 'shape_hw', 'dtype'.
    """
    if series_frames is not None and series_frames > 1:
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            meta = tf.imagej_metadata
            if not meta:
                return None
            declared = int(meta.get("images") or meta.get("slices") or 0)
            if declared <= 1:
                return None
            page0 = tf.pages[0]
            if (
                page0.compression != 1  # 1 == COMPRESSION_NONE
                or not page0.is_contiguous
                or len(page0.dataoffsets) != 1
            ):
                return None
            first_offset = int(page0.dataoffsets[0])
            frame_nbytes = int(page0.databytecounts[0])
            h, w = int(page0.shape[-2]), int(page0.shape[-1])
            # page0.dtype is native-byte-order regardless of the file's actual order --
            # pin to the file's recorded byte order or big-endian files silently byte-swap.
            dtype = page0.dtype.newbyteorder(tf.byteorder)
    except Exception:
        return None

    if frame_nbytes <= 0:
        return None
    available = os.path.getsize(path) - first_offset
    actual_frames = available // frame_nbytes
    if actual_frames <= 0 or actual_frames >= declared:
        return None
    return {
        "actual_frames": int(actual_frames),
        "declared_frames": declared,
        "first_offset": first_offset,
        "frame_nbytes": frame_nbytes,
        "shape_hw": (h, w),
        "dtype": dtype,
    }


def _memmap_truncated_imagej_stack(path: str, info: dict) -> np.memmap:
    """Build a (actual_frames, H, W) memmap directly over the raw frame
    bytes identified by _detect_truncated_imagej_stack(), bypassing
    tifffile's (broken, for this file) page-chain parsing entirely."""
    h, w = info["shape_hw"]
    return np.memmap(
        path,
        dtype=info["dtype"],
        mode="r",
        offset=info["first_offset"],
        shape=(info["actual_frames"], h, w),
    )


def get_tiff_truncation_warning(path: str) -> str | None:
    """Human-readable warning for a TIFF that load_tiff()/probe_media_info()
    will silently clamp due to truncation (see _detect_truncated_imagej_stack)
    -- for callers (e.g. the GUI's load dialog) that want to surface this
    directly in the UI instead of only the console note. None when the file
    isn't affected."""
    if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
        return None
    trunc = _detect_truncated_imagej_stack(path, None)
    if trunc is None:
        return None
    return (
        f"Warning: this file appears truncated. Its header declares "
        f"{trunc['declared_frames']} frames, but only "
        f"{trunc['actual_frames']} are fully present on disk -- "
        f"only those {trunc['actual_frames']} usable frames will be loaded."
    )


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
                # series[0] reflects the file's ImageJ/OME/shaped metadata, not raw page count -- some writers
                # pack multiple logical frames per physical page, which len(tf.pages) would misreport as 1 frame.
                # Frame count = product of every axis except the trailing spatial Y, X.
                series = tf.series[0]
                shape, axes = series.shape, series.axes
                h = int(shape[axes.index("Y")]) if "Y" in axes else None
                w = int(shape[axes.index("X")]) if "X" in axes else None
                frames = 1
                for ax, n in zip(axes, shape):
                    if ax not in ("Y", "X"):
                        frames *= int(n)
                reliable_frames = _reliable_series_frame_count(tf, frames)
                if reliable_frames != frames:
                    print(
                        f"Note: {path}'s metadata claims {frames} frames, but "
                        f"only {reliable_frames} real pages are actually "
                        f"present on disk; reporting {reliable_frames}."
                    )
                    frames = reliable_frames

            trunc = _detect_truncated_imagej_stack(path, frames)
            if trunc is not None:
                print(
                    f"Note: {path} appears truncated -- its ImageJ header "
                    f"declares {trunc['declared_frames']} frames but only "
                    f"{trunc['actual_frames']} are fully present on disk; "
                    f"reporting the {trunc['actual_frames']} usable frames."
                )
                frames = trunc["actual_frames"]
                h, w = trunc["shape_hw"]

            return (frames if frames else None), w, h
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


def get_tiff_axes_info(path: str) -> dict | None:
    """For a TIFF with more than one non-spatial axis (e.g. a genuine
    Z-and-T hyperstack), return the file's own metadata axis order and each
    axis's size so a caller can offer a "which axis is outermost" picker --
    load_tiff() otherwise just flattens axes in whatever order the file's
    own metadata happens to store them in, which is only guaranteed to be
    chronological if that order is already T-major.

    Returns None for non-TIFF files, files with 0-1 non-spatial axes (no
    ordering ambiguity possible), or on any read error. Otherwise:
        {"axes": [letter, ...], "sizes": {letter: int, ...]}
    with *axes* in the file's own native (metadata) order, outermost first.
    """
    if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            shape, axes = series.shape, series.axes
    except Exception:
        return None

    non_spatial = [a for a in axes if a not in ("Y", "X")]
    if len(non_spatial) <= 1:
        return None
    sizes = {a: int(n) for a, n in zip(axes, shape) if a not in ("Y", "X")}
    return {"axes": non_spatial, "sizes": sizes}
