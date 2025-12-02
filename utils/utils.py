import numpy as np
import cv2
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path
import tifffile
import h5py
from pathlib import Path
import hdf5plugin


def load_hdf5(h5_path, dataset_path, frames: int = 200, resize=True):
    if not Path(h5_path).exists():
        print(f"File not found: {h5_path}")
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        print("Top-level groups:", list(f.keys()))
        print("\nDatasets found:")
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                # name is relative (no leading slash), add it for a full path
                print(f"/{name}  shape={obj.shape}  dtype={obj.dtype}  compression={obj.compression}")
        f.visititems(show)

    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            print(f"Dataset '{dataset_path}' not found in file.")
            sys.exit(1)

        ds = h5[dataset_path]
        video = np.array(ds[:frames] )

    if resize:
        size = (256, 256) 
        video = np.array([cv2.resize(frame, size) for frame in video])

    return video[:20]

def load_tiff(path, frames: int = 50):
    mm = tifffile.memmap(path)
    print("shape:", mm.shape)
    print("dtype:", mm.dtype)

    if mm.ndim == 3:
        T = min(int(frames), mm.shape[0])
        # Convert each frame to native-endian uint16 (contiguous)
        video = np.stack([np.array(mm[i], dtype=np.uint16) for i in range(T)], axis=0)
    elif mm.ndim == 2:
        video = np.array(mm, dtype=np.uint16)[None, ...]  # make it (1,H,W)
    else:
        raise ValueError(f"Unsupported TIFF ndim={mm.ndim} with shape={mm.shape}")

    return video

def load_video(file: str, resize: bool = True, size=(256, 256), max_frames: int = 20):
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
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # <- This line is critical
        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")