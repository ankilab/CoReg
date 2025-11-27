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

    return video

def load_tiff(path, page: int = 0, vmin=None, vmax=None, frames: int = 100):
    mm = tifffile.memmap(path)
    print("shape:", mm.shape)
    print("dtype:", mm.dtype)
    video = []
    # Example 1: show a single slice from a 3D stack
    if mm.ndim == 3:
        for i in range(frames):
            video.append(mm[i])
    else:
        video = mm     # plain 2D image
    
    return np.array(video)

def load_video(file, resize=True):
    """
    Load a video from the given file path, resize it to the specified size,
    and return the video tensor and frames. 
    args:
        file (str): Path to the video file.
        resize: Flag to indicate whether to resize the video frames.
    returns:
        video (torch.Tensor): A tensor containing the video frames, resized to the specified size.
    """
    
    video = read_video_from_path(file)
    if resize:
        size = (256, 256) 
        video = np.array([cv2.resize(frame, size) for frame in video])
    else:
        video = np.array([frame for frame in video])
        size = video.shape[2:4]

    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    #video = F.interpolate(video, size=size, mode='bilinear', align_corners=False)[None]

    return video

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