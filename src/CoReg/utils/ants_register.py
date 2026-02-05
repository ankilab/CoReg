import gc
import os

import ants
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path

# Memory cleanup to prevent OOM errors
torch.cuda.empty_cache()
gc.collect()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def load_video(file, size=(256, 256)):
    """
    Load and resize a video for ANTs processing.

    :param file: Path to the video file.
    :param size: (width, height) tuple for resizing.
    :return: (video_tensor, video_frames_list)
    """
    video = read_video_from_path(file)
    video_frames = [cv2.resize(frame, size) for frame in video]
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    video_tensor = F.interpolate(
        video_tensor, size=size, mode="bilinear", align_corners=False
    )[None]

    return video_tensor, video_frames

def save_and_display_video(array, filename="output.mp4", fps=30):
    """Save a grayscale registration result to MP4."""
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min, array_max = array.min(), array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = cv2.cvtColor(array[i], cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")

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
        moving_image = ants.from_numpy(frames[i][..., 0])
        reg = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyNOnly")
        results.append(reg["warpedmovout"].numpy())

    return results


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    #define the video file path
    video_file = "/home/rzlin/va11moje/co-tracker/input/Drosophila_Ca_bad.avi"
    filename = (video_file.split('/')[-1].split('.')[0])
    if not os.path.exists(video_file):
        print(f"Video file {video_file} does not exist.")
    else:
        print(f"Video file {video_file} exists.")

    # Apply motion correction algorithm by
    # loading the video, running the CoTracker model, and warping the frames

    vid, video_frames = load_video(video_file, size=(256, 256))
    frames = np.array(video_frames)
    result = ants_registration(frames)
    save_and_display_video(np.array(result), f'output/{filename}_ants_registered.mp4', fps=15)


