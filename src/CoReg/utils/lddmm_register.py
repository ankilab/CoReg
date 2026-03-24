import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path
from scipy.ndimage import map_coordinates
import utils.lddmm_wrapper as lddmm_wrapper


def load_video(file, size=(256, 256)):
    """
    Load and resize a video for LDDMM processing.

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


def integrate_field(v, num_steps=10):
    """
    Forward-integrate a velocity field v(x,t) over [0,1].
    :param v: (2, H, W) velocity field.
    :param num_steps: Number of integration steps.
    :return: (2, H, W) deformation field phi.
    """
    # Basic coordinate grid
    _, H, W = v.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    # Initialize phi
    phi_x = grid_x.astype(np.float32)
    phi_y = grid_y.astype(np.float32)

    dt = 1.0 / num_steps
    for _ in range(num_steps):
        # Euler step: phi_{t+dt} = phi_t + dt * v(phi_t)
        vx = map_coordinates(v[0], [phi_y, phi_x], order=1, mode="nearest")
        vy = map_coordinates(v[1], [phi_y, phi_x], order=1, mode="nearest")
        phi_x += dt * vx
        phi_y += dt * vy

    return np.stack([phi_x, phi_y], axis=0)


def register_lddmm(fixed, moving):
    """
    Compute LDDMM registration between two frames.
    :param fixed: Reference/target frame.
    :param moving: Source/template frame.
    :return: Warped numpy array.
    """

    lddmm = lddmm_wrapper.LDDMM(
        template=[moving],
        target=[fixed],
        do_affine=0,
        do_lddmm=1,
        a=7,
        nt=5,
        niter=200,
        epsilon=4e0,
        sigma=20.0,
        sigmaR=40.0,
        optimizer="gdr",
        gpu_number=0,
        verbose=False,
        dx=None,
    )

    result = lddmm.run()
    return lddmm.outputDeformedTemplate()[0]


def lddmm_registration(frames):
    """
    Perform LDDMM registration on a sequence of frames, using frame 0 as reference.
    :param frames: List or array of frames.
    :return: List of warped numpy arrays.
    """
    lddmm_registrations = [frames[0]]
    for i in range(1, len(frames)):
        reg = register_lddmm(frames[0], frames[i])
        lddmm_registrations.append(reg)

    return lddmm_registrations


def save_and_display_video(array, filename="output.mp4", fps=30):
    """Save a grayscale registration result to MP4."""

    if len(array.shape) >= 3:
        array = array[
            ..., 0
        ]  # Assuming the last dimension is color channels, take the first channel
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min = array.min()
        array_max = array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # <- This line is critical
        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # define the video file path
    video_file = "/home/rzlin/va11moje/co-tracker/input/Drosophila_Ca_bad.avi"
    filename = video_file.split("/")[-1].split(".")[0]
    if not os.path.exists(video_file):
        print(f"Video file {video_file} does not exist.")
    else:
        print(f"Video file {video_file} exists.")

    # Apply motion correction algorithm by
    # loading the video, running the CoTracker model, and warping the frames

    vid, video_frames = load_video(video_file, size=(256, 256))
    frames = np.array(video_frames)
    result = lddmm_registration(frames)
    save_and_display_video(
        np.array(result), f"output/{filename}_lddm_registered.mp4", fps=15
    )
