import os
import torch
import numpy as np
from cotracker.utils.visualizer import read_video_from_path
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity as ssim
#  Visualization
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_filter, zoom
import numpy as np
import torch.nn.functional as F
import utils.torch_lddmm as torch_lddmm
import ants
from PIL import Image


def load_video(file, size=(256,256)):
    """
    Load a video from the given file path, resize it to the specified size,
    and return the video tensor and frames. 
    args:
        file (str): Path to the video file.
        size (tuple): Desired size for the video frames (width, height).
    returns:
        video (torch.Tensor): A tensor containing the video frames, resized to the specified size.
        video_frames (list): A list of resized video frames as numpy arrays.
    """
    video_path = file
    video = read_video_from_path(video_path)
    video_frames = [cv2.resize(frame, size) for frame in video]
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    
    video = F.interpolate(video, size=size, mode='bilinear', align_corners=False)[None]

    return video, video_frames

def integrate_field(v, num_steps=10):
    """
    Forward integrate the velocity field v(x,t) over [0,1] in small timesteps.
    For simplicity, we treat v as constant in time (Euler integration in time).
    v.shape = (2, H, W)

    Returns a deformation field phi, where
        phi[0] = x-coordinates of the final transformation
        phi[1] = y-coordinates of the final transformation
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
        vx = map_coordinates(v[0], [phi_y, phi_x], order=1, mode='nearest')
        vy = map_coordinates(v[1], [phi_y, phi_x], order=1, mode='nearest')
        phi_x += dt * vx
        phi_y += dt * vy

    return np.stack([phi_x, phi_y], axis=0)

def register_lddmm(fixed, moving):
    
    lddmm = torch_lddmm.LDDMM(template=[moving],
                          target=[fixed],
                          #outdir='../notebook/',
                          do_affine=0,
                          do_lddmm=1,
                          a=7,
                          nt=5,
                          niter=200,
                          epsilon=4e0,
                          sigma=20.0,
                          sigmaR=40.0,
                          optimizer='gdr',
                          gpu_number=0,
                          verbose=False,
                      dx=None)
    
    

    result = lddmm.run()
    return lddmm.outputDeformedTemplate()[0]

def lddmm_registration(frames):
    lddmm_registrations = [frames[0]]
    for i in range(1,len(frames)):
        reg = register_lddmm(frames[0], frames[i])
        lddmm_registrations.append(reg)
    
    return lddmm_registrations

def save_and_display_video(array, filename='output.mp4', fps=30):

    if len(array.shape) >= 3:
        array = array[..., 0]  # Assuming the last dimension is color channels, take the first channel
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
    result = lddmm_registration(frames)
    save_and_display_video(np.array(result), f'output/{filename}_lddm_registered.mp4', fps=15)


