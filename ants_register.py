import os
import numpy as np
from cotracker.utils.visualizer import read_video_from_path
import cv2
import torch
import torch.nn.functional as F
import ants


#tried to clear memory because of OOM errors
import gc
torch.cuda.empty_cache()
gc.collect()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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

def ants_registration(frames):
    # Load example images
    results = []

    for i in range(len(frames)-1):
        fixed_image = ants.from_numpy(frames[0][...,0]) #.transpose(1,0)
        moving_image = ants.from_numpy(frames[i+1][...,0]) #.transpose(1,0)
        # Perform registration
        registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyNOnly") #,  reg_iterations = [100,1000,20])
        # Transform the moving image
        warped_image = registration_result['warpedmovout'].numpy()
        results.append(np.array(warped_image))
    
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


