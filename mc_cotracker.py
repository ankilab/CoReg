import os
import numpy as np
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.models.core.model_utils import get_points_on_a_grid
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import math
from IPython.display import Video, display


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

def cotrack_model(video, grid_size):
    """"Run the CoTracker model on the provided video and grid points.
    Model defines which cotracker model is used for tracking.
    args:
        video (torch.Tensor): A tensor containing the video frames.
        grid_size (int): The size of the grid for tracking.
        points_x (list): List of x-coordinates for reference points.
        points_y (list): List of y-coordinates for reference points.
    returns:
        pred_tracks (torch.Tensor): Predicted tracks from the model.
        pred_visibility (torch.Tensor): Predicted visibility from the model.
        grid_pts (torch.Tensor): Points on the grid used for tracking.
    """

    import os
    os.environ["TORCH_HOME"] = "/data/users/va11moje/cache/cotracker"

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda")
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    model.model.model_resolution = video.shape[3:]
    grid_pts = get_points_on_a_grid(
                grid_size, model.model.model_resolution
            )

    pred_tracks, pred_visibility = model(
        video,
        grid_size=grid_size,
        grid_query_frame=0,
        backward_tracking=False
    )
    return pred_tracks, pred_visibility, grid_pts

def warping(predicted_tracks, frames):      
    """
    Warps the frames of a video based on predicted tracks.
    args:       
        predicted_tracks (torch.Tensor): Predicted tracks of shape (B, T, G, D).
        frames (torch.Tensor): Video frames of shape (T_orig, H, W, C).
    returns:
        warpeds (list): List of warped frames.
    """
    T_orig, H, W, C = frames.shape
    B, T, G, D = predicted_tracks.shape
    grid_size = int(math.sqrt(G))

    velocity = predicted_tracks[0].reshape(T, grid_size, grid_size, 2) #(24, 32, 32, 2)
    real_velocity = velocity-velocity[0] # (24, 32, 32, 2)
    v = real_velocity.transpose(0, 3, 1, 2) # (24, 2, 32, 32)
    vp = zoom(v, (1, 1, W/grid_size, H/grid_size))  #(24, 2, 256, 256) 
        
    warpeds = [frames[0][...,0]]

    for i in range(1,T_orig):
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        phi = np.diff(vp,axis=0)[0:i].sum(0)
        grid_x += phi[0]
        grid_y += phi[1]
  
        warped = map_coordinates(frames[i][...,0].astype(np.float32), [grid_y, grid_x], order=3, mode='nearest')
        warpeds.append(warped)
       
    return np.array(warpeds)


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

    vid = load_video(video_file, size=(256, 256))
    pred_tracks = cotrack_model(vid[0], grid_size=20)
    result = warping(pred_tracks[0].cpu().numpy(), np.array(vid[1]))
    save_and_display_video(np.array(result), f'output/{filename}_registered.mp4', fps=15)