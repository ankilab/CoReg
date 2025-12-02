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
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter


#tried to clear memory because of OOM errors
import gc
torch.cuda.empty_cache()
gc.collect()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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

def cotrack_model(video, queries, grid_size=0):
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
    model.model.model_resolution = video.shape[2:]

    pred_tracks, pred_visibility = model(
        video.unsqueeze(0).to(DEFAULT_DEVICE),
        queries=queries.to(DEFAULT_DEVICE),
        grid_size=0,
        grid_query_frame=0,      # ignored when queries is provided, but fine to keep
        backward_tracking=False
    )

    return pred_tracks, pred_visibility


### uniform grid warping
# def warping(predicted_tracks, frames):      
#     """
#     Warps the frames of a video based on predicted tracks.
#     args:       
#         predicted_tracks (torch.Tensor): Predicted tracks of shape (B, T, G, D).
#         frames (torch.Tensor): Video frames of shape (T_orig, H, W, C).
#     returns:
#         warpeds (list): List of warped frames.
#     """
#     frames = frames.transpose(0,2,3,1) 
#     T_orig, H, W, C = frames.shape
#     B, T, G, D = predicted_tracks.shape
#     print("predicted_tracks.shape", predicted_tracks.shape)
#     grid_size = int(math.sqrt(G))

#     velocity = predicted_tracks[0].reshape(T, grid_size, grid_size, 2) #(24, 32, 32, 2)
#     real_velocity = velocity-velocity[0] # (24, 32, 32, 2)
#     v = real_velocity.transpose(0, 3, 1, 2) # (24, 2, 32, 32)
#     vp = zoom(v, (1, 1, W/grid_size, H/grid_size))  #(24, 2, 256, 256) 
        
#     warpeds = [frames[0][...,0]]

#     for i in range(1,T_orig):
#         grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
#         grid_x = grid_x.astype(np.float32)
#         grid_y = grid_y.astype(np.float32)

#         phi = np.diff(vp,axis=0)[0:i].sum(0).transpose(0,2,1)
#         grid_x += phi[0]
#         grid_y += phi[1]
  
#         warped = map_coordinates(frames[i][...,0].astype(np.float32), [grid_y, grid_x], order=3, mode='nearest')
#         warpeds.append(warped)
       
#     return np.array(warpeds)



def warping(predicted_tracks, frames):
    """
    Returns:
        warpeds: (T,H,W) float32
        phis:    (T,H,W,2) float32  where phis[t,...] = (dx,dy) mapping ref->frame_t
    """
    if "torch" in str(type(predicted_tracks)):
        predicted_tracks = predicted_tracks.detach().cpu().numpy()
    if "torch" in str(type(frames)):
        frames = frames.detach().cpu().numpy()

    tracks = np.asarray(predicted_tracks)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"predicted_tracks must be (B,T,N,2), got {tracks.shape}")

    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[1] in (1, 3):  # (T,C,H,W)
        frames = frames.transpose(0, 2, 3, 1)            # (T,H,W,C)
    elif frames.ndim == 3:                               # (T,H,W)
        pass
    elif frames.ndim == 4 and frames.shape[-1] in (1, 3):# (T,H,W,C)
        pass
    else:
        raise ValueError(f"Unsupported frames shape: {frames.shape}")

    T_orig, H, W = frames.shape[0], frames.shape[1], frames.shape[2]

    B, T, N, _ = tracks.shape
    if B != 1:
        raise ValueError("This implementation assumes B=1")
    if T_orig > T:
        raise ValueError(f"frames has {T_orig} frames but predicted_tracks has only {T} timesteps")

    p_ref_all = tracks[0, 0].astype(np.float32)  # (N,2)

    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))

    # store channel-0 if RGB
    first = frames[0][..., 0] if frames.ndim == 4 else frames[0]
    warpeds = [first.astype(np.float32)]
    #phis = [np.zeros((H, W, 2), dtype=np.float32)]
    phis = []
    velo = []

    sparse_x = tracks[0, :, :, 0]  
    sparse_y = tracks[0, :, :, 1]  
    displacement = tracks[0] - tracks[0, 0]  # (T, G, 2)
    X, Y = np.meshgrid(W,H)
    dx_dense = np.zeros((T, H, W))
    dy_dense = np.zeros((T, H, W))
    for t in range(T):
        dx_dense[t] = griddata(
            (sparse_x[t], sparse_y[t]), displacement[t, :, 0], (X, Y), method='nearest', fill_value=0
        )    
        dy_dense[t] = griddata(
            (sparse_x[t], sparse_y[t]), displacement[t, :, 1], (X, Y), method='nearest', fill_value=0
        )
    # Stack dx and dy to calculate v
    vp = np.stack((dx_dense, dy_dense), axis=1)
    print(vp.shape)

    for i in range(1, T_orig):
        p_src_all = tracks[0, i].astype(np.float32)

        good = np.isfinite(p_ref_all).all(axis=1) & np.isfinite(p_src_all).all(axis=1)
        p_ref = p_ref_all[good]
        p_src = p_src_all[good]

        if p_ref.shape[0] < 3:
            src = frames[i][..., 0] if frames.ndim == 4 else frames[i]
            warpeds.append(src.astype(np.float32))
            #phis.append(phis[-1].copy())
            continue

        disp = (p_src - p_ref).astype(np.float32)  # dx,dy at ref points

        dx = griddata(p_ref, disp[:, 0], (grid_x, grid_y), method="linear",  fill_value=0)#fill_value=np.nan)
        dy = griddata(p_ref, disp[:, 1], (grid_x, grid_y), method="linear",  fill_value=0)#fill_value=np.nan)
        
        phi = np.diff(vp,axis=0)[0:i].sum(0)      
        phis.append(phi)

        u = vp[i, 0]  # dx
        v = vp[i, 0]  # dy
        velo.append(np.sqrt(u**2 + v**2))

        nan = np.isnan(dx) | np.isnan(dy)
        if np.any(nan):
            dx_nn = griddata(p_ref, disp[:, 0], (grid_x, grid_y), method="nearest")
            dy_nn = griddata(p_ref, disp[:, 1], (grid_x, grid_y), method="nearest")
            dx[nan] = dx_nn[nan]
            dy[nan] = dy_nn[nan]

        sample_x = grid_x + dx
        sample_y = grid_y + dy

        src = frames[i][..., 0] if frames.ndim == 4 else frames[i]
        warped = map_coordinates(
            src.astype(np.float32),
            [sample_y, sample_x],
            order=3,
            mode="nearest",
        )
        warpeds.append(warped.astype(np.float32))
    print(np.array(velo).shape)

    return np.asarray(warpeds, dtype=np.float32), np.array(velo)





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
    video_file = "/home/rzlin/va11moje/bimap_registration/input/Drosophila_Ca_bad.avi"
    filename = (video_file.split('/')[-1].split('.')[0])
    if not os.path.exists(video_file):
        print(f"Video file {video_file} does not exist.")
    else:
        print(f"Video file {video_file} exists.")

    # Apply motion correction algorithm by
    # loading the video, running the CoTracker model, and warping the frames

    vid = load_video(video_file, resize=False)
    pred_tracks = cotrack_model(vid, grid_size=25)
    result = warping(pred_tracks[0].cpu().numpy(), np.array(vid))
    save_and_display_video(np.array(result), f'output/{filename}_registered.mp4', fps=15)