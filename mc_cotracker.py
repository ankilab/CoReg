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

    return video[:10]

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
        grid_size=grid_size,
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



def warping(predicted_tracks, frames, smoothing=3,
            random=True, predicted=False, callback=None):
    """
    predicted_tracks: (B, T, G, 2)
    frames: torch or np, shape (T, C, H, W) or (T, H, W, C)
    callback: function(float), optional callback for progress (0..1)

    Returns:
        warpeds: (T, H, W) float32 (channel-0)
        phis:    (T-1, H, W, 2) float32  displacement field (dx,dy) from frame0 -> frame i
    """
    # ---- to numpy ----
    if torch.is_tensor(predicted_tracks):
        predicted_tracks = predicted_tracks.detach().cpu().numpy()
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()

    tracks = np.asarray(predicted_tracks)
    frames = np.asarray(frames)

    # ---- normalize frames shape ----
    if frames.ndim == 4 and frames.shape[1] in (1, 3):   # (T,C,H,W)
        T_orig, C, H, W = frames.shape
        frame0_gray = frames[0, 0].astype(np.float32)
        get_frame_gray = lambda i: frames[i, 0].astype(np.float32)
    elif frames.ndim == 4 and frames.shape[-1] in (1, 3): # (T,H,W,C)
        T_orig, H, W, C = frames.shape
        frame0_gray = frames[0, ..., 0].astype(np.float32)
        get_frame_gray = lambda i: frames[i, ..., 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported frames shape: {frames.shape}")

    # ---- input range (gray channel 0) ----
    if frames.ndim == 4 and frames.shape[1] in (1, 3):          # (T,C,H,W)
        gray_stack = frames[:, 0].astype(np.float32, copy=False)
    else:                                                       # (T,H,W,C)
        gray_stack = frames[..., 0].astype(np.float32, copy=False)

    in_min = float(np.nanmin(gray_stack))
    in_max = float(np.nanmax(gray_stack))
    do_clip = np.isfinite(in_min) and np.isfinite(in_max) and (in_max >= in_min)

    # clip the reference frame too (optional but keeps everything consistent)
    if do_clip:
        frame0_gray = np.clip(frame0_gray, in_min, in_max).astype(np.float32, copy=False)

    B, T, G, D = tracks.shape
    if D != 2:
        raise ValueError(f"predicted_tracks last dim must be 2, got {D}")
    grid_size = int(math.sqrt(G))

    if T_orig > T:
        raise ValueError(f"frames has {T_orig} frames but predicted_tracks has only {T} timesteps")

    sigma = float(smoothing) if smoothing is not None else 0.0
    do_smooth = sigma > 0.0

    def smooth_phi(phi_2hw: np.ndarray) -> np.ndarray:
        """phi_2hw: (2,H,W) -> smoothed (2,H,W)"""
        if not do_smooth:
            return phi_2hw.astype(np.float32, copy=False)
        out = np.empty_like(phi_2hw, dtype=np.float32)
        out[0] = cv2.GaussianBlur(phi_2hw[0].astype(np.float32), (0, 0), sigma)
        out[1] = cv2.GaussianBlur(phi_2hw[1].astype(np.float32), (0, 0), sigma)
        return out

    # ---- build dense displacement field vp: (T,2,H,W), displacement from t=0 ----
    if random:
        sparse_x = tracks[0, :, :, 0]  # (T, G)
        sparse_y = tracks[0, :, :, 1]  # (T, G)
        disp_sparse = tracks[0] - tracks[0, 0]  # (T, G, 2)

        X, Y = np.meshgrid(np.arange(W), np.arange(H))  # (H,W)
        dx_dense = np.zeros((T, H, W), dtype=np.float32)
        dy_dense = np.zeros((T, H, W), dtype=np.float32)

        for t in range(T):
            dx_dense[t] = griddata(
                (sparse_x[0], sparse_y[0]), # Use frame 0 coordinates as reference
                disp_sparse[t, :, 0],
                (X, Y),
                method="linear",
                fill_value=0
            ).astype(np.float32)

            dy_dense[t] = griddata(
                (sparse_x[0], sparse_y[0]),
                disp_sparse[t, :, 1],
                (X, Y),
                method="linear",
                fill_value=0
            ).astype(np.float32)

        vp = np.stack((dx_dense, dy_dense), axis=1)  # (T,2,H,W)

    elif predicted:
        v = tracks[0]
        if v.ndim == 4 and v.shape[1] == 2:
            vp = (v - v[0]).astype(np.float32)  # (T,2,H,W)
        else:
            v = v.reshape(T, grid_size, grid_size, 2)
            v = (v - v[0]).transpose(0, 3, 1, 2)                      # (T,2,gs,gs)
            vp = zoom(v, (1, 1, H / grid_size, W / grid_size)).astype(np.float32)
    else:
        v = tracks[0].reshape(T, grid_size, grid_size, 2)
        v = (v - v[0]).transpose(0, 3, 1, 2)                          # (T,2,gs,gs)
        vp = zoom(v, (1, 1, H / grid_size, W / grid_size)).astype(np.float32)  # (T,2,H,W)

    grid_x0, grid_y0 = np.meshgrid(np.arange(W, dtype=np.float32),
                                   np.arange(H, dtype=np.float32))

    warpeds = [frame0_gray]
    phis = []
    phis.append(np.zeros((H, W, 2), dtype=np.float32))

    for i in range(1, T_orig):
        
        if callback is not None:
             callback(i / T_orig)

        phi = vp[i]            # (2,H,W) raw displacement from t=0
        phi = smooth_phi(phi)  # <-- KEY CHANGE: smooth DF inside warping()

        phis.append(np.stack([phi[0], phi[1]], axis=-1))  # (H,W,2) (smoothed)

        sample_x = grid_x0 + phi[0]
        sample_y = grid_y0 + phi[1]

        src = get_frame_gray(i).astype(np.float32, copy=False)

        # mild anti-aliasing blur (tune sigma: 0.5 to 1.5 typically)
        src = cv2.GaussianBlur(src, (0, 0), 0.8)

        warped = map_coordinates(
            src,
            [sample_y, sample_x],
            order=3,
            mode="reflect"   
        ).astype(np.float32)

        if do_clip:
            warped = np.clip(warped, in_min, in_max, out=warped)

        warpeds.append(warped)

    return np.asarray(warpeds, dtype=np.float32), np.asarray(phis, dtype=np.float32)



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
    
    #define the video file pat
    video_file = r"C:\Users\luisa\Desktop\drosophila\Drosophila_Ca_bad.avi"
    filename = (video_file.split('/')[-1].split('.')[0])
    if not os.path.exists(video_file):
        print(f"Video file {video_file} does not exist.")
    else:
        print(f"Video file {video_file} exists.")

    # Apply motion correction algorithm by
    # loading the video, running the CoTracker model, and warping the frames

    vid = load_video(video_file, resize=False)

    queries = get_points_on_a_grid(5, vid.shape[2:])
    xy = queries  # (1,16,2)
    t = torch.zeros((*xy.shape[:-1], 1), device=xy.device, dtype=xy.dtype)      # (1,16,1)
    txy = torch.cat([t, xy], dim=-1)  # (1,16,3)
    pred_tracks = cotrack_model(vid, txy, grid_size=4)
    result,velo = warping(pred_tracks[0].cpu().numpy(), np.array(vid))
    
    save_and_display_video(np.array(result), f'output/registered.mp4', fps=15)
 
