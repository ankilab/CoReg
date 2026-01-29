# registration_gui

Motion-correction and image registration toolkit with a PyQt6 GUI. It provides three registration backends you can run from the GUI: CoTracker-based tracking + warping, ANTs (SyN), and Torch-LDDMM.

---

## Overview

- CoTracker tracking + scattered warping for fast motion correction.
- ANTs-based pairwise SyN registration (optional dependency).
- Torch-LDDMM registration (research-style, CPU/GPU via PyTorch).
- Interactive GUI (PyQt6 + pyqtgraph) for loading videos, editing points, running registration, and previewing results.

---

## Folder Structure

```
gui.py
README.md
input/
utils/
	ants_register.py
	lddmm_register.py
	mc_cotracker.py
	torch_lddmm.py
	utils.py
cotracker/
	... (bundled CoTracker code and helpers)
```

- Top-level GUI: start the app from gui.py.
- Registration backends: utilities and scripts live under utils/.
- CoTracker: vendorized sources live under cotracker/.
- Sample/input files: place your data in input/ (optional).

---

## Installation

Use a fresh virtual environment. Install PyTorch per your platform/CUDA, then common packages. Example on Windows (cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
REM Install PyTorch matching your CUDA/CPU: https://pytorch.org/get-started/locally/
pip install numpy opencv-python pyqt6 pyqtgraph scikit-image tifffile h5py hdf5plugin
REM Optional backends:
pip install antspyx  REM for ANTs support in the GUI
```

Notes:
- PyTorch is required (GPU recommended, falls back to CPU if unavailable).
- ANTs support is optional; antspyx wheels vary by platform—use conda/mamba if pip wheels are not available for your system.

---

## Running the GUI

From the repository root:

```cmd
python gui.py
```

Supported inputs: .avi, .mp4, .tiff/.tif, .h5 (HDF5)

Basic flow:
- Load File: select your video/stack/HDF5.
- Choose Method: CoTracker (default), ANTs, or LDDMM.
- Grid & Points: adjust grid density and optionally edit/add points on the first frame.
- Run Registration: executes the selected method and previews the warped result.
- Save Results: export warped video to MP4/TIFF/HDF5; tracked points are also saved as .npy and .csv next to the chosen output path.

Performance tips:
- For CoTracker, a GPU greatly improves speed. Large frames can require substantial VRAM—consider resizing or loading fewer frames.
- HDF5 reading may need hdf5plugin depending on your file compression.

---

## Examples

Minimal programmatic CoTracker run (headless):

```python
import torch
import numpy as np
from utils.utils import load_video, save_and_display_video
from cotracker.models.core.model_utils import get_points_on_a_grid
from utils.mc_cotracker import cotrack_model, warping

# 1) Load a video as (T, C, H, W)
video = load_video("input/sample.mp4", resize=True, size=(256, 256), max_frames=100)

# 2) Build grid points for queries, and prepend time index zeros
grid_xy = get_points_on_a_grid(8, video.shape[2:])  # 8x8 grid
t_q = torch.zeros((*grid_xy.shape[:-1], 1), device=grid_xy.device, dtype=grid_xy.dtype)
queries = torch.cat([t_q, grid_xy], dim=-1)        # (1, N, 3) with t,x,y

# 3) Track and warp
pred_tracks, pred_visibility = cotrack_model(video, queries)
warped_video, disp_field = warping(pred_tracks, video, smoothing=3.0)

# 4) Save result
save_and_display_video(np.asarray(warped_video), "output_registered.mp4", fps=30)
```

Install dependencies:

```cmd
pip install -r requirements.txt
REM Optional (for ANTs in GUI):
pip install antspyx
```

Use the GUI for ANTs and LDDMM methods by selecting the method in the dropdown and clicking Run Registration.

---

## Backend Scripts (utils/)

- utils/mc_cotracker.py: CoTracker model wrapper, tracking and warping helpers.
- utils/ants_register.py: ANTs SyN pairwise registration (requires antspyx).
- utils/lddmm_register.py + utils/torch_lddmm.py: LDDMM driver and implementation (PyTorch).
- utils/utils.py: Video I/O helpers (read MP4/TIFF/HDF5), saving to MP4/TIFF/HDF5, SSIM utilities.

While the GUI is the primary entry point, you can also import these modules in your own scripts.

---

## Troubleshooting

- PyTorch + CUDA: install a build matching your GPU drivers; otherwise the app runs on CPU.
- ANTs on Windows: if pip install fails, try conda-forge’s antspyx package.
- Video loading: if a file fails to load, try converting to a standard codec or use TIFF/HDF5.

---

## Acknowledgements

- Co-Tracker by Meta Research: https://github.com/facebookresearch/co-tracker
- This repo bundles/adapts CoTracker components for local use.



