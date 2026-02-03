# registration_gui

Motion-correction and image registration toolkit, which is CoTracker based tracking.

---

## Overview

- CoTracker tracking + scattered warping for fast motion correction.
- Interactive GUI (PyQt6 + pyqtgraph) for loading videos, editing points, running registration, and previewing results.

---


## Installation

Clone the repository. Use a fresh virtual environment. Install python >= 3.10, PyTorch per your platform/CUDA, then common packages. Example on Windows (cmd):

```cmd
git clone https://github.com/neuluna/registration_gui

pip install -e .
```

Notes:
- Check to install torch is with cuda support --> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126.
---

## Running the GUI

```cmd
python -m gui
```

Supported inputs: .avi, .mp4, .tiff/.tif, .h5 (HDF5)

Basic flow:
- Load File: select your video/stack/HDF5 (--> drag and drop).
- Grid & Points or totally manual: adjust grid density and optionally edit/add points on the first frame.
- Run Registration: executes the selected method and previews the warped result.
- Save Results: export warped video to MP4/TIFF/HDF5; tracked points are also saved as .npy and .csv next to the chosen output path.

Performance tips:
- For CoTracker, a GPU greatly improves speed.


---

## Acknowledgements

- Co-Tracker by Meta Research: https://github.com/facebookresearch/co-tracker
- This repo bundles/adapts CoTracker components for local use.



