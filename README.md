# registration_gui

Registration GUI is a small collection of Python tools for video motion-correction and image registration. This repository contains three main approaches to register a video sequence to a single reference frame:

- CoTracker-based motion tracking and scattered warping (fast learned trajectories): `mc_cotracker.py` + interactive GUI in `gui.py`.
- ANTs-based pairwise SyN registration: `ants_register.py` (ANTsPy wrapper).
- Torch-LDDMM geometric registration: `lddmm_register.py` + `torch_lddmm.py` (LDDMM implementation).

The repo also bundles utilities and light tests for functionality verification.

---

## Features ✅

- Motion correction via dense tracked points from a learned CoTracker model and scattered interpolation warping.
- Classic diffeomorphic registration via LDDMM and ANTs SyN algorithms.
- Interactive GUI (`wxPython`) to preview, edit seed points and run motion-correction with a visual output and playback controls.
- Utilities to load common video formats: AVI/MP4, TIFF, HDF5.
- Tests to check sampling and basic GUI wiring.

---

## Getting Started — Install 🔧

These are the main dependencies used across the scripts. Exact versions are not pinned in this repo, so using a new virtual environment is recommended.

Requirements (minimal):

- Python 3.8+
- numpy
- scipy
- matplotlib
- opencv-python
- torch (PyTorch) — with CUDA if you want GPU acceleration
- wxPython (for GUI)
- ants (ANTsPy) — optional for `ants_register.py`
- scikit-image
- tifffile
- h5py (+ hdf5plugin if needed)
- nibabel (for some LDDMM utilities)

Example Windows setup (using cmd):

```cmd
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python torch torchvision torchaudio wxPython scikit-image tifffile h5py nibabel ants
echo "If torch+CUDA is needed, follow official PyTorch installation instructions for CUDA-aware builds."
```

Note: `ants` is a wrapper for ANTsPy. On Windows you may need to install ANTs via conda or build instructions from the ANTsPy repository.

---

## How the scripts work — Quick overview 💡

- `mc_cotracker.py` — Loads a video, runs the CoTracker model to predict tracks for a uniform grid (and optional custom points), and warps frames using a scattered interpolation (griddata + map_coordinates) to produce a registered output.
- `gui.py` — A wxPython GUI that allows interactive point editing, loading videos, running the CoTracker-based registration, and visualizing results.
- `ants_register.py` — Uses ANTsPy to perform pairwise SyN registration of every frame to the first (reference) frame.
- `lddmm_register.py` + `torch_lddmm.py` — Performs LDDMM-based registration and helpers; `torch_lddmm.py` contains the implementation.
- `utils/utils.py` — Convenience helpers for loading/saving videos and reading TIFF/HDF5.

---

## Usage — Examples 🧭

The scripts often hardcode a `video_file` path under the `__main__` block. You can edit the script to use a custom path, or import functions and call them from other scripts.

Example: Run the CoTracker-based script directly (headless):

```cmd
python mc_cotracker.py
```

Or run the interactive GUI (requires `wxPython`):

```cmd
python gui.py
```

Example: Run ANTs pairwise registration script:

```cmd
python ants_register.py
```

Example: Run LDDMM registration (may need GPU and `torch_lddmm` compiled or available):

```cmd
python lddmm_register.py
```

Output: registered video files are written to the `output/` directory by default (e.g. `<filename>_registered.mp4`).

---

## GUI Usage (Motion Correction GUI) 🎛️

- Click `Load File` to choose a video (`.avi`, `.mp4`, `.tiff`, `.h5`).
- Optionally change `Grid size` and `Apply Grid` to use a coarser or denser seed grid.
- Toggle `Edit Points` to add/delete/move custom seed points in the first frame.
- Click `Run Registration` to start registration (runs CoTracker & warping).
- Playback controls (left/right stepping, play/pause, slider) let you review results.

Tips:
- Toggle `DEFAULT_DEVICE` in `mc_cotracker.py` or set `CUDA_VISIBLE_DEVICES` to choose the GPU.
- If video loading fails, try reducing frame count or resizing.

---

## Tests 🧪

Run the unit tests with Python's unittest (from repo root):

```cmd
python -m unittest discover -s tests
```

The test suite contains small checks for bilinear sampling and GUI import logic.

---

## Troubleshooting & Notes ⚠️

- Large videos and GPU memory: CoTracker processing requires significant GPU RAM for high resolutions — downsample or reduce frames for debugging.
- ANTs requires additional system dependencies to build/run; using conda often reduces friction.
- If `torch.hub` fails to load the CoTracker model, ensure you have network access and correct `TORCH_HOME` if caching to a custom path.
- The LDDMM implementation (`torch_lddmm.py`) is a heavy, research-style implementation; performance tuning and configuration is typically necessary.

---

## Acknowledgements & License 📜

- This repository uses or is inspired by Facebook/Meta Research's Co-Tracker: https://github.com/facebookresearch/co-tracker
- Some code and tests contain references to Meta's code licensing.

No explicit license is provided in this repository; please add a `LICENSE` file if you intend to make the usage terms clear.



