# CoReg

**CoReg** is a motion correction and image registration toolkit for fluorescence imaging data. It combines point-based tracking (CoTracker) with sparse-to-dense warping, and provides a graphical interface.

---

## Repository structure

```
src/CoReg/
    gui.py                    # GUI widgets and background worker
    controller.py             # Application logic and point management
    utils/
        mc_cotracker.py       # CoTracker tracking and warping
        presets.py            # Point-generation presets
        ants_wrapper.py       # ANTs registration
        lddmm_register.py     # LDDMM implementation
        lddmm_wrapper.py      # LDDMM wrapper
        normcorre_wrapper.py  # NoRMCorre wrapper
        pyflowreg_wrapper.py  # FlowReg wrapper
        cache_utils.py        # Result caching and comparison
        utils.py              # Data loading and image processing
requirements.txt
pyproject.toml
```

---

## Installation

**Requirements:** Python 3.10+, a CUDA-capable GPU (recommended).

1. Clone the repository:
    ```bash
    git clone https://github.com/neuluna/registration_gui.git
    cd registration_gui
    ```

2. Install PyTorch with CUDA support. Use a build that matches your driver:
    ```bash
    # CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # CUDA 12.4
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
    Verify GPU is detected:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

3. Install CoReg:
    ```bash
    pip install -e .
    ```

---

## GUI

```bash
python -m gui
```

**Workflow:**

1. Load a file (drag-and-drop or file browser). Supported formats: `.avi`, `.mp4`, `.tif`, `.tiff`, `.h5`, `.hdf5`.
2. Select a point-generation preset and adjust parameters.
3. Click **Registration** to run motion correction.
4. Review the result in the preview panel and export.

**Point presets:**

| Preset | Description |
|---|---|
| None | No seed points |
| Regular grid | Uniform rows x cols grid |
| Features (Shi-Tomasi) | Corner-based feature detection |
| Activity map | Points in high-motion regions |
| Segmentation centroids | Centroids of connected components |
| MatchAnything | Keypoint matching via MatchAnything model |

**Registration methods:** CoTracker, ANTs, LDDMM, NoRMCorre, FlowReg. Each method exposes a parameter dialog via the settings button.

Tracked points and registered video can be saved as `.h5`.

---

## Acknowledgements

- [CoTracker](https://github.com/facebookresearch/co-tracker) by Meta Research
- [ANTsPy](https://github.com/ANTsX/ANTsPy)
- [NoRMCorre](https://github.com/flatironinstitute/NoRMCorre)
