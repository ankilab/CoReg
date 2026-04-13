# CoReg

**CoReg** is a motion correction and image registration toolkit for fluorescence imaging data. It combines point-based tracking (CoTracker) with sparse-to-dense warping, and provides a graphical interface as well as a command-line interface for batch processing.

---

## Repository structure

```
src/CoReg/
    gui.py                    # GUI widgets and background worker
    controller.py             # Application logic and point management
    cli.py                    # Command-line interface
    utils/
        mc_cotracker.py       # CoTracker tracking and warping
        presets.py            # Point-generation presets (shared by GUI and CLI)
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

## CLI

After installation, `coreg-cli` is available on the path.

```bash
coreg-cli INPUT [options]
```

`INPUT` is either a single file or a directory. For directories, all supported files are processed and outputs are written to `<input>/registered/` by default.

### General options

| Option | Default | Description |
|---|---|---|
| `--method` | `cotracker` | Registration method: `cotracker`, `ants`, `lddmm`, `normcorre`, `flowreg` |
| `-o / --output` | auto | Output path; extension sets format (`.tif`, `.h5`, `.mp4`, ...) |
| `--output-dir` | `<input>/registered` | Output directory when INPUT is a folder |
| `--width W --height H` | original | Resize frames before processing |
| `--max-frames N` | all | Maximum frames to load |
| `--overwrite` | off | Overwrite existing output files |
| `--metrics-csv` | auto | Path for the metrics CSV (SSIM, MSE, ZNCC, MI) |

### Sequence sampling (folder input)

| Option | Description |
|---|---|
| `--seq-frames N` | Split each video into sub-sequences of N frames |
| `--seq-stride S` | Stride between sequence starts (default: non-overlapping) |
| `--num-seqs K` | Sample K random sequences per video (requires `--seq-frames`) |
| `--seed` | Random seed for reproducible sequence sampling |

### CoTracker options

| Option | Default | Description |
|---|---|---|
| `--cotracker-presets` | `regular_grid` | Comma-separated list of point presets to run |
| `--grid-rows` | `8` | Grid rows (used by `regular_grid` preset) |
| `--grid-cols` | `8` | Grid columns (used by `regular_grid` preset) |
| `--num-points` | `rows x cols` | Max seed points for non-grid presets |
| `--df-sigma` | `5.0` | Gaussian smoothing sigma for the displacement field |

**Available presets:** `regular_grid`, `features`, `activity`, `segmentation`, `matchanything`

When multiple presets are given, each is run as a separate pass and results are saved with the preset name appended. The metrics CSV is named accordingly (e.g. `metrics_cotracker_features.csv`).

### Examples

```bash
# Single file, default settings
coreg-cli data/movie.tif

# Shi-Tomasi features, 200 points
coreg-cli data/movie.tif --method cotracker --cotracker-presets features --num-points 200

# Activity preset, resize to 256x256
coreg-cli data/movie.tif --cotracker-presets activity --width 256 --height 256

# Run two presets in one call
coreg-cli data/movie.tif --cotracker-presets regular_grid,features

# Folder input: split into 100-frame sequences, 8x8 grid
coreg-cli data/ --seq-frames 100 --grid-rows 8 --grid-cols 8

# Folder input: sample 10 random sequences per video
coreg-cli data/ --seq-frames 50 --num-seqs 10 --seed 42

# ANTs registration
coreg-cli data/movie.tif --method ants -o data/movie_ants.tif
```

---

## Acknowledgements

- [CoTracker](https://github.com/facebookresearch/co-tracker) by Meta Research
- [ANTsPy](https://github.com/ANTsX/ANTsPy)
- [NoRMCorre](https://github.com/flatironinstitute/NoRMCorre)
