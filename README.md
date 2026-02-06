# CoReg

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CoReg** is a high-performance motion-correction and image registration toolkit powered by [CoTracker](https://github.com/facebookresearch/co-tracker). It combines advanced point tracking with scattered warping to provide fast and accurate motion correction for various imaging applications.

---

## Overview

- CoTracker tracking + scattered warping for fast motion correction.
- Interactive GUI (PyQt6 + pyqtgraph) for loading videos, editing points, running registration, and previewing results.

---

##  Repository Structure

```text
├── src/CoReg/                  # Main application package
│   ├── gui.py                  # GUI entry point and UI layout
│   ├── controller.py           # Core logic and process management
│   ├── cotracker/              # CoTracker model architecture
│   └── utils/                  # Registration algorithms and utilities
│       ├── ants_register.py    # ANTs-based registration
│       ├── lddmm_register.py   # LDDMM implementation
│       ├── mc_cotracker.py     # CoTracker motion correction
│       └── utils.py            # Data loading and image processing
├── requirements.txt            # Project dependencies
└── pyproject.toml              # Build and installation configuration
```



## 🚀 Installation

### Prerequisites

- Python 3.10 or higher.
- A virtual environment (recommended).

### Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/neuluna/registration_gui.git
    cd registration_gui
    ```

2.  **Install dependencies**:
    It is highly recommended to install PyTorch with CUDA support for optimal performance.
    ```bash
    # Example for Windows with CUDA 12.6
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```

3.  **Install CoReg**:
    ```bash
    pip install -e .
    ```

Notes:
- Check to install torch is with cuda support --> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126.
---

## Running the GUI

```cmd
python -m gui
```


### Basic Workflow

1.  **Load Data**: Drag and drop or browse to load your file (`.avi`, `.mp4`, `.tiff`, `.h5`).
2.  **Configure Points**: Adjust grid density or switch to manual mode to edit points on the first frame.
3.  **Process**: Click "Registration" to start the motion correction process.
4.  **Preview & Save**: Review the warped results and export them to your desired format.

---

## 📂 Supported Formats

-   `.avi`, `.mp4`, `.tiff`, `.tif`, `.h5`

> [!NOTE]
> Tracked points and Video can be saved as `.h5`.

---

## 💡 Performance Tips

-   **GPU Acceleration**: A CUDA-enabled GPU is strongly recommended for CoTracker.

---

##  Acknowledgements

-   **CoTracker** by Meta Research: [GitHub Repository](https://github.com/facebookresearch/co-tracker)
-   This project adapts CoTracker components for efficient local motion correction workflows.
