import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from gui import ProcessingWorker
from utils.utils import (
    load_hdf5,
    load_tiff,
    load_video,
    resize_video_tensor_to_wh,
    save_video_hdf5,
    save_video_tiff,
    save_and_display_video,
)


SUPPORTED_VIDEO_EXT = [".avi", ".mp4"]
SUPPORTED_TIFF_EXT = [".tiff", ".tif"]
SUPPORTED_HDF5_EXT = [".h5", ".hdf5"]
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXT + SUPPORTED_TIFF_EXT + SUPPORTED_HDF5_EXT


def _load_video_tensor(
    path: str,
    target_w: int | None = None,
    target_h: int | None = None,
    max_frames: int | None = None,
) -> torch.Tensor:
    """Load input file into a torch tensor shaped (T, C, H, W)."""
    ext = os.path.splitext(path)[1].lower()

    if ext in SUPPORTED_VIDEO_EXT:
        video_t = load_video(path, resize=False, max_frames=max_frames)

    elif ext in SUPPORTED_TIFF_EXT:
        video_t = load_tiff(path, resize=False)
        if max_frames is not None:
            video_t = video_t[: int(max_frames)]

    elif ext in SUPPORTED_HDF5_EXT:
        frames_arg = int(max_frames) if max_frames is not None else None
        video_np = load_hdf5(path, "/data", frames=frames_arg)

        if isinstance(video_np, torch.Tensor):
            video_t = video_np
        else:
            video_t = torch.from_numpy(np.asarray(video_np))

        if video_t.ndim == 3:  # (T,H,W)
            video_t = video_t.unsqueeze(1)

        if video_t.shape[1] == 1:
            # replicate grayscale to 3 channels for models expecting RGB
            video_t = video_t.repeat(1, 3, 1, 1)

        video_t = video_t.float()

    else:
        raise ValueError(f"Unsupported input extension '{ext}'")

    # Enforce torch and (T,C,H,W) for all types
    if not isinstance(video_t, torch.Tensor):
        video_t = torch.from_numpy(np.asarray(video_t))

    if video_t.ndim == 3:  # (T,H,W)
        video_t = video_t.unsqueeze(1)
    elif (
        video_t.ndim == 4
        and video_t.shape[1] not in (1, 3)
        and video_t.shape[-1] in (1, 3)
    ):
        # (T,H,W,C) -> (T,C,H,W)
        video_t = video_t.permute(0, 3, 1, 2)

    if ext in SUPPORTED_TIFF_EXT and max_frames is not None:
        video_t = video_t[: int(max_frames)]

    # Resize if requested
    if target_w is not None and target_h is not None:
        video_t = resize_video_tensor_to_wh(
            video_t, target_w=int(target_w), target_h=int(target_h)
        )

    return video_t.float()


def _create_regular_grid(h: int, w: int, rows: int, cols: int) -> np.ndarray:
    """Create a simple regular grid of (x,y) seed points in image coordinates."""
    rows = max(2, int(rows))
    cols = max(2, int(cols))
    xs = np.linspace(0.5, w - 0.5, cols, dtype=np.float32)
    ys = np.linspace(0.5, h - 0.5, rows, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    return pts.astype(np.float32)


def _run_worker(
    video_t: torch.Tensor,
    method: str,
    *,
    custom_grid_pts: torch.Tensor | None = None,
    df_sigma: float = 5.0,
    ants_params: dict | None = None,
    normcorre_params: dict | None = None,
    pyflowreg_params: dict | None = None,
):
    """Run ProcessingWorker synchronously and return its outputs."""
    result: dict[str, object | None] = {
        "tracks": None,
        "visibility": None,
        "confidence": None,
        "grid": None,
        "warped": None,
        "disp": None,
        "error": None,
    }

    worker = ProcessingWorker(
        video_in=video_t,
        video_out=video_t,
        base_grid_pts=None,
        custom_grid_pts=custom_grid_pts,
        df_sigma=float(df_sigma),
        tracks=None,
        parent=None,
        method=method,
        filter_name=None,
        existing_data=None,
        queries=None,
        ants_params=ants_params,
        normcorre_params=normcorre_params,
        pyflowreg_params=pyflowreg_params,
    )

    def _on_finished(tracks, vis, conf, grid, warped, disp):
        result["tracks"] = tracks
        result["visibility"] = vis
        result["confidence"] = conf
        result["grid"] = grid
        result["warped"] = warped
        result["disp"] = disp

    def _on_error(msg: str):
        result["error"] = msg

    def _on_status(text: str):
        print(text, file=sys.stderr)

    def _on_progress(pct: int):
        print(f"[{pct:3d}%]", file=sys.stderr, end="\r", flush=True)

    worker.finished.connect(_on_finished)
    worker.error.connect(_on_error)
    worker.status_update.connect(_on_status)
    worker.progress.connect(_on_progress)

    # Run synchronously in the current thread
    worker.run()
    print("", file=sys.stderr)  # newline after progress

    if result["error"]:
        raise RuntimeError(f"Processing failed: {result['error']}")

    return result


def _determine_output_path(
    input_path: str, output_arg: str | None, method_slug: str
) -> str:
    """Infer the output path if not explicitly provided."""
    if output_arg:
        return str(Path(output_arg).expanduser())

    in_path = Path(input_path)
    stem = in_path.stem
    return str(in_path.with_name(f"{stem}_{method_slug}.tiff"))


def _save_output(array, output_path: str) -> None:
    """Save the registered video stack based on output extension."""
    arr = np.asarray(array)
    ext = Path(output_path).suffix.lower()

    if ext in (".tif", ".tiff"):
        save_video_tiff(arr, output_path)
    elif ext in (".h5", ".hdf5"):
        save_video_hdf5(arr, output_path)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        save_and_display_video(arr, filename=output_path)
    else:
        raise SystemExit(
            f"Unsupported output extension '{ext}'. "
            "Use .tif/.tiff, .h5/.hdf5, or .mp4/.avi/.mov/.mkv."
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `coreg-cli` command."""
    parser = argparse.ArgumentParser(
        description="CoReg CLI: run motion correction/registration without the GUI.",
    )
    parser.add_argument(
        "input",
        help="Input file (.avi, .mp4, .tif, .tiff, .h5, .hdf5).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output file path. Extension determines format (.tif/.tiff/.h5/.hdf5/.mp4). "
            "Defaults to '<input>_<method>.tiff'."
        ),
    )
    parser.add_argument(
        "--method",
        default="cotracker",
        choices=["cotracker", "ants", "lddmm", "normcorre", "flowreg"],
        help="Registration method to use.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize width in pixels (default: keep original).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize height in pixels (default: keep original).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to load (default: all).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=4,
        help="Number of rows for CoTracker seed grid (cotracker only).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of columns for CoTracker seed grid (cotracker only).",
    )
    parser.add_argument(
        "--df-sigma",
        type=float,
        default=5.0,
        help="Smoothing sigma for CoTracker displacement field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    args = parser.parse_args(argv)

    input_path = str(Path(args.input).expanduser())
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    if (args.width is None) != (args.height is None):
        raise SystemExit(
            "Please provide both --width and --height for resizing, or neither."
        )

    target_w = args.width
    target_h = args.height
    max_frames = args.max_frames

    video_t = _load_video_tensor(
        input_path,
        target_w=target_w,
        target_h=target_h,
        max_frames=max_frames,
    )

    method_map = {
        "cotracker": "CoTracker",
        "ants": "ANTs",
        "lddmm": "LDDMM",
        "normcorre": "NoRMCorre",
        "flowreg": "FlowReg",
    }
    method = method_map[args.method.lower()]

    custom_grid: torch.Tensor | None = None
    ants_params: dict | None = None
    normcorre_params: dict | None = None
    pyflowreg_params: dict | None = None
    df_sigma = float(args.df_sigma)

    if method == "CoTracker":
        _, _, H, W = video_t.shape
        grid_pts = _create_regular_grid(H, W, args.grid_rows, args.grid_cols)
        custom_grid = torch.from_numpy(grid_pts)
    elif method == "ANTs":
        ants_params = {
            "type_of_transform": "SyNOnly",
            "aff_metric": "mattes",
            "syn_metric": "mattes",
            "reg_iterations": [40, 20, 0],
        }
    elif method == "NoRMCorre":
        normcorre_params = {
            "pw_rigid": True,
            "max_shifts": (6, 6),
            "strides": (96, 96),
            "overlaps": (32, 32),
            "upsample_factor_fft": 4,
            "max_deviation_rigid": 3,
        }
    elif method == "FlowReg":
        pyflowreg_params = {
            "alpha": 1.5,
            "min_level": 4,
            "buffer_size": 100,
            "save_w": True,
        }

    result = _run_worker(
        video_t,
        method=method,
        custom_grid_pts=custom_grid,
        df_sigma=df_sigma,
        ants_params=ants_params,
        normcorre_params=normcorre_params,
        pyflowreg_params=pyflowreg_params,
    )

    warped = result["warped"]
    if warped is None:
        raise SystemExit("Processing completed but no output video was produced.")

    output_path = _determine_output_path(input_path, args.output, args.method.lower())
    if os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    _save_output(warped, output_path)
    print(f"Saved registered video to: {output_path}")


if __name__ == "__main__":
    main()
