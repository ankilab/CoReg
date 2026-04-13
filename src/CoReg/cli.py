import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import presets as _presets

from gui import ProcessingWorker
from utils.utils import (
    load_hdf5,
    load_tiff,
    load_video,
    resize_video_tensor_to_wh,
    save_video_hdf5,
    save_video_tiff,
    save_and_display_video,
    _to_gray_float,
    video_ssim_score,
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


def _clip_to_gray_u8(clip_t: torch.Tensor) -> np.ndarray:
    """First frame of a (T,C,H,W) clip → uint8 grayscale array."""
    frame = clip_t[0]
    gray = (
        (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()
        if frame.shape[0] == 3
        else frame[0].numpy()
    )
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        return ((gray - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


def _clip_to_gray_video(clip_t: torch.Tensor) -> np.ndarray:
    """(T,C,H,W) clip → (T,H,W) float32 grayscale array."""
    gray = (
        0.299 * clip_t[:, 0] + 0.587 * clip_t[:, 1] + 0.114 * clip_t[:, 2]
        if clip_t.shape[1] == 3
        else clip_t[:, 0]
    )
    return gray.numpy().astype(np.float32)


COTRACKER_PRESETS = {"regular_grid", "features", "activity", "segmentation", "matchanything"}


def _generate_preset_points(
    clip_t: torch.Tensor,
    preset_name: str,
    rows: int,
    cols: int,
    df_sigma: float,
    num_points: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return (pts_xy, df_sigma) for the given preset.

    *num_points* caps the number of seed points for non-grid presets.
    When None, falls back to rows × cols.
    Falls back to a regular grid when the preset produces no points.
    """
    _, _, H, W = clip_t.shape
    if num_points is None or num_points <= 0:
        num_points = rows * cols
    min_dist = max(1.0, min(H, W) / (2.0 * max(rows, cols)))

    if preset_name == "regular_grid":
        return _presets.preset_regular_grid(H, W, rows, cols), df_sigma

    gray_u8 = _clip_to_gray_u8(clip_t)

    if preset_name == "features":
        pts = _presets.preset_features(gray_u8, num_points=num_points, min_dist=min_dist)
    elif preset_name == "activity":
        pts = _presets.preset_activity(
            _clip_to_gray_video(clip_t), num_points=num_points, min_dist=min_dist
        )
    elif preset_name == "segmentation":
        pts = _presets.preset_segmentation(gray_u8, num_points=num_points)
    elif preset_name == "matchanything":
        pts = _presets.preset_matchanything(gray_u8, num_points=num_points, min_dist=min_dist)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)

    if len(pts) == 0:
        print(
            f"Preset '{preset_name}' produced no points; falling back to regular grid.",
            file=sys.stderr,
        )
        pts = _presets.preset_regular_grid(H, W, rows, cols)

    return pts.astype(np.float32), df_sigma


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
        # Optional: print status messages to stderr; progress is handled at a higher level
        if text:
            print(str(text), file=sys.stderr)

    def _on_progress(pct: int):
        # CLI-wide progress is tracked per sequence/file, so we ignore per-frame updates here.
        return

    worker.finished.connect(_on_finished)
    worker.error.connect(_on_error)
    worker.status_update.connect(_on_status)
    worker.progress.connect(_on_progress)

    # Run synchronously in the current thread
    worker.run()

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


def _compute_video_metrics(video) -> dict[str, float]:
    """Compute temporal MSE, ZNCC, and MI over a video result.

    Metrics are computed between each frame and the first frame, then
    averaged over t = 1..T-1. This mirrors ``video_ssim_score`` with
    ``mode="ref_first"`` and provides a single summary value per clip.
    """

    gray = _to_gray_float(video)  # (T,H,W) float32
    T = int(gray.shape[0])
    if T < 2:
        return {"mse": 0.0, "zncc": 1.0, "mi": 0.0}

    ref = gray[0].ravel().astype(np.float64)
    mse_vals: list[float] = []
    zncc_vals: list[float] = []
    mi_vals: list[float] = []

    ref_mean = float(ref.mean())
    ref_centered = ref - ref_mean
    ref_var = float(np.dot(ref_centered, ref_centered))

    for t in range(1, T):
        cur = gray[t].ravel().astype(np.float64)

        # MSE
        diff = cur - ref
        mse_vals.append(float(np.mean(diff * diff)))

        # ZNCC (zero-mean normalized cross-correlation)
        cur_mean = float(cur.mean())
        cur_centered = cur - cur_mean
        cur_var = float(np.dot(cur_centered, cur_centered))
        denom = np.sqrt(ref_var * cur_var) + 1e-12
        zncc_vals.append(float(np.dot(ref_centered, cur_centered) / denom))

        # Mutual information via joint histogram
        bins = 64
        try:
            hist2d, xedges, yedges = np.histogram2d(
                ref, cur, bins=bins, density=False
            )
        except Exception:
            # If histogram computation fails, skip MI for this pair
            continue

        pxy = hist2d.astype(np.float64)
        total = pxy.sum()
        if total <= 0:
            continue
        pxy /= total

        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        px_nz = px[px > 0]
        py_nz = py[py > 0]
        pxy_nz = pxy[pxy > 0]

        Hx = -float(np.sum(px_nz * np.log2(px_nz))) if px_nz.size else 0.0
        Hy = -float(np.sum(py_nz * np.log2(py_nz))) if py_nz.size else 0.0
        Hxy = -float(np.sum(pxy_nz * np.log2(pxy_nz))) if pxy_nz.size else 0.0

        mi_vals.append(Hx + Hy - Hxy)

    def _safe_mean(values: list[float], default: float) -> float:
        return float(np.mean(values)) if values else float(default)

    return {
        "mse": _safe_mean(mse_vals, 0.0),
        "zncc": _safe_mean(zncc_vals, 1.0),
        "mi": _safe_mean(mi_vals, 0.0),
    }


def _process_directory_input(input_dir: str, args: argparse.Namespace) -> None:
    """Process all supported files in a directory, optionally in n-frame sequences.

    For each supported video/stack file in the directory, this function will:
    - load the data
    - optionally split it into sequences of length ``args.seq_frames``
    - run the chosen registration method on each full video / sequence
    - save results under ``args.output_dir`` or ``<input_dir>/registered``.
    """

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    target_w = args.width
    target_h = args.height
    max_frames = args.max_frames

    seq_frames = args.seq_frames
    seq_stride = args.seq_stride if args.seq_stride is not None else seq_frames
    if seq_frames is not None and seq_frames <= 0:
        raise SystemExit("--seq-frames must be a positive integer if provided.")
    if seq_frames is not None and (seq_stride is None or seq_stride <= 0):
        raise SystemExit("--seq-stride must be a positive integer when using --seq-frames.")

    num_seqs = getattr(args, "num_seqs", None)
    if num_seqs is not None:
        if seq_frames is None:
            raise SystemExit(
                "--num-seqs requires --seq-frames to specify the sequence length."
            )
        if int(num_seqs) <= 0:
            raise SystemExit("--num-seqs must be a positive integer if provided.")

    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_path / "registered"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    method_map = {
        "cotracker": "CoTracker",
        "ants": "ANTs",
        "lddmm": "LDDMM",
        "normcorre": "NoRMCorre",
        "flowreg": "FlowReg",
    }
    method_slug = args.method.lower()
    method = method_map[method_slug]

    # CoTracker presets: allow running multiple point-generation configurations
    cotracker_presets: list[str] | None = None
    if method == "CoTracker":
        preset_str = getattr(args, "cotracker_presets", None)
        if preset_str:
            names = [p.strip().lower() for p in str(preset_str).split(",") if p.strip()]
        else:
            names = ["regular_grid"]

        invalid = [n for n in names if n not in COTRACKER_PRESETS]
        if invalid:
            raise SystemExit(
                "Unknown CoTracker preset(s): "
                + ", ".join(sorted(set(invalid)))
                + f". Allowed: {', '.join(sorted(COTRACKER_PRESETS))}."
            )
        cotracker_presets = names

    # Prepare metrics CSV path (one per run)
    if getattr(args, "metrics_csv", None):
        metrics_path = Path(args.metrics_csv).expanduser()
    elif method == "CoTracker" and cotracker_presets:
        preset_tag = "_".join(cotracker_presets)
        metrics_path = output_root / f"metrics_{method_slug}_{preset_tag}.csv"
    else:
        metrics_path = output_root / f"metrics_{method_slug}.csv"
    if metrics_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Metrics CSV already exists: {metrics_path}\nUse --overwrite to replace it."
        )

    metrics_rows: list[list[object]] = []

    # Random generator for sequence starts (reproducible across runs)
    seed = getattr(args, "seed", 0)
    rng = np.random.default_rng(int(seed))

    # Collect supported files (non-recursive to keep behaviour predictable)
    files = [
        p
        for p in sorted(input_path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise SystemExit(
            f"No supported input files found in directory: {input_dir}. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Global progress bar over all clips (sequences * number of files * presets)
    pbar = tqdm(total=0, file=sys.stderr, desc=f"{method} clips", unit="clip")

    # Active preset list for this method (None means "no named preset")
    if method == "CoTracker" and cotracker_presets is not None:
        active_presets: list[str | None] = list(cotracker_presets)
    else:
        active_presets = [None]

    for fpath in files:
        pbar.write(f"Processing file: {fpath}")

        video_t = _load_video_tensor(
            str(fpath),
            target_w=target_w,
            target_h=target_h,
            max_frames=max_frames,
        )

        T = int(video_t.shape[0])

        def _run_for_clip(
            clip_t: torch.Tensor,
            seq_index: int | None = None,
            start_idx: int = 0,
            end_idx: int | None = None,
            preset_name: str | None = None,
        ) -> None:
            custom_grid: torch.Tensor | None = None
            ants_params: dict | None = None
            normcorre_params: dict | None = None
            pyflowreg_params: dict | None = None

            df_sigma = float(args.df_sigma)

            if method == "CoTracker":
                grid_pts, df_sigma = _generate_preset_points(
                    clip_t,
                    preset_name or "regular_grid",
                    int(args.grid_rows),
                    int(args.grid_cols),
                    float(args.df_sigma),
                    num_points=getattr(args, "num_points", None),
                )
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
                clip_t,
                method=method,
                custom_grid_pts=custom_grid,
                df_sigma=df_sigma,
                ants_params=ants_params,
                normcorre_params=normcorre_params,
                pyflowreg_params=pyflowreg_params,
            )

            warped = result["warped"]
            if warped is None:
                raise SystemExit(
                    "Processing completed but no output video was produced."
                )

            if seq_index is None:
                out_name = f"{fpath.stem}_{method_slug}.tiff"
            else:
                out_name = f"{fpath.stem}_seq{seq_index:03d}_{method_slug}.tiff"

            output_path = output_root / out_name
            if output_path.exists() and not args.overwrite:
                print(
                    f"Output file already exists, skipping: {output_path} (use --overwrite to replace)",
                    file=sys.stderr,
                )
                return

            _save_output(warped, str(output_path))
            print(f"Saved registered video to: {output_path}")

            # Compute metrics for this result and record CSV row
            try:
                ssim_val = video_ssim_score(warped, mode="ref_first")
            except Exception:
                ssim_val = float("nan")

            metrics = _compute_video_metrics(warped)
            # Derive frame range for this clip relative to original sequence
            if end_idx is None:
                gray = _to_gray_float(warped)
                T_clip = int(gray.shape[0]) if gray.ndim >= 3 else 0
                end_frame = int(start_idx + max(T_clip - 1, 0))
            else:
                end_frame = int(end_idx)

            row: list[object] = [
                fpath.name,
                int(start_idx),
                end_frame,
                float(ssim_val),
                float(metrics["mse"]),
                float(metrics["zncc"]),
                float(metrics["mi"]),
            ]
            if method == "CoTracker":
                row.append(preset_name or "regular_grid")

            metrics_rows.append(row)

            # One clip has completed
            pbar.update(1)

        # If sequence extraction is requested and feasible, run per-sequence
        if seq_frames is not None and T >= seq_frames:
            max_start = T - seq_frames

            if num_seqs is not None:
                total_positions = max_start + 1
                n_req = int(num_seqs)
                n = min(n_req, total_positions)
                if n <= 0:
                    print(
                        f"Skipping {fpath} because it is shorter than seq_frames.",
                        file=sys.stderr,
                    )
                    continue

                if n < n_req:
                    print(
                        f"Warning: requested {n_req} sequences but only {n} unique windows "
                        f"are available for {fpath.name}.",
                        file=sys.stderr,
                    )

                starts_arr = rng.choice(total_positions, size=n, replace=False)
                starts = sorted(int(s) for s in starts_arr)
            else:
                assert seq_stride is not None
                starts = list(range(0, max_start + 1, seq_stride))

            if not starts:
                print(
                    f"Skipping {fpath} because it is shorter than seq_frames.",
                    file=sys.stderr,
                )
                continue

            # Account for this file's sequences in the global progress total
            pbar.total += len(starts) * len(active_presets)
            pbar.refresh()

            for seq_idx, start in enumerate(starts):
                end = start + seq_frames
                clip = video_t[start:end]
                for preset_name in active_presets:
                    _run_for_clip(
                        clip,
                        seq_index=seq_idx,
                        start_idx=start,
                        end_idx=end - 1,
                        preset_name=preset_name,
                    )
        else:
            # Whole file is treated as a single clip (per preset)
            pbar.total += len(active_presets)
            pbar.refresh()
            for preset_name in active_presets:
                _run_for_clip(
                    video_t,
                    seq_index=None,
                    start_idx=0,
                    end_idx=T - 1,
                    preset_name=preset_name,
                )

    # Close global progress bar once all clips are processed
    pbar.close()

    # Write metrics CSV if any rows collected
    if metrics_rows:
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "video",
                "startframe",
                "endframe",
                "ssim",
                "mse",
                "zncc",
                "mi",
            ]
            if method == "CoTracker":
                header.append("preset")

            writer.writerow(header)
            writer.writerows(metrics_rows)
        print(f"Saved metrics CSV to: {metrics_path}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `coreg-cli` command."""
    parser = argparse.ArgumentParser(
        description="CoReg CLI: run motion correction/registration without the GUI.",
    )
    parser.add_argument(
        "input",
        help=(
            "Input path: either a single file (.avi, .mp4, .tif, .tiff, .h5, .hdf5) "
            "or a folder containing such files."
        ),
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
        "--seq-frames",
        type=int,
        default=None,
        help=(
            "If INPUT is a folder, split each video into sequences of this many "
            "frames and run registration on each sequence separately. "
            "If omitted, each file is processed as a single clip."
        ),
    )
    parser.add_argument(
        "--seq-stride",
        type=int,
        default=None,
        help=(
            "Stride between sequence starts when using --seq-frames. "
            "Defaults to seq-frames (non-overlapping)."
        ),
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=None,
        help=(
            "Number of sequences to sample per video when INPUT is a folder. "
            "Sequences have length --seq-frames and their starting frames are "
            "chosen at random using --seed for reproducibility."
        ),
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=8,
        help="Number of rows for CoTracker seed grid (cotracker only).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=8,
        help="Number of columns for CoTracker seed grid (cotracker only).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help=(
            "Maximum number of seed points for non-grid CoTracker presets "
            "(features, activity, segmentation, matchanything). "
            "Defaults to --grid-rows × --grid-cols when not set."
        ),
    )
    parser.add_argument(
        "--df-sigma",
        type=float,
        default=5.0,
        help="Smoothing sigma for CoTracker displacement field.",
    )
    parser.add_argument(
        "--cotracker-presets",
        help=(
            "Comma-separated list of CoTracker point-generation presets to run "
            f"(choices: {','.join(sorted(COTRACKER_PRESETS))}). "
            "Only used when --method=cotracker; default is 'regular_grid'."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "When INPUT is a folder, save all outputs under this directory "
            "(default: '<input>/registered'). Ignored for single-file input."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed used when sampling sequences with --num-seqs. "
            "The same seed yields the same sequence start frames across runs."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        help=(
            "Write a CSV with columns 'video,startframe,endframe,ssim,mse,zncc,mi'. "
            "For folder input, defaults to '<output-dir or input>/metrics_<method>.csv'. "
            "For single-file input, defaults to '<output>_metrics.csv'."
        ),
    )

    args = parser.parse_args(argv)

    input_path_str = str(Path(args.input).expanduser())
    if not os.path.exists(input_path_str):
        raise SystemExit(f"Input path not found: {input_path_str}")

    if (args.width is None) != (args.height is None):
        raise SystemExit(
            "Please provide both --width and --height for resizing, or neither."
        )

    # If the input is a directory, process all supported files inside it and exit.
    if os.path.isdir(input_path_str):
        if args.output is not None:
            raise SystemExit(
                "When INPUT is a directory, please use --output-dir instead of -o/--output."
            )
        _process_directory_input(input_path_str, args)
        return

    # Single-file behaviour (backwards compatible with previous CLI)
    target_w = args.width
    target_h = args.height
    max_frames = args.max_frames

    video_t = _load_video_tensor(
        input_path_str,
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
        preset_name = (
            (getattr(args, "cotracker_presets", None) or "regular_grid")
            .split(",")[0]
            .strip()
            .lower()
        )
        if preset_name not in COTRACKER_PRESETS:
            raise SystemExit(
                f"Unknown CoTracker preset '{preset_name}'. "
                f"Allowed: {', '.join(sorted(COTRACKER_PRESETS))}."
            )
        grid_pts, df_sigma = _generate_preset_points(
            video_t, preset_name, int(args.grid_rows), int(args.grid_cols), float(args.df_sigma),
            num_points=getattr(args, "num_points", None),
        )
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

    output_path = _determine_output_path(
        input_path_str, args.output, args.method.lower()
    )
    if os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    _save_output(warped, output_path)
    print(f"Saved registered video to: {output_path}")

    # Compute and save metrics CSV for this single-file run
    try:
        ssim_val = video_ssim_score(warped, mode="ref_first")
    except Exception:
        ssim_val = float("nan")

    metrics = _compute_video_metrics(warped)
    gray = _to_gray_float(warped)
    T = int(gray.shape[0]) if gray.ndim >= 3 else 0
    start_f = 0
    end_f = T - 1 if T > 0 else 0

    metrics_path = (
        Path(args.metrics_csv).expanduser()
        if getattr(args, "metrics_csv", None)
        else Path(output_path).with_name(Path(output_path).stem + "_metrics.csv")
    )
    if metrics_path.exists() and not args.overwrite:
        print(
            f"Metrics CSV already exists: {metrics_path} (use --overwrite to replace)",
            file=sys.stderr,
        )
    else:
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["video", "startframe", "endframe", "ssim", "mse", "zncc", "mi"]
            )
            writer.writerow(
                [
                    Path(input_path_str).name,
                    start_f,
                    end_f,
                    float(ssim_val),
                    float(metrics["mse"]),
                    float(metrics["zncc"]),
                    float(metrics["mi"]),
                ]
            )
        print(f"Saved metrics CSV to: {metrics_path}")


if __name__ == "__main__":
    main()
