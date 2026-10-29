#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Baseline registration benchmark CLI.

Mirrors cli.py but runs normcorre / pyflowreg / ANTs instead of CoTracker.
CSV output uses the same schema as cli.py's, so both runs' results can be
compared directly (see analysis/metrics/compute_comparison_metrics.py).

Usage
-----
python cli_baseline.py \\
    --folder /path/to/videos \\
    --methods normcorre_rigid normcorre_pwrigid pyflowreg ants \\
    --run-name baseline_comparison \\
    --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils.utils import (
    load_hdf5,
    load_tiff,
    load_video,
    memmap_or_load_tiff,
    resize_video_tensor_to_wh,
    save_video_hdf5,
)
from metrics_common import compute_metrics

SUPPORTED_VIDEO_EXT = {".avi", ".mp4"}
SUPPORTED_TIFF_EXT = {".tiff", ".tif"}
SUPPORTED_HDF5_EXT = {".h5", ".hdf5"}
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXT | SUPPORTED_TIFF_EXT | SUPPORTED_HDF5_EXT


BASELINE_METHODS: dict[str, dict] = {
    "normcorre_rigid": {
        "method": "normcorre",
        "pw_rigid": False,
        "max_shifts": (6, 6),
        "strides": (96, 96),
        "overlaps": (32, 32),
        "upsample_factor_fft": 4,
        "max_deviation_rigid": 3,
    },
    "normcorre_pwrigid": {
        "method": "normcorre",
        "pw_rigid": True,
        "max_shifts": (6, 6),
        "strides": (96, 96),
        "overlaps": (32, 32),
        "upsample_factor_fft": 4,
        "max_deviation_rigid": 3,
    },
    "pyflowreg": {
        "method": "pyflowreg",
        "alpha": 1.5,
        "min_level": 4,
    },
    "lddmm": {
        "method": "lddmm",
    },
    "ants": {
        "method": "ants",
    },
}


def get_total_frames(path: str) -> int:
    """Return total frame count for a file without loading all data into memory."""
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_TIFF_EXT:
        mm = memmap_or_load_tiff(path)
        return int(mm.shape[0]) if mm.ndim == 3 else 1
    if ext in SUPPORTED_HDF5_EXT:
        import h5py

        with h5py.File(path, "r") as f:
            for key in ("/data", "data", "/video", "video"):
                if key in f:
                    return int(f[key].shape[0])
        return 0
    if ext in SUPPORTED_VIDEO_EXT:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    return 0


def load_clip(
    path: str,
    num_frames: int | None = None,
    frame_start: int = 0,
    target_w: int | None = None,
    target_h: int | None = None,
) -> torch.Tensor:
    """Load a single file and return a (T, C, H, W) float32 tensor."""
    ext = Path(path).suffix.lower()

    if ext in SUPPORTED_VIDEO_EXT:
        video = load_video(path, resize=False, max_frames=None)
        end = (frame_start + num_frames) if num_frames is not None else None
        video = video[frame_start:end]

    elif ext in SUPPORTED_TIFF_EXT:
        mm = memmap_or_load_tiff(path)
        end = (frame_start + num_frames) if num_frames is not None else None
        chunk = mm[frame_start:end] if mm.ndim == 3 else mm[np.newaxis]
        video = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        if video.ndim == 3:
            video = video.unsqueeze(1)

    elif ext in SUPPORTED_HDF5_EXT:
        import h5py

        with h5py.File(path, "r") as f:
            ds = None
            for key in ("/data", "data", "/video", "video"):
                if key in f:
                    ds = f[key]
                    break
            if ds is None:
                raise KeyError(f"No suitable dataset found in {path}")
            end = (frame_start + num_frames) if num_frames is not None else ds.shape[0]
            chunk = ds[frame_start:end]
        video = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        if video.ndim == 3:
            video = video.unsqueeze(1)

    else:
        raise ValueError(f"Unsupported extension: '{ext}'")

    if not isinstance(video, torch.Tensor):
        video = torch.from_numpy(np.asarray(video))
    if video.ndim == 3:
        video = video.unsqueeze(1)
    if video.ndim == 4 and video.shape[1] not in (1, 3) and video.shape[-1] in (1, 3):
        video = video.permute(0, 3, 1, 2)
    video = video.float()

    if target_w is not None and target_h is not None:
        video = resize_video_tensor_to_wh(video, target_w=target_w, target_h=target_h)

    return video


def iter_folder(
    folder: str,
    num_frames: int | None = None,
    chunk_frames: int | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    file_filter: list[str] | None = None,
    file_nth: int = 1,
) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Yield (folder_name, chunk_name, clip_tensor) for every supported file in *folder*."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if file_filter is not None:
        filter_set = {f.lower() for f in file_filter}
        files = [
            p
            for p in files
            if p.name.lower() in filter_set or p.stem.lower() in filter_set
        ]
    if file_nth > 1:
        files = files[::file_nth]
    if not files:
        raise ValueError(
            f"No supported files found in {folder}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    for fpath in files:
        try:
            if chunk_frames is not None:
                total = get_total_frames(str(fpath))
                starts = range(0, max(1, total), chunk_frames)
                for i, start in enumerate(starts):
                    chunk_name = f"{fpath.stem}_chunk{i:03d}{fpath.suffix}"
                    try:
                        clip = load_clip(
                            str(fpath),
                            num_frames=chunk_frames,
                            frame_start=start,
                            target_w=target_w,
                            target_h=target_h,
                        )
                        if int(clip.shape[0]) < chunk_frames:
                            print(
                                f"Skipping {chunk_name}: incomplete chunk "
                                f"({int(clip.shape[0])}/{chunk_frames} frames).",
                                file=sys.stderr,
                            )
                            continue
                        yield folder_path.name, chunk_name, clip
                    except Exception as e:
                        print(
                            f"Warning: could not load chunk {i} of {fpath.name}: {e}",
                            file=sys.stderr,
                        )
            else:
                clip = load_clip(
                    str(fpath),
                    num_frames=num_frames,
                    target_w=target_w,
                    target_h=target_h,
                )
                if num_frames is not None and int(clip.shape[0]) != num_frames:
                    print(
                        f"Skipping {fpath.name}: expected {num_frames} frames, "
                        f"got {int(clip.shape[0])}",
                        file=sys.stderr,
                    )
                    continue
                yield folder_path.name, fpath.name, clip
        except Exception as e:
            print(f"Warning: could not load {fpath.name}: {e}", file=sys.stderr)


def _to_thwc(video: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        v = video.detach().cpu().numpy()
    else:
        v = np.asarray(video)
    if v.ndim == 4 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
        v = np.transpose(v, (0, 2, 3, 1))
    elif v.ndim == 3:
        v = v[..., None]
    elif v.ndim != 4:
        raise ValueError(f"Unsupported video shape: {v.shape}")
    return v.astype(np.float32)


def run_baseline(
    video: torch.Tensor,
    method_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a baseline method and return (warped, shift).

    shift is a (T, 2) per-frame (dx, dy) array when the backend exposes one
    (NoRMCorre's shifts come directly from jnormcorre; pyflowreg's are the
    mean absolute per-pixel displacement over its own valid-region mask),
    else None for ANTs/LDDMM. Used only to size the shared border-crop margin
    in video_metrics — not as the residual_flow metric itself, which stays
    independent of what any correction method reports about its own output.
    """
    cfg = BASELINE_METHODS[method_name]
    backend = cfg["method"]
    kwargs = {k: v for k, v in cfg.items() if k != "method"}

    if backend == "normcorre":
        from utils.normcorre_wrapper import register_video

        warped, shifts = register_video(video, reference_index=0, **kwargs)
        return np.asarray(warped, dtype=np.float32), np.asarray(
            shifts, dtype=np.float32
        )

    if backend == "pyflowreg":
        from utils.pyflowreg_wrapper import register_video

        warped, _, shift = register_video(video, reference_index=0, **kwargs)
        return np.asarray(warped, dtype=np.float32), np.asarray(
            shift, dtype=np.float32
        )

    if backend == "lddmm":
        from utils.lddmm_register import lddmm_registration

        v = _to_thwc(video)  # THWC
        T, H, W, C = v.shape
        frames = [v[i] for i in range(T)]  # list of HWC
        result = lddmm_registration(
            frames
        )  # list of T arrays (frame 0 included), each HWC or HW
        warped_frames = []
        for f in result:
            arr = np.asarray(f, dtype=np.float32)
            warped_frames.append(arr[..., 0] if arr.ndim == 3 else arr)
        warped_gray = np.stack(warped_frames, axis=0)  # THW
        warped = warped_gray[..., None]
        if C > 1:
            warped = np.repeat(warped, C, axis=-1)
        return warped.astype(np.float32), None

    if backend == "ants":
        from utils.ants_wrapper import ants_registration

        v = _to_thwc(video)  # THWC
        T, H, W, C = v.shape
        frames = [v[i] for i in range(T)]  # list of HWC
        warped_frames = ants_registration(frames)  # list of T-1 HW arrays
        # reconstruct: frame 0 is unchanged, frames 1..T-1 come from ants
        warped_gray = np.stack(
            [v[0, :, :, 0]] + [np.asarray(f, dtype=np.float32) for f in warped_frames],
            axis=0,
        )  # THW
        warped = warped_gray[..., None]
        if C > 1:
            warped = np.repeat(warped, C, axis=-1)
        return warped.astype(np.float32), None

    raise ValueError(f"Unknown backend: {backend}")


def run_evaluation(
    folders: list[str],
    methods: list[str],
    *,
    run_name: str | None = None,
    num_frames: int | None = None,
    chunk_frames: int | None = None,
    file_filter: list[str] | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    output_dir: str | None = None,
    output_csv: str | None = None,
    save_videos: bool = False,
    save_nth: int = 1,
    overwrite: bool = False,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = run_name if run_name else timestamp
    out_dir = Path(output_dir) if output_dir else Path("baseline_out") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": folder_name,
        "timestamp": timestamp,
        "folders": folders,
        "methods": methods,
        "num_frames": num_frames,
        "target_w": target_w,
        "target_h": target_h,
        "save_videos": save_videos,
        "save_nth": save_nth,
        "method_params": {m: BASELINE_METHODS[m] for m in methods},
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to: {out_dir / 'config.json'}")

    csv_path = Path(output_csv) if output_csv else out_dir / "benchmark_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and not overwrite:
        raise SystemExit(
            f"Output CSV already exists: {csv_path}\nUse --overwrite to replace it."
        )

    invalid = [m for m in methods if m not in BASELINE_METHODS]
    if invalid:
        raise SystemExit(
            f"Unknown method(s): {', '.join(invalid)}. "
            f"Available: {', '.join(BASELINE_METHODS)}."
        )

    clips: list[tuple[str, str, torch.Tensor]] = []
    for folder in folders:
        clips.extend(
            iter_folder(
                folder,
                num_frames=num_frames,
                chunk_frames=chunk_frames,
                target_w=target_w,
                target_h=target_h,
                file_filter=file_filter,
            )
        )

    total = len(clips) * len(methods)

    fieldnames = [
        "folder",
        "file",
        "preset",
        "n_frames",
        "n_points",
        "elapsed_s",
        "ssim",
        "psnr",
        "mse",
        "zncc",
        "mi",
        "crispness_mean",
        "corr_mean",
        "residual_flow",
    ]
    frame_fields = [
        "folder",
        "file",
        "preset",
        "frame",
        "ssim",
        "psnr",
        "mse",
        "zncc",
        "mi",
        "corr_mean",
        "residual_flow",
    ]
    frame_csv = csv_path.parent / (csv_path.stem + "_perframe.csv")

    csv_file = csv_path.open("w", newline="")
    frame_csv_file = frame_csv.open("w", newline="")
    try:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        frame_writer = csv.DictWriter(frame_csv_file, fieldnames=frame_fields)
        writer.writeheader()
        frame_writer.writeheader()
        csv_file.flush()
        frame_csv_file.flush()

        with tqdm(
            total=total, desc="Baseline benchmark", unit="run", file=sys.stderr
        ) as pbar:
            for file_idx, (folder_name_clip, filename, clip) in enumerate(clips):
                T = int(clip.shape[0])
                stem = Path(filename).stem
                video_dir = out_dir / folder_name_clip / stem
                video_dir.mkdir(parents=True, exist_ok=True)

                for method in methods:
                    pbar.set_postfix(file=filename, method=method)
                    elapsed_s = float("nan")
                    warped = shift = None
                    try:
                        t0 = time.perf_counter()
                        warped, shift = run_baseline(clip, method)
                        elapsed_s = time.perf_counter() - t0
                        metrics = compute_metrics(warped, drift=shift)
                        if save_videos and file_idx % save_nth == 0:
                            out_path = video_dir / f"{stem}_{method}.h5"
                            if not out_path.exists() or overwrite:
                                save_video_hdf5(warped, str(out_path))
                                print(f"Saved: {out_path}")
                            # Shift sidecar lets a later, separate comparison pass size a shared border-crop margin.
                            if shift is not None:
                                shift_path = video_dir / f"{stem}_{method}_shift.npy"
                                if not shift_path.exists() or overwrite:
                                    np.save(str(shift_path), np.asarray(shift))
                    except Exception as e:
                        print(
                            f"Error on {folder_name_clip}/{filename} / {method}: {e}",
                            file=sys.stderr,
                        )
                        metrics = {
                            "ssim": float("nan"),
                            "zncc": float("nan"),
                            "mi": float("nan"),
                            "psnr": float("nan"),
                            "mse": float("nan"),
                            "crispness_mean": float("nan"),
                            "corr_mean": float("nan"),
                            "residual_flow": float("nan"),
                            "per_frame": {
                                "ssim": [],
                                "psnr": [],
                                "mse": [],
                                "zncc": [],
                                "mi": [],
                                "corr_mean": [],
                                "residual_flow": [],
                            },
                        }

                    writer.writerow(
                        {
                            "folder": folder_name_clip,
                            "file": filename,
                            "preset": method,
                            "n_frames": T,
                            "n_points": 0,
                            "elapsed_s": elapsed_s,
                            "ssim": metrics["ssim"],
                            "psnr": metrics["psnr"],
                            "mse": metrics["mse"],
                            "zncc": metrics["zncc"],
                            "mi": metrics["mi"],
                            "crispness_mean": metrics["crispness_mean"],
                            "corr_mean": metrics["corr_mean"],
                            "residual_flow": metrics["residual_flow"],
                        }
                    )
                    csv_file.flush()

                    pf = metrics["per_frame"]
                    for t, (ssim_t, psnr_t, mse_t, zncc_t, mi_t, corr_t, flow_t) in enumerate(
                        zip(
                            pf["ssim"],
                            pf["psnr"],
                            pf["mse"],
                            pf["zncc"],
                            pf["mi"],
                            pf["corr_mean"],
                            pf["residual_flow"],
                        ),
                        start=0,
                    ):
                        frame_writer.writerow(
                            {
                                "folder": folder_name_clip,
                                "file": filename,
                                "preset": method,
                                "frame": t,
                                "ssim": ssim_t,
                                "psnr": psnr_t,
                                "mse": mse_t,
                                "zncc": zncc_t,
                                "mi": mi_t,
                                "corr_mean": corr_t,
                                "residual_flow": flow_t,
                            }
                        )
                    frame_csv_file.flush()

                    pbar.update(1)
    finally:
        csv_file.close()
        frame_csv_file.close()

    # Deduplicate summary CSV: keep only the max n_frames row per (folder, file, preset)
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["folder"], row["file"], row["preset"])
        try:
            n = int(row.get("n_frames", 0))
        except ValueError:
            n = 0
        if key not in best or n > int(best[key].get("n_frames", 0)):
            best[key] = row

    deduped = sorted(best.values(), key=lambda r: (r["folder"], r["file"], r["preset"]))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Saved benchmark CSV to: {csv_path}")
    print(f"Saved per-frame CSV to: {frame_csv}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Baseline registration benchmark: normcorre / pyflowreg / ANTs."
    )
    parser.add_argument(
        "--folder",
        nargs="+",
        required=True,
        help="One or more folders containing video/tiff/hdf5 files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(BASELINE_METHODS),
        help=(
            f"Methods to run. Available: {', '.join(BASELINE_METHODS)}. "
            "Default: all methods."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Only process these filenames (stem or full name) within the given folders.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames per clip (default: all).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help=(
            "Split each file into consecutive chunks of this many frames and "
            "process each chunk separately. Output names get _chunk000, _chunk001, ... suffix."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize width before processing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize height before processing.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run (subdirectory under --output-dir). "
        "Defaults to a timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_out/",
        help="Base directory for output. Results go into <output-dir>/<run-name>/.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the output CSV (default: <output-dir>/<run-name>/benchmark_baseline.csv).",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        default=False,
        help="Save registered videos as gzip-compressed HDF5 files.",
    )
    parser.add_argument(
        "--save-nth",
        type=int,
        default=1,
        metavar="N",
        help="When saving videos, keep only every Nth frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args(argv)

    if (args.width is None) != (args.height is None):
        raise SystemExit("Provide both --width and --height, or neither.")

    run_evaluation(
        folders=args.folder,
        methods=args.methods,
        run_name=args.run_name,
        num_frames=args.num_frames,
        chunk_frames=args.chunk_frames,
        file_filter=args.files,
        target_w=args.width,
        target_h=args.height,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        save_videos=args.save_videos,
        save_nth=args.save_nth,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
