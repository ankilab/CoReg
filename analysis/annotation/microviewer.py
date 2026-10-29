#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""MicroViewer — interactive viewer for registered TIFF stacks with per-frame metrics.

Opens a folder containing <stem>_<preset>.tiff files (e.g. from cli.py or
cli_baseline.py with --save-videos), shows all presets side-by-side as
synchronized image panels, and plots per-frame metrics vs frame 0 at the bottom.

Usage
-----
python microviewer.py [--tiff-dir /path/to/<stem>/tiff]
python microviewer.py                  # opens a folder-picker dialog on start
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from pyqtgraph import QtCore

from utils.utils import video_metrics

pg.setConfigOptions(imageAxisOrder="row-major")

# Okabe–Ito colorblind-safe palette (same as analysis.py)
_COLORS = [
    "#0072B2",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#D55E00",
    "#000000",
]

_METRIC_OPTS: list[tuple[str, str]] = [
    ("ssim", "SSIM  (↑ better)"),
    ("psnr", "PSNR dB  (↑ better)"),
    ("zncc", "ZNCC  (↑ better)"),
    ("mi", "Mutual Information  (↑ better)"),
    ("corr_mean", "Correlation w/ mean  (↑ better)"),
]

_EXCLUDE_TAGS = {"tracks", "warped", "zstack", "warped_zstack"}
_NCOLS = 3  # panels per row in the 2×3 grid


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_gray(path: Path) -> np.ndarray | None:
    """Load a TIFF file as (T, H, W) float32 grayscale, or None on failure.

    Handles (H,W), (T,H,W), (H,W,C), (T,H,W,C), and (T,C,H,W) layouts.
    """
    try:
        v = tifffile.imread(str(path)).astype(np.float32)
        if v.ndim == 2:
            return v[np.newaxis]  # (H,W) → (1,H,W)
        if v.ndim == 3:
            if v.shape[-1] in (1, 3, 4) and v.shape[0] > 4:
                return v.mean(axis=-1)[np.newaxis]  # (H,W,C) → (1,H,W)
            return v  # (T,H,W)
        if v.ndim == 4:
            if v.shape[1] in (1, 3, 4) and v.shape[-1] not in (1, 3, 4):
                return v.mean(axis=1)  # (T,C,H,W) → (T,H,W)
            return v.mean(axis=-1)  # (T,H,W,C) → (T,H,W)
    except Exception as exc:
        print(f"Could not load {path.name}: {exc}", file=sys.stderr)
    return None


def _perframe_metrics(video: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-frame SSIM/PSNR/ZNCC/MI vs frame 0 as float32 arrays."""
    per_frame = video_metrics(video, mode="ref_first", return_per_frame=True)[
        "per_frame"
    ]
    return {
        key: np.asarray(per_frame[key], dtype=np.float32) for key, _ in _METRIC_OPTS
    }


def _display_labels(labels: list[str]) -> list[str]:
    """Strip the common stem prefix so only the preset/method part is shown."""
    if not labels:
        return labels
    prefix = os.path.commonprefix(labels)
    if "_" in prefix:
        prefix = prefix[: prefix.rfind("_") + 1]
    else:
        prefix = ""
    return [lbl[len(prefix) :] if lbl.startswith(prefix) else lbl for lbl in labels]


# ── main widget ───────────────────────────────────────────────────────────────


class MicroViewer(QWidget):
    def __init__(self, tiff_dir: str | None = None):
        super().__init__()
        self.setWindowTitle("MicroViewer")
        self.resize(1280, 820)

        # Data state
        self._labels: list[str] = []
        self._dlabels: list[str] = []  # display (prefix-stripped) labels
        self._videos: list[np.ndarray] = []  # each (T, H, W) float32
        self._ranges: list[tuple] = []  # (lo, hi) for display normalisation
        self._metrics: list[dict] = []  # per-video per-frame metrics
        self._n_frames: int = 0
        self._frame: int = 0

        self._img_items: list[pg.ImageItem] = []
        self._viewboxes: list[pg.ViewBox] = []
        self._selected: int | None = None
        self._vline: pg.InfiniteLine | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._build_ui()

        if tiff_dir:
            self._load_dir(tiff_dir)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QHBoxLayout()
        tb.setSpacing(6)

        open_btn = QPushButton("Open folder…")
        open_btn.clicked.connect(self._on_open)
        tb.addWidget(open_btn)

        add_btn = QPushButton("Add file…")
        add_btn.clicked.connect(self._on_add_file)
        tb.addWidget(add_btn)

        self._dir_label = QLabel("No folder loaded")
        self._dir_label.setStyleSheet("color: #888;")
        tb.addWidget(self._dir_label, stretch=1)

        tb.addWidget(QLabel("FPS:"))
        self._fps = QSpinBox()
        self._fps.setRange(1, 60)
        self._fps.setValue(10)
        self._fps.setFixedWidth(55)
        tb.addWidget(self._fps)

        root.addLayout(tb)

        # ── frame controls ────────────────────────────────────────────────────
        fc = QHBoxLayout()
        fc.setSpacing(4)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        fc.addWidget(self._prev_btn)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(76)
        self._play_btn.clicked.connect(self._toggle_play)
        fc.addWidget(self._play_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(32)
        self._next_btn.clicked.connect(lambda: self._step(1))
        fc.addWidget(self._next_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        fc.addWidget(self._slider, stretch=1)

        self._flabel = QLabel("—")
        self._flabel.setFixedWidth(80)
        self._flabel.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        fc.addWidget(self._flabel)

        root.addLayout(fc)

        # ── splitter: image panels (top) / metrics (bottom) ──────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self._gfx = pg.GraphicsLayoutWidget()
        self._gfx.setBackground("#1a1a1a")
        self._gfx.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._gfx.scene().sigMouseClicked.connect(self._on_panel_click)
        splitter.addWidget(self._gfx)

        # Metrics panel
        mp = QWidget()
        ml = QVBoxLayout(mp)
        ml.setContentsMargins(2, 2, 2, 2)
        ml.setSpacing(2)

        mh = QHBoxLayout()
        mh.addWidget(QLabel("Metric:"))
        self._mcomb = QComboBox()
        for key, lbl in _METRIC_OPTS:
            self._mcomb.addItem(lbl, key)
        self._mcomb.currentIndexChanged.connect(self._refresh_plot)
        mh.addWidget(self._mcomb)
        mh.addStretch()
        ml.addLayout(mh)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1a1a1a")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("bottom", "Frame")
        self._plot.scene().sigMouseClicked.connect(self._on_plot_click)
        ml.addWidget(self._plot)

        splitter.addWidget(mp)
        splitter.setSizes([560, 240])
        root.addWidget(splitter, stretch=1)

    # ── loading ───────────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select TIFF folder")
        if d:
            self._load_dir(d)

    def _load_dir(self, path: str) -> None:
        folder = Path(path)
        self._dir_label.setText(f"Loading {folder.name} …")
        QApplication.processEvents()

        # Collect TIFFs: folder/, folder/tiff/, folder/*/tiff/, folder/split_videos/*/tiff/
        search_dirs = [folder, folder / "tiff"]
        for sub in sorted(folder.iterdir()) if folder.is_dir() else []:
            if sub.is_dir() and sub.name not in _EXCLUDE_TAGS:
                search_dirs.append(sub / "tiff")
        sv = folder / "split_videos"
        if sv.is_dir():
            for sub in sorted(sv.iterdir()):
                if sub.is_dir():
                    search_dirs.append(sub / "tiff")

        seen: set[Path] = set()
        tiffs: list[Path] = []
        for d in search_dirs:
            if d.is_dir():
                for ext in ("*.tiff", "*.tif"):
                    for f in sorted(d.glob(ext)):
                        if f not in seen:
                            seen.add(f)
                            tiffs.append(f)

        # Exclude tracks / zstack / warped overlays
        tiffs = [
            f
            for f in tiffs
            if not any(f.stem.endswith(f"_{tag}") for tag in _EXCLUDE_TAGS)
        ]

        if not tiffs:
            self._dir_label.setText(f"No TIFF files found in {folder}")
            return

        # Load and compute metrics
        labels, videos, metrics, ranges = [], [], [], []
        for p in tiffs:
            vid = _load_gray(p)
            if vid is None:
                continue
            m = _perframe_metrics(vid)
            finite = vid[np.isfinite(vid)]
            lo = float(np.percentile(finite, 1)) if finite.size else 0.0
            hi = float(np.percentile(finite, 99)) + 1e-8 if finite.size else 1.0
            labels.append(p.stem)
            videos.append(vid)
            metrics.append(m)
            ranges.append((lo, hi))

        if not videos:
            self._dir_label.setText("Could not load any TIFFs")
            return

        self._labels = labels
        self._dlabels = _display_labels(labels)
        self._videos = videos
        self._metrics = metrics
        self._ranges = ranges
        self._n_frames = min(v.shape[0] for v in videos)
        self._frame = 0

        self._selected = None
        self._dir_label.setText(str(folder))
        self.setWindowTitle(f"MicroViewer — {folder.name}")
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._flabel.setText(f"0 / {self._n_frames - 1}")

        self._rebuild_panels()
        self._refresh_plot()
        self._show_frame(0)

    def _on_add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add TIFF file", "", "TIFF files (*.tiff *.tif)"
        )
        if not path:
            return
        p = Path(path)
        vid = _load_gray(p)
        if vid is None:
            self._dir_label.setText(f"Could not load {p.name}")
            return
        m = _perframe_metrics(vid)
        finite = vid[np.isfinite(vid)]
        lo = float(np.percentile(finite, 1)) if finite.size else 0.0
        hi = float(np.percentile(finite, 99)) + 1e-8 if finite.size else 1.0

        self._labels.append(p.stem)
        self._dlabels = _display_labels(self._labels)
        self._videos.append(vid)
        self._metrics.append(m)
        self._ranges.append((lo, hi))
        self._n_frames = min(v.shape[0] for v in self._videos)
        self._frame = min(self._frame, max(0, self._n_frames - 1))

        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._flabel.setText(f"{self._frame} / {max(0, self._n_frames - 1)}")
        self._rebuild_panels()
        self._refresh_plot()
        self._show_frame(self._frame)

    # ── image panels ──────────────────────────────────────────────────────────

    def _rebuild_panels(self) -> None:
        self._gfx.clear()
        self._img_items.clear()
        self._viewboxes.clear()

        first_vb: pg.ViewBox | None = None
        font = QFont()
        font.setPointSize(8)

        n = len(self._dlabels)
        for i, dlabel in enumerate(self._dlabels):
            col = i % _NCOLS
            grid_row = i // _NCOLS
            title_row = grid_row * 2
            img_row = grid_row * 2 + 1

            lbl = self._gfx.addLabel(dlabel, row=title_row, col=col)
            lbl.item.setFont(font)

            vb = self._gfx.addViewBox(
                row=img_row, col=col, lockAspect=True, enableMouse=True
            )
            vb.setMenuEnabled(False)
            vb.setDefaultPadding(0.01)
            self._viewboxes.append(vb)

            img = pg.ImageItem()
            vb.addItem(img)
            self._img_items.append(img)

            if first_vb is None:
                first_vb = vb
            else:
                vb.setXLink(first_vb)
                vb.setYLink(first_vb)

        # Compact title rows, tall image rows
        layout = self._gfx.ci.layout
        n_grid_rows = (n + _NCOLS - 1) // _NCOLS
        for r in range(n_grid_rows):
            layout.setRowStretchFactor(r * 2, 1)
            layout.setRowStretchFactor(r * 2 + 1, 20)

    # ── panel selection ───────────────────────────────────────────────────────

    def _on_panel_click(self, ev) -> None:
        pos = ev.scenePos()
        for i, vb in enumerate(self._viewboxes):
            if vb.sceneBoundingRect().contains(pos):
                self._selected = None if self._selected == i else i
                self._update_selection()
                return

    def _update_selection(self) -> None:
        for i, vb in enumerate(self._viewboxes):
            vb.setBorder(
                pg.mkPen(_COLORS[i % len(_COLORS)], width=2)
                if i == self._selected
                else None
            )
        self._refresh_plot()

    # ── frame display ─────────────────────────────────────────────────────────

    def _show_frame(self, t: int) -> None:
        t = int(np.clip(t, 0, max(0, self._n_frames - 1)))
        self._frame = t
        self._flabel.setText(f"{t} / {max(0, self._n_frames - 1)}")

        for img_item, video, (lo, hi) in zip(
            self._img_items, self._videos, self._ranges
        ):
            frame = video[min(t, video.shape[0] - 1)]
            display = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
            img_item.setImage(display, autoLevels=False, levels=(0.0, 1.0))

        if self._vline is not None:
            self._vline.setValue(t)

    # ── metrics plot ──────────────────────────────────────────────────────────

    def _refresh_plot(self) -> None:
        key = self._mcomb.currentData() or "ssim"
        ylabel = next((l for k, l in _METRIC_OPTS if k == key), key)

        self._plot.clear()
        self._plot.addLegend(offset=(10, 10))
        self._plot.setLabel("left", ylabel)
        self._plot.setLabel("bottom", "Frame")

        for i, (dlabel, m) in enumerate(zip(self._dlabels, self._metrics)):
            ys = m.get(key, np.array([]))
            if not len(ys):
                continue
            xs = np.arange(1, len(ys) + 1, dtype=np.float32)
            color = pg.mkColor(_COLORS[i % len(_COLORS)])
            dimmed = self._selected is not None and i != self._selected
            if dimmed:
                color.setAlpha(50)
            self._plot.plot(
                xs,
                ys,
                pen=pg.mkPen(color, width=2.0 if i == self._selected else 1.5),
                name=dlabel,
                connect="finite",
            )

        if self._n_frames > 1:
            self._plot.setXRange(1, self._n_frames - 1, padding=0.02)

        self._vline = pg.InfiniteLine(
            pos=self._frame,
            angle=90,
            pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._plot.addItem(self._vline)

    # ── click on plot → seek ──────────────────────────────────────────────────

    def _on_plot_click(self, ev) -> None:
        if not self._n_frames:
            return
        pos = ev.scenePos()
        vb = self._plot.plotItem.vb
        pt = vb.mapSceneToView(pg.QtCore.QPointF(pos))
        t = int(round(float(pt.x())))
        t = max(0, min(self._n_frames - 1, t))
        self._slider.setValue(t)

    # ── playback ──────────────────────────────────────────────────────────────

    def _on_slider(self, val: int) -> None:
        if self._videos:
            self._show_frame(val)

    def _step(self, delta: int) -> None:
        if self._videos:
            self._slider.setValue(
                int(np.clip(self._frame + delta, 0, self._n_frames - 1))
            )

    def _advance(self) -> None:
        if self._videos:
            self._slider.setValue((self._frame + 1) % self._n_frames)

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            self._timer.start(max(1, 1000 // self._fps.value()))
            self._play_btn.setText("⏸  Pause")


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Interactive viewer for registered TIFF stacks with per-frame metrics."
    )
    ap.add_argument(
        "--tiff-dir",
        default=None,
        metavar="DIR",
        help="Folder containing <stem>_<preset>.tiff files (or a parent folder "
        "with a tiff/ subdirectory). Loaded on startup; omit to use the "
        "folder-picker dialog.",
    )
    args = ap.parse_args(argv)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MicroViewer(tiff_dir=args.tiff_dir)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
