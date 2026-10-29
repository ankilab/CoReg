#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""OverlayViewer — PyQt6/pyqtgraph multi-channel video overlay viewer.

Run standalone:
    python overlay_viewer.py --files a.h5 b.h5 c.tiff --gt-file gt.h5
    python overlay_viewer.py          # file-picker on start

A pyqtgraph port of roi_analysis.py's matplotlib-widget ``overlay_videos``:
same blend math (percentile-normalize -> per-channel LUT colorize -> alpha
composite), same ROI and frame-pairwise metrics, same PNG export — just
native Qt widgets instead of matplotlib Slider/Button, so it stays usable
without blocking a matplotlib event loop.

Controls
--------
← →          Step one frame back / forward
PgUp/PgDn    Jump 10 frames back / forward
Home / End   Jump to first / last frame
Space        Play / pause
D            Toggle ROI-draw mode on/off
Click        Add ROI vertex (while ROI-draw mode is on)
Right-click  Close ROI polygon (needs ≥ 3 points)
Enter        Close ROI polygon (needs ≥ 3 points)
Escape       Cancel in-progress ROI

Each loaded file is one channel: toggle it on/off, pick its LUT, and set its
opacity — channels are alpha-composited on top of each other in list order.
Three projection modes (Frame / Mean / Max) control what's shown when not
scrubbing a single frame. An optional GT file always shows its frame 0
unless "GT: Video" is toggled, so it stays a fixed reference while other
channels play. The Gauss σ slider Gaussian-blurs every channel's frame data
at the source, so it's reflected everywhere: the displayed overlay, ROI
stats, frame metrics, and Save PNG. Draw an ROI to see per-channel
mean/std/min/max there; toggle "Frame Metrics" to see MSE/PSNR/ZNCC/MI/SSIM
of every other active channel against the GT (if active) or the first
active channel. "Save PNG" exports the current composite with a channel
legend attached below the image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QColor, QFont, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGraphicsPolygonItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")

try:
    from skimage.metrics import structural_similarity as _ssim_fn

    _HAVE_SSIM = True
except Exception:
    _HAVE_SSIM = False

_FRAME_JUMP = 10
_PLAY_INTERVAL_MS = 100

_LUT_NAMES = [
    "gray", "green", "magenta", "cyan", "red", "blue", "yellow",
    "hot", "inferno", "viridis",
]
_DEFAULT_LUTS = ["gray", "green", "magenta", "cyan", "red", "blue", "yellow"]

# For the saved-PNG legend swatches only — the on-screen blend always uses
# the real matplotlib colormap via _get_cmap.
_LUT_SWATCH_RGB = {
    "gray": (200, 200, 200), "green": (0, 255, 0), "magenta": (255, 0, 255),
    "cyan": (0, 255, 255), "red": (255, 0, 0), "blue": (0, 128, 255),
    "yellow": (255, 255, 0), "hot": (255, 120, 0), "inferno": (180, 60, 180),
    "viridis": (68, 180, 120),
}

_CMAP_CACHE: dict[str, mcolors.Colormap] = {}


def _get_cmap(name: str) -> mcolors.Colormap:
    cm = _CMAP_CACHE.get(name)
    if cm is None:
        if name == "gray":
            cm = mpl.colormaps["gray"]
        elif name in ("hot", "inferno", "viridis"):
            cm = mpl.colormaps[name]
        else:
            hexes = {
                "green": "#00ff00", "magenta": "#ff00ff", "cyan": "#00ffff",
                "red": "#ff0000", "blue": "#0080ff", "yellow": "#ffff00",
            }
            cm = mcolors.LinearSegmentedColormap.from_list(
                f"_{name}", ["black", hexes[name]]
            )
        _CMAP_CACHE[name] = cm
    return cm


def _normalize(frame: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(frame, 1))
    hi = float(np.percentile(frame, 99))
    if hi > lo:
        return np.clip((frame.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(frame, dtype=np.float64)


def _colorize(frame: np.ndarray, lut_name: str, opacity: float) -> np.ndarray:
    rgba = np.asarray(_get_cmap(lut_name)(_normalize(frame)), dtype=np.float64).copy()
    rgba[..., 3] = opacity
    return rgba


def _resize_to_ref(frame: np.ndarray, ref_h: int, ref_w: int) -> np.ndarray:
    if frame.shape == (ref_h, ref_w):
        return frame
    return np.array(
        Image.fromarray(frame.astype(np.float32)).resize((ref_w, ref_h), Image.BILINEAR)
    ).astype(np.float32)


def _blend(rgbas: list[np.ndarray], ref_h: int, ref_w: int) -> np.ndarray:
    if not rgbas:
        return np.zeros((ref_h, ref_w, 3), dtype=np.float32)
    result = np.zeros((ref_h, ref_w, 3), dtype=np.float64)
    for rgba in rgbas:
        result += rgba[..., :3] * rgba[..., 3:4]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _load_video(path: Path) -> np.ndarray | None:
    """Load (T, H, W) float32 from HDF5 or TIFF."""
    try:
        ext = path.suffix.lower()
        if ext in (".h5", ".hdf5"):
            with h5py.File(str(path), "r") as f:
                key = "data" if "data" in f else list(f.keys())[0]
                arr = np.array(f[key]).astype(np.float32)
        else:
            arr = tifffile.imread(str(path)).astype(np.float32)
        if arr.ndim == 4:
            arr = arr.mean(axis=-1) if arr.shape[-1] in (1, 3, 4) else arr[:, 0]
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            return None
        return arr
    except Exception as exc:
        print(f"Could not load {path.name}: {exc}", file=sys.stderr)
        return None


class _FillPolygonItem(QGraphicsPolygonItem):
    """Non-interactive translucent ROI fill (drawing-only, not draggable)."""

    def __init__(self, polygon: QPolygonF, color: QColor):
        super().__init__(polygon)
        self.setBrush(color)
        self.setPen(pg.mkPen(None))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)


def _qpolygon(pts: np.ndarray) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in pts])


class OverlayViewer(QWidget):
    """Multi-channel video overlay viewer built on PyQt6 + pyqtgraph."""

    def __init__(
        self,
        files: list[str] | None = None,
        gt_file: str | None = None,
        output_dir: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle("OverlayViewer")
        self.resize(1200, 820)

        # ── data ──────────────────────────────────────────────────────────────
        self._names: list[str] = []
        self._paths: list[Path] = []  # parallel to _names, used only to default _out_dir
        self._videos: dict[str, np.ndarray] = {}
        self._gt_name: str | None = None
        self._ref_h = 1
        self._ref_w = 1
        self._max_t = 1

        # per-channel state
        self._active: list[bool] = []
        self._lut_idx: list[int] = []
        self._opacity: list[float] = []

        self._mode: str = "frame"  # "frame" | "mean" | "max"
        self._frame: int = 0
        self._gt_show_video: bool = False
        self._gaussian_sigma: float = 0.0
        self._show_frame_metrics: bool = False
        self._playing: bool = False

        self._out_dir = Path(output_dir) if output_dir else None

        # ── ROI state ─────────────────────────────────────────────────────────
        self._roi_draw_mode: bool = False
        self._roi_current_pts: list[tuple[float, float]] = []
        self._roi_progress_items: list = []
        self._roi_pts: np.ndarray | None = None
        self._roi_final_items: list = []

        self._roi_text: pg.TextItem | None = None
        self._metrics_text: pg.TextItem | None = None

        self._channel_rows_layout: QVBoxLayout | None = None
        self._toggle_btns: list[QPushButton] = []
        self._lut_combos: list[QComboBox] = []
        self._opacity_sliders: list[QSlider] = []
        self._opacity_lbls: list[QLabel] = []
        self._remove_btns: list[QPushButton] = []

        self._timer = QTimer(self)
        self._timer.setInterval(_PLAY_INTERVAL_MS)
        self._timer.timeout.connect(self._anim_step)

        self._build_ui()

        if files:
            self._load_all(files, gt_file)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(3)

        # toolbar
        tb = QHBoxLayout()
        tb.setSpacing(6)
        btn_open = QPushButton("Open files…")
        btn_open.clicked.connect(self._on_open)
        tb.addWidget(btn_open)
        btn_gt = QPushButton("Load GT…")
        btn_gt.setToolTip("Add a ground-truth channel that always shows frame 0.")
        btn_gt.clicked.connect(self._on_open_gt)
        tb.addWidget(btn_gt)
        self._status = QLabel("No files loaded")
        self._status.setStyleSheet("color:#888;")
        tb.addWidget(self._status, stretch=1)
        root.addLayout(tb)

        # frame row
        fr = QHBoxLayout()
        fr.setSpacing(4)
        for txt, cb, w in (
            ("◀◀", lambda: self._seek(self._frame - _FRAME_JUMP), 30),
            ("◀", lambda: self._seek(self._frame - 1), 30),
        ):
            b = QPushButton(txt)
            b.setFixedWidth(w)
            b.clicked.connect(cb)
            fr.addWidget(b)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        fr.addWidget(self._slider, stretch=1)

        for txt, cb, w in (
            ("▶", lambda: self._seek(self._frame + 1), 30),
            ("▶▶", lambda: self._seek(self._frame + _FRAME_JUMP), 30),
        ):
            b = QPushButton(txt)
            b.setFixedWidth(w)
            b.clicked.connect(cb)
            fr.addWidget(b)

        fr.addWidget(QLabel("Frame:"))
        self._fspin = QSpinBox()
        self._fspin.setMinimum(0)
        self._fspin.setMaximum(0)
        self._fspin.setFixedWidth(70)
        self._fspin.editingFinished.connect(lambda: self._seek(self._fspin.value()))
        fr.addWidget(self._fspin)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setFixedWidth(80)
        self._play_btn.clicked.connect(self._toggle_play)
        fr.addWidget(self._play_btn)
        root.addLayout(fr)

        # mode / ROI / save row
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        self._mode_group = QButtonGroup(self)
        self._mode_btns: dict[str, QPushButton] = {}
        for mode in ("frame", "mean", "max"):
            b = QPushButton(mode.capitalize())
            b.setCheckable(True)
            b.setChecked(mode == "frame")
            b.setFixedWidth(70)
            b.clicked.connect(lambda _c, m=mode: self._set_mode(m))
            self._mode_group.addButton(b)
            self._mode_btns[mode] = b
            mrow.addWidget(b)
        self._update_mode_styles()

        self._roi_btn = QPushButton("Draw ROI [D]")
        self._roi_btn.setFixedWidth(95)
        self._roi_btn.setCheckable(True)
        self._roi_btn.clicked.connect(self._toggle_roi_draw)
        mrow.addWidget(self._roi_btn)

        btn_clear_roi = QPushButton("Clear ROI")
        btn_clear_roi.setFixedWidth(80)
        btn_clear_roi.clicked.connect(self._clear_roi)
        mrow.addWidget(btn_clear_roi)

        self._metrics_btn = QPushButton("Frame Metrics")
        self._metrics_btn.setCheckable(True)
        self._metrics_btn.setToolTip("MSE / PSNR / ZNCC / MI / SSIM vs. reference channel.")
        self._metrics_btn.clicked.connect(self._toggle_frame_metrics)
        mrow.addWidget(self._metrics_btn)

        mrow.addWidget(QLabel("Gauss σ:"))
        self._sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self._sigma_slider.setRange(0, 50)  # sigma = value/10
        self._sigma_slider.setFixedWidth(100)
        self._sigma_slider.setToolTip(
            "Gaussian blur applied to every channel — affects the displayed "
            "overlay, ROI stats, frame metrics, and Save PNG."
        )
        self._sigma_slider.valueChanged.connect(self._on_sigma_changed)
        mrow.addWidget(self._sigma_slider)
        self._sigma_lbl = QLabel("0.0")
        self._sigma_lbl.setFixedWidth(30)
        mrow.addWidget(self._sigma_lbl)

        self._gt_mode_btn = QPushButton("GT: Frame 0")
        self._gt_mode_btn.setFixedWidth(90)
        self._gt_mode_btn.setVisible(False)
        self._gt_mode_btn.clicked.connect(self._toggle_gt_mode)
        mrow.addWidget(self._gt_mode_btn)

        mrow.addStretch(1)
        btn_save = QPushButton("Save PNG")
        btn_save.clicked.connect(self._save_png)
        mrow.addWidget(btn_save)
        root.addLayout(mrow)

        # image area
        self._gfx = pg.GraphicsLayoutWidget()
        self._gfx.setBackground("#111111")
        self._gfx.scene().sigMouseClicked.connect(self._on_click)
        self._vb = self._gfx.addViewBox(lockAspect=True, enableMouse=True)
        self._vb.setMenuEnabled(False)
        self._vb.setDefaultPadding(0.01)
        # ViewBox Y increases upward by default; image data is row-major top-down -- without this, view is Y-flipped.
        self._vb.invertY(True)
        self._img_item = pg.ImageItem()
        self._vb.addItem(self._img_item)

        # Scrollable, user-adjustable height via splitter -- a long channel list isn't stuck in a fixed small box.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(60)
        container = QWidget()
        self._channel_rows_layout = QVBoxLayout(container)
        self._channel_rows_layout.setSpacing(2)
        self._channel_rows_layout.addStretch(1)
        scroll.setWidget(container)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._gfx)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([600, 160])
        root.addWidget(splitter, stretch=1)

        hint = QLabel(
            "← →: frame  |  PgUp/PgDn: ±10 frames  |  Home/End: first/last  |  "
            "Space: play/pause  |  D: toggle ROI draw  |  click: add ROI vertex  |  "
            "right-click/Enter: close ROI  |  Esc: cancel ROI"
        )
        hint.setStyleSheet("color:#444; font-size:10px;")
        root.addWidget(hint)

    def _update_mode_styles(self) -> None:
        for m, b in self._mode_btns.items():
            b.setStyleSheet(
                "background:#0072B2; color:white;" if m == self._mode else ""
            )

    # ── loading ───────────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open video files", "",
            "Video files (*.h5 *.hdf5 *.tiff *.tif);;All files (*)",
        )
        if paths:
            self._load_all(paths, gt_path=None, append=True)

    def _on_open_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ground-truth file", "",
            "Video files (*.h5 *.hdf5 *.tiff *.tif);;All files (*)",
        )
        if path:
            self._load_all([], gt_path=path, append=True)

    def _load_all(
        self, paths: list[str], gt_path: str | None, append: bool = False
    ) -> None:
        if not append:
            self._names = []
            self._paths = []
            self._videos = {}
            self._gt_name = None
            self._active = []
            self._lut_idx = []
            self._opacity = []

        for p_s in paths:
            p = Path(p_s)
            v = _load_video(p)
            if v is None:
                self._status.setText(f"Failed to load: {p.name}")
                continue
            self._names.append(p.name)
            self._paths.append(p)
            self._videos[p.name] = v

        if gt_path is not None:
            p = Path(gt_path)
            v = _load_video(p)
            if v is not None:
                key = f"{p.name} (GT)"
                self._names.append(key)
                self._paths.append(p)
                self._videos[key] = v
                self._gt_name = key
            else:
                self._status.setText(f"Failed to load GT: {p.name}")

        if not self._videos:
            return

        n = len(self._names)
        while len(self._active) < n:
            i = len(self._active)
            self._active.append(True)
            self._lut_idx.append(_LUT_NAMES.index(_DEFAULT_LUTS[i % len(_DEFAULT_LUTS)]))
            self._opacity.append(1.0)

        if self._out_dir is None and self._paths:
            self._out_dir = self._paths[0].parent / "overlay"

        self._max_t = max(v.shape[0] for v in self._videos.values())
        self._ref_h, self._ref_w = next(iter(self._videos.values())).shape[1:]

        self._gt_mode_btn.setVisible(self._gt_name is not None)

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._max_t - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._max_t - 1))

        self._rebuild_channel_rows()
        self._status.setText(f"{n} channel(s)  —  {self._max_t} frames")
        # autoRange must run after _refresh() -- on first load the ImageItem has no data yet to fit to.
        self._refresh()
        self._vb.autoRange()

    def _rebuild_channel_rows(self) -> None:
        assert self._channel_rows_layout is not None
        while self._channel_rows_layout.count():
            item = self._channel_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._toggle_btns = []
        self._lut_combos = []
        self._opacity_sliders = []
        self._opacity_lbls = []
        self._remove_btns = []

        for i, name in enumerate(self._names):
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(220)
            lbl.setStyleSheet("color:black;" if self._gt_name != name else "color:#8a6d00; font-weight:bold;")
            row.addWidget(lbl)

            tog = QPushButton("●")
            tog.setCheckable(True)
            tog.setChecked(self._active[i])
            tog.setFixedWidth(28)
            tog.setStyleSheet(self._toggle_style(self._active[i]))
            tog.clicked.connect(lambda _c, idx=i: self._on_toggle(idx))
            self._toggle_btns.append(tog)
            row.addWidget(tog)

            combo = QComboBox()
            combo.addItems(_LUT_NAMES)
            combo.setCurrentIndex(self._lut_idx[i])
            combo.setFixedWidth(100)
            combo.currentIndexChanged.connect(lambda idx, ci=i: self._on_lut(ci, idx))
            self._lut_combos.append(combo)
            row.addWidget(combo)

            row.addWidget(QLabel("opacity"))
            op_slider = QSlider(Qt.Orientation.Horizontal)
            op_slider.setRange(0, 100)
            op_slider.setValue(round(self._opacity[i] * 100))
            op_slider.valueChanged.connect(lambda val, ci=i: self._on_opacity(ci, val))
            self._opacity_sliders.append(op_slider)
            row.addWidget(op_slider)

            op_lbl = QLabel(f"{self._opacity[i]:.2f}")
            op_lbl.setFixedWidth(35)
            self._opacity_lbls.append(op_lbl)
            row.addWidget(op_lbl)

            rem = QPushButton("✕")
            rem.setFixedWidth(24)
            rem.setToolTip("Remove this channel.")
            rem.setStyleSheet("color:#e34948;")
            rem.clicked.connect(lambda _c, idx=i: self._remove_channel(idx))
            self._remove_btns.append(rem)
            row.addWidget(rem)

            wrapper = QWidget()
            wrapper.setLayout(row)
            self._channel_rows_layout.insertWidget(i, wrapper)

    @staticmethod
    def _toggle_style(active: bool) -> str:
        color = "#009E73" if active else "#555555"
        return f"background:{color}; color:white;"

    def _remove_channel(self, idx: int) -> None:
        if idx >= len(self._names):
            return
        name = self._names[idx]
        # ref_h/ref_w come from the first-loaded video's shape (see _load_all);
        # removing that one means every other channel's resize target shifts.
        was_ref = idx == 0

        del self._names[idx]
        del self._paths[idx]
        del self._active[idx]
        del self._lut_idx[idx]
        del self._opacity[idx]
        self._videos.pop(name, None)

        if name == self._gt_name:
            self._gt_name = None
            self._gt_show_video = False
            self._gt_mode_btn.setVisible(False)
            self._gt_mode_btn.setText("GT: Frame 0")

        if not self._videos:
            self._reset_empty()
            return

        self._max_t = max(v.shape[0] for v in self._videos.values())
        if was_ref:
            new_h, new_w = next(iter(self._videos.values())).shape[1:]
            if (new_h, new_w) != (self._ref_h, self._ref_w):
                self._ref_h, self._ref_w = new_h, new_w
                # Existing ROI coords were drawn against the old reference size -- no longer meaningful.
                self._clear_roi()

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._max_t - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._max_t - 1))
        if self._frame >= self._max_t:
            self._seek(self._max_t - 1)

        self._rebuild_channel_rows()
        self._status.setText(f"{len(self._names)} channel(s)  —  {self._max_t} frames")
        self._refresh()

    def _reset_empty(self) -> None:
        self._videos = {}
        self._names = []
        self._paths = []
        self._active = []
        self._lut_idx = []
        self._opacity = []
        self._gt_name = None
        self._gt_show_video = False
        self._gt_mode_btn.setVisible(False)
        self._max_t = 1
        self._ref_h = 1
        self._ref_w = 1
        self._clear_roi()
        self._img_item.clear()
        self._slider.blockSignals(True)
        self._slider.setMaximum(0)
        self._slider.blockSignals(False)
        self._fspin.setMaximum(0)
        self._rebuild_channel_rows()
        self._status.setText("No files loaded")

    # ── channel state callbacks ───────────────────────────────────────────────

    def _on_toggle(self, idx: int) -> None:
        self._active[idx] = not self._active[idx]
        self._toggle_btns[idx].setChecked(self._active[idx])
        self._toggle_btns[idx].setStyleSheet(self._toggle_style(self._active[idx]))
        self._refresh()

    def _on_lut(self, idx: int, lut_idx: int) -> None:
        self._lut_idx[idx] = lut_idx
        self._refresh()

    def _on_opacity(self, idx: int, val: int) -> None:
        self._opacity[idx] = val / 100.0
        self._opacity_lbls[idx].setText(f"{self._opacity[idx]:.2f}")
        self._refresh()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._update_mode_styles()
        self._refresh()

    def _toggle_gt_mode(self) -> None:
        self._gt_show_video = not self._gt_show_video
        self._gt_mode_btn.setText("GT: Video" if self._gt_show_video else "GT: Frame 0")
        self._refresh()

    def _on_sigma_changed(self, val: int) -> None:
        self._gaussian_sigma = val / 10.0
        self._sigma_lbl.setText(f"{self._gaussian_sigma:.1f}")
        self._refresh()

    def _toggle_frame_metrics(self) -> None:
        self._show_frame_metrics = not self._show_frame_metrics
        self._refresh()

    # ── frame navigation ──────────────────────────────────────────────────────

    def _on_slider(self, val: int) -> None:
        if not self._videos:
            return
        self._fspin.blockSignals(True)
        self._fspin.setValue(val)
        self._fspin.blockSignals(False)
        self._frame = val
        self._refresh()

    def _seek(self, t: int) -> None:
        if self._max_t:
            self._slider.setValue(int(np.clip(t, 0, self._max_t - 1)))

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._play_btn.setText("⏸ Pause")
            self._timer.start()
        else:
            self._play_btn.setText("▶ Play")
            self._timer.stop()

    def _anim_step(self) -> None:
        if not self._max_t:
            return
        self._seek((self._frame + 1) % self._max_t)

    # ── blend / metrics ───────────────────────────────────────────────────────

    def _channel_frame(self, name: str, frame_idx: int) -> np.ndarray:
        v = self._videos[name]
        if name == self._gt_name and not self._gt_show_video:
            frame = v[0]
        elif self._mode == "mean":
            frame = v.mean(axis=0)
        elif self._mode == "max":
            frame = v.max(axis=0)
        else:
            frame = v[min(frame_idx, v.shape[0] - 1)]
        # Applied here so every consumer sees the same smoothed data -- previously only metrics got it, not the display.
        if self._gaussian_sigma > 0:
            frame = gaussian_filter(frame.astype(np.float32), sigma=self._gaussian_sigma)
        return frame

    def _compute_overlay(self, frame_idx: int) -> np.ndarray:
        rgbas = []
        for i, name in enumerate(self._names):
            if not self._active[i]:
                continue
            frame = _resize_to_ref(
                self._channel_frame(name, frame_idx), self._ref_h, self._ref_w
            )
            rgbas.append(_colorize(frame, _LUT_NAMES[self._lut_idx[i]], self._opacity[i]))
        return _blend(rgbas, self._ref_h, self._ref_w)

    def _compute_roi_metrics(self, frame_idx: int) -> dict[str, dict]:
        if self._roi_pts is None:
            return {}
        ys, xs = np.mgrid[0 : self._ref_h, 0 : self._ref_w]
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        mask = MplPath(self._roi_pts).contains_points(pts).reshape(self._ref_h, self._ref_w)
        if not mask.any():
            return {}
        n_pixels = int(mask.sum())
        metrics: dict[str, dict] = {}
        for i, name in enumerate(self._names):
            if not self._active[i]:
                continue
            frame = _resize_to_ref(
                self._channel_frame(name, frame_idx), self._ref_h, self._ref_w
            )
            vals = frame[mask]
            metrics[name] = {
                "mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max()),
                "n_pixels": n_pixels,
            }
        return metrics

    def _normalized_for_metrics(self, name: str, frame_idx: int) -> np.ndarray:
        frame = _resize_to_ref(
            self._channel_frame(name, frame_idx), self._ref_h, self._ref_w
        ).astype(np.float64)
        lo = float(np.percentile(frame, 1))
        hi = float(np.percentile(frame, 99))
        if hi > lo:
            return np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
        return np.zeros_like(frame)

    def _compute_frame_metrics(self, frame_idx: int) -> list[dict]:
        active_names = [self._names[i] for i in range(len(self._names)) if self._active[i]]
        if len(active_names) < 2:
            return []
        if self._gt_name is not None and self._gt_name in active_names:
            ref_name = self._gt_name
            cmp_names = [n for n in active_names if n != self._gt_name]
        else:
            ref_name, *cmp_names = active_names

        ref_frame = self._normalized_for_metrics(ref_name, frame_idx)
        ref_flat = ref_frame.ravel()
        results = []
        for other in cmp_names:
            cur_frame = self._normalized_for_metrics(other, frame_idx)
            cur_flat = cur_frame.ravel()

            mse = float(np.mean((ref_flat - cur_flat) ** 2))
            psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("nan")

            ref_c = ref_flat - ref_flat.mean()
            cur_c = cur_flat - cur_flat.mean()
            denom = np.sqrt(np.dot(ref_c, ref_c) * np.dot(cur_c, cur_c)) + 1e-12
            zncc = float(np.dot(ref_c, cur_c) / denom)

            try:
                hist2d, _, _ = np.histogram2d(ref_flat, cur_flat, bins=64)
                pxy = hist2d / (hist2d.sum() + 1e-12)
                px, py = pxy.sum(axis=1), pxy.sum(axis=0)

                def _ent(p: np.ndarray) -> float:
                    p = p[p > 0]
                    return -float(np.sum(p * np.log2(p)))

                mi = _ent(px) + _ent(py) - _ent(pxy.ravel())
            except Exception:
                mi = float("nan")

            ref_u8 = np.clip(ref_frame * 255.0, 0, 255).astype(np.uint8)
            cur_u8 = np.clip(cur_frame * 255.0, 0, 255).astype(np.uint8)
            ssim = (
                float(_ssim_fn(ref_u8, cur_u8, data_range=255)) if _HAVE_SSIM else float("nan")
            )

            results.append(
                {"ref": ref_name, "cmp": other, "mse": mse, "psnr": psnr,
                 "zncc": zncc, "mi": mi, "ssim": ssim}
            )
        return results

    # ── ROI drawing ───────────────────────────────────────────────────────────

    def _toggle_roi_draw(self, checked: bool) -> None:
        self._roi_draw_mode = checked
        self._vb.setMouseEnabled(x=not checked, y=not checked)
        if not checked:
            self._cancel_roi_draw()

    def _on_click(self, ev) -> None:
        if not self._roi_draw_mode:
            return
        pos = ev.scenePos()
        if not self._vb.sceneBoundingRect().contains(pos):
            return
        btn = ev.button()

        if btn == Qt.MouseButton.RightButton or ev.double():
            if len(self._roi_current_pts) >= 3:
                self._close_roi()
            else:
                self._cancel_roi_draw()
            ev.accept()
            return

        if btn != Qt.MouseButton.LeftButton:
            return
        pt = self._vb.mapSceneToView(QPointF(pos.x(), pos.y()))
        self._roi_current_pts.append((float(pt.x()), float(pt.y())))
        self._draw_roi_progress()
        ev.accept()

    def _draw_roi_progress(self) -> None:
        self._clear_roi_progress_items()
        pts = self._roi_current_pts
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        line = pg.PlotCurveItem(
            x=xs, y=ys,
            pen=pg.mkPen("#F0E442", width=1.5, style=Qt.PenStyle.DashLine),
            connect="all",
        )
        self._vb.addItem(line)
        self._roi_progress_items.append(line)
        scatter = pg.ScatterPlotItem(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            symbol="o", size=7,
            pen=pg.mkPen("#F0E442"), brush=pg.mkBrush("#F0E442"),
        )
        self._vb.addItem(scatter)
        self._roi_progress_items.append(scatter)

    def _clear_roi_progress_items(self) -> None:
        for item in self._roi_progress_items:
            try:
                self._vb.removeItem(item)
            except Exception:
                pass
        self._roi_progress_items = []

    def _cancel_roi_draw(self) -> None:
        self._clear_roi_progress_items()
        self._roi_current_pts = []

    def _close_roi(self) -> None:
        self._roi_pts = np.array(self._roi_current_pts, dtype=np.float32)
        self._cancel_roi_draw()
        self._draw_roi_final()
        self._roi_btn.setChecked(False)
        self._roi_draw_mode = False
        self._vb.setMouseEnabled(x=True, y=True)
        self._refresh()

    def _draw_roi_final(self) -> None:
        self._remove_roi_final_items()
        if self._roi_pts is None:
            return
        fill = _FillPolygonItem(_qpolygon(self._roi_pts), QColor(255, 255, 0, 20))
        self._vb.addItem(fill)
        xs = np.append(self._roi_pts[:, 0], self._roi_pts[0, 0])
        ys = np.append(self._roi_pts[:, 1], self._roi_pts[0, 1])
        outline = pg.PlotCurveItem(
            x=xs, y=ys, pen=pg.mkPen("#F0E442", width=1.5), connect="all"
        )
        self._vb.addItem(outline)
        self._roi_final_items = [fill, outline]

    def _remove_roi_final_items(self) -> None:
        for item in self._roi_final_items:
            try:
                self._vb.removeItem(item)
            except Exception:
                pass
        self._roi_final_items = []

    def _clear_roi(self) -> None:
        self._roi_pts = None
        self._remove_roi_final_items()
        self._cancel_roi_draw()
        self._roi_btn.setChecked(False)
        self._roi_draw_mode = False
        self._vb.setMouseEnabled(x=True, y=True)
        self._refresh()

    # ── on-screen metrics text ────────────────────────────────────────────────

    def _update_roi_text(self) -> None:
        if self._roi_text is not None:
            try:
                self._vb.removeItem(self._roi_text)
            except Exception:
                pass
            self._roi_text = None
        if self._roi_pts is None:
            return
        m = self._compute_roi_metrics(self._frame)
        if not m:
            return
        n_px = next(iter(m.values()))["n_pixels"]
        lines = [f"ROI ({n_px} px)"]
        for name, s in m.items():
            short = (name[:16] + "…") if len(name) > 16 else name
            lines.append(
                f"{short}  μ={s['mean']:.1f}  σ={s['std']:.1f}  [{s['min']:.0f}–{s['max']:.0f}]"
            )
        html = (
            '<div style="background-color:rgba(0,0,0,180); border:1px solid #F0E442; '
            'padding:4px; font-size:8pt; color:white;">' + "<br>".join(lines) + "</div>"
        )
        self._roi_text = pg.TextItem(html=html, anchor=(0, 1))
        self._roi_text.setPos(2, self._ref_h - 2)
        self._vb.addItem(self._roi_text)

    def _update_metrics_text(self) -> None:
        if self._metrics_text is not None:
            try:
                self._vb.removeItem(self._metrics_text)
            except Exception:
                pass
            self._metrics_text = None
        if not self._show_frame_metrics:
            return
        results = self._compute_frame_metrics(self._frame)
        if not results:
            html = (
                '<div style="background-color:rgba(0,0,0,180); border:1px solid #56B4E9; '
                'padding:4px; font-size:8pt; color:white;">Need ≥ 2 active channels</div>'
            )
        else:
            gauss_tag = f"  Gauss σ={self._gaussian_sigma:.1f}" if self._gaussian_sigma > 0 else ""
            ref_short = (
                (results[0]["ref"][:14] + "…") if len(results[0]["ref"]) > 14 else results[0]["ref"]
            )
            lines = [f"Frame metrics  ref: {ref_short}{gauss_tag}"]

            def _fmt(v: float, d: int = 3) -> str:
                return f"{v:.{d}f}" if np.isfinite(v) else "N/A"

            for r in results:
                cmp_short = (r["cmp"][:14] + "…") if len(r["cmp"]) > 14 else r["cmp"]
                lines.append(
                    f"vs {cmp_short}: MSE={_fmt(r['mse'], 4)} PSNR={_fmt(r['psnr'], 1)}dB "
                    f"ZNCC={_fmt(r['zncc'])} SSIM={_fmt(r['ssim'])} MI={_fmt(r['mi'])}"
                )
            html = (
                '<div style="background-color:rgba(0,0,0,180); border:1px solid #56B4E9; '
                'padding:4px; font-size:8pt; color:white;">' + "<br>".join(lines) + "</div>"
            )
        self._metrics_text = pg.TextItem(html=html, anchor=(1, 1))
        self._metrics_text.setPos(self._ref_w - 2, self._ref_h - 2)
        self._vb.addItem(self._metrics_text)

    # ── refresh / save ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self._videos:
            return
        self._frame = int(self._slider.value())
        overlay = self._compute_overlay(self._frame)
        self._img_item.setImage(overlay, autoLevels=False, levels=(0.0, 1.0))
        mode_str = {"frame": f"Frame {self._frame}", "mean": "Mean projection",
                    "max": "Max projection"}[self._mode]
        self.setWindowTitle(f"OverlayViewer — {mode_str}")
        self._update_roi_text()
        self._update_metrics_text()

    def _save_png(self) -> None:
        if not self._videos:
            self._status.setText("Nothing to save — load files first.")
            return
        fi = self._frame
        tag = self._mode if self._mode != "frame" else f"frame{fi:04d}"
        default_dir = self._out_dir if self._out_dir else Path("overlay").resolve()
        default_path = str(default_dir / f"overlay_{tag}.png")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save overlay PNG", default_path, "PNG files (*.png)"
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            self._save_png_to(path)
        except Exception as exc:
            # Without this, a bad output path fails silently -- Qt only prints the traceback to stderr.
            self._status.setText(f"Save failed: {exc}")
            print(f"Save PNG failed: {exc!r}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            # Remember the chosen folder as the default for next time.
            self._out_dir = path.parent

    def _save_png_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fi = self._frame

        arr = (self._compute_overlay(fi) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        lines = []
        for i, name in enumerate(self._names):
            if not self._active[i]:
                continue
            lut = _LUT_NAMES[self._lut_idx[i]]
            op = self._opacity[i]
            suffix = " (GT)" if name == self._gt_name else ""
            lines.append(f"{name}{suffix}  [{lut}  α={op:.2f}]")

        if lines:
            font = None
            for fname in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
                try:
                    font = ImageFont.truetype(fname, size=11)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()

            # Legend is a separate strip attached below the image (canvas is
            # extended), not an overlay composited on top of image content.
            pad, line_h, swatch = 4, 14, 8
            text_h = len(lines) * line_h + pad * 2
            banner = Image.new("RGB", (img.width, text_h), (0, 0, 0))
            draw = ImageDraw.Draw(banner)
            active_idxs = [i for i in range(len(self._names)) if self._active[i]]
            for row, i in enumerate(active_idxs):
                lut = _LUT_NAMES[self._lut_idx[i]]
                color = _LUT_SWATCH_RGB.get(lut, (255, 255, 255))
                y_txt = pad + row * line_h
                draw.rectangle(
                    [pad, y_txt + 2, pad + swatch, y_txt + 2 + swatch], fill=color
                )
                draw.text((pad + swatch + 3, y_txt), lines[row], fill=(255, 255, 255), font=font)

            combined = Image.new("RGB", (img.width, img.height + text_h))
            combined.paste(img, (0, 0))
            combined.paste(banner, (0, img.height))
            img = combined

        img.save(path)
        self._status.setText(f"Saved → {path}")
        print(f"Saved: {path}")

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k in (Qt.Key.Key_Right, Qt.Key.Key_Period):
            self._seek(self._frame + 1)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Comma):
            self._seek(self._frame - 1)
        elif k in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            step = _FRAME_JUMP if k == Qt.Key.Key_PageDown else -_FRAME_JUMP
            self._seek(self._frame + step)
        elif k == Qt.Key.Key_Home:
            self._seek(0)
        elif k == Qt.Key.Key_End:
            self._seek(self._max_t - 1)
        elif k == Qt.Key.Key_Space:
            self._toggle_play()
        elif k == Qt.Key.Key_D:
            self._roi_btn.setChecked(not self._roi_btn.isChecked())
            self._toggle_roi_draw(self._roi_btn.isChecked())
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._roi_draw_mode and len(self._roi_current_pts) >= 3:
                self._close_roi()
        elif k == Qt.Key.Key_Escape:
            if self._roi_current_pts:
                self._cancel_roi_draw()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev) -> None:
        self._timer.stop()
        super().closeEvent(ev)


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Interactive multi-channel overlay viewer for registered videos."
    )
    ap.add_argument(
        "--files", nargs="+", default=None, metavar="FILE",
        help="HDF5 or TIFF video file(s) to overlay.",
    )
    ap.add_argument(
        "--gt-file", default=None,
        help="Optional ground-truth HDF5/TIFF, always shown at frame 0 unless toggled.",
    )
    ap.add_argument("--out-dir", default=None, help="Where Save PNG writes files.")
    args = ap.parse_args(argv)

    display = (
        sys.platform == "darwin"
        or bool(
            __import__("os").environ.get("DISPLAY")
            or __import__("os").environ.get("WAYLAND_DISPLAY")
        )
    )
    if not display:
        sys.exit(
            "No display found.\n"
            "  • From a desktop terminal: just run this command there.\n"
            "  • Over SSH: reconnect with  ssh -X user@host  for X11 forwarding,\n"
            "    then set  export DISPLAY=:0  if needed."
        )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = OverlayViewer(files=args.files, gt_file=args.gt_file, output_dir=args.out_dir)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
