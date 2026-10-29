#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""Shared base for the cell-annotation tools (cell_activity.py, cell_area.py).

Not runnable standalone -- see those two modules for the actual entry
points. Everything here is common to both: loading video panels, drawing
and editing polygon "cells" by hand, CoTracker-based "Track cells", "Refine"
(edge-snapping), "Copy area to next frames" (freezing a static snapshot),
the "Areas:" sidebar, sessions, and the shared row/area/intensity
computation behind Export (`_compute_export_rows`). `CellActivity` adds
"Measure luminance" (ΔF/F) on top of this; `CellArea` adds nothing beyond
what's here. They each subclass `CellAnnotatorBase` and fill in two
extension points -- see `_build_measurement_toolbar` /
`_build_measurement_panel`, both no-ops in the base -- plus a couple of
stub methods (`_refresh_lum_list`, `_update_playhead`) that only
`CellActivity` overrides. `CellActivity` also fully overrides `_export()`
itself, to write a single combined CSV instead of this base class's
multi-file output (CSVs + JSON + images + plots) -- see its own _export().

A drawn region is one cell_id shared across every panel, which can have up
to three simultaneous *kinds* per panel: "manual" (as drawn/edited),
"tracked" (CoTracker, via "Track cells" -- available in both tools), and
"frozen" (a static snapshot held across N frames, via "Copy area to next
frames"). Tracking a moving cell is useful independent of luminance --
e.g. following its area/size over time as it drifts -- so it isn't
activity-specific.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
import warnings
from typing import Callable
from pathlib import Path

# pyqtgraph probes for torch on import -> noisy CUDA warning; suppressed before import.
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", message=".*cudaGetDeviceCount.*")

import cv2
import h5py
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPolygonF
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsPolygonItem,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")

# High-saturation palette -- plain black (prior Okabe-Ito palette) is invisible on the dark canvas.
_COLORS = [
    "#FF3B30",  # vivid red
    "#00E5FF",  # electric cyan
    "#39FF14",  # neon green
    "#FFD60A",  # vivid gold
    "#FF2D95",  # hot pink
    "#7C4DFF",  # vivid violet
    "#FF9500",  # vivid orange
    "#FFFFFF",  # white
]

_FRAME_JUMP = 10  # frames skipped by Page Up / Page Down
_FILL_ALPHA = round(255 * 0.6)  # cell-area fill opacity
# Default 5px vertex-handle radius is fiddly to grab; enlarged for easier dragging only.
_HANDLE_RADIUS = 9


def _enlarge_roi_handles(roi: "pg.PolyLineROI", radius: int = _HANDLE_RADIUS) -> None:
    """Resize every vertex handle on *roi* to *radius* px. ROI.handleSize
    (the constructor-time default, 5px) only affects handles added *after*
    it's changed, and PolyLineROI's own __init__ already adds all of its
    handles before returning -- so this instead mutates each already-created
    Handle directly and replays the same setup its own __init__ does after
    assigning a new radius (rebuild the path, drop the cached transformed
    shape, repaint), which is the only way to resize handles post hoc."""
    for h in roi.getHandles():
        h.radius = radius
        h.buildPath()
        h._shape = None
        h.update()


def _color(idx: int, alpha: int = 255) -> QColor:
    """Distinguishable color for palette index *idx* -- e.g. a cell_id via
    CellAnnotatorBase._cell_color_index, so distinct cells/plots stay
    visually distinguishable (see that method's docstring for why the index
    is cell_id alone). The first len(_COLORS) indices use the hand-picked,
    high-contrast palette above; any index beyond that generates a further
    color by stepping the hue by the golden angle (~137.5°) instead of
    wrapping back to _COLORS[0] -- consecutive hues chosen this way stay
    well-separated indefinitely, so a 9th, 10th, ... cell keeps getting its
    own distinct shade rather than reusing an earlier cell's color."""
    if idx < len(_COLORS):
        c = QColor(_COLORS[idx])
    else:
        hue = int((idx * 137.508) % 360)
        c = QColor.fromHsv(hue, 210, 255)
    c.setAlpha(alpha)
    return c


def _qpolygon(pts: np.ndarray) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in pts])


class _NoInsertPolyLineROI(pg.PolyLineROI):
    """A PolyLineROI with pyqtgraph's built-in "click an edge to insert a
    new vertex there" gesture (ROI.segmentClicked) disabled.

    Nothing in this app exposes or hints at that gesture, so a user has no
    reason to expect it — and it has no sanity check on the click position,
    so any stray click that lands on a segment (confirmed cause of a
    reported bug: a QMessageBox positioned over the video, dismissed with a
    double-click, let the second click bleed through onto the ROI
    underneath) silently splices in a vertex at that raw scene position,
    however far it is from the rest of the polygon. Overriding the method
    on a subclass (rather than reassigning the instance attribute after
    construction) is required — PolyLineROI.__init__ already connects each
    segment's sigClicked to self.segmentClicked before this class could
    monkeypatch the instance, so only a real method override is seen by
    that already-made connection.
    """

    def segmentClicked(self, segment, ev=None, pos=None) -> None:
        pass


class _FillPolygonItem(QGraphicsPolygonItem):
    """Translucent cell-area fill.

    pg.PolyLineROI.shape() is just the thin stroke through its handles, not
    the filled interior — clicking inside a cell (not exactly on a vertex or
    edge) doesn't hit the ROI at all and falls through to the ViewBox
    (panning the view instead of moving the cell). When *on_drag* is given,
    this item accepts left-button drags itself and forwards the per-move
    delta, so dragging anywhere inside the cell moves the whole polygon.
    """

    def __init__(
        self,
        polygon: QPolygonF,
        on_drag=None,
        roi: "pg.PolyLineROI | None" = None,
        on_select=None,
    ):
        super().__init__(polygon)
        self._on_drag = on_drag
        self._roi = roi
        self._on_select = on_select
        if on_drag is not None:
            self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif on_select is not None:
            # Selectable but not draggable -- e.g. a "frozen" area shown on
            # a frame other than its anchor frame: clicking it should
            # select it, but there's nowhere to store a per-frame drag edit
            # for a shape that's deliberately constant.
            self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        else:
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def _on_a_handle(self, item_pos: QPointF) -> bool:
        """Whether *item_pos* (this item's local coords) lands on a handle.

        A vertex handle sits spatially on top of the fill, but stacking
        order across nested item groups doesn't reliably give it priority
        here — so defer explicitly rather than stealing its click.
        """
        if self._roi is None:
            return False
        scene_pos = self.mapToScene(item_pos)
        for h in self._roi.getHandles():
            if h.boundingRect().contains(h.mapFromScene(scene_pos)):
                return True
        return False

    def mousePressEvent(self, ev) -> None:
        if self._on_a_handle(ev.pos()):
            ev.ignore()
            return
        # Consuming the press (needed for dragging) blocks the scene-level click-select signal -- notify directly instead.
        if self._on_select is not None:
            self._on_select()
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        delta = ev.pos() - ev.lastPos()
        if self._on_drag is not None and not delta.isNull():
            self._on_drag(delta)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        ev.accept()


def _fill_item(
    pts: np.ndarray, color_idx: int, alpha: int, on_drag=None, roi=None, on_select=None
) -> QGraphicsPolygonItem:
    """A translucent fill for a cell polygon.

    No explicit z-value: the image item is added once when the panel is
    built, before any per-frame overlay — so overlays (added later, each
    redraw) already paint on top of it via insertion order. Since this fill
    is always added before its matching outline/ROI, it stays visually
    beneath that outline too (so vertex handles/edges still take priority
    over whole-polygon dragging).
    """
    item = _FillPolygonItem(_qpolygon(pts), on_drag=on_drag, roi=roi, on_select=on_select)
    item.setBrush(QBrush(_color(color_idx, alpha)))
    item.setPen(pg.mkPen(None))
    return item


def _load_video(
    path: Path, frame_range: tuple[int, int] | None = None
) -> np.ndarray | None:
    """Load (T, H, W) float32 from HDF5 or TIFF.

    If *frame_range* (start, end), inclusive, is given, only those frames are
    read off disk (clamped to the file's own length) instead of the whole
    stack — used so additional stacks only load the section the user picked
    on the first one.
    """
    try:
        ext = path.suffix.lower()
        if ext in (".h5", ".hdf5"):
            with h5py.File(str(path), "r") as f:
                key = next(
                    (
                        k
                        for k in ("data", "video", "images", "stack", "compensated")
                        if k in f
                    ),
                    list(f.keys())[0],
                )
                ds = f[key]
                if frame_range is not None:
                    start, end = frame_range
                    end = min(end, ds.shape[0] - 1)
                    start = min(start, end)
                    arr = np.array(ds[start : end + 1], dtype=np.float32)
                else:
                    arr = np.array(ds, dtype=np.float32)
        elif frame_range is not None:
            start, end = frame_range
            with tifffile.TiffFile(str(path)) as tf:
                end = min(end, len(tf.pages) - 1)
                start = min(start, end)
            arr = tifffile.imread(str(path), key=range(start, end + 1)).astype(
                np.float32
            )
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


def _to_uint8_stack(video: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Percentile contrast-stretch a (T, H, W) array to uint8.

    Bounds are computed once from the whole stack, not per frame, so real
    intensity changes over time survive instead of being normalized away.
    Registered outputs are typically already 8-bit (registration algorithms
    quantize to uint8 internally), while a raw source video is often 16-bit
    with a much higher native scale — measuring mean_intensity straight off
    the raw arrays would make "mean intensity over time" incomparable across
    files. Converting every source to this same 8-bit scale first keeps the
    measurement on equal footing.
    """
    lo, hi = np.percentile(video, (p_low, p_high))
    if hi <= lo:
        return np.zeros(video.shape, dtype=np.uint8)
    v = np.clip((video.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (v * 255).astype(np.uint8)


def _cell_id_sort_key(cid: str) -> tuple[int, str]:
    """Sort "cell_id" export labels (see CellAnnotatorBase._cell_label) so
    "10" sorts after "3a"/"3b", not before them as a plain lexicographic
    string sort would (since "1" < "3" as characters)."""
    i = 0
    while i < len(cid) and cid[i].isdigit():
        i += 1
    return (int(cid[:i]) if i else 0, cid[i:])


# ── CoTracker background worker ───────────────────────────────────────────────


class _TrackWorker(QThread):
    """Run CoTracker in a background thread."""

    sig_progress = pyqtSignal(str)
    sig_done = pyqtSignal(object)   # np.ndarray (T, N, 2), or str error message on failure

    def __init__(
        self,
        video_thw: np.ndarray,
        query_points: np.ndarray,
        ref_frame: int,
    ):
        super().__init__()
        self._video = video_thw          # (T, H, W) float32
        self._query_points = query_points  # (N, 2) float32 (x, y)
        self._ref_frame = ref_frame

    def run(self) -> None:
        try:
            import torch

            _src = Path(__file__).resolve().parent
            src = _src.parent.parent / "src" / "CoReg"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from utils.cotracker_pipeline import run_cotracker_pipeline

            # (T, 1, H, W) — pipeline handles single→3-channel and normalisation
            video_t = torch.from_numpy(self._video[:, np.newaxis]).float()

            # queries (N, 3): [t_ref, x, y]
            N = len(self._query_points)
            queries_np = np.zeros((N, 3), dtype=np.float32)
            queries_np[:, 0] = float(self._ref_frame)
            queries_np[:, 1] = self._query_points[:, 0]   # x = col
            queries_np[:, 2] = self._query_points[:, 1]   # y = row
            queries_t = torch.from_numpy(queries_np)

            def _cb(msg: str) -> None:
                self.sig_progress.emit(msg)

            # compute_warp=False: skips the unused dense TPS warp, which used to OOM when tracking a 2nd cell.
            _, tracks, _ = run_cotracker_pipeline(
                video_t, queries_t, progress_cb=_cb, compute_warp=False
            )
            # tracks: (T, N, 2)
            self.sig_done.emit(np.asarray(tracks, dtype=np.float32))

        except Exception as exc:
            # QThread swallows uncaught exceptions silently -- print traceback so "see terminal" is actually true.
            import traceback
            traceback.print_exc()
            self.sig_done.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # Belt-and-suspenders: release GPU cache regardless of outcome, so a failed run doesn't pin memory.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ── Main widget ───────────────────────────────────────────────────────────────


class CellAnnotatorBase(QWidget):
    """Interactive polygon annotator built on PyQt6 + pyqtgraph -- shared by
    CellActivity and CellArea. See the module docstring for the split."""

    def __init__(self, files: list[str] | None = None):
        super().__init__()
        self.resize(1300, 820)

        # ── data ──────────────────────────────────────────────────────────────
        self._paths: list[Path] = []
        self._videos: list[np.ndarray] = []        # (T, H, W) float32
        self._ranges: list[tuple[float, float]] = []
        self._n_frames: int = 0
        self._frame: int = 0

        # Monotonic, never reused -- shared cell identity across panels, immune to per-panel list-index drift.
        self._next_cell_id: int = 0

        # polygons[panel][frame] = {cell_id: np.ndarray (N, 2)}
        self._polygons: list[dict[int, dict[int, np.ndarray]]] = []

        # ── section state ────────────────────────────────────────────────────
        # (start, end) of the first stack, locked once a 2nd file is added -- later stacks read only this range off disk.
        self._section: tuple[int, int] | None = None

        # Last save/load path -- lets Ctrl+S (_quick_save_session) re-save silently without a dialog.
        self._session_path: Path | None = None

        # ── drawing state ──────────────────────────────────────────────────────
        self._mode: str = "pan"
        self._active: int = 0
        self._current_pts: list[tuple[float, float]] = []

        # ── display state ─────────────────────────────────────────────────────
        self._fill_alpha: int = _FILL_ALPHA  # cell-area fill opacity, 0-255
        self._blur_sigma: float = 1.0        # display-only Gaussian blur, 0 = off

        # "kind" is load-bearing: one cell_id can have up to 3 simultaneous kinds (manual/tracked/frozen) on a panel.
        self._selected_area: tuple[int, int, str] | None = None

        # panel -> (id(video array), uint8-normalized copy) -- recomputed only when the array is replaced, not per frame.
        self._dff_video_cache: dict[int, tuple[int, np.ndarray]] = {}

        # (panel, cell_id, kind) -> stored "Measure luminance" results (CellActivity only; always empty for CellArea).
        # Kept unconditionally so shared code doesn't need to special-case the subclass.
        self._lum_measurements: dict[tuple[int, int, str], list[dict]] = {}
        # Overlay curves currently on the ΔF/F plot for every *checked*
        # area's stored measurements (CellActivity only).
        self._lum_overlay_curves: list = []
        # (panel, cell_id, kind, record) for each row of the measurements
        # list (CellActivity only), same order/length as _lum_overlay_curves.
        self._lum_list_records: list = []
        # (viewbox, outline item) pairs for checked fixed-region measurements -- separate from _lum_overlay_curves
        # since these live in the image ViewBox, not the ΔF/F plot.
        self._lum_region_overlay_items: list = []

        # (panel, frame, cell_id, kind) already run through Refine -- bulk "Refine" (nothing selected) skips these.
        self._refined_cells: set[tuple[int, int, int, str]] = set()

        # [(description, restore_fn), ...] -- each action pushes its exact reversal; old "guess highest cell_id" deleted
        # the wrong cell. Cleared on panel reindex (stale indices). Drag-edits push nothing (fire per mouse-move).
        self._undo_stack: list[tuple[str, Callable[[], None]]] = []

        # ── layout state ──────────────────────────────────────────────────────
        # None = grid of all files; otherwise the file index shown maximised.
        self._focused: int | None = None
        # file index shown at each grid position (== range(len(paths)) unless focused)
        self._visible_indices: list[int] = []
        # re-entrancy guard for mirroring pan/zoom across panels (see _on_range_changed)
        self._syncing_range: bool = False

        # ── tracking state (CellActivity only; always empty for CellArea) ──────
        # self._track_data[panel][cell_id] = {
        #   "tracks":    (n_tracked, n_vertices+1, 2) ndarray or None while in flight -- col 0 is centroid, 1.. vertices.
        #   "ref_frame": frame indexing self._polygons[panel][ref_frame][cell_id], source of the query points.
        #   "start":     absolute frame tracks[0] corresponds to (== ref_frame).
        #   "overrides": {frame: (N,2) ndarray} manual corrections, take precedence over CoTracker output.
        # }
        # Independent per cell_id so tracking one area can't clobber another's result.
        self._track_data: dict[int, dict[int, dict]] = {}
        self._pending_track_panel: int = -1    # panel the in-flight worker belongs to
        self._pending_track_cell_id: int = -1  # that panel's cell_id the in-flight worker is for
        # Pre-run snapshot so a failure or Undo can restore exact prior state (_start_track_for_panel/_on_track_done).
        self._pending_track_old_record: dict | None = None
        self._pending_track_old_checked: bool | None = None
        self._pending_track_old_refined: set[tuple[int, int, int, str]] = set()
        self._worker: _TrackWorker | None = None

        # (panel, cell_id, kind) -> checked?  Governs both canvas visibility and ΔF/F plot/measurement inclusion.
        # Absence means checked (new areas default on); see _is_area_checked.
        self._area_checked: dict[tuple[int, int, str], bool] = {}

        # self._frozen_cells[panel][cell_id] = (start_frame, end_frame, pts) -- one static polygon shown unchanged
        # over [start_frame, end_frame] (_effective_polygon). Deliberately not a self._polygons entry: start_frame
        # is whatever frame was on-screen at "Copy area", not tied to any self._polygons frame.
        self._frozen_cells: dict[int, dict[int, tuple[int, int, np.ndarray]]] = {}

        # Reuse cache for tracked-cell ROIs: rebuilding PolyLineROI from scratch per frame profiled at ~65% of
        # _redraw_overlays cost -- kept alive and just repositioned instead. See _get_or_update_track_roi.
        self._track_roi_cache: dict[int, dict[int, dict]] = {}

        # Same idea for manual polygons -- uncached cost measured at ~580ms/redraw with 20 cells across 4 panels.
        # See _get_or_update_manual_roi.
        self._manual_roi_cache: dict[int, dict[int, dict]] = {}

        # ── pyqtgraph items ────────────────────────────────────────────────────
        self._viewboxes: list[pg.ViewBox] = []
        self._img_items: list[pg.ImageItem] = []
        # _ov_items[panel][frame] = list of pg items for manual polygons
        self._ov_items: list[dict[int, list]] = []
        # _tr_items[panel] = list of pg items for tracked polygons (current frame)
        self._tr_items: list[list] = []
        self._prog_items: list = []
        self._prog_panel: int = -1

        self._build_ui()

        if files:
            for f in files:
                self._add_file(Path(f))

    def _alloc_cell_id(self) -> int:
        cid = self._next_cell_id
        self._next_cell_id += 1
        return cid

    def _is_area_checked(self, panel: int, cell_id: int, kind: str) -> bool:
        """Whether (panel, cell_id, kind) is checked in the Areas sidebar --
        the single source of truth for canvas visibility AND inclusion in
        the bottom plot / batch Measure luminance. Absence means checked
        (new areas default on); always go through this helper rather than
        an ad hoc `.get(..., True)` so the default can't drift out of sync
        between the three places that need it."""
        return self._area_checked.get((panel, cell_id, kind), True)

    # ── extension points (no-ops here; CellActivity overrides both) ───────────

    def _build_measurement_toolbar(self, tb: QHBoxLayout, sep) -> None:
        """Hook: activity-only toolbar buttons ("Measure luminance" /
        "Clear measurements"), including their own trailing separator.
        No-op in the base -- CellArea has no luminance measurement action.
        Tracking ("Track cells") is shared, not part of this hook -- see
        _build_ui, both tools can track a cell."""

    def _build_measurement_panel(self, root: QVBoxLayout) -> None:
        """Hook: activity-only bottom panel (ΔF/F plot + measurements
        list). No-op in the base -- CellArea has no such panel."""

    def _refresh_lum_list(self) -> None:
        """No-op in the base -- CellArea has no measurements list/plot to
        refresh. Called as a bare trailing statement from several shared
        methods below; safe to leave those call sites unchanged."""

    def _update_playhead(self, t: int) -> None:
        """No-op in the base -- CellArea has no ΔF/F plot to move a
        playhead on."""

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(3)

        def _sep() -> QFrame:
            """Thin vertical divider between clusters of related buttons in
            the single action toolbar below — a lighter-weight scan aid
            than splitting them into separate captioned rows (tried that,
            it read as a step-by-step wizard instead of a toolbar and made
            things worse: every action belongs in one place)."""
            line = QFrame()
            line.setFrameShape(QFrame.Shape.VLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            return line

        # ── action toolbar — every button lives here, in one place ───────
        tb = QHBoxLayout()
        tb.setSpacing(6)

        btn_open = QPushButton("Open files…")
        btn_open.clicked.connect(self._on_open)
        tb.addWidget(btn_open)
        tb.addWidget(_sep())

        self._draw_btn = QPushButton("Draw [D]")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setFixedWidth(90)
        self._draw_btn.clicked.connect(
            lambda checked: self._set_mode("draw" if checked else "pan")
        )
        tb.addWidget(self._draw_btn)

        btn_undo = QPushButton("Undo [Z]")
        btn_undo.setFixedWidth(75)
        btn_undo.setToolTip(
            "Undo the single most recent action: drawing a cell, a Refine, "
            "a Track cells run, \"Copy area to next frames\", or a Delete."
        )
        btn_undo.clicked.connect(self._undo)
        tb.addWidget(btn_undo)

        btn_clear = QPushButton("Clear frame")
        btn_clear.setFixedWidth(85)
        btn_clear.clicked.connect(self._clear_frame)
        tb.addWidget(btn_clear)

        btn_delete_cell = QPushButton("Delete cell [Del]")
        btn_delete_cell.setFixedWidth(105)
        btn_delete_cell.setToolTip(
            "Delete the selected area (click one to select it first). "
            "Deleting a manual area also removes its tracked/frozen areas "
            "on that same panel; deleting a tracked or frozen area directly "
            "removes just that one."
        )
        btn_delete_cell.clicked.connect(self._delete_selected_cell)
        tb.addWidget(btn_delete_cell)

        btn_refine = QPushButton("Refine [R]")
        btn_refine.setFixedWidth(80)
        btn_refine.setToolTip(
            "Shrink the selected cell to its nearest real edge: threshold "
            "against the actual background just outside it (never past the "
            "drawn outline), keep the largest connected region (the real "
            "cell, vs. scattered sensor-noise specks), and smooth its "
            "contour. With no cell "
            "selected, refines every cell on this frame."
        )
        btn_refine.clicked.connect(self._refine)
        tb.addWidget(btn_refine)
        tb.addWidget(_sep())

        self._track_btn = QPushButton("Track cells")
        self._track_btn.setFixedWidth(90)
        self._track_btn.setToolTip(
            "Track the selected area using CoTracker, forward to the end "
            "of the clip, on that area's own panel only (select a manual "
            "or tracked area in \"Areas:\" first) — creates or refreshes "
            "that cell's \"tracked\" area on this one panel. To track it "
            "on another panel too, select that panel's copy of the cell "
            "and run this again there."
        )
        self._track_btn.clicked.connect(self._on_track)
        tb.addWidget(self._track_btn)
        tb.addWidget(_sep())

        self._build_measurement_toolbar(tb, _sep)

        self._copy_area_btn = QPushButton("Copy area to next frames")
        self._copy_area_btn.setToolTip(
            "Duplicate the selected area's current-frame polygon onto "
            "every remaining frame to the end of the clip, as a frozen "
            "area."
        )
        self._copy_area_btn.clicked.connect(self._copy_area_to_next_frames)
        tb.addWidget(self._copy_area_btn)
        tb.addWidget(_sep())

        btn_export = QPushButton("Export…")
        btn_export.setFixedWidth(75)
        btn_export.clicked.connect(self._export)
        tb.addWidget(btn_export)

        btn_new_sess = QPushButton("New session")
        btn_new_sess.setFixedWidth(95)
        btn_new_sess.setToolTip(
            "Unload every panel and clear all annotations, tracking, and "
            "measurements — including cell_id allocation, which restarts "
            "at 0 — without closing this window. Save first if you want "
            "to keep the current work."
        )
        btn_new_sess.clicked.connect(self._on_new_session)
        tb.addWidget(btn_new_sess)

        btn_save_sess = QPushButton("Save session…")
        btn_save_sess.setFixedWidth(105)
        btn_save_sess.setToolTip(
            "Save every loaded file's path, polygons, tracking results and "
            "manual corrections to a .json file, to reload later.\n"
            "Ctrl+S re-saves over this same file without asking again."
        )
        btn_save_sess.clicked.connect(self._save_session)
        tb.addWidget(btn_save_sess)

        btn_load_sess = QPushButton("Load session…")
        btn_load_sess.setFixedWidth(105)
        btn_load_sess.setToolTip("Restore a previously saved annotation session.")
        btn_load_sess.clicked.connect(self._load_session)
        tb.addWidget(btn_load_sess)
        tb.addWidget(_sep())

        self._zoom_btn = QPushButton("Zoom out")
        self._zoom_btn.setFixedWidth(75)
        self._zoom_btn.setVisible(False)
        self._zoom_btn.clicked.connect(lambda: self._set_focused(None))
        tb.addWidget(self._zoom_btn)

        self._remove_panel_btn = QPushButton("Remove panel")
        self._remove_panel_btn.setToolTip(
            "Unload the active panel's video, along with its annotations, "
            "tracking, and stored luminance measurements — click a panel "
            "(or its cells) first to make it active"
        )
        self._remove_panel_btn.clicked.connect(self._remove_active_panel)
        tb.addWidget(self._remove_panel_btn)
        tb.addStretch(1)

        # One-row toolbar's min width can exceed real screens; adding tb straight to root would force the whole
        # window that wide. Scroll area lets the toolbar scroll instead of clipping, window stays resizable.
        tb_widget = QWidget()
        tb_widget.setLayout(tb)
        tb_scroll = QScrollArea()
        tb_scroll.setWidget(tb_widget)
        tb_scroll.setWidgetResizable(True)
        tb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tb_scroll.setFixedHeight(tb_widget.sizeHint().height() + 4)
        root.addWidget(tb_scroll)

        # Word-wrapped so a long status message doesn't force the window wider (same reasoning as tb_scroll above).
        self._status = QLabel("No files loaded")
        self._status.setStyleSheet("color:#888;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # display-adjustment row — fill opacity and display-only blur
        disp_row = QHBoxLayout()
        disp_row.setSpacing(4)

        disp_row.addWidget(QLabel("Fill α:"))
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(round(_FILL_ALPHA / 255 * 100))
        self._alpha_slider.setFixedWidth(120)
        self._alpha_slider.setToolTip("Cell-area fill opacity.")
        self._alpha_slider.valueChanged.connect(self._on_fill_alpha_changed)
        disp_row.addWidget(self._alpha_slider)
        self._alpha_lbl = QLabel(f"{self._alpha_slider.value()}%")
        self._alpha_lbl.setFixedWidth(35)
        disp_row.addWidget(self._alpha_lbl)

        disp_row.addSpacing(12)
        disp_row.addWidget(QLabel("Blur σ:"))
        self._blur_slider = QSlider(Qt.Orientation.Horizontal)
        self._blur_slider.setRange(0, 50)  # sigma = value / 10, i.e. 0.0-5.0
        self._blur_slider.setValue(round(self._blur_sigma * 10))
        self._blur_slider.setFixedWidth(120)
        self._blur_slider.setToolTip(
            "Gaussian blur applied to the displayed frame only (a visual aid "
            "for spotting cell boundaries) — measurements and exports always "
            "use the unblurred pixel data."
        )
        self._blur_slider.valueChanged.connect(self._on_blur_changed)
        disp_row.addWidget(self._blur_slider)
        self._blur_lbl = QLabel(f"{self._blur_sigma:.1f}")
        self._blur_lbl.setFixedWidth(35)
        disp_row.addWidget(self._blur_lbl)

        disp_row.addStretch(1)
        root.addLayout(disp_row)

        # frame slider row
        fr = QHBoxLayout()
        fr.setSpacing(4)

        btn_prev = QPushButton("◀")
        btn_prev.setFixedWidth(30)
        btn_prev.clicked.connect(lambda: self._seek(self._frame - 1))
        fr.addWidget(btn_prev)

        btn_next = QPushButton("▶")
        btn_next.setFixedWidth(30)
        btn_next.clicked.connect(lambda: self._seek(self._frame + 1))
        fr.addWidget(btn_next)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        fr.addWidget(self._slider, stretch=1)

        fr.addWidget(QLabel("Frame:"))
        self._fspin = QSpinBox()
        self._fspin.setMinimum(0)
        self._fspin.setMaximum(0)
        self._fspin.setFixedWidth(70)
        self._fspin.editingFinished.connect(lambda: self._seek(self._fspin.value()))
        fr.addWidget(self._fspin)

        self._panel_lbl = QLabel("Panel: —")
        self._panel_lbl.setFixedWidth(75)
        fr.addWidget(self._panel_lbl)

        root.addLayout(fr)

        # section row — frame range every loaded stack gets cropped to
        sect = QHBoxLayout()
        sect.setSpacing(4)

        sect_lbl_head = QLabel("Section:")
        sect_lbl_head.setToolTip(
            "Frame range to track. Load stacks (one or all at once), pick a "
            "range, then click Apply — every loaded stack is cropped to it, "
            "and any stack added afterwards only loads that same range."
        )
        sect.addWidget(sect_lbl_head)

        self._sect_start_spin = QSpinBox()
        self._sect_start_spin.setMinimum(0)
        self._sect_start_spin.setMaximum(0)
        self._sect_start_spin.setFixedWidth(70)
        sect.addWidget(self._sect_start_spin)

        btn_sect_start = QPushButton("↧ start")
        btn_sect_start.setFixedWidth(60)
        btn_sect_start.setToolTip("Use the current frame as the section start.")
        btn_sect_start.clicked.connect(
            lambda: self._sect_start_spin.setValue(self._frame)
        )
        sect.addWidget(btn_sect_start)

        sect.addWidget(QLabel("to"))

        self._sect_end_spin = QSpinBox()
        self._sect_end_spin.setMinimum(0)
        self._sect_end_spin.setMaximum(0)
        self._sect_end_spin.setFixedWidth(70)
        sect.addWidget(self._sect_end_spin)

        btn_sect_end = QPushButton("↧ end")
        btn_sect_end.setFixedWidth(60)
        btn_sect_end.setToolTip("Use the current frame as the section end.")
        btn_sect_end.clicked.connect(
            lambda: self._sect_end_spin.setValue(self._frame)
        )
        sect.addWidget(btn_sect_end)

        self._apply_sect_btn = QPushButton("Apply section")
        self._apply_sect_btn.setFixedWidth(95)
        self._apply_sect_btn.setEnabled(False)
        self._apply_sect_btn.setToolTip(
            "Crop every currently loaded stack to this frame range."
        )
        self._apply_sect_btn.clicked.connect(self._apply_section)
        sect.addWidget(self._apply_sect_btn)

        self._remove_sect_btn = QPushButton("Remove section")
        self._remove_sect_btn.setFixedWidth(105)
        self._remove_sect_btn.setEnabled(False)
        self._remove_sect_btn.setToolTip(
            "Undo the applied section — reloads every stack in full."
        )
        self._remove_sect_btn.clicked.connect(self._remove_section)
        sect.addWidget(self._remove_sect_btn)

        self._sect_lbl = QLabel("(load a stack, then pick a range and Apply)")
        self._sect_lbl.setStyleSheet("color:#888;")
        sect.addWidget(self._sect_lbl, stretch=1)

        root.addLayout(sect)

        # "Areas:" sidebar exists because overlapping shapes (e.g. tracked area over its manual source at the
        # ref frame) can't be click-disambiguated on canvas. Each checkbox controls both visibility and plot inclusion.
        self._gfx = pg.GraphicsLayoutWidget()
        self._gfx.setBackground("#1a1a1a")
        self._gfx.scene().sigMouseClicked.connect(self._on_click)

        areas_col = QVBoxLayout()
        areas_col.setSpacing(2)
        areas_lbl = QLabel("Areas:")
        areas_lbl.setToolTip(
            "Every area (cell, panel, kind) across every panel — check a "
            "row to show it on the canvas and include it in the bottom "
            "plot / \"Measure luminance\"; click a row to select it (same "
            "as clicking it on the canvas; Delete/Refine/Track/Copy area "
            "then act on it)"
        )
        areas_col.addWidget(areas_lbl)
        self._areas_list = QListWidget()
        self._areas_list.setFixedWidth(230)
        self._areas_list.itemChanged.connect(self._on_area_checked_changed)
        self._areas_list.itemClicked.connect(self._on_area_list_clicked)
        areas_col.addWidget(self._areas_list, stretch=1)

        img_row = QHBoxLayout()
        img_row.addWidget(self._gfx, stretch=1)
        img_row.addLayout(areas_col)
        root.addLayout(img_row, 1)

        self._build_measurement_panel(root)

        # hint bar
        hint = QLabel(
            "Two different clicks: click a panel's BACKGROUND to make it "
            "active (Refine/Export/Remove panel act on this)  —  "
            "click a CELL itself to select it (Delete/Refine/Copy "
            "area act on this)  |  "
            "D: draw mode  |  click: add point  |  right-click / Enter: close  "
            "|  Esc: cancel  |  Z: undo  |  Del: delete selected cell  |  "
            "R: refine selected (or all) cell(s)  |  "
            "← →: frame  |  PgUp/PgDn: ±10 frames  |  "
            "Home/End: first/last frame  |  Tab: next panel  |  "
            "outside draw mode: drag vertices/polygon to edit, "
            "right-click to add/remove a vertex, double-click a panel to zoom in/out"
        )
        hint.setStyleSheet("color:#444; font-size:10px;")
        # Unwrapped, this label's minimumSizeHint alone would force the window toolbar-wide (see tb_scroll above).
        hint.setWordWrap(True)
        root.addWidget(hint)

    # ── loading ───────────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open video files",
            "",
            "Video files (*.h5 *.hdf5 *.tiff *.tif);;All files (*)",
        )
        for p in paths:
            self._add_file(Path(p))

    def _add_file(self, path: Path) -> None:
        if not path.exists():
            self._status.setText(f"Not found: {path}")
            return

        is_first = not self._paths
        # Once a section is applied, new stacks only read that frame range off disk; before that they load in full.
        vid = _load_video(path, frame_range=self._section)
        if vid is None:
            self._status.setText(f"Failed to load: {path.name}")
            return

        finite = vid[np.isfinite(vid)]
        lo = float(np.percentile(finite, 1)) if finite.size else 0.0
        hi = float(np.percentile(finite, 99)) if finite.size else 1.0
        hi = max(hi, lo + 1e-8)

        self._paths.append(path)
        self._videos.append(vid)
        self._ranges.append((lo, hi))
        # Stacks are spatially aligned -- seed new panels with a copy of panel 0's annotations, same cell_ids.
        self._polygons.append(
            {}
            if is_first
            else {
                frame: {cid: pts.copy() for cid, pts in polys.items()}
                for frame, polys in self._polygons[0].items()
            }
        )
        self._ov_items.append({})
        self._tr_items.append([])

        self._n_frames = min(v.shape[0] for v in self._videos)
        self._active = len(self._paths) - 1
        self._panel_lbl.setText(f"Panel: {self._active + 1}")

        if self._section is None:
            # Section bounds track the shortest currently loaded stack until a section is applied.
            self._sect_start_spin.setMaximum(max(0, self._n_frames - 1))
            self._sect_end_spin.setMaximum(max(0, self._n_frames - 1))
            if is_first:
                self._sect_end_spin.setValue(max(0, self._n_frames - 1))
            self._apply_sect_btn.setEnabled(True)

        # Block signals: Qt auto-clamps + fires valueChanged if value exceeds new maximum, clobbering self._frame.
        target_frame = self._frame
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._n_frames - 1))
        self._status.setText(
            f"{len(self._paths)} file(s)  —  {self._n_frames} frames"
        )

        # Adding a file always shows the full grid, to reveal the new panel even if previously zoomed in.
        self._focused = None
        self._zoom_btn.setVisible(False)
        self._rebuild_panels()
        self._refresh_areas_list()
        self._slider.setValue(target_frame)
        self._show_frame(target_frame)
        self._autorange_panels()

    def _remove_active_panel(self) -> None:
        """"Remove panel" button: unload the active panel's video (see
        _remove_panel) -- click a panel, or one of its cells, first to make
        it active, same as Undo/Clear frame/Export already work off
        self._active."""
        if not self._paths:
            self._status.setText("No panels loaded.")
            return
        idx = self._active
        name = self._paths[idx].name
        has_data = bool(self._polygons[idx]) or bool(self._track_data.get(idx))
        if has_data:
            box = QMessageBox(self)
            box.setWindowTitle("Remove panel")
            box.setText(
                f"Remove \"{name}\"? Its annotated/tracked cells and stored "
                "luminance measurements will be lost. Other panels are "
                "unaffected."
            )
            box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Ok:
                self._status.setText("Remove panel: cancelled.")
                return
        self._remove_panel(idx)
        self._status.setText(f"Removed panel: {name}")

    def _remove_panel(self, idx: int) -> None:
        """Unload the video at panel *idx* and everything derived from it —
        polygons, tracking data, and stored luminance measurements — then
        reindex every panel-keyed structure for the panels that shift down
        to fill the gap (cell_ids themselves never need reindexing, only
        the outer panel-index layer)."""
        if idx >= len(self._paths):
            return

        # Panel reindexing below invalidates any undo closure's captured panel index -- stack can't survive this.
        self._undo_stack = []

        def _shift(p: int) -> int:
            return p - 1 if p > idx else p

        del self._paths[idx]
        del self._videos[idx]
        del self._ranges[idx]
        del self._polygons[idx]
        del self._ov_items[idx]
        del self._tr_items[idx]

        self._track_data = {_shift(p): v for p, v in self._track_data.items() if p != idx}
        self._dff_video_cache = {_shift(p): v for p, v in self._dff_video_cache.items() if p != idx}
        self._lum_measurements = {
            (_shift(p), c, k): v for (p, c, k), v in self._lum_measurements.items() if p != idx
        }
        self._refined_cells = {(_shift(p), f, c, k) for (p, f, c, k) in self._refined_cells if p != idx}
        self._frozen_cells = {_shift(p): v for p, v in self._frozen_cells.items() if p != idx}
        self._area_checked = {
            (_shift(p), c, k): v for (p, c, k), v in self._area_checked.items() if p != idx
        }

        if self._selected_area is not None:
            sp, sc, sk = self._selected_area
            self._selected_area = None if sp == idx else (_shift(sp), sc, sk)

        self._focused = None if self._focused == idx else (
            _shift(self._focused) if self._focused is not None else None
        )
        self._zoom_btn.setVisible(self._focused is not None)

        self._active = max(0, min(_shift(self._active), len(self._paths) - 1))

        # Removing a panel can only raise the shortest remaining stack's length -- recompute rather than leave stale.
        self._n_frames = min((v.shape[0] for v in self._videos), default=0)
        self._frame = min(self._frame, max(0, self._n_frames - 1))
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._n_frames - 1))

        self._rebuild_panels()
        self._panel_lbl.setText(f"Panel: {self._active + 1}" if self._paths else "Panel: —")
        self._slider.setValue(self._frame)
        self._show_frame(self._frame)
        self._refresh_lum_list()
        self._refresh_areas_list()
        self._autorange_panels()

    def _apply_section(self) -> None:
        """Crop every currently loaded stack down to the chosen frame range."""
        if not self._videos or self._section is not None:
            return

        total = self._n_frames  # shortest currently loaded stack
        end = max(0, min(self._sect_end_spin.value(), total - 1))
        start = max(0, min(self._sect_start_spin.value(), end))
        self._section = (start, end)
        # Frame indices shift below and track_data/frozen_cells get wiped -- any undo closure against the old state is invalid.
        self._undo_stack = []

        for i in range(len(self._videos)):
            self._videos[i] = self._videos[i][start : end + 1]
            self._polygons[i] = {
                f - start: polys
                for f, polys in self._polygons[i].items()
                if start <= f <= end
            }

        # Same seeding as _add_file: any panel still without annotations gets seeded with panel 0's.
        for i in range(1, len(self._polygons)):
            if not self._polygons[i]:
                self._polygons[i] = {
                    frame: {cid: pts.copy() for cid, pts in polys.items()}
                    for frame, polys in self._polygons[0].items()
                }

        self._n_frames = end - start + 1
        target_frame = max(0, min(self._frame - start, self._n_frames - 1))
        # Frame reindex invalidates tracked/frozen areas -- their frame indices are absolute, not remapped like
        # self._polygons/_refined_cells, so leaving them would show phantom entries in "Areas:".
        self._track_data = {}
        self._frozen_cells = {}
        self._area_checked = {}
        self._selected_area = None
        # Same frame-key remapping as self._polygons above, dropping marks outside the new range.
        self._refined_cells = {
            (p, f - start, c, k) for (p, f, c, k) in self._refined_cells if start <= f <= end
        }

        self._sect_start_spin.setEnabled(False)
        self._sect_end_spin.setEnabled(False)
        self._apply_sect_btn.setEnabled(False)
        self._remove_sect_btn.setEnabled(True)
        self._sect_lbl.setText(
            f"Applied: frames {start}–{end} ({end - start + 1} frames)"
        )

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._n_frames - 1))

        self._rebuild_panels()
        self._slider.setValue(target_frame)
        self._show_frame(target_frame)
        self._refresh_lum_list()
        self._refresh_areas_list()
        self._autorange_panels()

    def _remove_section(self) -> None:
        """Undo an applied section: reload every stack in full from disk."""
        if self._section is None or not self._paths:
            return

        # Same reasoning as _apply_section -- frame indices shift, invalidating any captured undo.
        self._undo_stack = []

        start, _end = self._section
        for i, path in enumerate(self._paths):
            full = _load_video(path)
            if full is None:
                self._status.setText(f"Could not reload {path.name} in full.")
                return
            self._videos[i] = full
            # Shift frame keys back to absolute indexing on the restored stack.
            self._polygons[i] = {
                f + start: polys for f, polys in self._polygons[i].items()
            }

        self._section = None
        self._n_frames = min(v.shape[0] for v in self._videos)
        target_frame = max(0, min(self._frame + start, self._n_frames - 1))
        # Frame reindex invalidates tracked/frozen areas -- same reasoning as _apply_section.
        self._track_data = {}
        self._frozen_cells = {}
        self._area_checked = {}
        self._selected_area = None
        # Same frame-key remapping as self._polygons above.
        self._refined_cells = {(p, f + start, c, k) for (p, f, c, k) in self._refined_cells}

        self._sect_start_spin.setEnabled(True)
        self._sect_end_spin.setEnabled(True)
        self._sect_start_spin.setMaximum(max(0, self._n_frames - 1))
        self._sect_end_spin.setMaximum(max(0, self._n_frames - 1))
        self._apply_sect_btn.setEnabled(True)
        self._remove_sect_btn.setEnabled(False)
        self._sect_lbl.setText("(section removed — pick a range and Apply)")
        self._status.setText(
            f"{len(self._paths)} file(s)  —  {self._n_frames} frames"
        )

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.blockSignals(False)
        self._fspin.setMaximum(max(0, self._n_frames - 1))

        self._rebuild_panels()
        self._slider.setValue(target_frame)
        self._show_frame(target_frame)
        self._refresh_lum_list()
        self._refresh_areas_list()
        self._autorange_panels()

    # ── panel grid ────────────────────────────────────────────────────────────

    def _set_focused(self, file_idx: int | None) -> None:
        """Show only *file_idx* maximised, or the full grid when None."""
        self._focused = file_idx
        self._zoom_btn.setVisible(file_idx is not None)
        if file_idx is not None:
            self._active = file_idx
        self._panel_lbl.setText(
            f"Panel: {self._active + 1}" + (" (zoomed)" if file_idx is not None else "")
        )
        self._rebuild_panels()
        self._show_frame(self._frame)
        self._autorange_panels()

    def _autorange_panels(self) -> None:
        """AutoRange every panel so images fit the view instead of the
        pyqtgraph default 0-1 range.

        Calling autoRange() only on the first panel and relying on
        _on_range_changed to mirror it to the rest is NOT equivalent: a
        ViewBox that only ever receives its range via setRange() (as
        opposed to autoRange() called directly on it) never properly
        initialises the transform its mouse handling uses, which silently
        makes ROIs on every panel but the first undraggable. So autoRange
        each one directly; _on_range_changed still keeps them aligned
        afterwards for ordinary pan/zoom.
        """
        for vb in self._viewboxes:
            vb.autoRange()

    def _grid_pos_of(self, file_idx: int) -> int | None:
        """Grid position currently showing *file_idx*, or None if not visible."""
        try:
            return self._visible_indices.index(file_idx)
        except ValueError:
            return None

    def _rebuild_panels(self) -> None:
        self._gfx.clear()
        self._img_items.clear()
        self._viewboxes.clear()
        for i in range(len(self._ov_items)):
            self._ov_items[i] = {}
        for i in range(len(self._tr_items)):
            self._tr_items[i] = []
        self._prog_items.clear()
        self._prog_panel = -1
        self._current_pts.clear()
        # _gfx.clear() already destroyed these scene items -- drop the dangling refs instead of removeItem().
        self._track_roi_cache = {}
        self._manual_roi_cache = {}
        self._lum_region_overlay_items = []

        self._visible_indices = (
            [self._focused] if self._focused is not None else list(range(len(self._paths)))
        )

        n = len(self._visible_indices)
        # Square-ish grid instead of maxing out columns first -- e.g. 5 files -> 3x2, not a near-empty 4+1 row.
        ncols = 1 if self._focused is not None else max(1, math.ceil(math.sqrt(n)))
        font = QFont()
        font.setPointSize(8)

        for grid_pos, file_idx in enumerate(self._visible_indices):
            path = self._paths[file_idx]
            gr = (grid_pos // ncols) * 2
            gc = grid_pos % ncols

            lbl = self._gfx.addLabel(path.name, row=gr, col=gc)
            lbl.item.setFont(font)

            vb = self._gfx.addViewBox(
                row=gr + 1, col=gc, lockAspect=True, enableMouse=True
            )
            vb.setMenuEnabled(False)
            vb.setDefaultPadding(0.01)
            self._viewboxes.append(vb)

            img = pg.ImageItem()
            vb.addItem(img)
            self._img_items.append(img)

            # Manual range mirroring, not ViewBox.setXLink/setYLink -- pyqtgraph's linking swallows ROI drag events
            # on non-master viewboxes, making cells on every panel but the first undraggable.
            vb.sigRangeChanged.connect(self._on_range_changed)

        self._update_borders()

    def _on_range_changed(self, vb: pg.ViewBox, view_range, changed=None) -> None:
        """Mirror one panel's pan/zoom range onto every other panel."""
        if self._syncing_range:
            return
        self._syncing_range = True
        try:
            for other in self._viewboxes:
                if other is not vb:
                    other.setRange(
                        xRange=view_range[0], yRange=view_range[1], padding=0
                    )
        finally:
            self._syncing_range = False

    def _update_borders(self) -> None:
        """Highlight the active panel — the one Refine/Export/Remove panel act on."""
        for grid_pos, vb in enumerate(self._viewboxes):
            file_idx = self._visible_indices[grid_pos]
            if file_idx == self._active:
                vb.setBorder(pg.mkPen(_color(file_idx), width=3))
            else:
                vb.setBorder(None)

    def _select_panel(self, file_idx: int) -> None:
        """Make *file_idx* the active stack for Undo/Clear/Export."""
        if file_idx != self._active:
            self._active = file_idx
            self._panel_lbl.setText(f"Panel: {file_idx + 1}")
            self._update_borders()

    def _select_area(self, panel: int, cell_id: int, kind: str) -> None:
        """Select (panel, cell_id, kind) (highlighted white) as the target
        for Delete/Refine/Track/Copy area/Clear measurements."""
        self._select_panel(panel)
        if self._selected_area == (panel, cell_id, kind):
            return
        old = self._selected_area
        self._selected_area = (panel, cell_id, kind)
        # Only the old/new selected ROI's pen needs to change -- drop just those 2 cache entries, not the whole
        # cache, or every selection click rebuilds all ROIs from scratch (hundreds of ms with many cells).
        for area in (old, self._selected_area):
            if area is None:
                continue
            a_panel, a_cell_id, a_kind = area
            cache_by_panel = {
                "tracked": self._track_roi_cache, "manual": self._manual_roi_cache,
            }.get(a_kind)
            if cache_by_panel is None:
                continue
            entry = cache_by_panel.get(a_panel, {}).pop(a_cell_id, None)
            if entry is not None:
                self._remove_track_roi_entry(entry)
        self._redraw_overlays()
        self._refresh_lum_list()
        self._refresh_areas_list()

    def _uint8_video(self, panel: int) -> np.ndarray:
        """Cached whole-stack percentile-normalized copy of panel *panel*'s
        video, same convention (and cache-by-array-identity, so a section
        apply/remove or session reload naturally invalidates it) as
        _to_uint8_stack's docstring explains."""
        video = self._videos[panel]
        cached = self._dff_video_cache.get(panel)
        if cached is None or cached[0] != id(video):
            cached = (id(video), _to_uint8_stack(video))
            self._dff_video_cache[panel] = cached
        return cached[1]

    def _cell_label(self, panel: int, cell_id: int) -> str:
        """Display label for *cell_id* -- just its plain number. Kind
        ("manual"/"tracked"/"frozen") is a separate axis, shown alongside
        this label in the Areas sidebar/canvas rather than folded into it,
        since one cell_id can have several simultaneous kinds now."""
        return str(cell_id + 1)

    def _cell_color_index(self, panel: int, cell_id: int, kind: str) -> int:
        """Palette index (see _color/_COLORS) for *cell_id* -- based on
        cell_id alone, deliberately ignoring *panel* and *kind* (kept in
        the signature since every call site naturally has them on hand,
        not because they're used here).

        cell_id is a single identity shared across every panel (see
        _close_polygon), so its color should be the same wherever it's
        drawn -- on any panel's canvas, and as a stored measurement's curve
        in CellActivity's ΔF/F plot, so a checked area's line color always
        matches its on-canvas outline. Folding panel/kind into the hash
        (the previous scheme) caused frequent same-palette collisions
        between *different* cells once a few were visible at once -- see
        the old `panel * 11 + cell_id * 3 + kind_rank` formula, which
        collided as early as 3 cells on one panel. Kind is still visually
        distinguished, just via line style (solid vs. dashed) and alpha
        instead of color -- see the three _redraw_overlays loops.

        Not reduced mod len(_COLORS) here -- _color() already does that."""
        return cell_id

    def _checked_areas(self) -> list[tuple[int, int, str]]:
        """Every (panel, cell_id, kind) that currently exists and is
        checked in the Areas sidebar. Used by CellActivity's batch "Measure
        luminance" and stored-measurement list. CellArea's export reads
        geometry directly and has no need for this, but it's kept here
        (not activity-only) since _redraw_overlays' checked-state checks
        and the Areas list itself are both shared, generic concepts."""
        out = []
        for panel in range(len(self._paths)):
            for cell_id, kind in self._all_cells_on_panel(panel):
                if self._is_area_checked(panel, cell_id, kind):
                    out.append((panel, cell_id, kind))
        return out

    def _copy_area_to_next_frames(self) -> None:
        """"Copy area to next frames" button: create (or refresh) the
        selected cell_id's "frozen" area on this panel -- its shape frozen
        to exactly what's shown right now, held constant from here to the
        end of the clip. The source area itself (manual or tracked) is
        left completely untouched -- this never writes into track_data, so
        it can't disturb tracking.

        Works on any selected area, tracked or not: a manual area's own
        static shape, or a tracked area's currently-tracked shape at this
        exact frame, are both valid things to freeze. Re-running this
        refreshes the same cell_id's existing frozen area (new shape/range)
        instead of creating a second one -- each cell_id has at most one
        frozen area per panel at a time."""
        if self._selected_area is None:
            self._status.setText("No area selected — click an area first.")
            return

        panel, cell_id, kind = self._selected_area
        pts, _resolved_kind = self._effective_polygon(panel, self._frame, cell_id, kind)
        if pts is None:
            self._status.setText("No polygon at the current frame to copy.")
            return

        end_frame = self._n_frames - 1  # always the rest of the clip
        if end_frame <= self._frame:
            self._status.setText("Not enough frames after the current one to copy across.")
            return

        # Refresh an existing frozen area in place rather than creating a
        # second one for the same cell_id -- just a dict overwrite now.
        old_frozen = self._frozen_cells.get(panel, {}).get(cell_id)
        old_selected = self._selected_area
        self._frozen_cells.setdefault(panel, {})[cell_id] = (self._frame, end_frame, pts.copy())
        self._selected_area = (panel, cell_id, "frozen")

        def _undo_copy_area(
            _panel=panel,
            _cell_id=cell_id,
            _old_frozen=old_frozen,
            _old_selected=old_selected,
        ) -> None:
            if _old_frozen is not None:
                self._frozen_cells.setdefault(_panel, {})[_cell_id] = _old_frozen
            else:
                self._frozen_cells.get(_panel, {}).pop(_cell_id, None)
            self._selected_area = _old_selected
            self._invalidate_roi_cache(_panel)

        self._push_undo(f"copying area for cell {self._cell_label(panel, cell_id)}", _undo_copy_area)

        self._invalidate_roi_cache(panel)
        self._redraw_overlays()
        self._refresh_lum_list()
        self._refresh_areas_list()
        self._status.setText(
            f"Cell {self._cell_label(panel, cell_id)}: frozen area for frames "
            f"{self._frame}–{end_frame} ({end_frame - self._frame + 1} frame(s))."
        )

    def _remove_area(self, panel: int, cell_id: int, kind: str) -> bool:
        """Remove (panel, cell_id, kind) entirely: its polygon (a
        self._polygons entry for a "manual" area, or a self._frozen_cells
        entry for a "frozen" one), tracking data (if "tracked"), refined
        marks, stored luminance measurements, checked-state, and cached ROI.
        Returns False if this (panel, cell_id, kind) didn't exist.

        A pure single-area primitive with no cascading and no
        self._selected_area/status-bar handling of its own -- shared by
        _delete_selected_cell (which adds cascade-to-other-kinds and the
        user-facing message). With stable cell_id keys, removal is just a
        dict-key delete -- no reindexing of anything else needed, unlike
        the old positional-index scheme this replaced."""
        if kind == "manual":
            for frame, polys in self._polygons[panel].items():
                if cell_id in polys:
                    del polys[cell_id]
                    break
            else:
                return False
        elif kind == "tracked":
            if cell_id not in self._track_data.get(panel, {}):
                return False
            del self._track_data[panel][cell_id]
        elif kind == "frozen":
            if cell_id not in self._frozen_cells.get(panel, {}):
                return False
            del self._frozen_cells[panel][cell_id]
        else:
            return False

        # (panel, frame, cell_id, kind) tuples -- drop any matching this
        # exact (panel, cell_id, kind), any frame.
        self._refined_cells = {
            (p, f, c, k) for (p, f, c, k) in self._refined_cells
            if not (p == panel and c == cell_id and k == kind)
        }
        self._lum_measurements.pop((panel, cell_id, kind), None)
        self._area_checked.pop((panel, cell_id, kind), None)
        if self._selected_area == (panel, cell_id, kind):
            self._selected_area = None

        self._invalidate_roi_cache(panel)

        return True

    def _delete_selected_cell(self) -> None:
        """Delete the selected area. Deleting a "manual" area also removes
        that same cell_id's "tracked"/"frozen" areas on that same panel --
        they're evaluations *of* it and are meaningless once it's gone.
        Deleting a "tracked" or "frozen" area directly removes just that
        one, leaving the manual area and any sibling kind untouched."""
        if self._selected_area is None:
            self._status.setText("No area selected — click an area, then Delete.")
            return
        panel, cell_id, kind = self._selected_area
        old_selected = self._selected_area

        kinds_to_remove = [kind] if kind != "manual" else ["manual", "tracked", "frozen"]

        # Snapshot everything _remove_area is about to delete for each kind,
        # so one combined Undo can put it all back exactly -- captured
        # *before* removal, since _remove_area doesn't keep what it deletes.
        snapshots: dict[str, dict] = {}
        for k in kinds_to_remove:
            if k == "manual":
                shape = None
                for frame, polys in self._polygons[panel].items():
                    if cell_id in polys:
                        shape = (frame, polys[cell_id].copy())
                        break
                if shape is None:
                    continue
                snapshots[k] = {"shape": shape}
            elif k == "tracked":
                record = self._track_data.get(panel, {}).get(cell_id)
                if record is None:
                    continue
                snapshots[k] = {"record": copy.deepcopy(record)}
            elif k == "frozen":
                frozen = self._frozen_cells.get(panel, {}).get(cell_id)
                if frozen is None:
                    continue
                start, end, pts = frozen
                snapshots[k] = {"frozen": (start, end, pts.copy())}
            if k in snapshots:
                snapshots[k]["refined"] = {
                    (p, f, c, kk) for (p, f, c, kk) in self._refined_cells
                    if p == panel and c == cell_id and kk == k
                }
                snapshots[k]["lum"] = self._lum_measurements.get((panel, cell_id, k))
                snapshots[k]["checked"] = self._area_checked.get((panel, cell_id, k))

        removed = [k for k in kinds_to_remove if self._remove_area(panel, cell_id, k)]

        self._selected_area = None

        if not removed:
            self._status.setText("Selected area no longer exists.")
            self._redraw_overlays()
            self._refresh_areas_list()
            return

        def _undo_delete(
            _panel=panel,
            _cell_id=cell_id,
            _snapshots=snapshots,
            _old_selected=old_selected,
        ) -> None:
            for k, snap in _snapshots.items():
                if k == "manual":
                    frame, pts = snap["shape"]
                    self._polygons[_panel].setdefault(frame, {})[_cell_id] = pts
                elif k == "tracked":
                    self._track_data.setdefault(_panel, {})[_cell_id] = snap["record"]
                elif k == "frozen":
                    self._frozen_cells.setdefault(_panel, {})[_cell_id] = snap["frozen"]
                self._refined_cells |= snap["refined"]
                if snap["lum"] is not None:
                    self._lum_measurements[(_panel, _cell_id, k)] = snap["lum"]
                if snap["checked"] is not None:
                    self._area_checked[(_panel, _cell_id, k)] = snap["checked"]
            self._selected_area = _old_selected
            self._invalidate_roi_cache(_panel)

        self._push_undo(f"deleting cell {cell_id + 1} ({', '.join(removed)})", _undo_delete)

        self._redraw_overlays()
        self._refresh_lum_list()
        self._refresh_areas_list()
        self._status.setText(
            f"Deleted cell {cell_id + 1} ({', '.join(removed)}) on "
            f"{self._paths[panel].stem}."
        )

    # ── refine (Otsu + connected components + smoothed contour) ────────────────

    def _visible_cell_ids(self, panel: int, frame: int) -> list[tuple[int, str]]:
        """(cell_id, kind) pairs with an effective polygon at (panel,
        frame) -- manual, tracked (each independently tracked -- see
        self._track_data), or frozen (shown wherever their range covers
        this frame). Used by bulk Refine, which processes every visible
        area on the current frame when nothing is explicitly selected."""
        ids: set[tuple[int, str]] = set()
        polys = self._polygons[panel].get(frame)
        if polys:
            ids.update((cell_id, "manual") for cell_id in polys)
        for cell_id, track in self._track_data.get(panel, {}).items():
            if track["tracks"] is None:
                continue
            rel = frame - track["start"]
            if 0 <= rel < len(track["tracks"]):
                ids.add((cell_id, "tracked"))
        for cell_id, (start_f, end_f, _pts) in self._frozen_cells.get(panel, {}).items():
            if start_f <= frame <= end_f:
                ids.add((cell_id, "frozen"))
        return sorted(ids)

    def _all_cells_on_panel(self, panel: int) -> list[tuple[int, str]]:
        """(cell_id, kind) pairs for every area that currently exists on
        *panel*, across every frame -- not just whichever ones have a shape
        at self._frame (see _visible_cell_ids for that). Used to populate
        the "Areas:" list, so an area whose only frame you've scrolled away
        from doesn't vanish from it."""
        ids: set[tuple[int, str]] = set()
        for polys in self._polygons[panel].values():
            ids.update((cell_id, "manual") for cell_id in polys)
        ids.update((cell_id, "tracked") for cell_id in self._track_data.get(panel, {}))
        ids.update((cell_id, "frozen") for cell_id in self._frozen_cells.get(panel, {}))
        return sorted(ids)

    def _refresh_areas_list(self) -> None:
        """Rebuild the "Areas:" sidebar list from every panel's areas (see
        _all_cells_on_panel) -- called any time an area might have been
        added, removed, tracked, or frozen. Each row's checkbox reflects
        self._area_checked (checked = shown + included in plot/measure);
        rebuilding doesn't change that state, just resyncs the widgets."""
        self._areas_list.blockSignals(True)
        self._areas_list.clear()
        for panel in range(len(self._paths)):
            panel_name = self._paths[panel].stem
            for cell_id, kind in self._all_cells_on_panel(panel):
                item = QListWidgetItem(
                    f"{panel_name}: {self._cell_label(panel, cell_id)} ({kind})"
                )
                item.setData(Qt.ItemDataRole.UserRole, (panel, cell_id, kind))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if self._is_area_checked(panel, cell_id, kind)
                    else Qt.CheckState.Unchecked
                )
                if (panel, cell_id, kind) == self._selected_area:
                    item.setSelected(True)
                self._areas_list.addItem(item)
        self._areas_list.blockSignals(False)

    def _on_area_checked_changed(self, item: QListWidgetItem) -> None:
        panel, cell_id, kind = item.data(Qt.ItemDataRole.UserRole)
        checked = item.checkState() == Qt.CheckState.Checked
        self._area_checked[(panel, cell_id, kind)] = checked
        self._redraw_overlays()
        self._refresh_lum_list()

    def _on_area_list_clicked(self, item: QListWidgetItem) -> None:
        panel, cell_id, kind = item.data(Qt.ItemDataRole.UserRole)
        self._select_area(panel, cell_id, kind)

    def _effective_polygon(
        self, panel: int, frame: int, cell_id: int, kind: str
    ) -> tuple[np.ndarray, str] | tuple[None, None]:
        """The polygon actually shown for (cell_id, kind) at *frame*, and
        how to write a refined version back: "manual" edits self._polygons
        directly; "tracked"/"override" edits are stored as a per-frame
        override, same as a manual drag-correction of a tracked area;
        "frozen" edits its one stored polygon in place (see
        _apply_refined_polygon), regardless of which frame within its
        range the edit came from -- it's the same shape everywhere. *kind*
        is required (not resolved by priority) since one cell_id can have
        several simultaneous kinds -- which one is meant has to come from
        the caller's selected/checked kind, not a guess."""
        if kind == "manual":
            polys = self._polygons[panel].get(frame)
            if polys is not None and cell_id in polys:
                return polys[cell_id], "manual"
            return None, None
        if kind == "tracked":
            track = self._track_data.get(panel, {}).get(cell_id)
            if track is not None and track["tracks"] is not None:
                rel = frame - track["start"]
                if 0 <= rel < len(track["tracks"]):
                    ref_polys = self._polygons[panel].get(track["ref_frame"], {})
                    if cell_id in ref_polys:
                        override = track["overrides"].get(frame)
                        if override is not None:
                            return override, "override"
                        pts = self._tracked_points(panel, cell_id, rel, ref_polys[cell_id])
                        if pts is not None:
                            return pts, "tracked"
            return None, None
        if kind == "frozen":
            frozen = self._frozen_cells.get(panel, {}).get(cell_id)
            if frozen is not None:
                start_f, end_f, pts = frozen
                if start_f <= frame <= end_f:
                    return pts, "frozen"
            return None, None
        return None, None

    def _apply_refined_polygon(
        self, panel: int, frame: int, cell_id: int, kind: str, new_pts: np.ndarray
    ) -> None:
        if kind == "manual":
            polys = self._polygons[panel].get(frame)
            if polys is not None and cell_id in polys:
                polys[cell_id] = new_pts
        elif kind == "frozen":
            start_f, end_f, _old_pts = self._frozen_cells[panel][cell_id]
            self._frozen_cells[panel][cell_id] = (start_f, end_f, new_pts)
        else:  # "tracked" or "override"
            self._track_data[panel][cell_id]["overrides"][frame] = new_pts

    @staticmethod
    def _refine_polygon_cv(
        video: np.ndarray, frame_idx: int, pts: np.ndarray
    ) -> np.ndarray | None:
        """Shrink a hand-drawn polygon to the nearest real cell edge.
        Anchors the cutoff to the *local contrast* between this specific
        cell and its immediate background — a ring of pixels just outside
        the drawn polygon vs. the polygon's own interior — rather than any
        fixed/absolute brightness level. The cutoff sits halfway between
        the two (a standard "half-max" edge definition): brighter than
        that counts as cell, dimmer counts as background. Because it's
        relative to this cell's own dynamic range, a dim/inactive cell
        with barely any contrast above background still gets a real edge
        at its own halfway point — an absolute cutoff (e.g. a fixed
        percentile of the background ring alone, tried previously) instead
        threw away most or all of a dim cell's interior since it never
        reached that fixed brightness level, and separately drifted between
        cutting cleanly at the true edge and cutting deep into solid bright
        interior depending on how tightly the polygon was drawn, since nothing
        tied the cutoff to how bright *this* cell actually is. Intersects
        the thresholded mask with the polygon's own rasterized shape so the
        result can only shrink inward, never grow past what was drawn.
        Takes the largest connected component within that intersection
        (the real cell is always far larger than any scattered single/few-
        pixel sensor-noise speck also passing the threshold), morphologically
        opens then closes it to shave off the thin spiky protrusions/notches
        that per-pixel noise right at the cutoff leaves along the boundary,
        and simplifies its contour to a smooth polygon. The crop is blurred
        before thresholding specifically so the mask's own boundary is
        rounded rather than following the raw pixel grid — approxPolyDP
        alone only reduces the *vertex count* of whatever boundary it's
        given, it doesn't remove genuinely spiky/staircased geometry, which
        is what the morphological open/close step is for.

        Before computing the background level, any *other* bright structure
        inside the ring (e.g. a neighboring cell close enough to intrude
        into it) is excluded from the statistic: a genuine other-cell
        footprint forms a sizeable connected blob with a gap between it and
        the drawn polygon, unlike the current cell's own edge glow, which
        stays connected to the polygon boundary with no gap. Returns None
        if nothing usable is found, or if there's no real contrast between
        the polygon's interior and its surrounding background at all.
        """
        img = video[min(frame_idx, video.shape[0] - 1)]
        H, W = img.shape
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        # Wider margin than the polygon -- intentional background reference ring, not rounding safety.
        pad = 15
        xa, xb = int(np.clip(x0 - pad, 0, W - 1)), int(np.clip(x1 + pad, 0, W - 1))
        ya, yb = int(np.clip(y0 - pad, 0, H - 1)), int(np.clip(y1 + pad, 0, H - 1))
        if xb <= xa or yb <= ya:
            return None

        crop = img[ya : yb + 1, xa : xb + 1].astype(np.float32)
        crop_blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.5)

        poly_local = pts.copy()
        poly_local[:, 0] -= xa
        poly_local[:, 1] -= ya
        poly_local_i = np.round(poly_local).astype(np.int32)
        poly_mask = np.zeros(crop.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_local_i], 255)
        if not (poly_mask > 0).any():
            return None

        outer = cv2.dilate(poly_mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        bg_ring = cv2.bitwise_and(outer, cv2.bitwise_not(poly_mask))
        if not (bg_ring > 0).any():
            return None

        # Median/MAD, not mean/percentile -- a single intruding blob would skew the statistic used to detect it.
        # Drop flagged components not touching the polygon: the gap marks them as a separate structure, not edge glow.
        ring_vals = crop_blur[bg_ring > 0]
        ring_median = float(np.median(ring_vals))
        ring_mad = float(np.median(np.abs(ring_vals - ring_median)))
        rough_cutoff = ring_median + 3 * 1.4826 * ring_mad
        ring_bright = cv2.bitwise_and(
            (crop_blur > rough_cutoff).astype(np.uint8) * 255, bg_ring
        )
        poly_dilated = cv2.dilate(poly_mask, np.ones((5, 5), np.uint8))
        clean_ring = bg_ring.copy()
        if (ring_bright > 0).any():
            n_rb, rb_labels, rb_stats, _ = cv2.connectedComponentsWithStats(
                ring_bright, connectivity=8
            )
            for label in range(1, n_rb):
                comp = (rb_labels == label).astype(np.uint8) * 255
                touches_polygon = bool(cv2.bitwise_and(comp, poly_dilated).any())
                if not touches_polygon and rb_stats[label, cv2.CC_STAT_AREA] >= 8:
                    clean_ring = cv2.bitwise_and(clean_ring, cv2.bitwise_not(comp))
        if not (clean_ring > 0).any():
            clean_ring = bg_ring

        # Half-max cutoff relative to this cell's own background/interior brightness, not an absolute level.
        # Median (not percentile) for robustness to stray outlier pixels.
        bg_level = float(np.median(crop_blur[clean_ring > 0]))
        fg_level = float(np.median(crop_blur[poly_mask > 0]))
        if fg_level <= bg_level:
            return None  # no real contrast here -- nothing to refine
        cutoff = bg_level + 0.5 * (fg_level - bg_level)
        bright_mask = (crop_blur >= cutoff).astype(np.uint8) * 255

        # Guarantees no growth beyond the hand-drawn polygon regardless of what the cutoff classified further out.
        mask = cv2.bitwise_and(bright_mask, poly_mask)

        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        biggest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        biggest_mask = (labels == biggest_label).astype(np.uint8) * 255

        # Open then close to shave off per-pixel-noise spikes/notches along the boundary -- approxPolyDP below
        # only reduces vertex count, not spiky geometry. Opening can split off a sliver, so re-pick the largest.
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(biggest_mask, cv2.MORPH_OPEN, smooth_kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, smooth_kernel)
        if (smoothed > 0).any():
            n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(smoothed, connectivity=8)
            if n2 > 1:
                biggest_label2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
                biggest_mask = (labels2 == biggest_label2).astype(np.uint8) * 255

        contours, _ = cv2.findContours(biggest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            return None

        refined = approx.astype(np.float32)
        refined[:, 0] += xa
        refined[:, 1] += ya
        return refined

    def _refine(self) -> None:
        panel, frame = self._active, self._frame
        if panel >= len(self._videos):
            self._status.setText("No video loaded on the active panel.")
            return
        video = self._videos[panel]

        selected = self._selected_area
        explicit = selected is not None and selected[0] == panel
        if explicit and selected is not None:
            targets = [(selected[1], selected[2])]
        else:
            # Bulk mode: skip already-refined areas so re-clicking "Refine" only picks up new ones.
            targets = [
                (cell_id, kind) for cell_id, kind in self._visible_cell_ids(panel, frame)
                if (panel, frame, cell_id, kind) not in self._refined_cells
            ]

        if not targets:
            msg = (
                "Selected area no longer exists."
                if explicit
                else "Nothing new to refine — every area on this frame is already refined."
            )
            self._status.setText(msg)
            return

        n_ok = 0
        # (panel, frame, cell_id, kind, resolved_kind, old_pts, was_already_marked) -- restored as one batch on Undo.
        undo_entries: list[tuple[int, int, int, str, str, np.ndarray, bool]] = []
        try:
            for cell_id, kind in targets:
                pts, resolved_kind = self._effective_polygon(panel, frame, cell_id, kind)
                if pts is None:
                    continue
                refined = self._refine_polygon_cv(video, frame, pts)
                if refined is None:
                    continue
                was_marked = (panel, frame, cell_id, kind) in self._refined_cells
                undo_entries.append(
                    (panel, frame, cell_id, kind, resolved_kind, pts.copy(), was_marked)
                )
                self._apply_refined_polygon(panel, frame, cell_id, resolved_kind, refined)
                self._refined_cells.add((panel, frame, cell_id, kind))
                n_ok += 1
        except Exception as exc:
            self._status.setText(f"Refine failed: {exc}")
            print(f"[refine] FAILED: {exc!r}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return

        if undo_entries:

            def _undo_refine(_entries=undo_entries) -> None:
                for p, f, cid, k, rk, old_pts, was_marked in _entries:
                    self._apply_refined_polygon(p, f, cid, rk, old_pts)
                    if not was_marked:
                        self._refined_cells.discard((p, f, cid, k))

            label = (
                f"refining cell {undo_entries[0][2] + 1}"
                if len(undo_entries) == 1
                else f"refining {len(undo_entries)} cell(s)"
            )
            self._push_undo(label, _undo_refine)

        # Cache entry has stale geometry after refine -- drop it so _redraw_overlays rebuilds fresh.
        self._invalidate_roi_cache(panel)
        self._redraw_overlays()
        self._status.setText(f"Refined {n_ok}/{len(targets)} cell(s).")
        print(f"[refine] done: {n_ok}/{len(targets)}", file=sys.stderr)

    # ── display adjustments ───────────────────────────────────────────────────

    def _on_fill_alpha_changed(self, val: int) -> None:
        self._fill_alpha = round(val / 100 * 255)
        self._alpha_lbl.setText(f"{val}%")
        # Cached fills bake their brush in at creation -- drop them all so _redraw_overlays rebuilds with the new alpha.
        for panel in set(self._track_roi_cache) | set(self._manual_roi_cache):
            self._invalidate_roi_cache(panel)
        self._redraw_overlays()

    def _on_blur_changed(self, val: int) -> None:
        self._blur_sigma = val / 10.0
        self._blur_lbl.setText(f"{self._blur_sigma:.1f}")
        self._show_frame(self._frame)

    # ── frame display ─────────────────────────────────────────────────────────

    def _on_slider(self, val: int) -> None:
        if not self._videos:
            return
        self._fspin.blockSignals(True)
        self._fspin.setValue(val)
        self._fspin.blockSignals(False)
        self._show_frame(val)

    def _seek(self, t: int) -> None:
        if self._n_frames:
            self._slider.setValue(int(np.clip(t, 0, self._n_frames - 1)))

    def _show_frame(self, t: int) -> None:
        if not self._videos:
            return
        t = int(np.clip(t, 0, max(0, self._n_frames - 1)))
        self._frame = t

        for grid_pos, file_idx in enumerate(self._visible_indices):
            video = self._videos[file_idx]
            lo, hi = self._ranges[file_idx]
            frame = video[min(t, video.shape[0] - 1)]
            disp = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
            if self._blur_sigma > 0:
                disp = gaussian_filter(disp, sigma=self._blur_sigma)
            self._img_items[grid_pos].setImage(disp, autoLevels=False, levels=(0.0, 1.0))

        self._cancel_progress()
        self._redraw_overlays()
        # Frame scrub never recomputes ΔF/F -- only an explicit "Measure luminance" click does; just move the playhead.
        self._update_playhead(t)

    # ── draw mode ─────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._draw_btn.setChecked(mode == "draw")
        for vb in self._viewboxes:
            vb.setMouseEnabled(x=(mode != "draw"), y=(mode != "draw"))
        if mode == "pan":
            self._cancel_progress()
        self._update_borders()
        self._redraw_overlays()

    def _on_click(self, ev) -> None:
        if not self._viewboxes:
            return

        pos = ev.scenePos()
        btn = ev.button()

        if self._mode != "draw":
            # Double-click a panel to maximise it (or return to the grid).
            if ev.double():
                for grid_pos, vb in enumerate(self._viewboxes):
                    if vb.sceneBoundingRect().contains(pos):
                        file_idx = self._visible_indices[grid_pos]
                        self._set_focused(
                            None if self._focused == file_idx else file_idx
                        )
                        ev.accept()
                        return
                return

            # Single click selects the panel as active (Undo/Clear/Export target) without entering draw mode.
            if btn == Qt.MouseButton.LeftButton:
                for grid_pos, vb in enumerate(self._viewboxes):
                    if vb.sceneBoundingRect().contains(pos):
                        self._select_panel(self._visible_indices[grid_pos])
                        return
            return

        if btn == Qt.MouseButton.RightButton or ev.double():
            if len(self._current_pts) >= 3:
                self._close_polygon()
            else:
                self._cancel_progress()
            ev.accept()
            return

        if btn != Qt.MouseButton.LeftButton:
            return

        for grid_pos, vb in enumerate(self._viewboxes):
            if vb.sceneBoundingRect().contains(pos):
                file_idx = self._visible_indices[grid_pos]
                if file_idx != self._active:
                    self._cancel_progress()
                self._select_panel(file_idx)

                pt = vb.mapSceneToView(QPointF(pos.x(), pos.y()))
                self._current_pts.append((float(pt.x()), float(pt.y())))
                self._draw_progress()
                ev.accept()
                return

    def _draw_progress(self) -> None:
        if not self._current_pts:
            return
        grid_pos = self._grid_pos_of(self._active)
        if grid_pos is None:
            return
        self._clear_progress_items()
        self._prog_panel = grid_pos

        vb = self._viewboxes[grid_pos]
        ci = self._active
        pts = self._current_pts
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]

        line = pg.PlotCurveItem(
            x=xs,
            y=ys,
            pen=pg.mkPen(_color(ci, 160), width=1.5, style=Qt.PenStyle.DashLine),
            connect="all",
        )
        vb.addItem(line)
        self._prog_items.append(line)

        scatter = pg.ScatterPlotItem(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            symbol="o",
            size=7,
            pen=pg.mkPen(_color(ci, 220), width=1),
            brush=pg.mkBrush(_color(ci, 200)),
        )
        vb.addItem(scatter)
        self._prog_items.append(scatter)

    def _clear_progress_items(self) -> None:
        if 0 <= self._prog_panel < len(self._viewboxes):
            vb = self._viewboxes[self._prog_panel]
            for item in self._prog_items:
                try:
                    vb.removeItem(item)
                except Exception:
                    pass
        self._prog_items.clear()
        self._prog_panel = -1

    def _cancel_progress(self) -> None:
        self._clear_progress_items()
        self._current_pts.clear()

    def _close_polygon(self) -> None:
        pts = np.array(self._current_pts, dtype=np.float32)
        panel, frame = self._active, self._frame
        if panel < len(self._polygons):
            # One cell_id shared across every panel's copy -- stacks are spatially/frame-aligned.
            cell_id = self._alloc_cell_id()
            self._polygons[panel].setdefault(frame, {})[cell_id] = pts
            for i in range(len(self._polygons)):
                if i != panel:
                    self._polygons[i].setdefault(frame, {})[cell_id] = pts.copy()

            def _undo_draw(_frame=frame, _cid=cell_id) -> None:
                for p in self._polygons:
                    p.get(_frame, {}).pop(_cid, None)

            self._push_undo(f"drawing cell {cell_id + 1}", _undo_draw)
        self._cancel_progress()
        self._redraw_overlays()
        self._refresh_areas_list()

    # ── overlay rendering ─────────────────────────────────────────────────────

    def _redraw_overlays(self) -> None:
        frame = self._frame

        # Drop last redraw's fixed-region measurement outlines before rebuilding (CellActivity's _measure_luminance).
        for vb_prev, item in self._lum_region_overlay_items:
            if item.scene() is not None:
                try:
                    vb_prev.removeItem(item)
                except Exception:
                    pass
        self._lum_region_overlay_items = []

        for grid_pos, vb in enumerate(self._viewboxes):
            pi = self._visible_indices[grid_pos]
            if pi >= len(self._ov_items):
                continue

            editable = self._mode != "draw"

            # In editable mode, cached items (_manual_roi_cache/_track_roi_cache) are NOT torn down here -- only
            # stray non-cached leftovers. In static (draw) mode nothing is cached, so everything is rebuilt.
            cached_ids = (
                {
                    id(o)
                    for cache in (
                        self._manual_roi_cache.get(pi, {}),
                        self._track_roi_cache.get(pi, {}),
                    )
                    for entry in cache.values()
                    for o in (entry["fill"], entry["roi"], entry["text"])
                }
                if editable
                else set()
            )
            for items in self._ov_items[pi].values():
                for item in items:
                    if id(item) not in cached_ids and item.scene() is not None:
                        try:
                            vb.removeItem(item)
                        except Exception:
                            pass
            self._ov_items[pi] = {}

            for item in self._tr_items[pi]:
                if id(item) not in cached_ids and item.scene() is not None:
                    try:
                        vb.removeItem(item)
                    except Exception:
                        pass
            self._tr_items[pi] = []

            # A checked "tracked" cell_id is skipped here even at its own ref frame -- the tracked-cell block below
            # renders it at every frame instead, avoiding a double-render. Unchecked tracked areas still render here
            # (fresh tracks start unchecked) so the cell doesn't vanish. No-op for CellArea (_track_data always empty).
            tracked_ids = {
                cid for cid, track in self._track_data.get(pi, {}).items()
                if track["tracks"] is not None and self._is_area_checked(pi, cid, "tracked")
            }
            polys = self._polygons[pi].get(frame, {})
            frame_items: list = []
            used_manual_cells: set = set()
            for cell_id, pts in polys.items():
                if cell_id in tracked_ids or not self._is_area_checked(pi, cell_id, "manual"):
                    continue
                is_sel = (pi, cell_id, "manual") == self._selected_area
                if editable:
                    used_manual_cells.add(cell_id)
                    frame_items += self._get_or_update_manual_roi(
                        vb, pi, cell_id, pts, frame, selected=is_sel
                    )
                else:
                    frame_items += self._add_polygon_items(
                        vb, pts, self._cell_color_index(pi, cell_id, "manual"), alpha=220, width=2,
                        selected=is_sel, label=self._cell_label(pi, cell_id),
                    )
            self._ov_items[pi][frame] = frame_items
            self._prune_roi_cache(self._manual_roi_cache, pi, used_manual_cells, editable)

            # Draw tracked polygons at every frame including their own ref frame -- this is their only rendering,
            # since the manual-polygon block above skips them there. Editable mode reuses cells from
            # _track_roi_cache (see _get_or_update_track_roi). No-op for CellArea (_track_data always empty).
            used_track_cells: set = set()
            for cell_id, track in self._track_data.get(pi, {}).items():
                if track["tracks"] is None or not self._is_area_checked(pi, cell_id, "tracked"):
                    continue
                rel = frame - track["start"]   # index into this cell's own tracked slice
                if not (0 <= rel < len(track["tracks"])):
                    continue
                ref_polys = self._polygons[pi].get(track["ref_frame"], {})
                if cell_id not in ref_polys:
                    continue
                override = track["overrides"].get(frame)
                if override is not None:
                    tracked_pts = override
                else:
                    tracked_pts = self._tracked_points(pi, cell_id, rel, ref_polys[cell_id])
                    if tracked_pts is None:
                        continue
                used_track_cells.add(cell_id)
                is_sel = (pi, cell_id, "tracked") == self._selected_area
                if editable:
                    items = self._get_or_update_track_roi(
                        vb, pi, cell_id, tracked_pts, frame, selected=is_sel
                    )
                else:
                    items = self._add_polygon_items(
                        vb, tracked_pts, self._cell_color_index(pi, cell_id, "tracked"), alpha=130,
                        width=1.5, dashed=True, selected=is_sel,
                        label=self._cell_label(pi, cell_id),
                    )
                self._tr_items[pi] += items
            self._prune_roi_cache(self._track_roi_cache, pi, used_track_cells, editable)

            # Frozen cells are self-contained (self._frozen_cells), no self._polygons lookup needed.
            # Always static, never draggable -- same shape on every frame by design -- but still clickable.
            for cell_id, (start_f, end_f, pts) in self._frozen_cells.get(pi, {}).items():
                if not (start_f <= frame <= end_f) or not self._is_area_checked(pi, cell_id, "frozen"):
                    continue
                is_sel = (pi, cell_id, "frozen") == self._selected_area
                self._tr_items[pi] += self._add_polygon_items(
                    vb, pts, self._cell_color_index(pi, cell_id, "frozen"), alpha=130, width=1.5,
                    dashed=True, selected=is_sel, label=self._cell_label(pi, cell_id),
                    on_select=lambda _pi=pi, _cid=cell_id: self._select_area(_pi, _cid, "frozen"),
                )

            # Draw each checked fixed-region measurement's anchor outline -- otherwise invisible, since a fixed
            # region isn't written into self._polygons/track overrides. No-op for CellArea (_lum_measurements empty).
            for (m_panel, m_cell_id, m_kind), recs in self._lum_measurements.items():
                if m_panel != pi or not self._is_area_checked(pi, m_cell_id, m_kind):
                    continue
                for rec in recs:
                    if rec.get("pts") is None or not rec.get("checked", False):
                        continue
                    pts = rec["pts"]
                    outline = pg.PlotCurveItem(
                        x=np.append(pts[:, 0], pts[0, 0]),
                        y=np.append(pts[:, 1], pts[0, 1]),
                        pen=pg.mkPen(rec["color"], width=2, style=Qt.PenStyle.DashLine),
                    )
                    vb.addItem(outline)
                    self._lum_region_overlay_items.append((vb, outline))

    def _add_polygon_items(
        self,
        vb: pg.ViewBox,
        pts: np.ndarray,
        color_idx: int,
        alpha: int = 220,
        width: float = 2.0,
        dashed: bool = False,
        selected: bool = False,
        label: str | None = None,
        on_select=None,
    ) -> list:
        """Add a filled closed polygon + centroid label to vb and return the
        items. *label* is the text shown (defaults to "color_idx + 1" for
        callers that haven't been updated to pass a real cell label --
        every current call site does). *on_select*, if given, makes the
        fill clickable (but not draggable) -- e.g. for a frozen cell shown
        on a frame other than its own reference one."""
        fill = _fill_item(pts, color_idx, self._fill_alpha, on_select=on_select)
        vb.addItem(fill)

        xs = np.append(pts[:, 0], pts[0, 0])
        ys = np.append(pts[:, 1], pts[0, 1])
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        outline_pen = (
            pg.mkPen(QColor("white"), width=3.5, style=style)
            if selected
            else pg.mkPen(_color(color_idx, alpha), width=width, style=style)
        )
        curve = pg.PlotCurveItem(x=xs, y=ys, pen=outline_pen, connect="all")
        vb.addItem(curve)

        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        txt = pg.TextItem(
            label if label is not None else str(color_idx + 1),
            color=_color(color_idx, alpha),
            anchor=(0.5, 0.5),
        )
        txt.setPos(cx, cy)
        vb.addItem(txt)
        return [fill, curve, txt]

    @staticmethod
    def _roi_points(roi: "pg.PolyLineROI") -> np.ndarray:
        """Absolute (N, 2) vertex coordinates of a PolyLineROI (pos + each point)."""
        state = roi.getState()
        px, py = state["pos"].x(), state["pos"].y()
        return np.array(
            [[px + p.x(), py + p.y()] for p in state["points"]], dtype=np.float32
        )

    def _add_editable_polygon(
        self,
        vb: pg.ViewBox,
        pts: np.ndarray,
        color_idx: int,
        on_edit,
        alpha: int = 220,
        dashed: bool = False,
        on_select=None,
        selected: bool = False,
        label: str | None = None,
    ) -> list:
        """Add a filled, draggable PolyLineROI; on_edit(new_pts) runs on
        every change. *label* is the text shown (defaults to "color_idx +
        1")."""
        color = _color(color_idx, alpha)
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        roi_pen = (
            pg.mkPen(QColor("white"), width=3, style=style)
            if selected
            else pg.mkPen(color, width=1.5 if dashed else 2, style=style)
        )

        roi = _NoInsertPolyLineROI(
            [tuple(p) for p in pts],
            closed=True,
            pen=roi_pen,
            handlePen=pg.mkPen(color),
        )
        _enlarge_roi_handles(roi)

        # PolyLineROI only hit-tests handles/edges -- fill forwards interior drags as a whole-ROI translate.
        fill = _fill_item(
            pts, color_idx, self._fill_alpha,
            on_drag=lambda delta: roi.translate(delta),
            roi=roi,
            on_select=on_select,
        )
        vb.addItem(fill)

        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        txt = pg.TextItem(
            label if label is not None else str(color_idx + 1),
            color=color, anchor=(0.5, 0.5),
        )
        txt.setPos(cx, cy)

        def _on_change(_roi=roi) -> None:
            new_pts = self._roi_points(_roi)
            if new_pts.shape[0] < 3:
                return  # ignore edits that would leave less than a polygon
            txt.setPos(float(new_pts[:, 0].mean()), float(new_pts[:, 1].mean()))
            fill.setPolygon(_qpolygon(new_pts))
            on_edit(new_pts)

        roi.sigRegionChanged.connect(_on_change)

        vb.addItem(roi)
        vb.addItem(txt)
        return [fill, roi, txt]

    def _remove_track_roi_entry(self, entry: dict) -> None:
        vb = entry["vb"]
        for item in (entry["fill"], entry["roi"], entry["text"]):
            # Caller may have already invalidated the cache (_select_area/_on_fill_alpha_changed) -- skip gone items.
            if item.scene() is not None:
                try:
                    vb.removeItem(item)
                except Exception:
                    pass

    def _prune_roi_cache(
        self, cache_by_panel: dict, panel: int, used_cells: set, editable: bool
    ) -> None:
        """Drop cached ROIs (manual or tracked, depending on which cache
        dict is passed) for *panel* that weren't reused this redraw.

        Happens when a cell falls out of the tracked range, its checkbox
        gets unchecked, tracking/polygon data is cleared, or the mode
        toggled away from editable (static rendering doesn't use — or
        update — either cache at all).
        """
        cache = cache_by_panel.get(panel)
        if not cache:
            return
        if not editable:
            for entry in cache.values():
                self._remove_track_roi_entry(entry)
            cache_by_panel[panel] = {}
            return
        stale = [ci for ci in cache if ci not in used_cells]
        for ci in stale:
            self._remove_track_roi_entry(cache.pop(ci))

    def _invalidate_roi_cache(self, panel: int) -> None:
        """Drop every cached editable ROI (manual and tracked) for *panel*,
        forcing the next _redraw_overlays to rebuild them fresh.

        Needed wherever a cell's points get rewritten through a code path
        other than dragging it on the canvas -- Refine, Undo, Delete, a
        finished/rolled-back Track run, Copy area, a display-setting change
        that rebakes each fill's brush -- since the reuse path in
        _get_or_update_manual_roi/_get_or_update_track_roi only repositions
        an *existing* ROI's handles to whatever points it's given; it has no
        way to know those points changed for a reason other than the drag
        that would normally trigger it.
        """
        for cache in (self._manual_roi_cache, self._track_roi_cache):
            for entry in cache.pop(panel, {}).values():
                self._remove_track_roi_entry(entry)

    def _get_or_update_track_roi(
        self,
        vb: pg.ViewBox,
        panel: int,
        ci: int,
        pts: np.ndarray,
        frame: int,
        selected: bool = False,
    ) -> list:
        """Reuse a cached tracked-cell ROI if possible, else create one.

        Repositioning an existing PolyLineROI's handles is far cheaper than
        rebuilding it (see the class-level note by _track_roi_cache), so this
        is what makes scrubbing through a heavily-tracked video fast.
        """
        cache = self._track_roi_cache.setdefault(panel, {})
        entry = cache.get(ci)

        if (
            entry is not None
            and entry["vb"] is vb
            and entry["n_verts"] == len(pts)
        ):
            entry["ctx"]["frame"] = frame
            roi = entry["roi"]
            # Block signals for the batch -- per-vertex sigRegionChanged would trigger O(n_handles) _on_change
            # per vertex, an O(n_handles^2) pattern per cell.
            roi.blockSignals(True)
            for h, p in zip(roi.getHandles(), pts):
                roi.movePoint(
                    h, (float(p[0]), float(p[1])), coords="parent", finish=False
                )
            roi.blockSignals(False)
            entry["fill"].setPolygon(_qpolygon(pts))
            entry["text"].setPos(float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            return [entry["fill"], entry["roi"], entry["text"]]

        if entry is not None:
            self._remove_track_roi_entry(entry)

        ctx = {"frame": frame}

        def _commit_track(new_pts, _panel=panel, _ci=ci, _ctx=ctx) -> None:
            f = _ctx["frame"]
            self._track_data[_panel][_ci]["overrides"][f] = new_pts
            # Manual edit supersedes prior auto-refine -- let a later bulk "Refine" pick it up again.
            self._refined_cells.discard((_panel, f, _ci, "tracked"))

        items = self._add_editable_polygon(
            vb, pts, self._cell_color_index(panel, ci, "tracked"), _commit_track,
            alpha=130, dashed=True,
            on_select=lambda _pi=panel, _ci=ci: self._select_area(_pi, _ci, "tracked"),
            selected=selected,
            label=self._cell_label(panel, ci),
        )
        fill, roi, text = items
        cache[ci] = {
            "fill": fill, "roi": roi, "text": text,
            "n_verts": len(pts), "vb": vb, "ctx": ctx,
        }
        return items

    def _get_or_update_manual_roi(
        self,
        vb: pg.ViewBox,
        panel: int,
        cell_id: int,
        pts: np.ndarray,
        frame: int,
        selected: bool = False,
    ) -> list:
        """Reuse a cached manual (hand-drawn) polygon's ROI if possible,
        else create one -- the same reuse-over-rebuild trick as
        _get_or_update_track_roi (see its docstring and the class-level
        note by _track_roi_cache/_manual_roi_cache), and for the same
        reason: rebuilding a PolyLineROI from scratch is expensive, and
        manual polygons previously had no cache at all, making them the
        dominant cost of _redraw_overlays once there were more than a
        handful of drawn cells.

        Keyed by cell_id alone; correct even though a manual polygon's
        points can differ per frame (unlike a tracked cell's one continuous
        shape) because the cached ROI's handles are always repositioned to
        whatever *pts* the caller resolved for the current frame -- same as
        the tracked cache does with its own per-frame reconstructed points.
        """
        cache = self._manual_roi_cache.setdefault(panel, {})
        entry = cache.get(cell_id)

        if (
            entry is not None
            and entry["vb"] is vb
            and entry["n_verts"] == len(pts)
        ):
            entry["ctx"]["frame"] = frame
            roi = entry["roi"]
            # Same batching rationale as _get_or_update_track_roi.
            roi.blockSignals(True)
            for h, p in zip(roi.getHandles(), pts):
                roi.movePoint(
                    h, (float(p[0]), float(p[1])), coords="parent", finish=False
                )
            roi.blockSignals(False)
            entry["fill"].setPolygon(_qpolygon(pts))
            entry["text"].setPos(float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            return [entry["fill"], entry["roi"], entry["text"]]

        if entry is not None:
            self._remove_track_roi_entry(entry)

        ctx = {"frame": frame}

        def _commit_manual(new_pts, _panel=panel, _cid=cell_id, _ctx=ctx) -> None:
            f = _ctx["frame"]
            p = self._polygons[_panel].get(f)
            if p is not None and _cid in p:
                p[_cid] = new_pts
            # Manual edit supersedes prior auto-refine -- let a later bulk "Refine" pick it up again.
            self._refined_cells.discard((_panel, f, _cid, "manual"))

        items = self._add_editable_polygon(
            vb, pts, self._cell_color_index(panel, cell_id, "manual"), _commit_manual,
            on_select=lambda _pi=panel, _cid=cell_id: self._select_area(_pi, _cid, "manual"),
            selected=selected,
            label=self._cell_label(panel, cell_id),
        )
        fill, roi, text = items
        cache[cell_id] = {
            "fill": fill, "roi": roi, "text": text,
            "n_verts": len(pts), "vb": vb, "ctx": ctx,
        }
        return items

    # ── tracking (CellActivity populates self._track_data; this just reads it) ─

    def _tracked_points(
        self, panel: int, ci: int, rel: int, ref_pts: np.ndarray
    ) -> np.ndarray | None:
        """Reconstruct cell *ci*'s polygon at tracked-slice index *rel* for
        *panel*, from its own independent tracking record (see
        self._track_data -- each tracked area has its own tracks array, not
        a shared one, so there's no "offsets" bookkeeping needed to find
        its slice of a bigger array: column 0 is always its centroid,
        columns 1..N its vertices).

        Each annotated vertex was tracked individually (see CellActivity's
        _start_track_for_panel), so by default this returns those
        per-vertex positions directly — letting the polygon deform rather
        than only rigidly translate. Falls back to a rigid centroid shift
        if *ref_pts* was edited (vertex count changed) since tracking ran,
        since the per-vertex correspondence no longer holds. Returns None
        if no tracking data exists for this panel/cell.
        """
        track = self._track_data.get(panel, {}).get(ci)
        if track is None or track["tracks"] is None:
            return None
        tracks = track["tracks"]
        n = tracks.shape[1] - 1
        if n == len(ref_pts):
            return tracks[rel, 1 : 1 + n]
        shift = tracks[rel, 0] - tracks[0, 0]
        return ref_pts + shift

    def _on_track(self) -> None:
        """"Track cells" button: create (or refresh) a "tracked" area for
        the *selected* area's cell_id, on that area's own panel only --
        select a manual or tracked area first (not frozen: re-tracking a
        static snapshot doesn't mean anything). Single area, single panel,
        one CoTracker run; to track the same cell on another panel too,
        select that panel's copy and run this again there."""
        if self._selected_area is None:
            self._status.setText("No area selected — click an area first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Tracking already in progress…")
            return

        panel, cell_id, kind = self._selected_area
        if kind == "frozen":
            self._status.setText("Select a manual or tracked area to track, not a frozen one.")
            return

        frame = self._frame
        polys = self._polygons[panel].get(frame, {})
        if cell_id not in polys:
            self._status.setText("Selected area has no shape on the current frame here.")
            return

        self._start_track_for_panel(panel, cell_id)

    def _start_track_for_panel(self, panel: int, cell_id: int) -> None:
        """Kick off CoTracker for *cell_id* on *panel*, from its manual
        shape at the current frame, forward to the end of the clip."""
        frame = self._frame
        polys = self._polygons[panel].get(frame, {})
        pts = polys.get(cell_id)
        if pts is None:
            self._status.setText("Selected area has no shape on the current frame here.")
            return

        full_video = self._videos[panel]   # (T, H, W)
        n_frames = full_video.shape[0] - frame  # always the rest of the clip
        if n_frames < 1:
            self._status.setText(f"{self._paths[panel].stem}: not enough frames to track.")
            return

        # Query points: centroid AND every vertex, so each is tracked individually and the shape can deform.
        query_points = [(float(pts[:, 0].mean()), float(pts[:, 1].mean()))]
        query_points.extend((float(x), float(y)) for x, y in pts)
        query_points_arr = np.array(query_points, dtype=np.float32)

        # Slice from the annotated frame forward; ref becomes index 0 in the slice
        video_slice = full_video[frame : frame + n_frames]

        # Snapshot as instance state, not a closure var, since the run finishes later/async in _on_track_done.
        self._pending_track_old_record = self._track_data.get(panel, {}).get(cell_id)
        self._pending_track_old_checked = self._area_checked.get((panel, cell_id, "tracked"))
        self._pending_track_old_refined = {
            (p, f, c, k) for (p, f, c, k) in self._refined_cells
            if p == panel and c == cell_id and k == "tracked"
        }

        # A fresh run replaces only the same cell_id's prior record; other cells' records are untouched.
        self._track_data.setdefault(panel, {})[cell_id] = {
            "tracks": None,
            "ref_frame": frame,
            "start": frame,
            "overrides": {},
        }
        # Freshly tracked area starts unchecked -- plot/measurement inclusion is an explicit opt-in, not a side effect.
        self._area_checked[(panel, cell_id, "tracked")] = False
        self._pending_track_panel = panel
        self._pending_track_cell_id = cell_id
        # Only this area's refine-marks go stale (tied to its previous tracking); other areas' marks stay valid.
        self._refined_cells = {
            (p, f, c, k) for (p, f, c, k) in self._refined_cells
            if not (p == panel and c == cell_id and k == "tracked")
        }

        self._track_btn.setEnabled(False)
        self._track_btn.setText("Tracking…")
        self._status.setText(
            f"CoTracker starting — {self._paths[panel].stem}: "
            f"cell {self._cell_label(panel, cell_id)}, "
            f"{n_frames} frames (#{frame} → #{frame + n_frames - 1})…"
        )
        # No redraw/plot update needed -- starts unchecked, nothing to show yet.
        self._refresh_areas_list()

        # ref_frame=0 because the slice already starts at the annotated frame
        self._worker = _TrackWorker(video_slice, query_points_arr, ref_frame=0)
        self._worker.sig_progress.connect(self._on_track_progress)
        self._worker.sig_done.connect(self._on_track_done)
        self._worker.start()

    def _on_track_progress(self, msg: str) -> None:
        self._status.setText(f"Tracking: {msg}")

    def _on_track_done(self, tracks: object) -> None:
        panel = self._pending_track_panel
        cell_id = self._pending_track_cell_id
        self._pending_track_panel = -1
        self._pending_track_cell_id = -1
        old_record = self._pending_track_old_record
        old_checked = self._pending_track_old_checked
        old_refined = self._pending_track_old_refined
        self._pending_track_old_record = None
        self._pending_track_old_checked = None
        self._pending_track_old_refined = set()
        self._track_btn.setEnabled(True)
        self._track_btn.setText("Track cells")

        if isinstance(tracks, str) or tracks is None or not isinstance(tracks, np.ndarray):
            # Roll back the pending record _start_track_for_panel created -- restore the prior record or drop the key.
            data = self._track_data.get(panel)
            if data is not None:
                if old_record is not None:
                    data[cell_id] = old_record
                else:
                    data.pop(cell_id, None)
            if old_checked is not None:
                self._area_checked[(panel, cell_id, "tracked")] = old_checked
            else:
                self._area_checked.pop((panel, cell_id, "tracked"), None)
            self._refined_cells |= old_refined
            # tracks is a str when _TrackWorker.run() hit an exception -- record the real reason, not "see terminal".
            reason = tracks if isinstance(tracks, str) else "see terminal for details"
            self._status.setText(f"{self._paths[panel].stem}: tracking FAILED ({reason})")
            self._refresh_areas_list()  # drop the rolled-back entry from the list
            return

        record = self._track_data.get(panel, {}).get(cell_id)
        if record is None:
            return  # this area's data was cleared (e.g. section changed) mid-run
        record["tracks"] = tracks   # (T, n_vertices+1, 2)
        self._status.setText(
            f"Tracking finished — {self._paths[panel].stem}: "
            f"cell {self._cell_label(panel, cell_id)}"
        )

        def _undo_track(
            _panel=panel,
            _cell_id=cell_id,
            _old_record=old_record,
            _old_checked=old_checked,
            _old_refined=old_refined,
        ) -> None:
            data = self._track_data.get(_panel)
            if data is not None:
                if _old_record is not None:
                    data[_cell_id] = _old_record
                else:
                    data.pop(_cell_id, None)
            if _old_checked is not None:
                self._area_checked[(_panel, _cell_id, "tracked")] = _old_checked
            else:
                self._area_checked.pop((_panel, _cell_id, "tracked"), None)
            self._refined_cells |= _old_refined
            for entry in self._track_roi_cache.pop(_panel, {}).values():
                self._remove_track_roi_entry(entry)

        self._push_undo(f"tracking cell {self._cell_label(panel, cell_id)}", _undo_track)

        # No redraw/plot update -- still unchecked, check it in "Areas:" to see it.
        self._refresh_areas_list()

    # ── undo / clear ──────────────────────────────────────────────────────────

    _UNDO_STACK_LIMIT = 50

    def _push_undo(self, description: str, restore: Callable[[], None]) -> None:
        """Register *restore* as what "Undo" reverts next. Called by every
        discrete action (draw, refine, track, copy area, delete) right
        after it succeeds, so Undo always reverses exactly that action --
        never "whatever cell happens to look newest", which used to delete
        an unrelated cell if the actual last action wasn't a draw."""
        self._undo_stack.append((description, restore))
        del self._undo_stack[: -self._UNDO_STACK_LIMIT]

    def _undo(self) -> None:
        """"Undo" button / Z: reverse the single most recent discrete
        action (see _push_undo) -- drawing a cell, a Refine, a Track cells
        run, "Copy area to next frames", or a Delete. Each of those pushed
        its own precise reversal when it completed; this just pops and
        runs the last one, so "Undo" always undoes what you actually just
        did, not a guess based on whatever data looks newest."""
        if not self._undo_stack:
            self._status.setText("Nothing to undo.")
            return
        description, restore = self._undo_stack.pop()
        restore()
        self._redraw_overlays()
        self._refresh_areas_list()
        self._refresh_lum_list()
        self._status.setText(f"Undid: {description}.")

    def _clear_frame(self) -> None:
        self._cancel_progress()
        panel, frame = self._active, self._frame
        if panel < len(self._polygons):
            self._polygons[panel].pop(frame, None)
            self._redraw_overlays()
            self._refresh_areas_list()

    # ── export ────────────────────────────────────────────────────────────────

    def _compute_export_rows(self) -> tuple[list[dict], list[dict], dict]:
        """Build (rows, tracked_rows, ann) -- one row per (file, frame,
        cell) for every manual/frozen area (`rows`) and every tracked
        area's tracked frames (`tracked_rows`), plus the same information
        restructured as `ann` (per-file JSON annotations). Shared by
        _export() (CellArea's multi-file output) and CellActivity's
        override (a single combined CSV) so the mask-rasterization/area/
        intensity computation only lives in one place."""
        from matplotlib.path import Path as MplPath

        rows: list[dict] = []
        ann: dict = {}

        # mean_intensity is always measured off a uint8-normalized copy (_to_uint8_stack) so a raw 16-bit source
        # and an already-8-bit output are measured on the same scale.
        uint8_videos = [_to_uint8_stack(v) for v in self._videos]

        for pi, (path, video) in enumerate(zip(self._paths, uint8_videos)):
            _, H, W = video.shape
            grid_xy = np.column_stack(
                [np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()]
            )
            fname = path.name
            ann[fname] = []
            for frame_idx, polys in sorted(self._polygons[pi].items()):
                img = video[min(frame_idx, video.shape[0] - 1)]
                for ci, pts in polys.items():
                    mask = MplPath(pts).contains_points(grid_xy).reshape(H, W)
                    area = int(mask.sum())
                    mean_i = float(img[mask].mean()) if area else float("nan")
                    max_i = float(img[mask].max()) if area else float("nan")
                    rows.append(
                        {
                            "file": fname,
                            "frame": frame_idx,
                            "cell_id": self._cell_label(pi, ci),
                            "kind": "manual",
                            "area_pixels": area,
                            "mean_intensity": round(mean_i, 4),
                            "max_intensity": round(max_i, 4),
                            "n_vertices": len(pts),
                            "polygon_vertices": json.dumps(pts.tolist()),
                        }
                    )
                    ann[fname].append(
                        {
                            "frame": frame_idx,
                            "cell_id": self._cell_label(pi, ci),
                            "kind": "manual",
                            "vertices": pts.tolist(),
                        }
                    )

        # Tracked positions (CellActivity only): area/intensity measured from the actual deforming polygon at
        # every frame, a real per-frame time series, not just the reference frame.
        tracked_rows: list[dict] = []
        for panel, cells in self._track_data.items():
            if panel >= len(self._polygons):
                continue
            video = uint8_videos[panel]
            _, H, W = video.shape
            grid_xy = np.column_stack(
                [np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()]
            )
            fname = self._paths[panel].name if panel < len(self._paths) else ""
            for ci, track in cells.items():
                if track["tracks"] is None:
                    continue
                ref = track["ref_frame"]
                start = track["start"]
                ref_polys = self._polygons[panel].get(ref, {})
                if ci not in ref_polys:
                    continue
                ref_pts = ref_polys[ci]
                n_tracked = track["tracks"].shape[0]
                for rel_t in range(n_tracked):
                    abs_frame = start + rel_t
                    img = video[min(abs_frame, video.shape[0] - 1)]
                    override = track["overrides"].get(abs_frame)
                    if override is not None:
                        # Manually corrected: use the edited polygon/centroid instead of the raw CoTracker point.
                        tracked_pts = override
                        cx, cy = float(override[:, 0].mean()), float(override[:, 1].mean())
                    else:
                        tracked_pts = self._tracked_points(panel, ci, rel_t, ref_pts)
                        if tracked_pts is None:
                            continue
                        cx = float(track["tracks"][rel_t, 0, 0])
                        cy = float(track["tracks"][rel_t, 0, 1])
                    mask = MplPath(tracked_pts).contains_points(grid_xy).reshape(H, W)
                    area = int(mask.sum())
                    mean_i = float(img[mask].mean()) if area else float("nan")
                    max_i = float(img[mask].max()) if area else float("nan")
                    tracked_rows.append(
                        {
                            "panel": panel,
                            "file": fname,
                            "ref_frame": ref,
                            "frame": abs_frame,
                            "cell_id": self._cell_label(panel, ci),
                            "kind": "tracked",
                            "track_x": round(cx, 2),
                            "track_y": round(cy, 2),
                            "area_pixels": area,
                            "mean_intensity": round(mean_i, 4),
                            "max_intensity": round(max_i, 4),
                            "manually_corrected": override is not None,
                            "polygon_vertices": json.dumps(tracked_pts.tolist()),
                        }
                    )
        # Frozen cells aren't in self._polygons, so the loop above never sees them -- export separately here.
        for pi, fc in self._frozen_cells.items():
            if pi >= len(self._paths):
                continue
            video = uint8_videos[pi]
            _, H, W = video.shape
            grid_xy = np.column_stack(
                [np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()]
            )
            fname = self._paths[pi].name
            for ci, (start_f, end_f, pts) in fc.items():
                mask = MplPath(pts).contains_points(grid_xy).reshape(H, W)
                area = int(mask.sum())
                label = self._cell_label(pi, ci)
                for frame_idx in range(start_f, min(end_f, video.shape[0] - 1) + 1):
                    img = video[frame_idx]
                    mean_i = float(img[mask].mean()) if area else float("nan")
                    max_i = float(img[mask].max()) if area else float("nan")
                    rows.append(
                        {
                            "file": fname,
                            "frame": frame_idx,
                            "cell_id": label,
                            "kind": "frozen",
                            "area_pixels": area,
                            "mean_intensity": round(mean_i, 4),
                            "max_intensity": round(max_i, 4),
                            "n_vertices": len(pts),
                            "polygon_vertices": json.dumps(pts.tolist()),
                        }
                    )
                    ann.setdefault(fname, []).append(
                        {"frame": frame_idx, "cell_id": label, "kind": "frozen", "vertices": pts.tolist()}
                    )

        return rows, tracked_rows, ann

    def _export(self) -> None:
        """"Export…" button: write `rows`/`tracked_rows` (see
        _compute_export_rows) to CSV(s), plus a JSON annotations file,
        per-frame annotated PNGs, and summary/over-time metric plots.
        CellActivity overrides this entirely with a single combined CSV --
        see its own _export()."""
        rows, tracked_rows, ann = self._compute_export_rows()
        if not rows and not tracked_rows:
            self._status.setText("No polygons to export.")
            return

        # Name output files after the first stack, e.g. "1czi_part000_cells.csv".
        stem = self._paths[0].stem
        default = (
            str(self._paths[0].parent / "nerve_cell_analysis") if self._paths else ""
        )
        out_str = QFileDialog.getExistingDirectory(
            self, "Select output folder", default
        )
        if not out_str:
            return
        out_dir = Path(out_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        if tracked_rows:
            csv_t = out_dir / f"{stem}_cells_tracked.csv"
            with open(csv_t, "w", newline="") as fh:
                w2 = csv.DictWriter(fh, fieldnames=list(tracked_rows[0].keys()))
                w2.writeheader()
                w2.writerows(tracked_rows)
            print(f"Saved: {csv_t}")

        if rows:
            csv_p = out_dir / f"{stem}_cells.csv"
            with open(csv_p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Saved: {csv_p}")

        json_p = out_dir / f"{stem}_cells_annotations.json"
        with open(json_p, "w") as fh:
            json.dump(ann, fh, indent=2)
        print(f"Saved: {json_p}")

        # One PNG per manually-annotated (panel, frame), same scope as `rows`/`ann` -- tracked frames are already in the CSV.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_paths = self._export_metric_plots(rows, tracked_rows, out_dir, stem, plt)

        img_dir = out_dir / f"{stem}_cells_images"
        n_images = 0
        for pi, (path, video) in enumerate(zip(self._paths, self._videos)):
            lo, hi = self._ranges[pi]
            file_stem = path.stem
            for frame_idx, polys in sorted(self._polygons[pi].items()):
                if not polys:
                    continue
                img_dir.mkdir(parents=True, exist_ok=True)
                frame_img = video[min(frame_idx, video.shape[0] - 1)]
                disp = np.clip((frame_img - lo) / (hi - lo), 0.0, 1.0)
                H, W = disp.shape
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                ax.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
                for ci, pts in polys.items():
                    color = _color(self._cell_color_index(pi, ci, "manual")).name()
                    closed = np.vstack([pts, pts[:1]])
                    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5)
                    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                    ax.text(cx, cy, self._cell_label(pi, ci), color=color, fontsize=9,
                            ha="center", va="center")
                ax.set_axis_off()
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                out_png = img_dir / f"{file_stem}_frame{frame_idx:04d}.png"
                fig.savefig(out_png, dpi=100)
                plt.close(fig)
                n_images += 1
        if n_images:
            print(f"Saved {n_images} annotation image(s) -> {img_dir}")

        self._status.setText(
            f"Exported {len(rows)} cell(s) + {n_images} image(s) + "
            f"{len(plot_paths)} plot(s) → {out_dir.name}/"
        )

    _METRIC_PLOT_PALETTE = [
        "#2a78d6", "#1baf7a", "#eda100", "#008300",
        "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
    ]

    def _export_metric_plots(
        self,
        rows: list[dict],
        tracked_rows: list[dict],
        out_dir: Path,
        stem: str,
        plt,
    ) -> list[Path]:
        """Save PNG visualizations of the exported CSV metrics.

        One combined figure (area + intensity over time, one line per file,
        one row per cell) when tracking data exists; a reference-frame bar
        chart, always, covering every annotated cell/file — the only option
        when nothing has been tracked yet.
        """
        palette = self._METRIC_PLOT_PALETTE
        saved: list[Path] = []

        if tracked_rows:
            files = list(dict.fromkeys(r["file"] for r in tracked_rows))
            # Group by (cell_id, kind), not cell_id alone -- manual/frozen views of the same id can share a label.
            cells = sorted(
                {(r["cell_id"], r["kind"]) for r in tracked_rows},
                key=lambda ck: _cell_id_sort_key(ck[0]),
            )
            colors = dict(zip(files, palette))

            fig, axes = plt.subplots(
                len(cells), 2, figsize=(11, 2.6 * len(cells)), squeeze=False
            )
            for row, (cid, kind) in enumerate(cells):
                ax_area, ax_int = axes[row]
                for fname in files:
                    sub = sorted(
                        (r for r in tracked_rows
                         if r["cell_id"] == cid and r["kind"] == kind and r["file"] == fname),
                        key=lambda r: r["frame"],
                    )
                    if not sub:
                        continue
                    frames = [r["frame"] for r in sub]
                    c = colors[fname]
                    ax_area.plot(frames, [r["area_pixels"] for r in sub], color=c, lw=1.4, label=fname)
                    ax_int.plot(frames, [r["mean_intensity"] for r in sub], color=c, lw=1.4, label=fname)
                ax_area.set_ylabel(f"cell {cid} ({kind})\narea (px)", fontsize=9)
                ax_int.set_ylabel("mean intensity", fontsize=9)
                for ax in (ax_area, ax_int):
                    ax.grid(True, linestyle="--", alpha=0.3)
                    ax.tick_params(labelsize=8)
                    for spine in ("top", "right"):
                        ax.spines[spine].set_visible(False)
                if row == 0:
                    ax_area.set_title("Area over time", fontsize=11)
                    ax_int.set_title("Mean intensity over time", fontsize=11)
                if row == len(cells) - 1:
                    ax_area.set_xlabel("frame", fontsize=9)
                    ax_int.set_xlabel("frame", fontsize=9)

            handles = [plt.Line2D([0], [0], color=colors[f], lw=2, label=f) for f in files]
            fig.legend(
                handles=handles, loc="upper center", ncol=min(len(files), 6),
                bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False,
            )
            fig.suptitle(f"{stem}: per-cell metrics over time", fontsize=12, fontweight="bold", y=1.05)
            fig.tight_layout()
            out_png = out_dir / f"{stem}_cells_metrics_over_time.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_png)
            print(f"Saved: {out_png}")

        if rows:
            files = list(dict.fromkeys(r["file"] for r in rows))
            # Group by (cell_id, kind) -- see the tracked_rows block above for why.
            cells = sorted(
                {(r["cell_id"], r["kind"]) for r in rows},
                key=lambda ck: _cell_id_sort_key(ck[0]),
            )
            colors = dict(zip(files, palette))
            x = np.arange(len(cells))
            n_files = max(len(files), 1)
            width = 0.8 / n_files

            fig, (ax_area, ax_int) = plt.subplots(
                1, 2, figsize=(max(8, 1.4 * len(cells) * n_files), 4)
            )
            for fi, fname in enumerate(files):
                areas, intens = [], []
                for cid, kind in cells:
                    match = next(
                        (r for r in rows
                         if r["cell_id"] == cid and r["kind"] == kind and r["file"] == fname), None
                    )
                    areas.append(match["area_pixels"] if match else 0)
                    intens.append(match["mean_intensity"] if match else 0)
                offset = (fi - (n_files - 1) / 2) * width
                c = colors[fname]
                ax_area.bar(x + offset, areas, width=width * 0.9, color=c, label=fname)
                ax_int.bar(x + offset, intens, width=width * 0.9, color=c, label=fname)

            for ax, title, ylabel in (
                (ax_area, "Area at reference frame", "area (px)"),
                (ax_int, "Mean intensity at reference frame", "mean intensity"),
            ):
                ax.set_title(title, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_xlabel("cell", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{cid}({kind})" for cid, kind in cells], fontsize=8)
                ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                ax.tick_params(labelsize=8)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)

            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in files]
            fig.legend(
                handles, files, loc="upper center", ncol=min(n_files, 6),
                bbox_to_anchor=(0.5, 1.08), fontsize=8, frameon=False,
            )
            fig.suptitle(f"{stem}: reference-frame metrics", fontsize=12, fontweight="bold", y=1.15)
            fig.tight_layout()
            out_png = out_dir / f"{stem}_cells_metrics_summary.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_png)
            print(f"Saved: {out_png}")

        return saved

    # ── session save/load ────────────────────────────────────────────────────

    def _session_dict(self) -> dict:
        """Full annotation state: file paths, section, polygons, tracking
        results and manual corrections — enough to resume exactly where the
        session left off, unlike _export()'s measurement CSVs. No backward
        compatibility with the old per-panel-index session format -- an old
        file will fail to load cleanly (see _load_session)."""
        return {
            "next_cell_id": self._next_cell_id,
            "paths": [str(p) for p in self._paths],
            "section": list(self._section) if self._section is not None else None,
            "frame": self._frame,
            "fill_alpha": self._fill_alpha,
            "blur_sigma": self._blur_sigma,
            "polygons": [
                {
                    str(frame): {str(cid): pts.tolist() for cid, pts in polys.items()}
                    for frame, polys in panel_polys.items()
                }
                for panel_polys in self._polygons
            ],
            "track_data": {
                str(panel): {
                    str(cid): {
                        "tracks": (
                            track["tracks"].tolist() if track["tracks"] is not None else None
                        ),
                        "ref_frame": track["ref_frame"],
                        "start": track["start"],
                        "overrides": {
                            str(frame): pts.tolist()
                            for frame, pts in track["overrides"].items()
                        },
                    }
                    for cid, track in cells.items()
                }
                for panel, cells in self._track_data.items()
            },
            "refined_cells": [list(t) for t in sorted(self._refined_cells)],
            "frozen_cells": {
                str(panel): {
                    str(cid): [v[0], v[1], v[2].tolist()] for cid, v in fc.items()
                }
                for panel, fc in self._frozen_cells.items()
            },
            "area_checked": {
                f"{p}:{c}:{k}": v for (p, c, k), v in self._area_checked.items()
            },
        }

    def _write_session_to(self, path: Path) -> None:
        with open(path, "w") as fh:
            json.dump(self._session_dict(), fh)
        self._session_path = path

    def _save_session(self) -> None:
        if not self._paths:
            self._status.setText("Nothing to save — load files first.")
            return
        # Same "{stem}_..." naming convention as _export() -- avoids a generic "session.json" colliding across folders.
        stem = self._paths[0].stem
        default = str(self._paths[0].parent / f"{stem}_session.json")
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Save session", default, "Session files (*.json)"
        )
        if not out_str:
            return
        self._write_session_to(Path(out_str))
        self._status.setText(f"Session saved → {Path(out_str).name}")

    def _quick_save_session(self) -> None:
        """Ctrl+S: re-save over the session file this session was last
        saved to or loaded from, with no dialog -- for a quick "save my
        progress" without breaking flow to pick a filename again. Falls
        back to the normal "Save session…" dialog (same as clicking the
        button) the first time, when there's nowhere "already stored" yet
        to save back to."""
        if not self._paths:
            self._status.setText("Nothing to save — load files first.")
            return
        if self._session_path is None:
            self._save_session()
            return
        self._write_session_to(self._session_path)
        self._status.setText(f"Session saved (Ctrl+S) → {self._session_path.name}")

    def _reset_state(self, next_cell_id: int = 0) -> None:
        """Clear every loaded panel and all annotation state, back to
        exactly what __init__ starts with (cell_id allocation included) --
        shared by _new_session() and _load_session() (which resets before
        loading the saved file's own state back in). Cancels any in-flight
        tracking worker first, same as closeEvent."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._paths = []
        self._videos = []
        self._ranges = []
        self._polygons = []
        self._ov_items = []
        self._tr_items = []
        self._track_data = {}
        self._track_roi_cache = {}
        self._manual_roi_cache = {}
        self._refined_cells = set()
        self._n_frames = 0
        self._frame = 0
        self._focused = None
        self._active = 0
        self._selected_area = None
        self._dff_video_cache = {}
        self._lum_measurements = {}
        self._frozen_cells = {}
        self._area_checked = {}
        self._section = None
        self._next_cell_id = next_cell_id
        self._undo_stack = []
        self._session_path = None

    def _on_new_session(self) -> None:
        """"New session" button: confirm first if there's anything to
        lose (same pattern as _remove_active_panel), since this wipes
        every loaded panel and all annotation state."""
        has_data = bool(self._paths) or any(self._polygons) or bool(self._track_data)
        if has_data:
            box = QMessageBox(self)
            box.setWindowTitle("New session")
            box.setText(
                "Start a new session? Every loaded panel and all "
                "annotations, tracking, and stored measurements will be "
                "lost — including cell_id allocation, which restarts at "
                "0. Save first if you want to keep this work."
            )
            box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Ok:
                self._status.setText("New session: cancelled.")
                return
        self._new_session()

    def _new_session(self) -> None:
        """Clear every loaded panel and all annotation state -- including
        cell_id allocation, which restarts at 0 -- without closing and
        reopening the window. Same reset _load_session performs before
        loading a file, just with nothing loaded back in afterward.
        Display preferences (fill opacity, blur) are left as they are --
        those aren't session/annotation state."""
        self._reset_state(next_cell_id=0)

        self._zoom_btn.setVisible(False)
        self._panel_lbl.setText("Panel: —")

        self._slider.blockSignals(True)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._fspin.setMaximum(0)

        self._sect_start_spin.setEnabled(True)
        self._sect_end_spin.setEnabled(True)
        self._sect_start_spin.setMaximum(0)
        self._sect_end_spin.setMaximum(0)
        self._sect_start_spin.setValue(0)
        self._sect_end_spin.setValue(0)
        self._apply_sect_btn.setEnabled(False)
        self._remove_sect_btn.setEnabled(False)
        self._sect_lbl.setText("(load a stack, then pick a range and Apply)")

        self._rebuild_panels()
        self._refresh_areas_list()
        self._refresh_lum_list()
        self._status.setText("New session — load files to begin.")

    def _load_session(self) -> None:
        in_str, _ = QFileDialog.getOpenFileName(
            self, "Load session", "", "Session files (*.json)"
        )
        if not in_str:
            return
        with open(in_str) as fh:
            data = json.load(fh)

        if "next_cell_id" not in data:
            self._status.setText(
                "Old session format, incompatible with this version — re-annotate."
            )
            return

        self._reset_state(next_cell_id=int(data["next_cell_id"]))

        section = data.get("section")
        # Set before _add_file so each file's _load_video reads only this frame range off disk.
        self._section = tuple(section) if section is not None else None

        for p in data.get("paths", []):
            self._add_file(Path(p))

        if not self._paths:
            self._status.setText("Session referenced no loadable files.")
            return

        # _add_file seeds new panels with a copy of panel 0's polygons -- overwrite with the real saved state.
        for i, panel_polys in enumerate(data.get("polygons", [])):
            if i < len(self._polygons):
                self._polygons[i] = {
                    int(frame): {
                        int(cid): np.array(pts, dtype=np.float32)
                        for cid, pts in poly_map.items()
                    }
                    for frame, poly_map in panel_polys.items()
                }

        for panel_str, cells in data.get("track_data", {}).items():
            panel = int(panel_str)
            self._track_data[panel] = {}
            for cid_str, track in cells.items():
                self._track_data[panel][int(cid_str)] = {
                    "tracks": (
                        np.array(track["tracks"], dtype=np.float32)
                        if track["tracks"] is not None
                        else None
                    ),
                    "ref_frame": track["ref_frame"],
                    "start": track["start"],
                    "overrides": {
                        int(frame): np.array(pts, dtype=np.float32)
                        for frame, pts in track["overrides"].items()
                    },
                }

        self._refined_cells = {tuple(t) for t in data.get("refined_cells", [])}

        self._frozen_cells = {
            int(panel_str): {
                int(cid_str): (v[0], v[1], np.array(v[2], dtype=np.float32))
                for cid_str, v in fc.items()
            }
            for panel_str, fc in data.get("frozen_cells", {}).items()
        }
        self._area_checked = {}
        for key, checked in data.get("area_checked", {}).items():
            p_str, c_str, kind = key.split(":", 2)
            self._area_checked[(int(p_str), int(c_str), kind)] = checked

        self._fill_alpha = data.get("fill_alpha", _FILL_ALPHA)
        self._alpha_slider.blockSignals(True)
        self._alpha_slider.setValue(round(self._fill_alpha / 255 * 100))
        self._alpha_slider.blockSignals(False)
        self._alpha_lbl.setText(f"{self._alpha_slider.value()}%")

        self._blur_sigma = data.get("blur_sigma", 0.0)
        self._blur_slider.blockSignals(True)
        self._blur_slider.setValue(round(self._blur_sigma * 10))
        self._blur_slider.blockSignals(False)
        self._blur_lbl.setText(f"{self._blur_sigma:.1f}")

        if self._section is not None:
            start, end = self._section
            self._sect_start_spin.setMaximum(max(0, self._n_frames - 1))
            self._sect_end_spin.setMaximum(max(0, self._n_frames - 1))
            self._sect_start_spin.setValue(start)
            self._sect_end_spin.setValue(end)
            self._sect_start_spin.setEnabled(False)
            self._sect_end_spin.setEnabled(False)
            self._apply_sect_btn.setEnabled(False)
            self._remove_sect_btn.setEnabled(True)
            self._sect_lbl.setText(
                f"Applied: frames {start}–{end} ({end - start + 1} frames)"
            )

        target_frame = max(0, min(int(data.get("frame", 0)), self._n_frames - 1))
        self._focused = None
        self._zoom_btn.setVisible(False)
        self._active = 0
        self._panel_lbl.setText(f"Panel: {self._active + 1}")
        self._rebuild_panels()
        self._slider.setValue(target_frame)
        self._show_frame(target_frame)
        # track_data/frozen_cells/area_checked are only restored after the _add_file loop, so this was stale until now.
        self._refresh_areas_list()
        self._autorange_panels()
        # So Ctrl+S (_quick_save_session) re-saves over this file by default.
        self._session_path = Path(in_str)
        self._status.setText(
            f"Session loaded — {len(self._paths)} file(s), {self._n_frames} frames"
        )

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k == Qt.Key.Key_S and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._quick_save_session()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Period):
            self._seek(self._frame + 1)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Comma):
            self._seek(self._frame - 1)
        elif k in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            step = _FRAME_JUMP if k == Qt.Key.Key_PageDown else -_FRAME_JUMP
            self._seek(self._frame + step)
        elif k == Qt.Key.Key_Home:
            self._seek(0)
        elif k == Qt.Key.Key_End:
            self._seek(self._n_frames - 1)
        elif k == Qt.Key.Key_D:
            self._set_mode("pan" if self._mode == "draw" else "draw")
        elif k == Qt.Key.Key_Z:
            self._undo()
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected_cell()
        elif k == Qt.Key.Key_R:
            self._refine()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._mode == "draw" and len(self._current_pts) >= 3:
                self._close_polygon()
        elif k == Qt.Key.Key_Escape:
            if self._current_pts:
                self._cancel_progress()
            elif self._focused is not None:
                self._set_focused(None)
        elif k == Qt.Key.Key_Tab:
            # Only meaningful in the full grid -- zoomed in, only one panel is visible.
            if self._paths and self._focused is None:
                self._cancel_progress()
                self._active = (self._active + 1) % len(self._paths)
                self._panel_lbl.setText(f"Panel: {self._active + 1}")
                self._update_borders()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        super().closeEvent(ev)
