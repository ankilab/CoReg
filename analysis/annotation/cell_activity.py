#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""CellActivity — PyQt6/pyqtgraph tool for measuring cell luminance (ΔF/F)
in microscopy videos, with CoTracker-based motion tracking.

Run standalone:
    python cell_activity.py --files a.h5 b.h5 c.tiff
    python cell_activity.py          # file-picker on start

Controls
--------
D            Toggle draw mode on/off
Click        Add vertex to current polygon
Right-click  Close polygon (needs ≥ 3 points)
Enter        Close polygon (needs ≥ 3 points)
Escape       Cancel in-progress polygon
Z            Undo the last action (draw, refine, track, copy area, delete)
Delete       Delete the selected cell (click a cell to select it first)
Tab          Cycle active panel
← →          Step one frame back / forward
PgUp/PgDn    Jump 10 frames back / forward
Home / End   Jump to first / last frame
Ctrl+S       Save session — re-saves the last save/load file, no dialog

Editing (outside draw mode)
----------------------------
Drag a vertex       Move that point
Drag the outline     Move the whole polygon
Right-click an edge  Add a vertex there
Right-click a vertex Remove that vertex (≥ 3 must remain)

The same dragging/right-click editing also applies to CoTracker's tracked
polygons (dashed outlines) on any non-reference frame — a manual correction
there is remembered per cell/frame and overrides the CoTracker position in
both the display and the exported CSV, until the next "Track cells" run.

CoTracker tracks each cell's centroid *and* every annotated vertex
individually, so tracked polygons deform with the cell instead of only
rigidly translating (falls back to a rigid centroid shift if the reference
polygon's vertex count changed after tracking ran). "Track cells" tracks
the selected area, forward to the end of the clip, on that area's own panel
only — select a manual (or already-tracked) area first, then click it again
to track the same cell on another panel.

Click a cell (in draw-off mode), or a row in the "Areas:" sidebar, to select
it (highlighted white) — Delete or the "Delete cell" button then removes it.
A drawn region is one cell_id shared across every panel, which can have up
to three simultaneous *kinds* per panel: "manual" (as drawn/edited),
"tracked" (CoTracker), and "frozen" (a static snapshot held across N
frames, via "Copy area to next frames"). Deleting a selected manual area
also removes that same cell_id's tracked/frozen areas on that panel;
deleting a tracked or frozen area directly removes just that one. Each
Areas row's checkbox controls both canvas visibility and inclusion in the
bottom ΔF/F plot / batch "Measure luminance" — check exactly the areas you
want to compare.

"Measure luminance" computes ΔF/F (normalized intensity change from a
baseline) from the current frame to the end of the clip, for every checked
area, and stores the result (shown as a dashed curve on the bottom plot and
a row in the measurements list) — nothing is computed just by checking a
box; only this explicit click ever calculates anything. A raw
(unregistered) panel uses a tracked area's actual tracked shape; every
other panel (already registered), or a manual/frozen area anywhere, always
measures the fixed current-frame region instead — tracking further
wouldn't add signal there, just noise.

"Refine" shrinks a hand-drawn polygon to the nearest real cell edge: blurs
the crop, then thresholds against the *actual local background* (a ring
just outside the drawn polygon) rather than an Otsu split of the
polygon's own interior — cell brightness here typically fades gradually
into the background with no sharp edge, so a bimodal split of the
interior alone can land deep inside the cell and cut off real signal.
Intersects that with the polygon's own shape so it can only shrink
inward, never grow past what was drawn, keeps the largest connected
region within that (the real cell is always far larger than any scattered
single/few-pixel sensor-noise speck also passing the threshold), and
simplifies its contour to a smooth polygon — the blur is what rounds the
mask's own boundary rather than leaving it blocky, not just the contour
simplification afterward. With a cell selected it refines just that one;
with none selected it refines every cell visible on the active panel's
current frame. Works on tracked cells too (stored as a per-frame override,
exactly like a manual drag-correction).

The fill opacity and a display-only Gaussian blur (doesn't affect
measurements or exports) are both adjustable via sliders. "New session"
clears every loaded panel and all annotation state (including cell_id
allocation, which restarts at 0) without closing this window. "Save
session…" / "Load session…" persist and restore every loaded file's path,
polygons, tracking results and manual corrections. "Export…" writes a
single {stem}_activity.csv — one row per (file, frame, cell) covering every
manual/frozen/tracked area, with area, mean/max intensity, delta_f_f (F0 =
mean of each cell's first 4 frames), and polygon vertices — no images or
separate plot files, since those can always be regenerated from the
vertices this one file already stores.

Workflow: load stacks (one, several, or all at once), annotate cells on the
first stack, pick a frame range in the Section row and click "Apply section"
— every loaded stack is cropped to that range, and any stack added
afterwards only loads that same range. Each newly added stack is also seeded
with a copy of the first stack's polygons (same pixel coords, same frame),
since aligned stacks share the same cell positions. Drawing a new cell on
any one panel afterwards likewise adds a copy of it to every other loaded
panel at the same frame — edit each panel's copy independently afterwards
if needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)
import numpy as np
import pyqtgraph as pg

from annotator_base import CellAnnotatorBase, _color


def _add_delta_f_f_column(rows: list[dict]) -> None:
    """In place: add a "delta_f_f" key to every row in *rows*, grouped by
    (file, cell_id, kind) and ordered by frame within each group — F0 =
    mean of that cell's first 4 frames' mean_intensity, same convention as
    CellActivity._compute_dff_trace / figures/cell_analysis.py.

    A trace with fewer than 2 frames, or an all-zero F0, has no meaningful
    ΔF/F (same bail-out as _compute_dff_trace) — those rows get an empty
    string instead of a computed value.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["file"], row["cell_id"], row.get("kind")), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["frame"])
        vals = [r["mean_intensity"] for r in group]
        if len(vals) < 2:
            for r in group:
                r["delta_f_f"] = ""
            continue
        f0 = sum(vals[:4]) / len(vals[:4])
        for r, v in zip(group, vals):
            r["delta_f_f"] = round((v - f0) / f0, 4) if f0 else ""


def _is_raw_file(name: str) -> bool:
    """True for an unregistered/ground-truth video: a bare "<clip>.tif" or
    "<clip>.tiff" file whose stem doesn't end in "_corrected". Matches the
    same raw-vs-method filename convention figures/cell_analysis.py's
    parse_file_column() already uses for the exported CSVs, so "raw" means
    the same thing here as it does throughout the rest of this project's
    analysis tooling — an ".h5"/".hdf5" file is always some registration
    method's output, never raw, regardless of its name."""
    lower = name.lower()
    if not (lower.endswith(".tif") or lower.endswith(".tiff")):
        return False
    stem = name.rsplit(".", 1)[0]
    return not stem.endswith("_corrected")


# ── Main widget ───────────────────────────────────────────────────────────────


class CellActivity(CellAnnotatorBase):
    """CellAnnotatorBase (which already includes CoTracker tracking) +
    "Measure luminance" (ΔF/F) + a combined-CSV-only Export."""

    def __init__(self, files: list[str] | None = None):
        super().__init__(files=files)
        self.setWindowTitle("CellActivity")

    def _expand_static_manual_rows(self, rows: list[dict]) -> list[dict]:
        """A manual (non-tracked, non-frozen) cell normally has one export
        row per frame it was actually drawn/edited on -- for the common
        case of "drawn once on frame N, never touched again", that's a
        single row, which _add_delta_f_f_column then can't compute a real
        ΔF/F from (needs >= 2 points to have a trace at all). "Measure
        luminance" already treats exactly this case as a *fixed region*:
        the shape as drawn, reused unchanged for every later frame (see
        _compute_dff_trace's fixed_region branch) -- but that's only ever
        applied to the live ΔF/F plot, never to this export. Mirror it
        here: a manual cell with only one recorded frame gets expanded
        into one row per frame from there to the end of its own panel's
        clip, reusing that same mask -- same technique _compute_export_rows
        already uses for "frozen" cells, just triggered automatically
        instead of requiring an explicit "Copy area to next frames" click
        first. A cell genuinely hand-edited on several distinct frames is
        left untouched -- those per-frame shapes were drawn deliberately,
        not the "drawn once and forgotten" case this is for."""
        from matplotlib.path import Path as MplPath

        file_to_panel = {p.name: i for i, p in enumerate(self._paths)}
        frame_counts: dict[tuple, int] = {}
        for r in rows:
            if r["kind"] == "manual":
                key = (r["file"], r["cell_id"])
                frame_counts[key] = frame_counts.get(key, 0) + 1

        expanded: list[dict] = []
        for r in rows:
            key = (r["file"], r["cell_id"])
            if r["kind"] != "manual" or frame_counts.get(key, 0) != 1:
                expanded.append(r)
                continue

            panel = file_to_panel.get(r["file"])
            if panel is None or panel >= len(self._videos):
                expanded.append(r)
                continue

            video = self._uint8_video(panel)
            n_frames = video.shape[0]
            start_f = r["frame"]
            if start_f >= n_frames - 1:
                expanded.append(r)  # already the last frame -- nothing to extend into
                continue

            pts = np.array(json.loads(r["polygon_vertices"]), dtype=np.float32)
            H, W = video.shape[1:]
            grid_xy = np.column_stack(
                [np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()]
            )
            mask = MplPath(pts).contains_points(grid_xy).reshape(H, W)
            if not mask.any():
                expanded.append(r)
                continue

            for frame_idx in range(start_f, n_frames):
                img = video[frame_idx]
                expanded.append({
                    **r,
                    "frame": frame_idx,
                    "mean_intensity": round(float(img[mask].mean()), 4),
                    "max_intensity": round(float(img[mask].max()), 4),
                })

        return expanded

    def _export(self) -> None:
        """"Export…" button: a single {stem}_activity.csv -- one row per
        (file, frame, cell) covering every manual/frozen/tracked area, with
        area, mean/max intensity, delta_f_f, and (for tracked rows) the
        tracked centroid, alongside the polygon vertices. No JSON, no
        per-frame annotated PNGs, no separate metric plots -- all of those
        can always be regenerated from the vertices this one file already
        stores, so there's nothing they'd capture that isn't already here."""
        rows, tracked_rows, _ann = self._compute_export_rows()
        if not rows and not tracked_rows:
            self._status.setText("No polygons to export.")
            return

        rows = self._expand_static_manual_rows(rows)
        _add_delta_f_f_column(rows)
        _add_delta_f_f_column(tracked_rows)

        # One shared column set -- tracked rows lack vertex count, so it's recomputed from polygon_vertices' length.
        combined: list[dict] = []
        for r in rows:
            combined.append({
                "file": r["file"],
                "frame": r["frame"],
                "cell_id": r["cell_id"],
                "kind": r["kind"],
                "area_pixels": r["area_pixels"],
                "mean_intensity": r["mean_intensity"],
                "max_intensity": r["max_intensity"],
                "delta_f_f": r["delta_f_f"],
                "n_vertices": r["n_vertices"],
                "track_x": "",
                "track_y": "",
                "manually_corrected": "",
                "polygon_vertices": r["polygon_vertices"],
            })
        for r in tracked_rows:
            combined.append({
                "file": r["file"],
                "frame": r["frame"],
                "cell_id": r["cell_id"],
                "kind": r["kind"],
                "area_pixels": r["area_pixels"],
                "mean_intensity": r["mean_intensity"],
                "max_intensity": r["max_intensity"],
                "delta_f_f": r["delta_f_f"],
                "n_vertices": len(json.loads(r["polygon_vertices"])),
                "track_x": r["track_x"],
                "track_y": r["track_y"],
                "manually_corrected": r["manually_corrected"],
                "polygon_vertices": r["polygon_vertices"],
            })
        combined.sort(key=lambda r: (r["file"], r["frame"], str(r["cell_id"]), r["kind"]))

        stem = self._paths[0].stem
        default = str(self._paths[0].parent / f"{stem}_activity.csv")
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Export activity CSV", default, "CSV files (*.csv)"
        )
        if not out_str:
            return

        with open(out_str, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
            w.writeheader()
            w.writerows(combined)

        self._status.setText(
            f"Exported {len(combined)} row(s) → {Path(out_str).name}"
        )

    def _panel_is_raw(self, panel: int) -> bool:
        """Whether panel *panel*'s video is the raw/unregistered one (see
        _is_raw_file) -- used by _measure_luminance to decide whether a
        per-frame tracked shape is a meaningful thing to measure there at
        all, versus an already-corrected/registered output where it isn't."""
        return panel < len(self._paths) and _is_raw_file(self._paths[panel].name)

    # ── toolbar / bottom-panel extension points ────────────────────────────

    def _build_measurement_toolbar(self, tb: QHBoxLayout, sep) -> None:
        self._lum_btn = QPushButton("Measure luminance")
        self._lum_btn.setToolTip(
            "Compute ΔF/F from the current frame to the end of the clip, "
            "for every area checked in \"Areas:\" — decide what to "
            "measure by checking it there, then click once to measure and "
            "store all of them. Each result is stored separately (see the "
            "measurements list) so you can compare across areas/panels.\n\n"
            "Fully automatic, no prompt: a raw (unregistered) panel uses a "
            "tracked area's tracked shape; every other panel (already "
            "corrected/registered), or a manual/frozen area, always uses "
            "the fixed current-frame region instead."
        )
        self._lum_btn.clicked.connect(self._measure_luminance)
        tb.addWidget(self._lum_btn)

        self._clear_lum_btn = QPushButton("Clear measurements")
        self._clear_lum_btn.setToolTip(
            "Discard the selected area's stored \"Measure luminance\" "
            "results"
        )
        self._clear_lum_btn.clicked.connect(self._clear_lum_measurements)
        tb.addWidget(self._clear_lum_btn)
        tb.addWidget(sep())

    def _build_measurement_panel(self, root: QVBoxLayout) -> None:
        # ΔF/F plot shows exactly what's checked in "Areas:" -- but is only computed on explicit "Measure luminance" click.
        self._dff_plot = pg.PlotWidget()
        self._dff_plot.setBackground("#1a1a1a")
        self._dff_plot.setFixedHeight(130)
        self._dff_plot.setLabel("left", "ΔF/F")
        self._dff_plot.setLabel("bottom", "frame")
        self._dff_plot.showGrid(x=True, y=True, alpha=0.2)
        self._dff_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(255, 255, 255, 120))
        self._dff_plot.addItem(self._dff_playhead)
        self._dff_placeholder = pg.TextItem(
            "check an area, then click \"Measure luminance\"", color=(150, 150, 150)
        )
        self._dff_plot.addItem(self._dff_placeholder)
        self._dff_placeholder.setPos(0, 0)

        # Stored "Measure luminance" results for every checked area -- check a row to overlay it on self._dff_plot.
        self._lum_list = QListWidget()
        self._lum_list.setFixedWidth(230)
        self._lum_list.setToolTip(
            "Stored luminance measurements for every checked area — check "
            "a row to overlay it on the plot for comparison"
        )
        self._lum_list.itemChanged.connect(self._on_lum_item_changed)

        dff_row = QHBoxLayout()
        dff_row.addWidget(self._dff_plot, stretch=1)
        dff_row.addWidget(self._lum_list)
        root.addLayout(dff_row)

    def _update_playhead(self, t: int) -> None:
        self._dff_playhead.setPos(t)

    # ── luminance measurement ───────────────────────────────────────────────

    def _compute_dff_trace(
        self,
        panel: int,
        cell_id: int,
        kind: str,
        frame_range: range | None = None,
        fixed_region: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """ΔF/F for (panel, cell_id, kind) -- F0 = mean of the trace's first
        4 frames, same convention as figures/cell_analysis.py's
        compute_delta_f_f. Returns None if there isn't enough data yet.

        *frame_range*, if given (see _measure_luminance), restricts the
        trace to just those frames (clipped to the video's own length)
        instead of the automatic default: every frame *kind* has an
        effective polygon for (the tracked range for "tracked", the frozen
        range for "frozen", or whichever frame(s) it was manually drawn on
        for "manual").

        *fixed_region*, if True, rasterizes the polygon only once (from the
        first frame in *frame_range*) and reuses that same mask for every
        frame instead of asking _effective_polygon per frame -- this is
        what "Measure luminance" falls back to for a non-tracked area,
        since its effective polygon is only ever defined on the frame(s) it
        was actually drawn/frozen on."""
        if panel >= len(self._videos) or panel >= len(self._polygons):
            return None
        uint8_video = self._uint8_video(panel)
        H, W = uint8_video.shape[1:]
        grid_xy = np.column_stack([np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()])

        if frame_range is not None:
            frames = [f for f in frame_range if 0 <= f < uint8_video.shape[0]]
            if not frames:
                return None
        elif kind == "tracked":
            track = self._track_data.get(panel, {}).get(cell_id)
            if track is None or track.get("tracks") is None:
                return None
            frames = range(track["start"], track["start"] + track["tracks"].shape[0])
        elif kind == "frozen":
            frozen = self._frozen_cells.get(panel, {}).get(cell_id)
            if frozen is None:
                return None
            start_f, end_f, _pts = frozen
            frames = range(start_f, end_f + 1)
        else:  # "manual"
            frames = sorted(
                f for f, polys in self._polygons[panel].items() if cell_id in polys
            )
            if not frames:
                return None

        from matplotlib.path import Path as MplPath

        xs, ys = [], []

        if fixed_region:
            pts, _ = self._effective_polygon(panel, frames[0], cell_id, kind)
            if pts is None:
                return None
            mask = MplPath(pts).contains_points(grid_xy).reshape(H, W)
            if not mask.any():
                return None
            for f in frames:
                xs.append(f)
                ys.append(float(uint8_video[min(f, uint8_video.shape[0] - 1)][mask].mean()))
        else:
            for f in frames:
                pts, _ = self._effective_polygon(panel, f, cell_id, kind)
                if pts is None:
                    continue
                mask = MplPath(pts).contains_points(grid_xy).reshape(H, W)
                if not mask.any():
                    continue
                xs.append(f)
                ys.append(float(uint8_video[min(f, uint8_video.shape[0] - 1)][mask].mean()))

        if len(ys) < 2:
            return None
        ys_arr = np.asarray(ys, dtype=np.float32)
        f0 = float(ys_arr[:4].mean())
        if f0 == 0:
            return None
        return np.asarray(xs, dtype=np.float32), (ys_arr - f0) / f0

    def _measure_luminance(self) -> None:
        """"Measure luminance" button: compute ΔF/F from the current frame
        to the end of the clip, for *every area checked* in "Areas:" —
        decide what to measure by checking it there, then one click
        measures and stores all of them. Each result is *stored*
        separately (see self._lum_measurements) rather than overwriting
        anything, so they pile up and can be ticked on/off in
        self._lum_list to compare.

        Which region gets measured is fully determined per area, not
        asked: only a *raw* (unregistered) video's own motion makes a
        per-frame tracked shape meaningful. An already-corrected/registered
        output should already have this cell roughly stationary, so
        tracking it further doesn't add real signal and just risks
        conflating tracking noise with registration quality; it's always
        measured as the fixed region instead. A raw panel uses the tracked
        shape for a tracked area, else falls back to the fixed region the
        same as a manual/frozen area anywhere."""
        checked = self._checked_areas()
        if not checked:
            self._status.setText("Nothing checked in \"Areas:\" — check an area first.")
            return

        n = self._n_frames - self._frame  # always the rest of the clip
        frame_range = range(self._frame, self._frame + n)
        stored = 0
        failed: list[str] = []
        for panel, cell_id, kind in checked:
            is_tracked = kind == "tracked"
            fixed_region = not (is_tracked and self._panel_is_raw(panel))
            result = self._compute_dff_trace(
                panel, cell_id, kind, frame_range=frame_range, fixed_region=fixed_region
            )
            if result is None:
                panel_name = self._paths[panel].stem if panel < len(self._paths) else f"panel {panel + 1}"
                failed.append(f"{panel_name}:{self._cell_label(panel, cell_id)}({kind})")
                continue

            xs, dff = result
            mode_label = "fixed region" if fixed_region else "tracked shape"
            # Grab the anchor polygon so _redraw_overlays can draw its outline -- otherwise no visual trace of
            # where a "fixed region" reading came from.
            region_pts = None
            if fixed_region:
                region_pts, _ = self._effective_polygon(panel, self._frame, cell_id, kind)

            key = (panel, cell_id, kind)
            existing = self._lum_measurements.setdefault(key, [])
            record = {
                "xs": xs,
                "dff": dff,
                "mode": mode_label,
                "n": n,
                "final_pct": float(dff[-1]) * 100.0,
                "color": self._lum_record_color(panel, cell_id, kind, len(existing)),
                "pts": None if region_pts is None else np.asarray(region_pts).copy(),
                # Newest measurement starts checked/visible.
                "checked": True,
            }
            existing.append(record)
            stored += 1

        self._refresh_lum_list()
        self._redraw_overlays()
        msg = (
            f"Stored {stored} measurement(s) across {len(checked)} checked area(s) "
            f"— frames [{self._frame}, {self._frame + n - 1}]."
        )
        if failed:
            # "No data" distinct from silent failure -- usually a tracked area whose range excludes self._frame.
            msg += f" No data here for: {', '.join(failed)}."
        self._status.setText(msg)

    def _lum_record_color(self, panel: int, cell_id: int, kind: str, repeat_idx: int) -> tuple[int, int, int]:
        """RGB tuple for a stored measurement record -- base hue tied to
        the cell's identity (matching its canvas outline, see
        _cell_color_index) so a stored measurement always visually ties
        back to "this cell", even across a batch covering several
        different areas at once. Repeat measurements of the *same* area
        are distinguished by alternating lightness instead of hue."""
        base = _color(self._cell_color_index(panel, cell_id, kind))
        cycle = repeat_idx % 3
        if cycle == 1:
            base = base.lighter(140)
        elif cycle == 2:
            base = base.darker(140)
        return (base.red(), base.green(), base.blue())

    def _lum_records_for_checked_areas(self) -> list[tuple[int, int, str, dict]]:
        """(panel, cell_id, kind, record) tuples for every stored
        measurement of a currently *checked* area -- matches the same scope
        as batch Measure luminance (see _checked_areas), so what's checked
        is exactly what shows up here too."""
        out = []
        for panel, cell_id, kind in self._checked_areas():
            for rec in self._lum_measurements.get((panel, cell_id, kind), []):
                out.append((panel, cell_id, kind, rec))
        return out

    def _on_lum_item_changed(self, item: QListWidgetItem) -> None:
        """A row's checkbox toggled in self._lum_list: show/hide the
        corresponding stored measurement's overlay curve on self._dff_plot,
        and (for a fixed-region measurement) its region outline on that
        record's own panel, drawn by _redraw_overlays."""
        idx = self._lum_list.row(item)
        checked = item.checkState() == Qt.CheckState.Checked
        if 0 <= idx < len(self._lum_overlay_curves):
            self._lum_overlay_curves[idx].setVisible(checked)
        if 0 <= idx < len(self._lum_list_records):
            _p, _c, _k, rec = self._lum_list_records[idx]
            rec["checked"] = checked
        self._redraw_overlays()

    def _refresh_lum_list(self) -> None:
        """Rebuild self._lum_list (and its plot overlay curves) for every
        stored measurement of a currently checked area (see
        _lum_records_for_checked_areas), and update the plot's
        placeholder/title/playhead to match -- the only thing that ever
        puts a curve on self._dff_plot is a stored "Measure luminance"
        result; nothing is computed just by checking a box. Called on
        selection/checkbox change and whenever a measurement is added or
        cleared. Each record's own "checked" flag (not the list widget) is
        the source of truth for whether it's shown — this list only needs
        to reflect it, not decide it. The region-outline overlay itself is
        drawn by _redraw_overlays, not here, since that's keyed off the
        same "checked" flag."""
        for curve in self._lum_overlay_curves:
            self._dff_plot.removeItem(curve)
        self._lum_overlay_curves = []
        self._lum_list_records = []

        self._lum_list.blockSignals(True)
        self._lum_list.clear()
        titles = []
        n_shown = 0
        for panel, cell_id, kind, rec in self._lum_records_for_checked_areas():
            curve = self._dff_plot.plot(
                rec["xs"],
                rec["dff"],
                pen=pg.mkPen(rec["color"], width=2, style=Qt.PenStyle.DashLine),
            )
            shown = rec.get("checked", False)
            curve.setVisible(shown)
            self._lum_overlay_curves.append(curve)
            self._lum_list_records.append((panel, cell_id, kind, rec))

            panel_name = self._paths[panel].stem if panel < len(self._paths) else f"panel {panel + 1}"
            label = f"{panel_name}: {self._cell_label(panel, cell_id)}({kind})"
            item = QListWidgetItem(
                f"{label} f{int(rec['xs'][0])}–{int(rec['xs'][-1])} (N={rec['n']}) "
                f"{rec['mode']}: {rec['final_pct']:+.1f}%"
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if shown else Qt.CheckState.Unchecked
            )
            item.setForeground(QColor(*rec["color"]))
            self._lum_list.addItem(item)
            if shown:
                n_shown += 1
                titles.append(label)
        self._lum_list.blockSignals(False)

        if n_shown == 0:
            self._dff_plot.setTitle(None)
            self._dff_placeholder.setText(
                "check an area, then click \"Measure luminance\""
            )
            self._dff_placeholder.show()
        else:
            self._dff_placeholder.hide()
            self._dff_plot.setTitle(" | ".join(titles))
        self._dff_playhead.setPos(self._frame)

    def _clear_lum_measurements(self) -> None:
        """"Clear measurements" button: drop all stored luminance
        measurements for exactly the selected area -- deliberately scoped
        to the single selection (not the whole checked set), staying
        narrow/precise unlike the additive/batch Measure luminance."""
        if self._selected_area is None:
            self._status.setText("No area selected.")
            return
        had = self._lum_measurements.pop(self._selected_area, None) is not None
        self._refresh_lum_list()
        self._redraw_overlays()
        self._status.setText(
            "Cleared stored luminance measurements for this area."
            if had else "No stored measurements for the selected area."
        )


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Cell luminance (ΔF/F) measurement tool, with CoTracker-based tracking."
    )
    ap.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="FILE",
        help="HDF5 or TIFF video file(s) to annotate.",
    )
    args = ap.parse_args(argv)

    # Qt exits at C-level (no traceback) when it can't open a display -- check early for a readable message.
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
    w = CellActivity(files=args.files)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
