#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

"""CellArea — PyQt6/pyqtgraph tool for measuring cell area/size over time
in microscopy videos. Area/size only, no luminance measurement (see
cell_activity.py). Shared behavior (drawing, tracking, refine, sessions,
export, workflow) documented in annotator_base.py.

Run standalone:
    python cell_area.py --files a.h5 b.h5 c.tiff
    python cell_area.py          # file-picker on start

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
Drag a vertex        Move that point
Drag the outline      Move the whole polygon
Right-click an edge   Add a vertex there
Right-click a vertex  Remove that vertex (≥ 3 must remain)
"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from annotator_base import CellAnnotatorBase


class CellArea(CellAnnotatorBase):
    """CellAnnotatorBase (which already includes CoTracker tracking) with
    no luminance measurement -- area/size export only (see the module
    docstring). Uses both extension points' default (no-op) behavior from
    the base: no "Measure luminance"/"Clear measurements" buttons, no ΔF/F
    plot, no delta_f_f export column."""

    def __init__(self, files: list[str] | None = None):
        super().__init__(files=files)
        self.setWindowTitle("CellArea")


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Cell area/size measurement tool for microscopy videos."
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
    w = CellArea(files=args.files)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
