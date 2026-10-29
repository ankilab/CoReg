#!/usr/bin/env python3

# Copyright 2026 AnkiLab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
# All rights reserved.

# SPDX-License-Identifier: MIT

import csv
import io
import json
import math
import os
import sys
import zipfile
import cv2
import numpy as np
import pyqtgraph as pg
import tifffile
import torch

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
except Exception:
    AutoImageProcessor = None
    AutoModelForKeypointMatching = None
from PyQt6.QtCore import QElapsedTimer, QTimer, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QSpinBox,
)

from gui import (
    AntsRegistrationDialog,
    CutsExportWorker,
    MotionProfileWidget,
    NormCorreDialog,
    ParameterDialog,
    PGImagePanel,
    ProcessingWorker,
    PyFlowRegDialog,
    TimeSliceDialog,
    VideoLoadSettingsDialog,
)
from utils.mc_cotracker import cotrack_model, release_gpu_memory
from utils.cache_utils import RegistrationCache
import utils.presets as _presets

from utils.utils import (
    LazyVideo,
    load_hdf5,
    load_tiff,
    load_video,
    get_tiff_axes_info,
    get_tiff_truncation_warning,
    probe_media_info,
    resize_video_tensor_to_wh,
    video_metrics,
    save_and_display_video,
    to_uint8_contrast,
    _to_gray_float,
    _to_numpy,
)

# Configure pyqtgraph for row-major image display (matches numpy indexing)
pg.setConfigOptions(imageAxisOrder="row-major")

SUPPORTED_VIDEO_EXT = [".avi", ".mp4"]
SUPPORTED_TIFF_EXT = [".tiff", ".tif"]
SUPPORTED_HDF5_EXT = [".h5", ".hdf5"]
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXT + SUPPORTED_TIFF_EXT + SUPPORTED_HDF5_EXT


class _StatusRedirect(io.TextIOBase):
    """Forwards every write() to a QLabel, showing the last non-empty line."""

    def __init__(self, label, original):
        super().__init__()
        self._label = label
        self._original = original

    def write(self, text):
        self._original.write(text)
        stripped = text.rstrip("\n").strip()
        if stripped:
            self._label.setText(stripped)
        return len(text)

    def flush(self):
        self._original.flush()


class ControllerGUI(QMainWindow):
    """
    Main Application Window for Motion Correction and Video Registration.
    Features:
    - Video loading (Video, TIFF, HDF5)
    - Manual and automated point tracking (CoTracker)
    - Rigid/Non-rigid registration (ANTs, LDDMM)
    - Interactive track editing and manual corrections
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoReg")

        screen = QApplication.primaryScreen()
        self._initial_zoom_done = False
        self._auto_fit_done = False

        # Capped regardless of reported screen size -- some X11-forwarding setups report 2 monitors as one combined screen.
        _MAX_WINDOW_W, _MAX_WINDOW_H = 1920, 1080

        if screen is not None:
            g = screen.availableGeometry()
            w = min(int(g.width() * 0.8), _MAX_WINDOW_W)
            h = min(int(g.height() * 0.8), _MAX_WINDOW_H)
            self.resize(w, h)
            self.move(
                g.left() + (g.width() - w) // 2,
                g.top() + (g.height() - h) // 2,
            )
        else:
            # Fallback size if no screen is available (headless environments)
            self.resize(1280, 720)

        self.setAcceptDrops(True)
        self._tracks_user_edited = False
        self.video_path = None
        self.video_data = None
        self._original_video_data = None
        self._frame_start = 0
        self._frame_end = None
        self._last_preset_index = 0

        self.pred_tracks = None
        self.pred_visibility = None
        self.pred_confidence = None
        self.grid_pts = None
        self.warped_video = None
        self.ants_video = None
        self.lddmm_video = None
        self.normcorre_video = None
        self.flowreg_video = None
        self.disp_field = None
        self.drift_xy = None

        # Initialize registration cache
        self.registration_cache = RegistrationCache()

        self.grid_rows = 4
        self.grid_cols = 4
        self.grid_spacing = 0  # pixels; 0 means disabled and use rows/cols
        # base_grid_pts/base_grid_active/track_meta are parallel arrays indexed by point index; must stay same length.
        self.base_grid_pts = None
        self.base_grid_active = None
        self.base_grid_pts0 = None
        self.custom_grid_pts = []
        # (start, end) slice of base_grid_pts from the most recent preset -- lets a tweak regenerate just those points.
        self._last_preset_range = None

        self._drag = None
        self._drag_start_tracks = None
        self._drag_start_pos = None
        self._img_cids = {}

        # Keep track of which tracks were edited in the current session
        self._edited_track_indices = set()
        self._last_edit_frame = None

        self.edit_points_mode = False
        self.edit_tracks_mode = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        # Contrast levels frozen for the playback run -- per-frame auto-levels would flicker as brightness varies.
        self._play_levels_orig = None
        self._play_levels_filt = None

        self.viz_data = {}

        self.worker = None
        self._cuts_export_worker = None
        self.current_filter = "None"
        self.invert_display = False
        self.fps = 30
        self.ants_params = {
            "type_of_transform": "SyNOnly",
            "aff_metric": "mattes",
            "syn_metric": "mattes",
            "reg_iterations": [40, 20, 0],
        }
        self.normcorre_params = {
            "pw_rigid": True,
            "max_shifts": (6, 6),
            "strides": (96, 96),
            "overlaps": (32, 32),
            "upsample_factor_fft": 4,
            "max_deviation_rigid": 3,
        }
        self.pyflowreg_params = {
            "alpha": 1.5,
            "min_level": 4,
            "buffer_size": 100,
            "save_w": True,
        }

        self.ssim_score = None
        self.track_active = None
        self.selected_track_idx = None
        self.base_grid_tracked = None
        self.custom_grid_tracked = []

        self.track_meta = []
        self._pending_track_meta = []
        self._pending_base_idxs_last_run = None
        self._pending_custom_idxs_last_run = None

        # ---- Deferred Warping Recalculation (Debounced) ----
        self._recalc_timer = QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.timeout.connect(self._recalculate_warping_from_tracks)
        self._recalc_inflight = False
        self._pending_recalc = False

        self._debounce_ms = 5000
        self._edit_clock = QElapsedTimer()
        self._edit_clock.start()
        self._target_recalc_time_ms = 0  # absolute time in elapsed-ms domain

        # Incremental tracking state
        self.all_query_points_at_frame0 = None  # (N, 2) array matching pred_tracks

        self.init_ui()

    def _schedule_recalc_warping(self, delay_ms=None):
        """
        Schedule a warping recalculation after an edit.
        Implements debouncing to prevent excessive processing during interaction.
        """
        if self.video_data is None or self.pred_tracks is None:
            return

        if delay_ms is None:
            delay_ms = self._debounce_ms

        # Set deadline = now + delay
        now = self._edit_clock.elapsed()
        self._target_recalc_time_ms = now + int(max(0, delay_ms))

        # If a recalculation is currently in flight, queue a follow-up
        if self._recalc_inflight:
            self._pending_recalc = True
            return

        # Start/restart timer for the remaining time until deadline
        remaining = max(0, self._target_recalc_time_ms - self._edit_clock.elapsed())
        self._recalc_timer.stop()
        self._recalc_timer.start(int(remaining))

    # ---------- display helpers ----------
    @staticmethod
    def flow_to_color(flow):
        """Map optical flow magnitude to 8-bit intensity for visualization."""
        if flow is None:
            return None
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        if mag.max() > 0:
            return (mag / mag.max() * 255.0).astype(np.uint8)
        return np.zeros_like(mag, dtype=np.uint8)

    def disp_scalar_u8(self, disp_hw, clip=None):
        """Convert displacement array to 8-bit intensity with optional clipping."""
        d = np.asarray(disp_hw, dtype=np.float32)
        finite = np.isfinite(d)
        if not finite.any():
            return np.zeros(d.shape, dtype=np.uint8)

        if clip is None:
            a = np.abs(d[finite])
            clip = float(np.percentile(a, 99.5))
            if clip <= 1e-6:
                clip = 1.0

        d = np.clip(d, -clip, clip)
        out = ((d + clip) / (2 * clip) * 255.0).astype(np.uint8)
        return out

    def _tracks_to_numpy(self):
        """Robustly convert prediction tracks to a numpy array."""
        if self.pred_tracks is None:
            return None
        if isinstance(self.pred_tracks, np.ndarray):
            return self.pred_tracks
        try:
            return self.pred_tracks.detach().cpu().numpy()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return np.asarray(self.pred_tracks)

    # ---------- UI ----------
    def init_ui(self):
        """Build the main GUI layout, toolbars, and view panels."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Infrequent actions live in dropdown menus; toolbar keeps only constantly-tuned controls. Each menu entry
        # wraps the same widget instance (via QWidgetAction), so existing .isChecked()/.setEnabled() refs keep working.
        menu_bar = self.menuBar()

        # One shared flat style so QPushButton/QCheckBox read as the same kind of row; also overrides load_btn's
        # old bold toolbar styling, which no longer fits as a menu entry.
        _MENU_WIDGET_STYLE = """
            QPushButton, QCheckBox {
                font-size: 12px;
                font-weight: normal;
                padding: 4px 8px;
                border: none;
                background: transparent;
                text-align: left;
            }
            QPushButton:hover, QCheckBox:hover {
                background: rgba(128, 128, 128, 60);
            }
            QPushButton:disabled, QCheckBox:disabled {
                color: gray;
            }
        """

        def _menu_widget(menu, widget):
            widget.setStyleSheet(_MENU_WIDGET_STYLE)
            action = QWidgetAction(menu)
            action.setDefaultWidget(widget)
            menu.addAction(action)
            return action

        file_menu = menu_bar.addMenu("File")
        edit_menu = menu_bar.addMenu("Edit")
        view_menu = menu_bar.addMenu("View")
        processing_menu = menu_bar.addMenu("Processing")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        main_layout.addLayout(toolbar)

        # 1. Video Loading
        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.select_file)
        self.load_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        _menu_widget(file_menu, self.load_btn)

        # 2. Filter
        toolbar.addWidget(QLabel("Input Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Gaussian", "Sobel", "Canny"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        toolbar.addWidget(self.filter_combo)

        # 3a. Edit Points (grid/custom)
        self.edit_points_btn = QCheckBox("Edit Points")
        self.edit_points_btn.stateChanged.connect(self.on_edit_modes_changed)
        self.edit_points_btn.setEnabled(False)
        _menu_widget(edit_menu, self.edit_points_btn)

        # 3b. Edit Tracks (post-registration)
        self.edit_tracks_btn = QCheckBox("Edit Tracks")
        self.edit_tracks_btn.stateChanged.connect(self.on_edit_modes_changed)
        self.edit_tracks_btn.setEnabled(False)  # enabled only when tracks exist
        _menu_widget(edit_menu, self.edit_tracks_btn)

        # Retrack button: retrack all tracks from the last edited frame
        self.retrack_btn = QPushButton("Retrack from Frame")
        self.retrack_btn.setEnabled(False)
        self.retrack_btn.clicked.connect(self.on_retrack_from_edited_frame_clicked)
        _menu_widget(edit_menu, self.retrack_btn)

        # AutoWarp button: recompute deformation using current tracks
        self.recalculate_btn = QPushButton("AutoWarp")
        self.recalculate_btn.setEnabled(False)
        self.recalculate_btn.clicked.connect(self.on_autowarp_clicked)
        _menu_widget(edit_menu, self.recalculate_btn)

        # self.apply_all_tracks_chk = QCheckBox("Apply to all")
        # self.apply_all_tracks_chk.setEnabled(False)
        # top_bar.addWidget(self.apply_all_tracks_chk)

        # 4. Preset
        toolbar.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "None (no points)",
                "Regular grid",
                "Features",
                "Motion",
                "Greedy motion",
                "Segmentation",
                "MatchAnything",
            ]
        )
        self._last_preset_index = self.preset_combo.currentIndex()
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        toolbar.addWidget(self.preset_combo)

        # 4b. Area selector: drag a square/rectangle on the left view to
        # restrict where preset-generated points are placed.
        self.roi_select_chk = QCheckBox("Restrict to area")
        self.roi_select_chk.setToolTip(
            "Drag out a square on the image; new preset points are only placed inside it"
        )
        self.roi_select_chk.toggled.connect(self.on_roi_select_toggled)
        toolbar.addWidget(self.roi_select_chk)

        # 5. Parameters
        self.settings_btn = QPushButton("⚙️ Parameters")
        self.settings_btn.clicked.connect(self.open_parameter_dialog)
        toolbar.addWidget(self.settings_btn)

        # 6. Registration Process
        self.process_btn = QPushButton("Registration")
        self.process_btn.clicked.connect(self.on_process_clicked)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 5px;"
        )
        toolbar.addWidget(self.process_btn)

        # 7. Stop
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)

        # 8. Save
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setToolTip("Save corrected outputs and/or session (.coreg)")
        self.save_btn.clicked.connect(self.save_results)
        _menu_widget(file_menu, self.save_btn)

        # 9. Load session (.coreg)
        self.load_session_btn = QPushButton("Load Session")
        self.load_session_btn.setToolTip("Restore configuration and points from a .coreg file")
        self.load_session_btn.clicked.connect(self.load_coreg)
        _menu_widget(file_menu, self.load_session_btn)

        # 10. Export current frame as an image
        self.export_frame_btn = QPushButton("Export Frame...")
        self.export_frame_btn.setToolTip("Save the currently displayed frame as an image at a chosen DPI")
        self.export_frame_btn.clicked.connect(self.export_frame_dialog)
        _menu_widget(file_menu, self.export_frame_btn)

        self.force_cpu_chk = QCheckBox("Force CPU")
        self.force_cpu_chk.setToolTip("Run all processing on CPU instead of GPU")
        _menu_widget(processing_menu, self.force_cpu_chk)

        # --- Parameter Widgets (Detailed) ---
        # These are created here but added to the ParameterDialog instead of main window
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(10, 5000)
        self.num_points_spin.setValue(50)
        self.num_points_spin.valueChanged.connect(self.on_preset_params_changed)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(1, 100)
        self.min_dist_spin.setValue(10)
        self.min_dist_spin.valueChanged.connect(self.on_preset_params_changed)

        self.match_min_score_spin = QDoubleSpinBox()
        self.match_min_score_spin.setRange(0.0, 1.0)
        self.match_min_score_spin.setSingleStep(0.05)
        self.match_min_score_spin.setDecimals(2)
        self.match_min_score_spin.setValue(0.20)
        self.match_min_score_spin.valueChanged.connect(self.on_preset_params_changed)

        self.restart_matchanything_btn = QPushButton("Restart MatchAnything")
        self.restart_matchanything_btn.clicked.connect(self.restart_matchanything)
        self.restart_matchanything_btn.setVisible(False)
        self.restart_matchanything_btn.setEnabled(False)

        # Greedy motion parameters
        def _dbl(lo, hi, step, val, decimals=3):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(decimals)
            w.setValue(val)
            w.valueChanged.connect(self.on_preset_params_changed)
            return w

        def _int(lo, hi, val):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            w.valueChanged.connect(self.on_preset_params_changed)
            return w

        self.lbl_gm_sigma          = QLabel("Sigma (smoothing):")
        self.gm_sigma_spin         = _dbl(0.1, 20.0, 0.1, 1.8, 1)
        self.lbl_gm_gamma          = QLabel("Gamma (motion bias):")
        self.gm_gamma_spin         = _dbl(0.05, 5.0, 0.05, 0.5, 2)
        self.lbl_gm_bg_weight      = QLabel("Background weight:")
        self.gm_bg_weight_spin     = _dbl(0.001, 0.5, 0.001, 0.015, 3)
        self.lbl_gm_peak_min_dist  = QLabel("Peak min distance (px):")
        self.gm_peak_min_dist_spin = _int(1, 100, 3)
        self.lbl_gm_peak_thr       = QLabel("Peak threshold (rel):")
        self.gm_peak_thr_spin      = _dbl(0.01, 1.0, 0.01, 0.06, 2)
        self.lbl_gm_peak_frac      = QLabel("Max peak fraction:")
        self.gm_peak_frac_spin     = _dbl(0.0, 1.0, 0.05, 0.8, 2)
        self.lbl_gm_cand_factor    = QLabel("Candidate factor:")
        self.gm_cand_factor_spin   = _int(1, 50, 5)
        self.lbl_gm_samp_dist      = QLabel("Sample min distance (px):")
        self.gm_samp_dist_spin     = _dbl(1.0, 100.0, 0.5, 8.0, 1)
        self.lbl_gm_seed           = QLabel("Random seed:")
        self.gm_seed_spin          = _int(0, 9999, 0)

        self._gm_widgets = [
            self.lbl_gm_sigma,          self.gm_sigma_spin,
            self.lbl_gm_gamma,          self.gm_gamma_spin,
            self.lbl_gm_bg_weight,      self.gm_bg_weight_spin,
            self.lbl_gm_peak_min_dist,  self.gm_peak_min_dist_spin,
            self.lbl_gm_peak_thr,       self.gm_peak_thr_spin,
            self.lbl_gm_peak_frac,      self.gm_peak_frac_spin,
            self.lbl_gm_cand_factor,    self.gm_cand_factor_spin,
            self.lbl_gm_samp_dist,      self.gm_samp_dist_spin,
            self.lbl_gm_seed,           self.gm_seed_spin,
        ]

        self.grid_rows_ctrl = QSpinBox()
        self.grid_rows_ctrl.setRange(2, 100)
        self.grid_rows_ctrl.setValue(int(getattr(self, "grid_rows", 4)))
        self.grid_rows_ctrl.valueChanged.connect(self.on_preset_params_changed)

        self.grid_cols_ctrl = QSpinBox()
        self.grid_cols_ctrl.setRange(2, 100)
        self.grid_cols_ctrl.setValue(int(getattr(self, "grid_cols", 4)))
        self.grid_cols_ctrl.valueChanged.connect(self.on_preset_params_changed)

        # Spacing parameter (pixels); 0 disables spacing and uses rows/cols
        self.grid_spacing_spin = QSpinBox()
        self.grid_spacing_spin.setRange(-1000, 1000)
        self.grid_spacing_spin.setSingleStep(5)
        self.grid_spacing_spin.setValue(int(getattr(self, "grid_spacing", 0)))
        self.grid_spacing_spin.valueChanged.connect(self.on_preset_params_changed)

        # Center offset controls (pixels) to shift grid center
        self.lbl_offset_x = QLabel("Center offset X (px):")
        self.grid_offset_x_spin = QSpinBox()
        self.grid_offset_x_spin.setRange(-2000, 2000)
        self.grid_offset_x_spin.setSingleStep(10)
        self.grid_offset_x_spin.setValue(int(getattr(self, "grid_offset_x", 0)))
        self.grid_offset_x_spin.valueChanged.connect(self.on_preset_params_changed)

        self.lbl_offset_y = QLabel("Center offset Y (px):")
        self.grid_offset_y_spin = QSpinBox()
        self.grid_offset_y_spin.setRange(-2000, 2000)
        self.grid_offset_y_spin.setSingleStep(10)
        self.grid_offset_y_spin.setValue(int(getattr(self, "grid_offset_y", 0)))
        self.grid_offset_y_spin.valueChanged.connect(self.on_preset_params_changed)

        # self.apply_preset_btn = QPushButton("Apply to Grid")
        # self.apply_preset_btn.clicked.connect(self.apply_point_preset)

        self.df_sigma_spin = QSpinBox()
        self.df_sigma_spin.setRange(0, 20)
        self.df_sigma_spin.setValue(5)
        self.df_sigma_spin.valueChanged.connect(lambda *_: self.update_plot())
        self.df_sigma_spin.valueChanged.connect(
            lambda: self.recalculate_btn.setEnabled(True)
        )

        self.lbl_N = QLabel("Max Points:")
        self.lbl_minDist = QLabel("Min Distance:")
        self.lbl_match_min_score = QLabel("Match Min Score:")
        self.lbl_rows = QLabel("Rows:")
        self.lbl_cols = QLabel("Cols:")
        self.lbl_spacing = QLabel("Spacing (px):")

        self.show_tracks_chk = QCheckBox("Show Tracks")
        self.show_tracks_chk.setChecked(True)
        self.show_tracks_chk.stateChanged.connect(lambda *_: self.update_plot())

        self.sync_view_chk = QCheckBox("Sync view")
        self.sync_view_chk.stateChanged.connect(self.on_sync_view_toggled)

        self.clear_points_btn = QPushButton("Clear Custom Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)

        self.show_contrast_chk = QCheckBox("Contrast controls")
        self.show_contrast_chk.setChecked(True)
        self.show_contrast_chk.stateChanged.connect(self.on_show_contrast_toggled)

        self.undo_btn = QPushButton("Reset to Preset")
        self.undo_btn.clicked.connect(self.undo_all_changes)
        self.undo_btn.setEnabled(False)

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_all_zoom)

        # Display-only brightness flip (swaps min/max levels to pyqtgraph) -- doesn't touch the underlying data.
        self.invert_display_chk = QCheckBox("Invert display (dark ⇄ light)")
        self.invert_display_chk.setChecked(False)
        self.invert_display_chk.stateChanged.connect(self.on_invert_display_toggled)

        # --- Initialize Dialog ---
        self.param_dialog = ParameterDialog(self)
        pd_layout = self.param_dialog.content_layout

        # Add widgets to dialog layout

        # Parameters section
        pd_layout.addWidget(self.lbl_rows)
        pd_layout.addWidget(self.grid_rows_ctrl)
        pd_layout.addWidget(self.lbl_cols)
        pd_layout.addWidget(self.grid_cols_ctrl)
        pd_layout.addWidget(self.lbl_spacing)
        pd_layout.addWidget(self.grid_spacing_spin)
        pd_layout.addWidget(self.lbl_offset_x)
        pd_layout.addWidget(self.grid_offset_x_spin)
        pd_layout.addWidget(self.lbl_offset_y)
        pd_layout.addWidget(self.grid_offset_y_spin)

        pd_layout.addWidget(self.lbl_N)
        pd_layout.addWidget(self.num_points_spin)
        pd_layout.addWidget(self.lbl_minDist)
        pd_layout.addWidget(self.min_dist_spin)
        pd_layout.addWidget(self.lbl_match_min_score)
        pd_layout.addWidget(self.match_min_score_spin)
        pd_layout.addWidget(self.restart_matchanything_btn)

        for w in self._gm_widgets:
            pd_layout.addWidget(w)

        _menu_widget(view_menu, self.show_tracks_chk)
        _menu_widget(view_menu, self.sync_view_chk)
        _menu_widget(view_menu, self.reset_zoom_btn)

        # _menu_widget(view_menu, self.clear_points_btn)
        # pd_layout.addWidget(self.undo_btn)
        _menu_widget(view_menu, self.show_contrast_chk)
        _menu_widget(view_menu, self.invert_display_chk)
        # DF sigma is tuned frequently -- stays in the toolbar rather than behind the View menu.
        toolbar.addWidget(QLabel("DF σ:"))
        toolbar.addWidget(self.df_sigma_spin)
        toolbar.addStretch(1)
        # Move close button to bottom
        pd_layout.removeWidget(self.param_dialog.close_btn)
        pd_layout.addWidget(self.param_dialog.close_btn)

        self.points_label = QLabel("Points: 0")
        self.status_label = QLabel("Ready - Drag & drop video or use Load button")
        self.status_label.setWordWrap(True)

        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.points_label)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        _redirect = _StatusRedirect(self.status_label, self._orig_stdout)
        sys.stdout = _redirect
        sys.stderr = _StatusRedirect(self.status_label, self._orig_stderr)

        # --- Two-panel viewer (left/right) with dropdown selectors ---
        viewer_widget = QWidget(central_widget)
        viewer_layout = QGridLayout(viewer_widget)
        viewer_layout.setContentsMargins(6, 6, 6, 6)
        viewer_layout.setHorizontalSpacing(8)
        viewer_layout.setVerticalSpacing(6)
        viewer_layout.setColumnStretch(0, 1)
        viewer_layout.setColumnStretch(1, 1)
        viewer_layout.setRowStretch(1, 1)

        # View selectors (same entries, in the order of the processing
        # chain: input -> tracked points -> deformation field -> results).
        # Baseline methods (ANTs/LDDMM/NoRMCorre/PyFlowReg) are set off from
        # CoReg with a separator and a disabled, greyed-out "Baselines"
        # header item, so they don't read as more CoReg output options.
        def _populate_view_combo(combo: QComboBox) -> None:
            combo.addItems(["Input", "Tracked Points", "Deformation Field", "CoReg"])
            combo.insertSeparator(combo.count())
            combo.addItem("Baselines")
            header = combo.model().item(combo.count() - 1)
            header.setEnabled(False)
            header.setForeground(QColor("gray"))
            header_font = combo.font()
            header_font.setPointSize(max(7, header_font.pointSize() - 1))
            header.setFont(header_font)
            combo.addItems(["ANTs", "LDDMM", "NoRMCorre", "PyFlowReg"])

        self.view_left_combo = QComboBox()
        _populate_view_combo(self.view_left_combo)
        self.view_left_combo.setCurrentText("Tracked Points")
        self.view_left_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        self.view_right_combo = QComboBox()
        _populate_view_combo(self.view_right_combo)
        self.view_right_combo.setCurrentText("CoReg")
        self.view_right_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        # Image panels
        self.view_left = PGImagePanel(show_histogram=True)
        self.view_right = PGImagePanel(show_histogram=True)
        self.view_left.clicked_outside_image.connect(self._exit_edit_modes_if_enabled)
        self.view_right.clicked_outside_image.connect(self._exit_edit_modes_if_enabled)
        self.view_left.set_parent_gui(self)
        self.view_right.set_parent_gui(self)

        viewer_layout.addWidget(self.view_left_combo, 1, 0)
        viewer_layout.addWidget(self.view_right_combo, 1, 1)
        viewer_layout.addWidget(self.view_left, 2, 0)
        viewer_layout.addWidget(self.view_right, 2, 1)

        main_layout.addWidget(viewer_widget, stretch=1)
        playback_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 24px;
                color: #0078D7;
                font-weight: bold;
                border: 2px solid #0078D7;
                border-radius: 5px;
                padding: 5px;
                min-width: 40px;
            }
            QPushButton:disabled {
                color: #A0A0A0;
                border-color: #A0A0A0;
            }
        """
        )

        self.step_back_btn = QPushButton("◀")
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setEnabled(False)

        self.step_fwd_btn = QPushButton("▶")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        self.step_fwd_btn.setEnabled(False)

        # Preset multipliers instead of a continuous slider -- easier to hit an exact common speed.
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x", "3x", "5x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_change)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setEnabled(False)

        self.frame_label = QLabel("0/0")

        # Row height is set by frames_layout, much taller than these buttons -- bottom-align so the "Speed" block
        # and arrow buttons share the same baseline instead of each centering independently.
        playback_layout.addWidget(self.play_btn)
        playback_layout.setAlignment(self.play_btn, Qt.AlignmentFlag.AlignBottom)
        playback_layout.addWidget(self.step_back_btn)
        playback_layout.setAlignment(self.step_back_btn, Qt.AlignmentFlag.AlignBottom)
        playback_layout.addWidget(self.step_fwd_btn)
        playback_layout.setAlignment(self.step_fwd_btn, Qt.AlignmentFlag.AlignBottom)

        # Tight spacing so "Speed" sits directly above the dropdown.
        speed_layout = QVBoxLayout()
        speed_layout.setSpacing(2)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.addWidget(QLabel("Speed"))
        speed_layout.addWidget(self.speed_combo)
        playback_layout.addLayout(speed_layout)
        playback_layout.setAlignment(speed_layout, Qt.AlignmentFlag.AlignBottom)

        # Frames control with label on top
        frames_layout = QVBoxLayout()
        frames_header_layout = QHBoxLayout()  # To put Frames and 0/0 next to each other
        frames_lbl = QLabel("Frames")
        # Fixed policy so labels hug their text instead of claiming equal leftover width (else "Frames"/"0/50" gap).
        frames_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        frames_header_layout.addWidget(frames_lbl)
        self.frame_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        frames_header_layout.addWidget(self.frame_label)
        frames_header_layout.addStretch(1)
        self.slice_btn = QPushButton("✂ Slice")
        self.slice_btn.setToolTip(
            "Select a sub-range of frames to use "
            "(or Shift+drag on the motion plot below)"
        )
        self.slice_btn.setEnabled(False)
        self.slice_btn.clicked.connect(self.open_slice_dialog)
        frames_header_layout.addWidget(self.slice_btn)
        frames_layout.addLayout(frames_header_layout)

        self.motion_plot = MotionProfileWidget()
        self.motion_plot.frame_changed.connect(self.slider.setValue)
        self.motion_plot.range_selected.connect(self._on_motion_plot_range_selected)
        frames_layout.addWidget(self.motion_plot)

        frames_layout.addWidget(self.slider)
        playback_layout.addLayout(frames_layout, stretch=1)


        main_layout.addLayout(playback_layout)

        self._update_preset_controls()
        self.status_label.setText("UI loaded. Click 'Load File' or drag & drop a file.")

    def _exit_edit_modes_if_enabled(self):
        """Disable edit modes if a click occurs outside the image area."""
        if not (self.edit_points_btn.isChecked() or self.edit_tracks_btn.isChecked()):
            return

        # Turn both options off without triggering signals
        self.edit_points_btn.blockSignals(True)
        self.edit_tracks_btn.blockSignals(True)
        self.edit_points_btn.setChecked(False)
        self.edit_tracks_btn.setChecked(False)
        self.edit_points_btn.blockSignals(False)
        self.edit_tracks_btn.blockSignals(False)

        # Update internal flags and disconnect handlers
        self.edit_points_mode = False
        self.edit_tracks_mode = False
        self.toggle_edit_mode()

        self.status_label.setText("Edit mode off (clicked outside image).")

    def on_process_clicked(self):
        """Run registration with method inferred from panel dropdown selections."""
        method = "CoTracker"

        choices = []
        if hasattr(self, "view_left_combo") and self.view_left_combo is not None:
            choices.append(self.view_left_combo.currentText())
        if hasattr(self, "view_right_combo") and self.view_right_combo is not None:
            choices.append(self.view_right_combo.currentText())

        method_map = {
            "LDDMM": "LDDMM",
            "ANTs": "ANTs",
            "NoRMCorre": "NoRMCorre",
            "PyFlowReg": "PyFlowReg",
        }
        for c in choices:
            mapped = method_map.get(c)
            if mapped is not None:
                method = mapped
                break

        ants_params = None
        if method == "ANTs":
            dlg = AntsRegistrationDialog(parent=self, current=self.ants_params)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.status_label.setText("ANTs registration cancelled.")
                return
            self.ants_params = dlg.result_params()
            ants_params = self.ants_params

        normcorre_params = None
        if method == "NoRMCorre":
            dlg = NormCorreDialog(parent=self, current=self.normcorre_params)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.status_label.setText("NoRMCorre registration cancelled.")
                return
            self.normcorre_params = dlg.result_params()
            normcorre_params = self.normcorre_params

        pyflowreg_params = None
        if method == "PyFlowReg":
            dlg = PyFlowRegDialog(parent=self, current=self.pyflowreg_params)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.status_label.setText("PyFlowReg registration cancelled.")
                return
            self.pyflowreg_params = dlg.result_params()
            pyflowreg_params = self.pyflowreg_params

        self.visualize(
            method=method,
            ants_params=ants_params,
            normcorre_params=normcorre_params,
            pyflowreg_params=pyflowreg_params,
        )

    def on_sync_view_toggled(self, state):
        master_vb = self.view_left.vb
        vb_r = self.view_right.vb

        if state == Qt.CheckState.Checked.value:
            vb_r.setXLink(master_vb)
            vb_r.setYLink(master_vb)
            self.status_label.setText("View sync: ON")
        else:
            vb_r.setXLink(None)
            vb_r.setYLink(None)
            self.status_label.setText("View sync: OFF")

    def on_show_contrast_toggled(self, state):
        visible = state == Qt.CheckState.Checked.value
        self.view_left.set_histogram_visible(visible)
        self.view_right.set_histogram_visible(visible)

    def on_invert_display_toggled(self, *_):
        """Flip the display-level mapping (see _calc_levels) -- a pure
        rendering toggle, never touches video_data/_original_video_data."""
        self.invert_display = self.invert_display_chk.isChecked()
        try:
            self.update_plot()
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass

    def _set_visible(self, widgets, visible: bool):
        for w in widgets:
            w.setVisible(visible)

    def _make_disp_preview(self, disp_frame):
        """Generate the display preview with adjustable Gaussian smoothing for the DF."""
        d = np.asarray(disp_frame)

        # Read sigma value from the GUI spinner to control smoothing
        sigma = float(self.df_sigma_spin.value())
        if sigma > 0:
            if d.ndim == 2:
                d = cv2.GaussianBlur(d.astype(np.float32), (0, 0), sigma)
            elif d.ndim == 3 and d.shape[-1] == 2:
                u = cv2.GaussianBlur(d[..., 0].astype(np.float32), (0, 0), sigma)
                v = cv2.GaussianBlur(d[..., 1].astype(np.float32), (0, 0), sigma)
                d = np.stack([u, v], axis=-1)

        if d.ndim == 2:
            return self.disp_scalar_u8(d)
        if d.ndim == 3 and d.shape[-1] == 2:
            return self.flow_to_color(d)
        return None

    def _update_preset_controls(self):
        preset = self.preset_combo.currentText()
        is_none = preset.startswith("None")
        is_grid = preset.startswith("Regular grid")
        is_features = preset.startswith("Features")
        is_motion = preset.startswith("Motion")
        is_greedy = preset.startswith("Greedy motion")
        is_seg = preset.startswith("Segmentation")
        is_matchanything = preset.startswith("MatchAnything")

        show_rows_cols = is_grid and not is_none
        show_n = (
            is_features or is_motion or is_greedy or is_seg or is_matchanything
        ) and not is_none
        show_mindist = (is_features or is_matchanything) and not is_none
        show_match_score = is_matchanything and not is_none
        show_spacing = is_grid and not is_none

        self._set_visible(
            [self.lbl_rows, self.grid_rows_ctrl, self.lbl_cols, self.grid_cols_ctrl],
            show_rows_cols,
        )
        self._set_visible([self.lbl_N, self.num_points_spin], show_n)
        self._set_visible([self.lbl_minDist, self.min_dist_spin], show_mindist)
        self._set_visible(
            [self.lbl_match_min_score, self.match_min_score_spin], show_match_score
        )
        self._set_visible([self.restart_matchanything_btn], show_match_score)
        self.restart_matchanything_btn.setEnabled(
            show_match_score and self.video_data is not None
        )
        self._set_visible(self._gm_widgets, is_greedy)
        # Spacing control should only be enabled for Regular grid
        if hasattr(self, "lbl_spacing") and hasattr(self, "grid_spacing_spin"):
            self._set_visible([self.lbl_spacing, self.grid_spacing_spin], show_spacing)
            self.grid_spacing_spin.setEnabled(show_spacing)

        # Center offset controls visible only for Regular grid
        if (
            hasattr(self, "lbl_offset_x")
            and hasattr(self, "grid_offset_x_spin")
            and hasattr(self, "lbl_offset_y")
            and hasattr(self, "grid_offset_y_spin")
        ):
            self._set_visible(
                [
                    self.lbl_offset_x,
                    self.grid_offset_x_spin,
                    self.lbl_offset_y,
                    self.grid_offset_y_spin,
                ],
                show_spacing,
            )
            self.grid_offset_x_spin.setEnabled(show_spacing)
            self.grid_offset_y_spin.setEnabled(show_spacing)

        # self.apply_preset_btn.setVisible(True)
        # self.apply_preset_btn.setEnabled(True)

    def on_preset_changed(self, *_):
        """Handle changes to the point-generation preset dropdown."""
        self._update_preset_controls()
        preset = self.preset_combo.currentText()

        # Apply immediately if no video is loaded or "None" is selected
        if self.video_data is None or preset.startswith("None"):
            self.apply_point_preset(mode="replace")
            self._last_preset_index = self.preset_combo.currentIndex()
            return

        # Prompt user to replace or augment existing points
        box = QMessageBox(self)
        box.setWindowTitle("Change Preset")
        box.setText(
            "Should the new preset replace existing points or be added to them?\n\n"
            "• Replace: Clears all current points, tracks, and results.\n"
            "• Add: Appends new points (tracks/results will be reset)."
        )

        btn_replace = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        btn_add = box.addButton("Add", QMessageBox.ButtonRole.ActionRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_replace)

        box.exec()
        clicked = box.clickedButton()

        # Cancel -> revert preset selection
        if clicked == btn_cancel:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(self._last_preset_index)
            self.preset_combo.blockSignals(False)
            return

        if clicked == btn_replace:
            mode = "replace"
        elif clicked == btn_add:
            mode = "add"
        else:
            mode = "replace"

        # Always jump to and display frame 0.
        self._pause_if_playing()
        self.current_frame = 0
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_frame_label()

        self.apply_point_preset(mode=mode, keep_custom=False)
        self._last_preset_index = self.preset_combo.currentIndex()

    def _invalidate_filtered_video_cache(self):
        """Clear cached filtered versions of the video data."""
        for k in list(self.viz_data.keys()):
            if k.startswith("filtered_video::"):
                self.viz_data.pop(k, None)

    def on_filter_changed(self, text):
        """Handle intensity filter selection changes."""
        self.current_filter = text
        self._invalidate_filtered_video_cache()
        self.update_plot()

        if self.video_data is None:
            return

        n_base = (
            int(self.base_grid_active.sum()) if self.base_grid_active is not None else 0
        )
        n_custom = len(self.custom_grid_pts) if self.custom_grid_pts else 0
        if (n_base + n_custom) == 0:
            return

        # Prompt user to re-track points with the new filter
        box = QMessageBox(self)
        box.setWindowTitle("Filter Changed")
        box.setText(
            "Input filter has changed.\n\n"
            "Would you like to restart registration and re-track ALL points "
            "using the filtered video?"
        )
        btn_yes = box.addButton("Restart", QMessageBox.ButtonRole.AcceptRole)
        btn_no = box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_yes)

        box.exec()
        clicked = box.clickedButton()

        if clicked == btn_yes:
            self.status_label.setText(
                f"Filter set to {self.current_filter}. Restarting tracking..."
            )
            self._track_all_points_now(method="CoTracker")
        elif clicked == btn_no:
            self.status_label.setText(
                f"Filter set to {self.current_filter}. (Tracking not restarted)"
            )

    def on_preset_params_changed(self):
        if self.video_data is None:
            return

        preset = self.preset_combo.currentText()
        if preset.startswith("MatchAnything"):
            self.restart_matchanything_btn.setEnabled(True)
            self.status_label.setText(
                "MatchAnything parameters changed. Click 'Restart MatchAnything' to recalculate."
            )
            return

        self.apply_point_preset(mode="update_current", keep_custom=True)

    def restart_matchanything(self):
        """Explicitly re-run MatchAnything preset after parameter edits."""
        if self.video_data is None:
            self.status_label.setText("Load data first to run MatchAnything.")
            return

        preset = self.preset_combo.currentText()
        if not preset.startswith("MatchAnything"):
            self.status_label.setText(
                "Select the 'MatchAnything' preset to use Restart MatchAnything."
            )
            return

        self.apply_point_preset(mode="update_current", keep_custom=True)
        self.restart_matchanything_btn.setEnabled(False)

    def _track_all_points_now(self, method="CoTracker"):
        """
        Force a full re-tracking of ALL existing points with the current filter.
        Keeps the point locations, but clears tracks/warping results and marks all points as pending.
        """
        if self.video_data is None:
            return

        # Don’t overlap with an ongoing worker
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.status_label.setText(
                        "Tracking is already running. Stop it first to restart."
                    )
                    return
            except (RuntimeError, AttributeError):
                pass

        # Keep points, but clear tracking/warping outputs
        self._reset_results_and_tracks()

        # Ensure flags exist and mark everything as NOT tracked (so visualize() will track them)
        if self.base_grid_pts is None:
            self.base_grid_pts = np.zeros((0, 2), np.float32)
        if self.base_grid_active is None:
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)

        if self.custom_grid_pts is None:
            self.custom_grid_pts = []
        self.custom_grid_tracked = [False] * len(self.custom_grid_pts)

        # Also reset track_meta so mapping matches fresh tracks
        self.track_meta = []

        # Run full registration/tracking (will use filtered input via _get_video_input_for_tracking)
        self.visualize(method=method)

    # ---------- presets of points ----------
    def _get_gray_video_np(self):
        vd = self.video_data
        if isinstance(vd, LazyVideo):
            # Explicit materialize() rather than relying on np.asarray's implicit sequence-protocol conversion.
            vd = vd.materialize()
        if isinstance(vd, torch.Tensor):
            vd = vd.detach().cpu().numpy()
        vd = np.asarray(vd)

        if vd.ndim == 4 and vd.shape[1] in (1, 3):
            return vd[:, 0].astype(np.float32)
        if vd.ndim == 3:
            return vd.astype(np.float32)
        raise ValueError(f"Unsupported video_data shape for grayscale: {vd.shape}")

    def _reference_image(self, gray_video, n=10):
        n = int(min(max(n, 1), gray_video.shape[0]))
        ref = gray_video[:n].mean(axis=0)
        ref = cv2.GaussianBlur(ref, (0, 0), 1.0)
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ref_u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ref_u8)
        return ref_u8

    def _preset_features(self, ref_u8_filt, num_points=200, min_dist=10):
        return _presets.preset_features(ref_u8_filt, num_points=num_points, min_dist=min_dist)

    def _preset_motion(self, gray_video, num_points=200):
        return _presets.preset_motion(gray_video, num_points=num_points)

    def _preset_greedy_motion(self, gray_video, num_points=200):
        return _presets.preset_greedy_motion(
            gray_video,
            num_points=num_points,
            sigma=float(self.gm_sigma_spin.value()),
            gamma=float(self.gm_gamma_spin.value()),
            background_weight=float(self.gm_bg_weight_spin.value()),
            peak_min_distance=int(self.gm_peak_min_dist_spin.value()),
            peak_threshold_rel=float(self.gm_peak_thr_spin.value()),
            max_peak_fraction=float(self.gm_peak_frac_spin.value()),
            candidate_factor=int(self.gm_cand_factor_spin.value()),
            sample_min_distance=float(self.gm_samp_dist_spin.value()),
            random_seed=int(self.gm_seed_spin.value()),
        )

    def _preset_segmentation_centroids(self, ref_u8_filt, num_points=200):
        return _presets.preset_segmentation(ref_u8_filt, num_points=num_points)

    def _preset_matchanything(self, frame_u8, ref_u8, num_points=200, min_score=0.2, min_dist=10):
        return _presets.preset_matchanything(
            frame_u8,
            ref_u8,
            num_points=num_points,
            min_score=min_score,
            min_dist=min_dist,
            status_cb=self.status_label.setText,
        )

    def _preset_regular_grid(self, H, W, rows, cols, offset_x=0, offset_y=0):
        return _presets.preset_regular_grid(H, W, rows, cols, offset_x=offset_x, offset_y=offset_y)

    def _preset_regular_grid_spacing(self, H: int, W: int, spacing_px: int):
        return _presets.preset_regular_grid_spacing(H, W, spacing_px)

    def apply_point_preset(self, mode="replace", keep_custom=False):
        """
        Applies a predefined strategy to select tracking points (Grid, Features, Motion, etc.).
        'replace' clears existing base points; 'add' keeps them; 'update_current'
        regenerates only the currently-focused preset's own contribution (see
        on_preset_params_changed), leaving points from an earlier 'add' of a
        different preset untouched.
        'keep_custom' determines if manually added points in self.custom_grid_pts are preserved.
        """
        if self.video_data is None:
            return

        preset = self.preset_combo.currentText()
        if preset.startswith("None"):
            mode = "replace"
            keep_custom = False

        # "replace" clears everything; "add" keeps existing registrations/tracks; "update_current" still resets
        # since its own preset's points are about to change.
        if mode == "replace":
            self._reset_results_and_tracks()
            self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            self.base_grid_active = np.zeros((0,), dtype=bool)
            self.base_grid_pts0 = self.base_grid_pts.copy()
            if not keep_custom:
                self.custom_grid_pts = []
                self.custom_grid_tracked = []
        elif mode == "add":
            if self.base_grid_pts is None:
                self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            if self.base_grid_active is None:
                self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)
        elif mode == "update_current":
            self._reset_results_and_tracks()
            if self.base_grid_pts is None:
                self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            if self.base_grid_active is None:
                self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        # 2) Handle 'None' preset
        if preset.startswith("None"):
            self.status_label.setText("Preset applied: None (no points)")
            self.process_btn.setEnabled(True)
            self._update_points_label()
            self.update_grid_display_pre_registration()
            self.update_plot()
            return

        # 3) Calculate new points based on image content
        gray = self._get_gray_video_np()
        H, W = gray.shape[1], gray.shape[2]
        ref_u8 = self._reference_image(gray)
        ref_u8_filt = self.apply_filter(ref_u8, self.current_filter)

        if preset.startswith("Regular grid"):
            offset_x = (
                int(self.grid_offset_x_spin.value())
                if hasattr(self, "grid_offset_x_spin")
                else int(getattr(self, "grid_offset_x", 0))
            )
            offset_y = (
                int(self.grid_offset_y_spin.value())
                if hasattr(self, "grid_offset_y_spin")
                else int(getattr(self, "grid_offset_y", 0))
            )
            rows = max(
                2,
                (
                    self.grid_rows_ctrl.value()
                    if hasattr(self, "grid_rows_ctrl")
                    else self.grid_rows
                ),
            )
            cols = max(
                2,
                (
                    self.grid_cols_ctrl.value()
                    if hasattr(self, "grid_cols_ctrl")
                    else self.grid_cols
                ),
            )
            self.grid_rows, self.grid_cols = rows, cols
            # spacing_px=0 means disabled (use rows/cols); nonzero derives rows/cols from fixed pixel spacing.
            # Restored here (not in the shared preset_regular_grid) after commit 12e8616 dropped it -- CLI's
            # rows x cols behavior must stay unchanged.
            spacing = (
                int(self.grid_spacing_spin.value())
                if hasattr(self, "grid_spacing_spin")
                else int(getattr(self, "grid_spacing", 0))
            )
            self.grid_spacing = spacing
            if spacing > 0:
                pts_new = self._preset_regular_grid_spacing(H, W, spacing)
            else:
                pts_new = self._preset_regular_grid(
                    H,
                    W,
                    rows,
                    cols,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            min_dist = max(
                5,
                (
                    int(self.min_dist_spin.value())
                    if self.min_dist_spin is not None
                    else 10
                ),
            )
        else:
            num_points = int(self.num_points_spin.value())
            min_dist = int(self.min_dist_spin.value())
            if preset.startswith("Features"):
                pts_new = self._preset_features(
                    ref_u8_filt, num_points=num_points, min_dist=min_dist
                )
            elif preset.startswith("Motion"):
                pts_new = self._preset_motion(
                    gray, num_points=num_points
                )
            elif preset.startswith("Greedy motion"):
                pts_new = self._preset_greedy_motion(
                    gray, num_points=num_points
                )
            elif preset.startswith("MatchAnything"):
                min_score = (
                    float(self.match_min_score_spin.value())
                    if hasattr(self, "match_min_score_spin")
                    else 0.2
                )
                pts_new = self._preset_matchanything(
                    ref_u8_filt,
                    ref_u8,
                    num_points=num_points,
                    min_score=min_score,
                    min_dist=min_dist,
                )
            else:
                pts_new = self._preset_segmentation_centroids(
                    ref_u8_filt, num_points=num_points
                )

        pts_new = np.asarray(pts_new, dtype=np.float32).reshape(-1, 2)

        # Area restriction applied uniformly after generation, covering every preset without touching utils/presets.py.
        if self.roi_select_chk.isChecked() and len(pts_new):
            roi = self.view_left.get_roi_bounds()
            if roi is not None:
                x0, y0, x1, y1 = roi
                inside = (
                    (pts_new[:, 0] >= x0) & (pts_new[:, 0] < x1)
                    & (pts_new[:, 1] >= y0) & (pts_new[:, 1] < y1)
                )
                pts_new = pts_new[inside]

        # 4) Integrate new points into point storage
        if mode == "replace":
            self.base_grid_pts = pts_new
            self.base_grid_active = np.ones((len(pts_new),), dtype=bool)
            self.base_grid_pts0 = pts_new.copy()
            self._last_preset_range = (0, len(pts_new))
        elif mode == "add":
            old = np.asarray(self.base_grid_pts, dtype=np.float32).reshape(-1, 2)
            pts_new_f = self._filter_new_points(pts_new, min_dist=min_dist)
            self.base_grid_pts = np.vstack([old, pts_new_f]).astype(np.float32)
            self.base_grid_active = np.concatenate(
                [
                    self.base_grid_active.astype(bool),
                    np.ones((len(pts_new_f),), dtype=bool),
                ]
            )
            self._last_preset_range = (len(old), len(old) + len(pts_new_f))
        elif mode == "update_current":
            all_pts = np.asarray(self.base_grid_pts, dtype=np.float32).reshape(-1, 2)
            prev_range = self._last_preset_range
            if (
                prev_range is not None
                and prev_range[0] > 0
                and prev_range[1] == len(all_pts)
            ):
                # Keep points an earlier "Add" contributed before this preset's range; regenerate only this slice.
                preceding = all_pts[: prev_range[0]]
                preceding_active = np.asarray(
                    self.base_grid_active[: prev_range[0]], dtype=bool
                )
                # Filter against only preceding (other-preset) points, not this preset's own stale ones.
                self.base_grid_pts = preceding
                self.base_grid_active = preceding_active
                pts_new_f = self._filter_new_points(pts_new, min_dist=min_dist)
                self.base_grid_pts = np.vstack([preceding, pts_new_f]).astype(
                    np.float32
                )
                self.base_grid_active = np.concatenate(
                    [preceding_active, np.ones((len(pts_new_f),), dtype=bool)]
                )
                self._last_preset_range = (
                    len(preceding),
                    len(preceding) + len(pts_new_f),
                )
            else:
                # No preceding "add" -- plain full replace, same as mode="replace".
                self.base_grid_pts = pts_new
                self.base_grid_active = np.ones((len(pts_new),), dtype=bool)
                self.base_grid_pts0 = pts_new.copy()
                self._last_preset_range = (0, len(pts_new))

        # 5) Keep 'tracked' status flags aligned with the point list
        self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
        if mode == "replace":
            self.custom_grid_tracked = []

        # Enable registration if we have at least one valid point
        has_points = len(self.base_grid_pts) > 0 or len(self.custom_grid_pts) > 0
        self.process_btn.setEnabled(has_points)

        # 6) Refresh UI
        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.update_plot()
        self.status_label.setText(f"Preset applied: {preset} ({mode})")

    # ---------- file loading ----------
    def select_file(self):
        """Open a file dialog to select a video/image stack for processing."""
        file_dialog = QFileDialog(self, "Select Video File")
        ext_filter = " ".join([f"*{e}" for e in SUPPORTED_EXTENSIONS])
        file_dialog.setNameFilters([f"Video files ({ext_filter})", "All files (*.*)"])
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if not file_dialog.exec():
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        self.video_path = selected_files[0]
        self.status_label.setText(f"Selected: {os.path.basename(self.video_path)}")
        QApplication.processEvents()

        # Ask for load settings (resize / max frames)
        settings = self._show_load_settings_dialog()
        if settings is None:
            self.status_label.setText("Load canceled.")
            return

        # Load the data
        self.status_label.setText(f"Loading: {os.path.basename(self.video_path)}")
        QApplication.processEvents()
        self.load_data(load_settings=settings)

    def clear_all_data(self, keep_video=False):
        """
        Reset the application state and clear loaded data/results.
        :param keep_video: If True, preserves the loaded video_data and its UI state.
        """
        self._initial_zoom_done = False
        self.viz_data = {}

        if not keep_video:
            self.video_data = None
            self.total_frames = 0
            self.current_frame = 0
            # Clear cache when video changes
            self.registration_cache.clear()

        # Reset processing results
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None
        self.drift_xy = None
        self.ants_video = None
        self.lddmm_video = None
        self.normcorre_video = None
        self.flowreg_video = None

        # Reset panel displays
        if hasattr(self, "view_left"):
            self.view_left.set_image(None)
        if hasattr(self, "view_right"):
            self.view_right.set_image(None)

        # Reset motion profile widget
        if hasattr(self, "motion_plot"):
            self.motion_plot.set_current_frame(0)
            self.motion_plot.set_deformation_data([])
            if not keep_video:
                self.motion_plot.set_data([])

        # Reset UI control states
        if not keep_video:
            self.slider.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.step_fwd_btn.setEnabled(False)
            self.step_back_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            pass  # save always available
            self.edit_points_btn.setEnabled(False)
            self.clear_points_btn.setEnabled(False)
            self.frame_label.setText("0/0")
            self.edit_tracks_btn.setEnabled(False)
            self.edit_tracks_btn.setChecked(False)

    def _reset_results_and_tracks(self, keep_input_cache: bool = True):
        """
        Clear anything that depends on tracking/warping results, but keep:
        - loaded video_data
        - base/custom point presets (those are handled by apply_point_preset)

        *keep_input_cache* preserves the cached display frames (viz_data["input"])
        for speed when video_data itself hasn't changed. Pass False whenever
        video_data has just been replaced (e.g. after applying a frame slice) —
        otherwise update_plot() keeps rendering from the stale pre-change cache,
        indexed 0-based, which looks like the slice always starts at frame 0.
        """
        # Tracking outputs
        self.pred_tracks = None
        self.pred_visibility = None
        self.pred_confidence = None
        self.grid_pts = None

        # Warping / registration outputs
        self.warped_video = None
        self.disp_field = None
        self.drift_xy = None
        self.ants_video = None
        self.lddmm_video = None
        self.normcorre_video = None
        self.flowreg_video = None
        self.ssim_score = None

        # Per-track UI state
        self.track_active = None
        self.selected_track_idx = None

        # Viz cache for derived views
        if hasattr(self, "viz_data") and isinstance(self.viz_data, dict):
            # keep input cache if you want (faster), remove everything else
            inp = self.viz_data.get("input", None) if keep_input_cache else None
            self.viz_data.clear()
            if inp is not None:
                self.viz_data["input"] = inp

        # UI toggles/buttons that depend on results
        if hasattr(self, "edit_tracks_btn"):
            self.edit_tracks_btn.setEnabled(False)
            self.edit_tracks_btn.setChecked(False)
        if hasattr(self, "recalculate_btn"):
            self.recalculate_btn.setEnabled(False)
        if hasattr(self, "save_btn"):
            pass  # save always available

        # stop playback if running
        if hasattr(self, "timer"):
            self.timer.stop()
        self.is_playing = False
        self._play_levels_orig = None
        self._play_levels_filt = None
        if hasattr(self, "play_btn"):
            self.play_btn.setText("▶")

    # ---------- time slice ----------
    def open_slice_dialog(self):
        if self._original_video_data is None:
            return
        total = int(self._original_video_data.shape[0])
        current_end = self._frame_end if self._frame_end is not None else total - 1
        try:
            motion_scores = self._compute_motion_scores(self._original_video_data)
        except Exception as e:
            print(f"Error computing motion scores for slice dialog: {e}")
            motion_scores = []
        dlg = TimeSliceDialog(
            self,
            total_frames=total,
            current_start=self._frame_start,
            current_end=current_end,
            motion_scores=motion_scores,
        )
        # "Export Slices" writes files immediately in the background, independent of Apply/Reset/Cancel below.
        dlg.export_requested.connect(lambda: self._export_cuts(dlg))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        start, end = dlg.result_slice()
        if start is None:
            self._frame_start = 0
            self._frame_end = None
        else:
            if start > end:
                start, end = end, start
            self._frame_start = max(0, start)
            self._frame_end = min(total - 1, end)
        self._apply_frame_slice()

    def _export_cuts(self, dlg: "TimeSliceDialog"):
        """Save every cut currently in *dlg* as its own .h5 video under a
        chosen directory, plus one CSV logging every cut's frame range.

        The actual per-cut work (frame decode + gzip-9 compressed write) runs
        in a background CutsExportWorker thread -- doing this synchronously
        on the GUI thread used to freeze the whole window ("not responding")
        for however long a sizeable cut took to write."""
        if getattr(self, "_cuts_export_worker", None) is not None:
            dlg.set_export_status("An export is already running — wait for it to finish.")
            return

        cuts = dlg.result_cuts()
        if not cuts:
            dlg.set_export_status("No cuts to export yet — use + Add Cut first.")
            return

        default_dir = os.path.dirname(self.video_path) if getattr(self, "video_path", None) else ""
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Folder for Cuts", default_dir)
        if not out_dir:
            dlg.set_export_status("Export canceled.")
            return

        base_name = (
            os.path.splitext(os.path.basename(self.video_path))[0]
            if getattr(self, "video_path", None)
            else "video"
        )

        dlg.set_export_enabled(False)
        dlg.set_export_status(f"Exporting 0/{len(cuts)}…")

        worker = CutsExportWorker(self._original_video_data, cuts, out_dir, base_name, parent=self)
        worker.progress.connect(lambda i, n, d=dlg: self._on_cuts_export_progress(d, i, n))
        worker.done.connect(lambda rows, failed, d=dlg, od=out_dir, bn=base_name: self._on_cuts_export_done(d, rows, failed, od, bn))
        self._cuts_export_worker = worker
        worker.start()

    def _on_cuts_export_progress(self, dlg, completed: int, total: int):
        try:
            dlg.set_export_status(f"Exporting {completed}/{total}…")
        except RuntimeError:
            pass  # dialog's C++ object was already destroyed

    def _on_cuts_export_done(self, dlg, rows: list[dict], failed: list[str], out_dir: str, base_name: str):
        self._cuts_export_worker = None

        if rows:
            csv_path = os.path.join(out_dir, f"{base_name}_cuts.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "file", "start_frame", "end_frame", "n_frames"])
                writer.writeheader()
                writer.writerows(rows)

        if failed:
            msg = f"Exported {len(rows)}/{len(rows) + len(failed)} cuts to {out_dir} — failed: {'; '.join(failed)}"
        else:
            msg = f"Exported {len(rows)} cut(s) + CSV log to {out_dir}"
        self.status_label.setText(msg)
        try:
            dlg.set_export_status(msg)
            dlg.set_export_enabled(True)
        except RuntimeError:
            pass  # dialog's C++ object was already destroyed

    def _on_motion_plot_range_selected(self, start: int, end: int):
        """Shift+drag on the motion plot selected [start, end] (inclusive) —
        extract it as a new slice. These indices are relative to the
        currently displayed video_data, which may itself already be a slice
        of _original_video_data, so offset by the active slice's start
        before converting to absolute indices."""
        if self._original_video_data is None:
            return
        offset = int(self._frame_start)
        total = int(self._original_video_data.shape[0])
        new_start = max(0, min(offset + start, total - 1))
        new_end = max(0, min(offset + end, total - 1))
        if new_start > new_end:
            new_start, new_end = new_end, new_start

        self._frame_start = new_start
        self._frame_end = new_end
        self._apply_frame_slice()

    def _apply_frame_slice(self):
        if self._original_video_data is None:
            return
        total = int(self._original_video_data.shape[0])
        start = int(self._frame_start)
        end = int(self._frame_end) if self._frame_end is not None else total - 1
        end = min(end, total - 1)
        start = max(0, min(start, end))

        self.motion_plot.clear_selection()
        self.video_data = self._original_video_data[start : end + 1]
        self.registration_cache.set_input(self.video_data)

        self.total_frames = int(self.video_data.shape[0])
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setValue(0)
        self.current_frame = 0

        self._reset_results_and_tracks(keep_input_cache=False)
        # Slicing only changes time range, not spatial content -- don't regenerate points (would discard edits).
        self._update_points_label()
        self.update_plot()
        self.update_frame_label()

        self._recompute_motion_bar()

        is_full = start == 0 and end == total - 1
        if is_full:
            self.slice_btn.setText("✂ Slice")
            self.status_label.setText(
                f"Full video: {total} frames"
            )
        else:
            self.slice_btn.setText(f"✂ {start}–{end}")
            self.status_label.setText(
                f"Slice: frames {start}–{end}  ({self.total_frames} frames of {total})"
            )

    def _recompute_motion_bar(self):
        """Recompute the motion profile bar from the current video_data slice."""
        try:
            scores = self._compute_motion_scores(self.video_data)
            self.motion_plot.set_data(scores)
        except Exception as e:
            print(f"Error computing motion bar: {e}")
            self.motion_plot.set_data([])
        # refresh view
        try:
            self.update_plot()
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass

    def _compute_motion_scores(self, vd) -> np.ndarray:
        """Per-frame motion score (mean abs diff to the previous frame,
        downsampled and normalized to [0, 1]) for an arbitrary video tensor
        or LazyVideo -- shared by _recompute_motion_bar (current video_data
        slice) and the Slice dialog's embedded motion plot (full,
        unsliced _original_video_data, so its drag-selected indices line up
        with the dialog's own absolute Start/End frame spinboxes)."""
        try:
            target_dim = 64

            if isinstance(vd, LazyVideo):
                # Stream one frame at a time and downsample immediately -- materializing the full video here
                # OOM-killed the app on large H5 files.
                T = len(vd)
                H, W = vd._H, vd._W
                scale = min(target_dim / H, target_dim / W)
                dsize = (
                    (max(1, int(W * scale)), max(1, int(H * scale)))
                    if scale < 1.0
                    else None
                )

                scores = [0.0]
                prev_small = None
                for i in range(T):
                    frame = vd.get_frame_hwc(i)[:, :, 0]  # (H,W) float32 [0,1]
                    small = (
                        cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
                        if dsize is not None
                        else frame
                    )
                    if prev_small is not None:
                        scores.append(float(np.abs(small - prev_small).mean()))
                    prev_small = small

                scores = np.asarray(scores, dtype=np.float32)
                if T > 1:
                    mx = scores.max()
                    if mx > 0:
                        scores /= mx
                    return scores
                return np.asarray([])
            else:
                gray_t = vd
                if isinstance(gray_t, torch.Tensor):
                    gray_t = gray_t.detach().cpu().numpy()
                gray_t = np.asarray(gray_t)
                if gray_t.ndim == 4 and gray_t.shape[1] in (1, 3):
                    gray_full = gray_t[:, 0].astype(np.float32)
                elif gray_t.ndim == 3:
                    gray_full = gray_t.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported video shape for motion score: {gray_t.shape}")
                T, H, W = gray_full.shape
                scale = min(target_dim / H, target_dim / W)

                if scale < 1.0:
                    dsize = (int(W * scale), int(H * scale))
                    gray_small = np.array([
                        cv2.resize(gray_full[i], dsize, interpolation=cv2.INTER_AREA)
                        for i in range(T)
                    ])
                else:
                    gray_small = gray_full

                flat = gray_small.reshape(T, -1).astype(np.float32)
                if T > 1:
                    diffs = np.abs(flat[1:] - flat[:-1]).mean(axis=1).astype(np.float32)
                    scores = np.concatenate([[0.0], diffs])
                    mx = scores.max()
                    if mx > 0:
                        scores /= mx
                    return scores
                return np.asarray([])
        except Exception as e:
            print(f"Error computing motion scores: {e}")
            return np.asarray([])

    def _gather_existing_points(self):
        """Consolidate all active grid and custom points into a single array."""
        pts = []
        if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
            if self.base_grid_active is None:
                pts.append(np.asarray(self.base_grid_pts, np.float32))
            else:
                act = np.asarray(self.base_grid_active, bool)
                if act.any():
                    pts.append(np.asarray(self.base_grid_pts[act], np.float32))

        if self.custom_grid_pts:
            pts.append(np.asarray(self.custom_grid_pts, np.float32))

        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(pts).astype(np.float32)

    def _filter_new_points(self, new_pts, min_dist):
        """Remove new points that are too close to existing points."""
        new_pts = np.asarray(new_pts, dtype=np.float32)
        if new_pts.size == 0:
            return new_pts.reshape(0, 2)

        exist = self._gather_existing_points()
        if exist.shape[0] == 0:
            return new_pts

        thr2 = (float(min_dist) / 2.0) ** 2
        keep = []
        for p in new_pts:
            dist2 = np.sum((exist - p) ** 2, axis=1)
            if dist2.size == 0 or float(np.min(dist2)) > thr2:
                keep.append(p)

        return np.asarray(keep, dtype=np.float32).reshape(-1, 2)

    def _retrack_all_points_with_current_filter(self):
        if self.video_data is None:
            return

        n_base = (
            int(self.base_grid_pts.shape[0]) if self.base_grid_pts is not None else 0
        )
        n_custom = len(self.custom_grid_pts) if self.custom_grid_pts else 0
        if (n_base + n_custom) == 0:
            return

        # laufenden worker stoppen
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.quit()
                    self.worker.wait(500)
            except (RuntimeError, AttributeError):
                pass
            self.worker = None

        # Invalidate "Corrected/DF" -- input has changed.
        self.warped_video = None
        self.disp_field = None
        self.drift_xy = None
        self.ants_video = None
        self.lddmm_video = None
        self.normcorre_video = None
        self.flowreg_video = None
        self.ssim_score = None
        if isinstance(self.viz_data, dict):
            self.viz_data.pop("warped", None)

        # --- Alle Punkte als pending markieren ---
        if self.base_grid_pts is None:
            self.base_grid_pts = np.zeros((0, 2), np.float32)
        if self.base_grid_active is None:
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
        self.custom_grid_tracked = [False] * (
            len(self.custom_grid_pts) if self.custom_grid_pts else 0
        )

        # Mapping reset
        self.track_meta = []
        self._pending_track_meta = []

        # Reset tracks, but not the filter/input cache.
        self.pred_tracks = None
        self.pred_visibility = None
        self.pred_confidence = None
        self.track_active = None
        self.selected_track_idx = None

        # Keine Auto-Warp-Flags setzen
        self._recalc_inflight = False
        self._pending_recalc = False

        self.status_label.setText(
            f"Filter changed → re-tracking all points ({self.current_filter})…"
        )

        # Jetzt neu tracken (nutzt _get_video_input_for_tracking -> gefiltert)
        self.visualize(method="CoTracker")

    def load_data(self, load_settings=None):
        self._auto_fit_done = False
        if not self.video_path:
            return

        if load_settings is None:
            load_settings = {
                "target_w": None,
                "target_h": None,
                "max_frames": None,
                "axes_order": None,
            }

        target_w = load_settings.get("target_w", None)  # None => keep original
        target_h = load_settings.get("target_h", None)  # None => keep original
        max_frames = load_settings.get("max_frames", None)  # None => all frames
        axes_order = load_settings.get("axes_order", None)  # None => file's own metadata order

        self.clear_all_data()
        ext = os.path.splitext(self.video_path)[1].lower()
        self.status_label.setText("Loading data...")
        QApplication.processEvents()

        try:
            # -----------------------------
            # 1) LOAD (as torch tensor)
            # -----------------------------
            if ext in SUPPORTED_VIDEO_EXT:
                # load_video returns torch in your pipeline (T,C,H,W) likely
                video_t = load_video(
                    self.video_path, resize=False, max_frames=max_frames
                )

            elif ext in SUPPORTED_TIFF_EXT:
                video_t = load_tiff(self.video_path, resize=False, axes_order=axes_order)
                if max_frames is not None:
                    # video_t might be numpy or torch depending on load_tiff implementation
                    # but we enforce torch below
                    pass

            elif ext in SUPPORTED_HDF5_EXT:
                frames_arg = int(max_frames) if max_frames is not None else None
                # Without resize=False, load_hdf5's defaults (256x256, cv2.INTER_AREA) silently compress the
                # intensity histogram -- root cause of H5 outputs differing from the TIFF path.
                # resize=False also returns a LazyVideo (frames decoded on demand) instead of materializing the
                # whole recording in RAM -- large H5 files were getting OOM-killed otherwise.
                video_t = load_hdf5(
                    self.video_path, "/data", frames=frames_arg, resize=False
                )

            else:
                self.status_label.setText(f"Unsupported file extension: {ext}")
                return

            # Leave a LazyVideo alone -- converting it here forces full materialization, defeating streaming.
            # Materialized later only when processing needs it (ProcessingWorker.run in gui.py).
            if isinstance(video_t, LazyVideo):
                pass
            elif not isinstance(video_t, torch.Tensor):
                video_t = torch.from_numpy(np.asarray(video_t))

            if isinstance(video_t, LazyVideo):
                pass
            elif video_t.ndim == 3:  # (T,H,W)
                video_t = video_t.unsqueeze(1)
            elif (
                video_t.ndim == 4
                and video_t.shape[1] not in (1, 3)
                and video_t.shape[-1] in (1, 3)
            ):
                # (T,H,W,C) -> (T,C,H,W)
                video_t = video_t.permute(0, 3, 1, 2)

            if ext in SUPPORTED_TIFF_EXT and max_frames is not None:
                if isinstance(video_t, LazyVideo):
                    video_t._T = min(video_t._T, int(max_frames))
                else:
                    video_t = video_t[: int(max_frames)]

            # -----------------------------
            # 2) RESIZE by target W/H
            # -----------------------------
            if (target_w is not None) and (target_h is not None):
                if isinstance(video_t, LazyVideo):
                    video_t = video_t.materialize()
                video_t = resize_video_tensor_to_wh(
                    video_t, target_w=int(target_w), target_h=int(target_h)
                )

            self.video_data = video_t
            self._original_video_data = video_t
            self._frame_start = 0
            self._frame_end = None
            self.slice_btn.setEnabled(True)
            self.slice_btn.setText("✂ Slice")

            # Update cache with new input video
            self.registration_cache.set_input(self.video_data)

            # -----------------------------
            # 3) Continue with existing UI logic
            # -----------------------------
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self._last_preset_index = 0
            self.preset_combo.blockSignals(False)
            self._update_preset_controls()
            self.apply_point_preset()
            self.undo_btn.setEnabled(True)

            self.total_frames = int(self.video_data.shape[0])
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)

            self.current_frame = 0
            self.update_frame_label()

            self.process_btn.setEnabled(True)
            self.edit_points_btn.setEnabled(True)
            self.edit_tracks_btn.setEnabled(False)
            self.edit_tracks_btn.setChecked(False)
            self.clear_points_btn.setEnabled(True)

            self.play_btn.setEnabled(True)
            self.step_fwd_btn.setEnabled(True)
            self.step_back_btn.setEnabled(True)

            # Show what was actually loaded
            T = int(self.video_data.shape[0])
            H = int(self.video_data.shape[-2])
            W = int(self.video_data.shape[-1])
            self.status_label.setText(
                f"Loaded: {os.path.basename(self.video_path)} | Frames: {T} | Size: {W}×{H}"
            )

            QApplication.processEvents()
            self.update_plot()
            QTimer.singleShot(100, self.reset_all_zoom)

        except Exception as e:
            self.status_label.setText(f"Error loading data: {e}")
            self.video_data = None
            self.registration_cache.clear()
            return

        # -----------------------------
        # 4) Diff-to-first-frame code
        # -----------------------------
        self._recompute_motion_bar()

    def on_roi_select_toggled(self, checked: bool) -> None:
        """Show/hide the square area selector on the reference view; its
        bounds (if enabled) are read back in apply_point_preset() to
        restrict where newly generated preset points are kept."""
        self.view_left.set_roi_select_mode(checked)

    def toggle_edit_mode(self):
        """Update signal connections based on current edit mode checkboxes."""
        # Ensure clean state before reconnecting
        for panel in [self.view_left, self.view_right]:
            try:
                panel.pressed.disconnect(self.on_image_press)
                panel.moved.disconnect(self.on_image_move)
                panel.released.disconnect(self.on_image_release)
            except (TypeError, RuntimeError):
                pass

        if self.edit_points_mode or self.edit_tracks_mode:
            for panel in [self.view_left, self.view_right]:
                panel.pressed.connect(self.on_image_press)
                panel.moved.connect(self.on_image_move)
                panel.released.connect(self.on_image_release)

            self._pause_if_playing()
            self.status_label.setText(
                "Edit mode: "
                + ("Points " if self.edit_points_mode else "")
                + ("Tracks" if self.edit_tracks_mode else "")
            )
        else:
            self._img_cids.clear()
            self._drag = None
            self.status_label.setText(
                f"Loaded: {os.path.basename(self.video_path) if self.video_path else 'Ready'}"
            )

        if self.pred_tracks is not None:
            self.update_plot()
        else:
            self.update_grid_display_pre_registration()

    def _nearest_point(self, pts, x, y):
        """
        Finds the index of the point in 'pts' closest to (x, y).
        Returns: (index, euclidean_distance)
        """
        if pts is None or len(pts) == 0:
            return None, float("inf")
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def _find_nearest_point_any(self, x, y, tol=15.0):
        """
        Search for the nearest neighbor among base grid and custom points.
        :return: (kind, index, distance) or (None, None, None) if exceeds tolerance.
        """
        b_idx, b_dist = None, 1e9
        if self.base_grid_pts is not None and self.base_grid_active is not None:
            active_idxs = np.where(self.base_grid_active)[0]
            if len(active_idxs) > 0:
                loc, b_dist = self._nearest_point(self.base_grid_pts[active_idxs], x, y)
                b_idx = int(active_idxs[loc])

        c_idx, c_dist = None, 1e9
        if len(self.custom_grid_pts) > 0:
            cpts = np.array(self.custom_grid_pts, dtype=np.float32)
            c_idx, c_dist = self._nearest_point(cpts, x, y)

        if b_dist < tol and b_dist <= c_dist:
            return "base", b_idx, b_dist
        if c_dist < tol:
            return "custom", c_idx, c_dist
        return None, None, None

    def _handle_point_deletion(self, kind: str, idx: int):
        """
        Deletes a point (base or custom) and its associated track if it exists.
        Handles index shifts in track_meta to maintain consistency.
        """
        # 1. If this point has an existing track (from a previous run), delete it
        tidx = self._find_track_index_for_point(kind, idx)
        if tidx is not None:
            self._delete_track_idx(tidx)

        # 2. Remove the point from its respective internal storage
        if kind == "custom":
            self.custom_grid_pts.pop(idx)
            if self.custom_grid_tracked is not None and idx < len(
                self.custom_grid_tracked
            ):
                self.custom_grid_tracked.pop(idx)
        else:  # base
            self._delete_base_point(idx)
            if self.base_grid_tracked is not None and idx < len(self.base_grid_tracked):
                self.base_grid_tracked = np.delete(self.base_grid_tracked, idx)

        # 3. CRITICAL: Adjust track_meta mapping because indices shift after pop/delete
        if hasattr(self, "track_meta") and self.track_meta is not None:
            fixed = []
            for k, i in self.track_meta:
                if k == kind and i > idx:
                    fixed.append((k, i - 1))
                else:
                    fixed.append((k, i))
            self.track_meta = fixed

        # 4. Refresh UI to reflect changes
        self._update_points_label()
        self.update_plot()

        # Update registration button state
        has_points = len(self.base_grid_pts) > 0 or len(self.custom_grid_pts) > 0
        self.process_btn.setEnabled(has_points)

    def _set_ui_processing_state(self, processing: bool):
        """
        Toggles UI elements (buttons, sliders) between 'processing' and 'idle' states.
        Ensures the user cannot trigger conflicting actions while a worker is running.
        """
        self.process_btn.setEnabled(not processing)
        self.load_btn.setEnabled(not processing)
        self.settings_btn.setEnabled(not processing)
        self.stop_btn.setEnabled(processing)
        if not processing:
            pass

    def _squeeze_batch_dim(self, data):
        """
        Utility to normalize model outputs (tracks, etc.) by removing the batch dimension
        and converting to a standard numpy format if needed. Support torch tensors.
        """
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            try:
                data = data.detach().cpu().numpy()
            except (RuntimeError, AttributeError, TypeError):
                pass
        arr = np.asarray(data)
        # Normalize (1, T, N, 2) -> (T, N, 2) or (1, T, N) -> (T, N)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _cleanup_worker(self):
        """
        Safely shuts down the background processing worker if it exists.
        Waits for the thread to exit before clearing the reference.
        """
        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(500)
            except (RuntimeError, AttributeError):
                pass
        self.worker = None

    def _exit_edit_modes(self, reason: str = ""):
        if getattr(self, "edit_points_btn", None) is not None:
            self.edit_points_btn.blockSignals(True)
            self.edit_points_btn.setChecked(False)
            self.edit_points_btn.blockSignals(False)

        if getattr(self, "edit_tracks_btn", None) is not None:
            self.edit_tracks_btn.blockSignals(True)
            self.edit_tracks_btn.setChecked(False)
            self.edit_tracks_btn.blockSignals(False)

        self.edit_points_mode = False
        self.edit_tracks_mode = False
        self.toggle_edit_mode()

        if reason:
            self.status_label.setText(reason)

    def on_image_press(self, x, y, button_value):
        """
        Handle mouse button press events on the image panels.
        Supports point addition/deletion and track selection/dragging.
        """
        if not (self.edit_points_mode or self.edit_tracks_mode):
            return

        x_f, y_f = float(x), float(y)
        btn = int(button_value)
        tol = 15.0

        # Point editing is restricted to the first frame
        if self.edit_points_mode and self.current_frame != 0:
            self.status_label.setText("Point editing is only available on frame 0.")
            return

        # Multi-touch/redundant drag guard
        if self._drag is not None:
            return

        # RIGHT CLICK (Button 3) -> Trigger deletion or deactivation
        if btn == 3:
            # In track edit mode: toggle active state of nearest trajectory
            if self.edit_tracks_mode and self.pred_tracks is not None:
                tracks = self._squeeze_batch_dim(self.pred_tracks)
                current_pts = tracks[self.current_frame]
                t_idx, t_dist = self._nearest_point(current_pts, x_f, y_f)
                if t_idx is not None and t_dist < tol:
                    self._tracks_user_edited = True
                    self.selected_track_idx = t_idx
                    if (
                        self.track_active is not None
                        and len(self.track_active) == current_pts.shape[0]
                    ):
                        self.track_active[t_idx] = ~self.track_active[t_idx]
                    self.update_plot()
                    # Enable explicit AutoWarp instead of auto-recalc
                    if hasattr(self, "recalculate_btn"):
                        self.recalculate_btn.setEnabled(True)
                    self.status_label.setText(
                        "Track activation changed. Click AutoWarp to update warping."
                    )
                return

            # If in Point Edit mode: delete the nearest base or custom point
            if self.edit_points_mode:
                kind, idx, _ = self._find_nearest_point_any(x_f, y_f, tol=20.0)
                if kind:
                    self._handle_point_deletion(kind, idx)
                return
            return

        # LEFT CLICK (Button 1) -> Select for Dragging or Add New Point
        if btn == 1:
            # First, check if user clicked on an existing track to move it
            if self.edit_tracks_mode and self.pred_tracks is not None:
                tracks = self._squeeze_batch_dim(self.pred_tracks)
                current_pts = tracks[self.current_frame]
                tidx, tdist = self._nearest_point(current_pts, x_f, y_f)
                if tidx is not None and tdist < 15.0:
                    self._drag = ("track", tidx)
                    # remember which track and frame were edited
                    self._edited_track_indices.add(int(tidx))
                    self._last_edit_frame = int(self.current_frame)
                    return

            # Second, check if user clicked on an existing base/custom grid point
            kind, idx, _ = self._find_nearest_point_any(x_f, y_f, tol=tol)
            if kind:
                self._drag = (kind, idx)
                # Update position immediately on click for better feedback
                if kind == "base":
                    self.base_grid_pts[idx] = (x_f, y_f)
                else:
                    self.custom_grid_pts[idx] = (int(x_f), int(y_f))
                self.update_grid_display_pre_registration()
                return

            # Third, if no existing point was clicked, add a new custom point
            if self.edit_points_mode:
                self.custom_grid_pts.append((int(x_f), int(y_f)))
                self.custom_grid_tracked.append(False)  # Not yet processed
                self._update_points_label()
                self.update_grid_display_pre_registration()
                self.update_plot()

                # Enable registration button as we now have at least one point
                self.process_btn.setEnabled(True)

    def on_image_move(self, x, y):
        """Handle mouse movement events for dragging points or tracks."""
        if not (self.edit_points_mode or self.edit_tracks_mode) or self._drag is None:
            return

        kind, idx = self._drag

        if kind == "track":
            self._tracks_user_edited = True
            if not self.edit_tracks_mode:
                return

            if self.pred_tracks is not None:
                if isinstance(self.pred_tracks, torch.Tensor):
                    self.pred_tracks[self.current_frame, idx, 0] = x
                    self.pred_tracks[self.current_frame, idx, 1] = y
                else:
                    self.pred_tracks[self.current_frame, idx] = (float(x), float(y))

                if "tracks" in self.viz_data:
                    self.viz_data["tracks"][self.current_frame, idx] = (
                        float(x),
                        float(y),
                    )

                self.update_plot()
                # Do not auto-schedule warping; user triggers via AutoWarp button
                if hasattr(self, "recalculate_btn"):
                    self.recalculate_btn.setEnabled(True)
                self.status_label.setText(
                    "Track modified. Use Retrack or AutoWarp to apply changes."
                )
            return

        if kind == "base":
            self.base_grid_pts[idx] = (float(x), float(y))
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()

    def on_image_release(self):
        """Finalize dragging operations and clear interaction state."""
        if (
            self._drag is not None
            and self._drag[0] == "track"
            and self.pred_tracks is not None
        ):
            # After a drag on a track, just enable explicit buttons
            if hasattr(self, "recalculate_btn"):
                self.recalculate_btn.setEnabled(True)
            if hasattr(self, "retrack_btn"):
                self.retrack_btn.setEnabled(True)
            self.status_label.setText(
                "Track modified. Use Retrack to retrack from this frame, or AutoWarp to only update warping."
            )

        self._drag = None
        self._drag_start_tracks = None
        self._drag_start_pos = None

    def clear_points(self):
        self.custom_grid_pts = []
        self.custom_grid_tracked = []
        self._update_points_label()
        self.update_grid_display_pre_registration()

    def undo_all_changes(self):
        if self.base_grid_pts0 is None:
            return
        self.base_grid_pts = self.base_grid_pts0.copy()
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.custom_grid_pts = []
        self.custom_grid_tracked = []
        self.update_grid_display_pre_registration()

    def reset_all_zoom(self):
        """Reset zoom and auto-range for both viewer panels."""
        for panel in [self.view_left, self.view_right]:
            panel.reset_view()
        self.status_label.setText("View reset on both panels.")

    # ---------- visualization ----------
    def _get_current_tracks(self):
        tr = self.viz_data.get("tracks", None)
        if tr is None or not hasattr(tr, "ndim") or tr.ndim != 3 or tr.shape[0] == 0:
            return None
        idx = min(self.current_frame, tr.shape[0] - 1)
        return tr[idx]

    def update_grid_display_pre_registration(self):
        """
        Pre-registration preview: base grid + custom points on top of the current input frame.
        With the new 2-panel UI, we don't draw to a fixed 'grid panel' anymore.
        Instead we just refresh both viewports via update_plot(), which renders according to dropdown choices.
        """
        if self.video_data is None:
            return
        self.update_plot()

    def _calc_levels(self, img, already_normalized: bool = False):
        """Calculate display intensity levels for an image.

        Percentile-based (1st/99th) by default, not strict min/max: a
        single hot outlier pixel otherwise stretches the whole display
        range and washes out real signal contrast — matches the convention
        analysis/cell_annotator.py already uses (_add_file's percentile
        levels, _to_uint8_stack).

        *already_normalized* must be set for a frame that's already been
        contrast-stretched upstream — a LazyVideo frame (see
        utils.utils.LazyVideo.get_frame_hwc / to_uint8_contrast), which is
        *itself* a percentile stretch to [0, 1], computed once per video
        (from sampled bounds) rather than per frame. Percentile-stretching
        it a second time here would compound the two: each frame gets its
        own slightly different re-stretch on top of an already-stretched
        base, so contrast drifts frame to frame in a way the eager
        (unstretched-source) path never did. For an already-normalized
        frame this just uses its own actual min/max instead."""
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.dtype.kind == "f":
            finite = arr[np.isfinite(arr)]
            if finite.size:
                if already_normalized:
                    lo, hi = float(np.min(finite)), float(np.max(finite))
                else:
                    lo, hi = np.percentile(finite, (1, 99))
                    lo, hi = float(lo), float(hi)
                if hi <= lo:
                    hi = lo + 1.0
                return (hi, lo) if self.invert_display else (lo, hi)
        else:
            if already_normalized:
                lo, hi = float(np.min(arr)), float(np.max(arr))
            else:
                lo, hi = np.percentile(arr, (1, 99))
                lo, hi = float(lo), float(hi)
            if hi <= lo:
                hi = lo + 1.0
            return (hi, lo) if self.invert_display else (lo, hi)
        return None

    def update_plot(self):
        """Render current frame into the two selected panels."""
        if self.video_data is None:
            return

        T = int(self.video_data.shape[0])
        self.current_frame = int(np.clip(self.current_frame, 0, T - 1))
        idx = self.current_frame

        is_lazy = isinstance(self.video_data, LazyVideo)
        if is_lazy:
            # Decode only the current frame — avoid loading the full video into RAM
            original_frame = self.video_data.get_frame_hwc(idx)  # (H,W,1) float32
        else:
            # Eager path: cache the full video as THWC numpy on first call
            if "input" not in self.viz_data:
                self.viz_data["input"] = (
                    self.video_data.permute(0, 2, 3, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            original_frame = self.viz_data["input"][idx]  # (H,W,C)

        fv = self._get_filtered_video_for_filter(self.current_filter)
        filtered_frame = (
            fv[idx] if fv is not None else self.apply_filter(original_frame, "None")
        )
        # No real filter video yet -> filtered_frame is original_frame through a no-op filter, still pre-stretched.
        filtered_is_normalized = is_lazy and fv is None

        # While playing, freeze contrast at the first frame's levels -- otherwise brightness flickers per-frame.
        if self.is_playing:
            if self._play_levels_orig is None:
                self._play_levels_orig = self._calc_levels(original_frame, already_normalized=is_lazy)
            if self._play_levels_filt is None:
                self._play_levels_filt = self._calc_levels(
                    filtered_frame, already_normalized=filtered_is_normalized
                )
            levels_orig = self._play_levels_orig
            levels_filt = self._play_levels_filt
        else:
            levels_orig = self._calc_levels(original_frame, already_normalized=is_lazy)
            levels_filt = self._calc_levels(filtered_frame, already_normalized=filtered_is_normalized)

        left_choice = self.view_left_combo.currentText()
        right_choice = self.view_right_combo.currentText()

        self._render_view(
            self.view_left,
            left_choice,
            idx,
            filtered_frame,
            original_frame,
            levels_filt,
            levels_orig,
        )
        self._render_view(
            self.view_right,
            right_choice,
            idx,
            filtered_frame,
            original_frame,
            levels_filt,
            levels_orig,
        )

        if not getattr(self, "_initial_zoom_done", False):
            self.view_left.reset_view()
            self.view_right.reset_view()
            self._initial_zoom_done = True

        # cursor in motion plot
        if hasattr(self, "motion_plot"):
            self.motion_plot.set_current_frame(self.current_frame)

    def _render_view(
        self,
        panel,
        which,
        idx,
        filtered_frame,
        original_frame,
        levels_filt=None,
        levels_orig=None,
    ):
        """
        Populate a PGImagePanel with data based on the selected view mode.
        """
        which = (which or "").strip()

        def clear_overlays():
            """Reset all point and trajectory overlays on the panel."""
            panel.points_item.setData([], [])
            panel.cross_item.setData([], [])
            if hasattr(panel, "pending_item"):
                panel.pending_item.setData([], [])
            panel.set_trajectories(None, 0)

        # ---------- helpers ----------
        def draw_points_preview():
            """Render current grid and custom points for preview."""
            pts_active = None
            crosses = None

            if (
                self.base_grid_pts is not None
                and self.base_grid_active is not None
                and np.any(self.base_grid_active)
            ):
                pts_active = self.base_grid_pts[self.base_grid_active]

            if self.custom_grid_pts:
                crosses = np.asarray(self.custom_grid_pts, dtype=np.float32)

            panel.set_points(pts_xy=pts_active, crosses_xy=crosses)
            panel.set_trajectories(None, 0)  # Tracks are not shown in preview

        # ---------- INPUT VIEW ----------
        if which == "Input":
            # Always pass levels_orig -- without it, switching from a uint8 view's 0-255 range makes float
            # [0-1] data appear completely black.
            panel.set_image(original_frame, auto_range=False, levels=levels_orig)
            clear_overlays()
            return

        # ---------- TRACKED POINTS VIEW ----------
        if which == "Tracked Points":
            lev = None if (panel.hist and panel.hist.isVisible()) else levels_filt
            panel.set_image(filtered_frame, auto_range=False, levels=lev)

            tracks = self.viz_data.get("tracks", None)
            if tracks is None:
                draw_points_preview()
                return

            tr = np.asarray(tracks)
            if tr.ndim == 4:
                tr = tr[0]  # (T, N, 2)
            if tr.ndim != 3 or idx >= tr.shape[0]:
                draw_points_preview()
                return

            pts_curr = tr[idx]
            n_pts = pts_curr.shape[0]
            colors = np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (n_pts, 1))
            panel.set_points_colored(pts_curr, colors_rgba=colors, size=8)

            # Auto-fit view to points if not already done
            if (
                not self.edit_points_mode
                and not self.edit_tracks_mode
                and not self.is_playing
                and not getattr(self, "_auto_fit_done", False)
            ):
                panel.fit_view_to_image_and_points(pts_curr, margin=20)
                self._auto_fit_done = True

            # draw trajectories ONLY here
            if self.show_tracks_chk.isChecked():
                panel.set_trajectories(tr, idx)
            else:
                panel.set_trajectories(None, idx)

            # ---- draw pending (untracked) points so they don't "disappear" ----
            pending_pts = []

            # pending base = active & not tracked
            if (
                self.base_grid_pts is not None
                and self.base_grid_active is not None
                and self.base_grid_tracked is not None
            ):
                pend_mask = self.base_grid_active.astype(bool) & (
                    ~np.asarray(self.base_grid_tracked, bool)
                )
                if np.any(pend_mask):
                    pending_pts.append(self.base_grid_pts[pend_mask].astype(np.float32))

            # pending custom = not tracked
            if self.custom_grid_pts and self.custom_grid_tracked is not None:
                pend_custom = [
                    p
                    for p, trk in zip(self.custom_grid_pts, self.custom_grid_tracked)
                    if not trk
                ]
                if len(pend_custom) > 0:
                    pending_pts.append(np.asarray(pend_custom, dtype=np.float32))

            if len(pending_pts) > 0:
                P = np.vstack(pending_pts)
                panel.pending_item.setData(P[:, 0], P[:, 1])
            else:
                panel.pending_item.setData([], [])

            # (optional) also show custom points as crosses while editing points
            if self.edit_points_mode and self.custom_grid_pts:
                c = np.asarray(self.custom_grid_pts, dtype=np.float32)
                panel.cross_item.setData(c[:, 0], c[:, 1])
            else:
                panel.cross_item.setData([], [])

            return

        # ---------- CORRECTED ----------
        if which == "CoReg":
            warped = self.viz_data.get("warped", None)
            if warped is not None and idx < len(warped):
                wf = warped[idx]
                panel.set_image(
                    wf,
                    auto_range=False,
                    levels=self._calc_levels(wf),
                )
            else:
                panel.set_image(None)
            clear_overlays()

            return

        # ---------- DEFORMATION FIELD ----------
        if which == "Deformation Field":
            if self.disp_field is not None:
                disp_frame = (
                    self.disp_field[idx]
                    if idx < len(self.disp_field)
                    else self.disp_field[0]
                )
                disp_preview = self._make_disp_preview(disp_frame)
                panel.set_image(
                    disp_preview,
                    auto_range=False,
                    levels=self._calc_levels(disp_preview),
                )
            else:
                panel.set_image(None)
            clear_overlays()

            return

        # ---------- Corrected (ANTs) ----------
        if which == "ANTs":
            v = self.ants_video
            if v is not None and idx < len(v):
                frame = v[idx]
                panel.set_image(frame, auto_range=False, levels=self._calc_levels(frame))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # ---------- Corrected (LDDMM) ----------
        if which == "LDDMM":
            v = self.lddmm_video
            if v is not None and idx < len(v):
                frame = v[idx]
                panel.set_image(frame, auto_range=False, levels=self._calc_levels(frame))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # ---------- Corrected (NoRMCorre) ----------
        if which == "NoRMCorre":
            v = getattr(self, "normcorre_video", None)
            if v is not None and idx < len(v):
                frame = v[idx]
                panel.set_image(frame, auto_range=False, levels=self._calc_levels(frame))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # ---------- Corrected (PyFlowReg) ----------
        if which == "PyFlowReg":
            v = getattr(self, "flowreg_video", None)
            if v is not None and idx < len(v):
                frame = v[idx]
                panel.set_image(frame, auto_range=False, levels=self._calc_levels(frame))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # fallback
        panel.set_image(original_frame, auto_range=False, levels=levels_orig)
        panel.fit_view_to_image_and_points(None, margin=20)
        clear_overlays()

    def visualize(
        self,
        method="CoTracker",
        ants_params=None,
        normcorre_params=None,
        pyflowreg_params=None,
    ):
        """
        Initiate the motion correction processing via the specified method.
        """
        if self.video_data is None:
            return

        self._initial_zoom_done = False
        QTimer.singleShot(0, self.reset_all_zoom)

        # Ensure edit modes are disabled during processing
        if self.edit_points_btn.isChecked():
            self.edit_points_btn.setChecked(False)
        if self.edit_tracks_btn.isChecked():
            self.edit_tracks_btn.setChecked(False)

        self.timer.stop()
        self.is_playing = False
        self._play_levels_orig = None
        self._play_levels_filt = None
        self.play_btn.setText("▶")

        if "input" not in self.viz_data:
            self.viz_data["input"] = (
                self.video_data.permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if self.warped_video is not None:
            self.viz_data["warped"] = self.warped_video

        # Lock UI to prevent user interference during complex processing
        self.status_label.setText("Processing... (This may take a while)")
        self._set_ui_processing_state(True)

        # Check cache for slow direct registration methods.
        # ANTs/LDDMM reuse cached results; NoRMCorre/PyFlowReg always recompute so the user can "re-run" easily.
        if method in ["ANTs", "LDDMM"]:
            # Collect parameters for this method
            method_params = {}
            if method == "ANTs":
                method_params = ants_params or {}
            elif method == "NoRMCorre":
                method_params = normcorre_params or {}
            elif method == "PyFlowReg":
                method_params = pyflowreg_params or {}

            # Check if we have cached results
            cached_result = self.registration_cache.get(method, method_params)
            if cached_result is not None:
                warped_np, disp_arr = cached_result
                self.status_label.setText(f"Using cached {method} results")
                self.on_processing_finished(None, None, None, None, warped_np, disp_arr)
                return

        video_in = self._get_video_input_for_tracking()
        video_out = self.video_data

        # Ensure point tracking flags are synchronized with current point sets
        if self.base_grid_pts is None:
            self.base_grid_pts = np.zeros((0, 2), np.float32)

        if self.base_grid_active is None:
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        if self.base_grid_tracked is None or len(self.base_grid_tracked) != len(
            self.base_grid_pts
        ):
            tmp = np.zeros((len(self.base_grid_pts),), dtype=bool)
            if self.base_grid_tracked is not None:
                m_len = min(len(tmp), len(self.base_grid_tracked))
                tmp[:m_len] = self.base_grid_tracked[:m_len]
            self.base_grid_tracked = tmp

        if self.custom_grid_tracked is None:
            self.custom_grid_tracked = []

        if len(self.custom_grid_tracked) != len(self.custom_grid_pts):
            if len(self.custom_grid_tracked) < len(self.custom_grid_pts):
                self.custom_grid_tracked = list(self.custom_grid_tracked) + [False] * (
                    len(self.custom_grid_pts) - len(self.custom_grid_tracked)
                )
            else:
                self.custom_grid_tracked = list(
                    self.custom_grid_tracked[: len(self.custom_grid_pts)]
                )

        # Build lists of points that require tracking
        active_mask = self.base_grid_active.astype(bool)
        pending_base_mask = active_mask & (~self.base_grid_tracked)
        pending_base_idxs = np.where(pending_base_mask)[0]
        pending_base_pts = (
            self.base_grid_pts[pending_base_idxs].astype(np.float32)
            if len(pending_base_idxs)
            else np.zeros((0, 2), np.float32)
        )

        pending_custom_idxs = [
            i for i, trk in enumerate(self.custom_grid_tracked) if not trk
        ]
        pending_custom_pts = (
            np.asarray(
                [self.custom_grid_pts[i] for i in pending_custom_idxs], dtype=np.float32
            )
            if len(pending_custom_idxs)
            else np.zeros((0, 2), np.float32)
        )

        new_queries = None
        pending_meta = []  # <-- THIS will become self._pending_track_meta

        if pending_base_pts.shape[0] > 0 or pending_custom_pts.shape[0] > 0:
            all_new = []
            if pending_base_pts.shape[0] > 0:
                all_new.append(pending_base_pts)
                pending_meta.extend([("base", int(bi)) for bi in pending_base_idxs])

            if pending_custom_pts.shape[0] > 0:
                all_new.append(pending_custom_pts)
                pending_meta.extend([("custom", int(ci)) for ci in pending_custom_idxs])

            new_queries = torch.from_numpy(np.vstack(all_new)).unsqueeze(
                0
            )  # (1,N_new,2)

        # remember what this run tracked
        self._pending_base_idxs_last_run = pending_base_idxs
        self._pending_custom_idxs_last_run = pending_custom_idxs
        self._pending_track_meta = pending_meta  # <-- IMPORTANT

        # -------------------------------
        # Existing tracks to merge (worker will concat old + new)
        # -------------------------------
        existing_data = None
        if self.pred_tracks is not None:
            tr_old = np.asarray(self.pred_tracks)
            if tr_old.ndim == 4 and tr_old.shape[0] == 1:
                tr_old = tr_old[0]

            vis_old = None
            if self.pred_visibility is not None:
                vis_old = np.asarray(self.pred_visibility)
                if vis_old.ndim == 3 and vis_old.shape[0] == 1:
                    vis_old = vis_old[0]
                if vis_old.ndim == 3 and vis_old.shape[-1] == 1:
                    vis_old = vis_old[..., 0]

            conf_old = None
            if getattr(self, "pred_confidence", None) is not None:
                conf_old = np.asarray(self.pred_confidence)
                if conf_old.ndim == 3 and conf_old.shape[0] == 1:
                    conf_old = conf_old[0]

            existing_data = (tr_old, vis_old, conf_old)

            # Ensure track_meta matches existing old tracks length (critical for appending)
            self._ensure_track_meta_len(tr_old.shape[1])

        # If nothing new to track and no existing tracks, abort
        if (
            new_queries is None
            and existing_data is None
            and method not in ["ANTs", "LDDMM", "NoRMCorre", "PyFlowReg"]
        ):
            self.status_label.setText("No points to track.")
            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.settings_btn.setEnabled(True)
            return

        # Prepare processing worker
        sigma = (
            float(self.df_sigma_spin.value()) if hasattr(self, "df_sigma_spin") else 5.0
        )

        self.worker = ProcessingWorker(
            video_in,
            video_out=video_out,
            parent=self,
            existing_data=existing_data,
            method=method,
            queries=new_queries,
            df_sigma=sigma,
            ants_params=ants_params,
            normcorre_params=normcorre_params,
            pyflowreg_params=pyflowreg_params,
            force_cpu=self.force_cpu_chk.isChecked(),
        )

        # your worker expects new_queries in custom_grid_pts sometimes:
        if new_queries is not None:
            self.worker.custom_grid_pts = new_queries

        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.start()

        self.stop_btn.setEnabled(True)

    def recalculate_warping(self):
        """Immediately trigger or schedule a warping recalculation."""
        self._schedule_recalc_warping(delay_ms=0)

    def _recalculate_warping_from_tracks(self):
        if self.video_data is None or self.pred_tracks is None:
            return

        if self._recalc_inflight:
            self._pending_recalc = True
            return

        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self._pending_recalc = True
                    return
            except (RuntimeError, AttributeError):
                pass

        self._recalc_inflight = True
        self._pending_recalc = False

        self.status_label.setText("Auto: recalculating warping...")
        self._set_ui_processing_state(True)

        # Tracks -> torch (T, N, 2)
        tr_np = self._squeeze_batch_dim(self.pred_tracks)

        # Filter tracks to only include active ones for warping
        if self.track_active is not None and len(self.track_active) == tr_np.shape[1]:
            tr_np_use = tr_np[:, self.track_active, :]
        else:
            tr_np_use = tr_np

        tracks_t = torch.from_numpy(np.asarray(tr_np_use, dtype=np.float32))

        # Dummy base grid tensor (worker requirement)
        base_active = (
            self.base_grid_pts[self.base_grid_active].astype(np.float32)
            if (self.base_grid_pts is not None and self.base_grid_active is not None)
            else np.zeros((0, 2), np.float32)
        )
        base_grid_tensor = torch.from_numpy(base_active).unsqueeze(0)

        # Ensure any currently running worker is terminated cleanly before starting a new one
        self._cleanup_worker()

        sigma = (
            float(self.df_sigma_spin.value()) if hasattr(self, "df_sigma_spin") else 5.0
        )

        self.worker = ProcessingWorker(
            video_in=self.video_data,  # Use original for re-warping
            video_out=self.video_data,
            base_grid_pts=base_grid_tensor,
            tracks=tracks_t,
            parent=self,
            filter_name=None,
            df_sigma=sigma,
            force_cpu=self.force_cpu_chk.isChecked(),
        )

        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_retrack_from_edited_frame_clicked(self):
        """Handler for the 'Retrack from Frame' button.

        Retracks all tracks starting from the last edited frame (if any),
        otherwise from the current frame. This uses the current track
        locations on that frame (including any manual edits) as seeds.
        """
        if self.pred_tracks is None:
            return

        tracks_arr = self._squeeze_batch_dim(self.pred_tracks)
        if tracks_arr is None or tracks_arr.ndim != 3:
            return

        T, N, _ = tracks_arr.shape
        if T == 0 or N == 0:
            return

        # Choose frame: prefer the last edit frame, fall back to current
        frame_idx = (
            int(self._last_edit_frame)
            if self._last_edit_frame is not None
            else int(self.current_frame)
        )
        frame_idx = int(np.clip(frame_idx, 0, T - 1))

        # Retrack all tracks from this frame onward
        track_ids = list(range(N))
        self._retrack_tracks_from_frame(frame_idx=frame_idx, track_indices=track_ids)

        # After retracking, we can disable the retrack button until next edit
        self.retrack_btn.setEnabled(False)
        self._edited_track_indices.clear()
        self._last_edit_frame = None

    def on_autowarp_clicked(self):
        """Explicit AutoWarp button handler: recompute warping only."""
        self.recalculate_warping()

    def _retrack_tracks_from_frame(self, frame_idx: int, track_indices):
        """Retrack a subset (or all) tracks starting from a given frame.

        The manual edits on the chosen frame are used as new seeds, and only
        frames >= frame_idx are overwritten in the existing trajectories.
        """
        if self.video_data is None or self.pred_tracks is None:
            return

        tracks_arr = self._squeeze_batch_dim(self.pred_tracks)
        if tracks_arr is None or tracks_arr.ndim != 3:
            return

        T, N, _ = tracks_arr.shape
        if T == 0 or N == 0:
            return

        if track_indices is None:
            return

        # Normalize frame and indices
        f = int(np.clip(int(frame_idx), 0, T - 1))
        track_ids = [int(i) for i in track_indices if 0 <= int(i) < N]
        if not track_ids:
            return

        # Run CoTracker on the video suffix starting at frame f, so queries can use time index 0.
        video_in = self._get_video_input_for_tracking()  # (T,C,H,W)
        if isinstance(video_in, np.ndarray):
            video_t_full = torch.from_numpy(np.asarray(video_in))
        else:
            video_t_full = video_in

        # Slice from frame f onward for retracking
        try:
            video_t_sub = video_t_full[f:]
        except Exception:
            self.status_label.setText("Retracking failed: cannot slice video tensor.")
            return

        # Device selection
        if self.force_cpu_chk.isChecked():
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        try:
            video_t_sub = video_t_sub.to(device)
        except Exception:
            # Fallback to CPU if move fails for some reason
            device = "cpu"
            video_t_sub = video_t_sub.to(device)

        # Ensure 3-channel input for CoTracker (it expects RGB-like input)
        if (
            isinstance(video_t_sub, torch.Tensor)
            and video_t_sub.ndim == 4
            and video_t_sub.shape[1] == 1
        ):
            video_t_sub = video_t_sub.repeat(1, 3, 1, 1)

        # Seeds: use the (possibly edited) positions at frame f, but set time=0
        xy0 = tracks_arr[f, track_ids, :].astype(np.float32)  # (K,2)
        q = torch.from_numpy(xy0).to(device).float().unsqueeze(0)  # (1,K,2)
        t_coord = torch.zeros((1, q.shape[1], 1), device=device, dtype=q.dtype)
        queries = torch.cat([t_coord, q], dim=-1)  # (1,K,3)

        # Run CoTracker once for the selected tracks on the suffix
        self.status_label.setText(
            f"Retracking {len(track_ids)} track(s) from frame {f}…"
        )
        try:
            pred_tracks_new, _ = cotrack_model(
                video_t_sub,
                queries=queries,
                device=device,
                grid_size=0,
                progress_cb=None,
            )
        except Exception as e:
            self.status_label.setText(f"Retracking failed: {e}")
            return

        # Convert and merge back into existing trajectories
        new_tr = pred_tracks_new.detach().cpu().numpy()
        if new_tr.ndim == 4 and new_tr.shape[0] == 1:
            new_tr = new_tr[0]  # (T_sub, K, 2)

        if new_tr.ndim != 3 or new_tr.shape[-1] != 2:
            self.status_label.setText(
                f"Retracking failed: unexpected track shape {new_tr.shape}"
            )
            return

        T_sub, K, _ = new_tr.shape
        if K != len(track_ids):
            # Something is inconsistent; avoid corrupting tracks
            self.status_label.setText(
                "Retracking failed: inconsistent number of returned tracks."
            )
            return

        # Only overwrite frames >= f; keep earlier frames untouched
        for j, tid in enumerate(track_ids):
            for dt in range(T_sub):
                t_global = f + dt
                if t_global >= T:
                    break
                tracks_arr[t_global, tid, :] = new_tr[dt, j, :]

        # tracks_arr[f, track_ids, :] already reflects the edited positions -- kept exact, not just CoTracker's output.

        self.pred_tracks = tracks_arr
        self.viz_data["tracks"] = self.pred_tracks

        # Immediately re-warp using updated trajectories
        self.recalculate_btn.setEnabled(True)
        self.status_label.setText("Retracking finished. Updating warping…")
        self.recalculate_warping()

    def _ensure_track_meta_len(self, count: int):
        """Validate and adjust track_meta list to match point count."""
        if not hasattr(self, "track_meta") or self.track_meta is None:
            self.track_meta = []
        if len(self.track_meta) < count:
            self.track_meta.extend([("unknown", -1)] * (count - len(self.track_meta)))
        elif len(self.track_meta) > count:
            self.track_meta = self.track_meta[:count]

    def _find_track_index_for_point(self, kind: str, idx: int):
        """
        Find the track index corresponding to a specific point kind and index.
        :param kind: Point category ("base" or "custom").
        :param idx: Point index within the category.
        :return: Integer index in track_meta/pred_tracks, or None if not found.
        """
        if not hasattr(self, "track_meta") or self.track_meta is None:
            return None
        target = (str(kind), int(idx))
        for t_idx, meta in enumerate(self.track_meta):
            if meta == target:
                return t_idx
        return None

    def _delete_base_point(self, idx: int):
        """Remove a base grid point and keep related arrays in sync.
        Does not modify `base_grid_tracked` (caller handles that).
        """
        if self.base_grid_pts is None or len(self.base_grid_pts) == 0:
            return
        n = int(self.base_grid_pts.shape[0])
        if idx < 0 or idx >= n:
            return

        try:
            self.base_grid_pts = np.delete(self.base_grid_pts, idx, axis=0)
        except (AttributeError, TypeError, ValueError):
            arr = np.asarray(self.base_grid_pts, dtype=np.float32).reshape(-1, 2)
            self.base_grid_pts = np.delete(arr, idx, axis=0)

        if self.base_grid_active is not None and len(self.base_grid_active) >= n:
            self.base_grid_active = np.delete(self.base_grid_active, idx, axis=0)

        if self.base_grid_pts0 is not None:
            try:
                if len(self.base_grid_pts0) >= n:
                    self.base_grid_pts0 = np.delete(self.base_grid_pts0, idx, axis=0)
            except (AttributeError, TypeError, ValueError):
                pass

    def _delete_track_idx(self, tidx: int):
        """
        Permanently remove a track trajectory and its associated metadata.
        """
        if self.pred_tracks is None:
            return

        tr = self._squeeze_batch_dim(self.pred_tracks)
        if tr is None or tr.ndim != 3:
            return

        _, n_pts, _ = tr.shape
        if tidx < 0 or tidx >= n_pts:
            return

        # 1. Remove from coordinate arrays
        self.pred_tracks = np.delete(tr, tidx, axis=1)
        self.viz_data["tracks"] = self.pred_tracks

        # 2. Remove from visibility/confidence maps
        if self.pred_visibility is not None:
            vis = self._squeeze_batch_dim(self.pred_visibility)
            if vis is not None and vis.shape[1] > tidx:
                self.pred_visibility = np.delete(vis, tidx, axis=1)

        if getattr(self, "pred_confidence", None) is not None:
            conf = self._squeeze_batch_dim(self.pred_confidence)
            if conf is not None and conf.shape[1] > tidx:
                self.pred_confidence = np.delete(conf, tidx, axis=1)

        # 3. Update active track mask
        if self.track_active is not None and len(self.track_active) > tidx:
            self.track_active = np.delete(self.track_active, tidx)
        else:
            self.track_active = np.ones((self.pred_tracks.shape[1],), dtype=bool)

        # 4. Maintain alignment with track_meta metadata
        if hasattr(self, "track_meta") and self.track_meta is not None:
            if tidx < len(self.track_meta):
                self.track_meta.pop(tidx)

        # 5. Handle selection index shift
        if self.selected_track_idx is not None:
            if self.selected_track_idx == tidx:
                self.selected_track_idx = None
            elif self.selected_track_idx > tidx:
                self.selected_track_idx -= 1

    def on_tracks_ready(self, pred_tracks, pred_visibility, pred_confidence, grid_pts):
        """
        Callback triggered when tracker trajectories are available.
        """
        tr = self._squeeze_batch_dim(pred_tracks)
        if tr is None or tr.ndim != 3 or tr.shape[-1] != 2:
            raise ValueError(
                f"Unexpected tracks array shape: {getattr(tr, 'shape', 'None')}"
            )

        t_size, n_pts, _ = tr.shape
        self.pred_tracks = tr
        self.pred_visibility = self._squeeze_batch_dim(pred_visibility)
        self.pred_confidence = self._squeeze_batch_dim(pred_confidence)

        self.track_active = np.ones((n_pts,), dtype=bool)
        self.selected_track_idx = None

        if grid_pts is not None:
            self.grid_pts = grid_pts

        self.viz_data["tracks"] = tr

        # Update frame constraints
        v_data = self.video_data
        self.total_frames = int(v_data.shape[0]) if v_data is not None else t_size
        self.current_frame = int(np.clip(self.current_frame, 0, self.total_frames - 1))

        self.update_plot()
        self.edit_tracks_btn.setEnabled(True)
        self._tracks_user_edited = False

    def _to_thwc(self, vid):
        v = np.asarray(vid)
        # (T,3,H,W) -> (T,H,W,3)
        if v.ndim == 4 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
            v = np.transpose(v, (0, 2, 3, 1))
        return v

    def on_edit_modes_changed(self, *_):
        sender = self.sender()
        # Enforce mutual exclusivity: if one is checked, uncheck the other
        if sender == self.edit_points_btn and self.edit_points_btn.isChecked():
            if self.edit_tracks_btn.isChecked():
                self.edit_tracks_btn.blockSignals(True)
                self.edit_tracks_btn.setChecked(False)
                self.edit_tracks_btn.blockSignals(False)
        elif sender == self.edit_tracks_btn and self.edit_tracks_btn.isChecked():
            if self.edit_points_btn.isChecked():
                self.edit_points_btn.blockSignals(True)
                self.edit_points_btn.setChecked(False)
                self.edit_points_btn.blockSignals(False)

        self._auto_fit_done = False
        self.edit_points_mode = self.edit_points_btn.isChecked()
        self.edit_tracks_mode = self.edit_tracks_btn.isChecked()

        if self.edit_points_mode:
            self._pause_if_playing()
            if self.current_frame != 0:
                self.current_frame = 0
                self.slider.blockSignals(True)
                self.slider.setValue(0)
                self.slider.blockSignals(False)
                self.update_frame_label()
                self.update_plot()
                self.status_label.setText(
                    "Point editing is only available on frame 0 "
                    "(automatically jumped to frame 0)."
                )

            # Enable registration button if points exist or we are in edit mode
            has_points = (
                self.base_grid_pts is not None and len(self.base_grid_pts) > 0
            ) or len(self.custom_grid_pts) > 0
            self.process_btn.setEnabled(has_points or self.edit_points_mode)

        if self.edit_tracks_mode:
            self.view_left_combo.blockSignals(True)
            self.view_right_combo.blockSignals(True)

            self.view_left_combo.setCurrentText("Tracked Points")
            self.view_right_combo.setCurrentText("CoReg")

            self.view_left_combo.blockSignals(False)
            self.view_right_combo.blockSignals(False)

            self.update_plot()

        self.toggle_edit_mode()

    def on_processing_finished(
        self,
        pred_tracks,
        pred_visibility,
        pred_confidence,
        _grid_pts,
        warped_video,
        disp_field,
        drift_field=None,
    ):
        """
        Final callback when a registration/warping task completes.
        Handles merging new tracks, updating the warped video, and unlocking the UI.
        """
        self._auto_fit_done = False

        # 1) Handle Non-Tracking Methods (ANTs / LDDMM / NoRMCorre / PyFlowReg)
        if pred_tracks is None and warped_video is not None:
            method = getattr(self.worker, "method", "ANTs") if self.worker else "ANTs"

            # Collect method parameters for caching
            method_params = {}
            if method == "ANTs":
                method_params = self.ants_params
            elif method == "NoRMCorre":
                method_params = self.normcorre_params
            elif method == "PyFlowReg":
                method_params = self.pyflowreg_params

            # Cache the result
            warped_np = np.asarray(warped_video, dtype=np.float32)
            disp_np = (
                np.asarray(disp_field, dtype=np.float32)
                if disp_field is not None
                else np.zeros_like(warped_np)[..., :2]
            )
            self.registration_cache.put(method, warped_np, disp_np, method_params)

            if method == "LDDMM":
                v = self._to_thwc(warped_video)
                v_arr = np.asarray(v)
                if v_arr.dtype == np.uint8:
                    pass
                else:
                    # register_lddmm() preserves the input's own range, commonly uint16-scale -- a min/max
                    # threshold would clip almost everything. Percentile-stretch instead.
                    v_arr = to_uint8_contrast(v_arr)

                self.lddmm_video = v_arr.astype(np.uint8)
                self.status_label.setText("LDDMM registration finished")
            elif method == "NoRMCorre":
                v = self._to_thwc(warped_video)
                v_arr = np.asarray(v)
                if np.issubdtype(v_arr.dtype, np.floating):
                    vmin = float(np.nanmin(v_arr))
                    vmax = float(np.nanmax(v_arr))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        v_arr = np.clip(v_arr, 0, 255)
                    # data in [0,1] -> scale to [0,255]; cutoff at 1.5, not 1.0, gives slack for float noise above 1.0
                    elif vmax <= 1.5:
                        v_arr = (v_arr - vmin) / (vmax - vmin + 1e-8)
                        v_arr = np.clip(v_arr * 255.0, 0, 255)
                    else:
                        v_arr = np.clip(v_arr, 0, 255)
                else:
                    vmax = int(v_arr.max()) if v_arr.size > 0 else 0
                    if vmax <= 1:
                        v_arr = v_arr.astype(np.float32) * 255.0
                    v_arr = np.clip(v_arr, 0, 255)

                self.normcorre_video = v_arr.astype(np.uint8)
                self.status_label.setText("NoRMCorre registration finished")
            elif method == "PyFlowReg":
                v = self._to_thwc(warped_video)
                v_arr = np.asarray(v)
                if np.issubdtype(v_arr.dtype, np.floating):
                    vmin = float(np.nanmin(v_arr))
                    vmax = float(np.nanmax(v_arr))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        v_arr = np.clip(v_arr, 0, 255)
                    # data in [0,1] -> scale to [0,255]; cutoff at 1.5, not 1.0, gives slack for float noise above 1.0
                    elif vmax <= 1.5:
                        v_arr = (v_arr - vmin) / (vmax - vmin + 1e-8)
                        v_arr = np.clip(v_arr * 255.0, 0, 255)
                    else:
                        v_arr = np.clip(v_arr, 0, 255)
                else:
                    vmax = int(v_arr.max()) if v_arr.size > 0 else 0
                    if vmax <= 1:
                        v_arr = v_arr.astype(np.float32) * 255.0
                    v_arr = np.clip(v_arr, 0, 255)

                self.flowreg_video = v_arr.astype(np.uint8)
                self.status_label.setText("PyFlowReg registration finished")
            else:
                v = self._to_thwc(warped_video)
                v_arr = np.asarray(v)
                if v_arr.dtype != np.uint8:
                    # ants_registration()'s warpedmovout preserves the input's own range -- same as LDDMM above.
                    v_arr = to_uint8_contrast(v_arr)

                self.ants_video = v_arr.astype(np.uint8)
                self.status_label.setText("ANTs registration finished")

            self.update_plot()
            self._cleanup_worker()
            self._set_ui_processing_state(False)
            QTimer.singleShot(0, self.reset_all_zoom)
            return

        # 2) Normalize and Store results
        new_tr = self._squeeze_batch_dim(pred_tracks)
        if new_tr is None:
            self.status_label.setText("Processing finished; no tracks returned.")
            self._cleanup_worker()
            self._set_ui_processing_state(False)
            self._recalc_inflight = False
            return

        _, n_pts, _ = new_tr.shape
        self.pred_tracks = new_tr
        self.pred_visibility = self._squeeze_batch_dim(pred_visibility)
        self.pred_confidence = self._squeeze_batch_dim(pred_confidence)

        # 3) Update Point Status (mark points as successfully tracked)
        if hasattr(self, "_pending_base_idxs_last_run"):
            if self.base_grid_tracked is None:
                self.base_grid_tracked = np.zeros(
                    (len(self.base_grid_pts),), dtype=bool
                )
            for b_idx in self._pending_base_idxs_last_run:
                if 0 <= b_idx < len(self.base_grid_tracked):
                    self.base_grid_tracked[b_idx] = True
            del self._pending_base_idxs_last_run

        if hasattr(self, "_pending_custom_idxs_last_run"):
            if self.custom_grid_tracked is None:
                self.custom_grid_tracked = [False] * len(self.custom_grid_pts)
            while len(self.custom_grid_tracked) < len(self.custom_grid_pts):
                self.custom_grid_tracked.append(False)

            for c_idx in self._pending_custom_idxs_last_run:
                if 0 <= c_idx < len(self.custom_grid_tracked):
                    self.custom_grid_tracked[c_idx] = True
            del self._pending_custom_idxs_last_run

        # 4) Maintain track_meta mapping consistency
        if not hasattr(self, "track_meta") or self.track_meta is None:
            self.track_meta = []

        pending_meta = getattr(self, "_pending_track_meta", [])
        old_n_est = max(0, n_pts - len(pending_meta))

        # Pad or trim meta to match the new number of tracks
        if len(self.track_meta) < old_n_est:
            self.track_meta.extend(
                [("unknown", -1)] * (old_n_est - len(self.track_meta))
            )
        elif len(self.track_meta) > old_n_est and len(pending_meta) > 0:
            self.track_meta = self.track_meta[:old_n_est]

        if len(pending_meta) > 0:
            self.track_meta.extend(pending_meta)

        self.track_meta = (
            self.track_meta[:n_pts]
            if len(self.track_meta) >= n_pts
            else self.track_meta + [("unknown", -1)] * (n_pts - len(self.track_meta))
        )
        self._pending_track_meta = []

        # 5) Store registration results
        self.warped_video = (
            self._to_thwc(warped_video) if warped_video is not None else None
        )
        if self.warped_video is not None:
            v = np.asarray(self.warped_video)

            # Normalize to uint8 for consistent display.
            if np.issubdtype(v.dtype, np.floating):
                vmin = float(np.nanmin(v))
                vmax = float(np.nanmax(v))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    v = np.clip(v, 0, 255)
                elif vmax <= 1.5:
                    # data in [0,1] -> scale to [0,255]; cutoff at 1.5, not 1.0, gives slack for float noise above 1.0
                    v = (v - vmin) / (vmax - vmin + 1e-8)
                    v = np.clip(v * 255.0, 0, 255)
                else:
                    # Assume already roughly in 0-255 range
                    v = np.clip(v, 0, 255)
            else:
                # Integer types
                vmax = int(v.max()) if v.size > 0 else 0
                if vmax <= 1:
                    # Binary 0/1 -> stretch to 0/255
                    v = v.astype(np.float32) * 255.0
                v = np.clip(v, 0, 255)

            self.warped_video = v.astype(np.uint8)
            self.viz_data["warped"] = self.warped_video

        self.disp_field = disp_field
        self.drift_xy = drift_field
        self.viz_data["tracks"] = self.pred_tracks

        # 6) Calculate SSIM registration quality metric
        try:
            if self.warped_video is not None:
                self.registration_metrics = video_metrics(
                    self.warped_video,
                    mode="ref_first",
                    drift=self.drift_xy,
                )
                self.ssim_score = self.registration_metrics["ssim"]
                self.status_label.setText("Done")
            else:
                self.status_label.setText("Warping updated")
        except Exception:
            self.status_label.setText("Done")

        # 7) Cleanup and Unlock UI
        pass  # save always available
        self.edit_tracks_btn.setEnabled(True)
        self._cleanup_worker()
        self._set_ui_processing_state(False)
        self._recalc_inflight = False

        self.update_plot()
        QTimer.singleShot(50, self.view_left.reset_view)
        QTimer.singleShot(50, self.view_right.reset_view)

        # Trigger deferred recalc if input was modified during processing
        if getattr(self, "_pending_recalc", False):
            self._pending_recalc = False
            elapsed = self._edit_clock.elapsed()
            target = getattr(self, "_target_recalc_time_ms", 0)
            remaining = max(0, target - elapsed)
            self._schedule_recalc_warping(delay_ms=remaining)

        self._tracks_user_edited = False
        self.worker = None
        self._recalc_inflight = False

    def on_processing_error(self, err_msg):
        """Callback for worker errors. Unlocks the UI and displays the message."""
        self.status_label.setText(f"Error: {err_msg}")
        self._cleanup_worker()
        self._set_ui_processing_state(False)
        self._recalc_inflight = False

    def closeEvent(self, event):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        super().closeEvent(event)

    def stop_processing(self):
        """Interrupt and terminate the current background processing worker."""
        self._cleanup_worker()
        self._set_ui_processing_state(False)
        self.status_label.setText("Stopped.")

        release_gpu_memory()

    # ---------- PLAY Logic of video ----------
    def _pause_if_playing(self):
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            self._play_levels_orig = None
            self._play_levels_filt = None

    def step_forward(self):
        """Advance the video to the next frame."""
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.slider.setValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def step_backward(self):
        """Reverse the video to the previous frame."""
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame - 1) % self.total_frames
        self.slider.setValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def toggle_playback(self):
        if self.total_frames <= 0:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            return

        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            self._play_levels_orig = None
            self._play_levels_filt = None
        else:
            interval = max(1, int(50 / self._current_speed()))
            self.timer.start(interval)
            self.is_playing = True
            self._play_levels_orig = None
            self._play_levels_filt = None
            self.play_btn.setText("⏸")

    def on_timer(self):
        """Update logic for the playback timer."""
        if self.video_data is None:
            self._pause_if_playing()
            return

        # Synchronize frame counts
        self.total_frames = int(
            self.video_data.shape[0] if hasattr(self.video_data, "shape") else 0
        )
        if self.total_frames <= 0:
            self._pause_if_playing()
            return

        self.current_frame = (self.current_frame + 1) % self.total_frames

        # Block signals to prevent redundant recursion from slider callbacks
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

        self.update_plot()
        self.update_frame_label()

    def on_slider_change(self, value):
        if not self.is_playing:
            self.current_frame = int(value)
            self.update_plot()
            self.update_frame_label()

    def update_frame_label(self):
        self.frame_label.setText(f"{self.current_frame}/{self.total_frames}")

    def _current_speed(self) -> float:
        return float(self.speed_combo.currentText().rstrip("x"))

    def on_speed_change(self, *_):
        if self.is_playing:
            interval = max(1, int(50 / self._current_speed()))
            self.timer.start(interval)

    # ----------Visualization adjustment of input data----------
    def _show_load_settings_dialog(self):
        if not self.video_path:
            return None

        frames, w, h = probe_media_info(self.video_path)
        axes_info = get_tiff_axes_info(self.video_path)
        truncation_warning = get_tiff_truncation_warning(self.video_path)
        dlg = VideoLoadSettingsDialog(
            self, frames, w, h, axes_info=axes_info, truncation_warning=truncation_warning
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.result_settings()
        return None

    def _get_filtered_video_for_filter(self, filt_name: str):
        """
        Retrieve a filtered version of the video data from cache or generate it.
        """
        if self.video_data is None:
            return None

        key = f"filtered_video::{(filt_name or 'None')}"
        if key in self.viz_data:
            return self.viz_data[key]

        # Ensure input cache exists
        if "input" not in self.viz_data:
            self.viz_data["input"] = (
                self.video_data.permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        inp = self.viz_data["input"]
        t_size = inp.shape[0]
        out = []
        for t_idx in range(t_size):
            # apply_filter returns (H, W) uint8
            out.append(self.apply_filter(inp[t_idx], filt_name))

        stacked = np.stack(out, axis=0).astype(np.uint8)
        self.viz_data[key] = stacked
        return stacked

    def apply_filter(self, frame, filt: str):
        """
        Apply a specified image filter to a single frame.
        :return: (H, W) uint8 grayscale image.
        """
        if frame is None:
            return None

        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray = arr[..., 0]
        else:
            gray = arr

        # Normalize to [0, 255] for standard OpenCV filter behavior
        gray_f = gray.astype(np.float32)
        mn_val, mx_val = float(np.nanmin(gray_f)), float(np.nanmax(gray_f))
        if mx_val > mn_val:
            gray_f = (gray_f - mn_val) / (mx_val - mn_val) * 255.0
        gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)

        filt = (filt or "None").lower()
        if filt in ("none", "raw"):
            return gray_u8
        if filt in ("gaus", "gaussian"):
            return cv2.GaussianBlur(gray_u8, (0, 0), 1.2)
        if filt in ("sobel",):
            sx_grad = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
            sy_grad = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(sx_grad, sy_grad)
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if filt in ("edge", "canny"):
            return cv2.Canny(gray_u8, 50, 150)

        return gray_u8

    def _update_points_label(self):
        n_base = (
            int(self.base_grid_active.sum()) if self.base_grid_active is not None else 0
        )
        n_custom = len(self.custom_grid_pts) if self.custom_grid_pts else 0
        self.points_label.setText(
            f"Points: {n_base + n_custom} "
        )  # (base {n_base} + custom {n_custom})

    def _get_video_input_for_tracking(self):
        """
        Prepare the video input (potentially filtered) for the tracking model.
        :return: torch.Tensor of shape (T, C, H, W).
        """
        filt = (self.current_filter or "None").lower()
        if filt in ("none", "raw"):
            return self.video_data

        # Build/retrieve filtered video as (T, H, W) uint8
        filt_vid = self._get_filtered_video_for_filter(self.current_filter)
        if filt_vid is None:
            return self.video_data

        # Convert to torch (T, 1, H, W) and repeat for 3-channel models
        tens = torch.from_numpy(filt_vid).unsqueeze(1).float()
        tens = tens.repeat(1, 3, 1, 1)

        # Normalize to matched input range (heuristic: check if original is [0, 1])
        try:
            v_ref = self.video_data
            if isinstance(v_ref, LazyVideo):
                tens = tens / 255.0  # LazyVideo is always normalised to [0, 1]
            elif isinstance(v_ref, torch.Tensor) and v_ref.dtype.is_floating_point:
                ref_max = float(v_ref.detach().max().cpu())
                if ref_max <= 1.5:
                    tens = tens / 255.0
        except (TypeError, RuntimeError):
            pass

        return tens

    # ---------- drag&drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    self.video_path = file_path
                    self.status_label.setText(
                        f"Selected: {os.path.basename(self.video_path)}"
                    )

                    settings = self._show_load_settings_dialog()
                    if settings is None:
                        self.status_label.setText("Load canceled.")
                        return

                    self.load_data(load_settings=settings)
                    event.acceptProposedAction()
                    return

    def open_parameter_dialog(self):
        self.param_dialog.show()
        self.param_dialog.raise_()
        self.param_dialog.activateWindow()

    # -----Save Logic-----
    def save_results(self):
        """Save corrected outputs as HDF5 and/or session as .coreg."""
        available = {
            "Raw input (current slice)": self.video_data,
            "CoReg": self.warped_video,
            "ANTs": self.ants_video,
            "LDDMM": self.lddmm_video,
            "NoRMCorre": self.normcorre_video,
            "PyFlowReg": self.flowreg_video,
        }
        available = {k: v for k, v in available.items() if v is not None}

        # Build selection dialog
        sel_dlg = QDialog(self)
        sel_dlg.setWindowTitle("Save")
        sel_layout = QVBoxLayout(sel_dlg)

        tiff_checks: dict[str, QCheckBox] = {}
        if available:
            sel_layout.addWidget(QLabel("Corrected outputs (HDF5):"))
            for name in available:
                chk = QCheckBox(name)
                chk.setChecked(True)
                sel_layout.addWidget(chk)
                tiff_checks[name] = chk
            sel_layout.addWidget(QLabel(""))  # spacer

        sel_layout.addWidget(QLabel("Session:"))
        session_chk = QCheckBox("Save session (.coreg)")
        session_chk.setChecked(True)
        sel_layout.addWidget(session_chk)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(sel_dlg.accept)
        cancel_btn.clicked.connect(sel_dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        sel_layout.addLayout(btn_row)

        if sel_dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Save canceled.")
            return

        selected_tiffs = [n for n, c in tiff_checks.items() if c.isChecked()]
        save_session = session_chk.isChecked()

        if not selected_tiffs and not save_session:
            self.status_label.setText("Nothing selected for saving.")
            return

        # Ask for base path (drive the dialog from the TIFF side when applicable)
        suggested = "output_corrected.h5"
        if getattr(self, "video_path", None):
            dir_in = os.path.dirname(self.video_path)
            base_in = os.path.splitext(os.path.basename(self.video_path))[0]
            suggested = os.path.join(dir_in, f"{base_in}_corrected.h5")

        base_path, _ = QFileDialog.getSaveFileName(
            self, "Choose Base Output Path", suggested, "HDF5 File (*.h5 *.hdf5)"
        )
        if not base_path:
            self.status_label.setText("Save canceled.")
            return

        root_p, ext_p = os.path.splitext(base_path)
        if ext_p.lower() not in [".h5", ".hdf5"]:
            ext_p = ".h5"

        saved_paths = []
        try:
            if selected_tiffs:
                from utils.utils import save_video_hdf5

                for name in selected_tiffs:
                    if len(selected_tiffs) == 1:
                        out_path = root_p + ext_p
                    else:
                        out_path = f"{root_p}_{name.replace(' ', '_')}{ext_p}"
                    data = available[name]
                    if isinstance(data, LazyVideo):
                        data = data.materialize()
                    save_video_hdf5(data, out_path)
                    saved_paths.append(out_path)

            if save_session:
                coreg_path = root_p + ".coreg"
                self._write_coreg(coreg_path)
                saved_paths.append(coreg_path)

            self.status_label.setText(f"Saved {len(saved_paths)} file(s).")

        except Exception as err:
            self.status_label.setText(f"Save error: {err}")

    # ------------------------------------------------------------------
    # single-frame image export
    # ------------------------------------------------------------------

    def _frame_array_for_export(self, which: str, idx: int):
        """Return the raw (H,W[,C]) array shown by view *which* at frame
        *idx*, or None if that view has nothing at this frame. Mirrors the
        per-view lookups in _render_view/update_plot, kept separate (rather
        than refactoring those) so this new export path can't regress the
        already-working live display."""
        which = (which or "").strip()

        if which in ("Input", "Tracked Points"):
            if isinstance(self.video_data, LazyVideo):
                original_frame = self.video_data.get_frame_hwc(idx)
            else:
                if "input" not in self.viz_data:
                    self.viz_data["input"] = (
                        self.video_data.permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                original_frame = self.viz_data["input"][idx]
            if which == "Input":
                return original_frame
            fv = self._get_filtered_video_for_filter(self.current_filter)
            return fv[idx] if fv is not None else self.apply_filter(original_frame, "None")

        if which == "CoReg":
            warped = self.viz_data.get("warped", None)
            return warped[idx] if warped is not None and idx < len(warped) else None

        if which == "Deformation Field":
            if self.disp_field is None:
                return None
            disp_frame = (
                self.disp_field[idx] if idx < len(self.disp_field) else self.disp_field[0]
            )
            return self._make_disp_preview(disp_frame)

        video_attr = {
            "ANTs": "ants_video",
            "LDDMM": "lddmm_video",
            "NoRMCorre": "normcorre_video",
            "PyFlowReg": "flowreg_video",
        }.get(which)
        if video_attr is not None:
            v = getattr(self, video_attr, None)
            return v[idx] if v is not None and idx < len(v) else None

        return None

    def _frame_to_uint8(self, frame, already_display_ready: bool = False) -> np.ndarray:
        """Convert a raw display frame to a uint8 (H,W) or (H,W,3) array ready
        to save as an image. Percentile-normalized the same way _calc_levels
        drives on-screen contrast, unless *already_display_ready* (the
        Deformation Field preview is already a color-mapped uint8 image)."""
        arr = _to_numpy(frame)
        arr = np.squeeze(arr)  # (H,W,1) -> (H,W)

        if already_display_ready:
            return arr.astype(np.uint8)

        levels = self._calc_levels(arr)
        if levels is None:
            return np.zeros(arr.shape[:2], dtype=np.uint8)
        lo, hi = levels
        norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return (norm * 255).astype(np.uint8)

    def _read_micron_per_pixel(self) -> float | None:
        """Read the ImageJ XResolution tag (pixels per micron) off the
        source TIFF, if the loaded video came from one and the tag is
        present. Same convention figures/cell_crops_clean.py's
        _micron_per_pixel uses, so exported frames read consistently with
        the offline figure scripts. Returns None for non-TIFF sources or a
        missing tag -- callers fall back to a plain pixel-count scale bar."""
        path = getattr(self, "video_path", None)
        if not path or os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
            return None
        try:
            with tifffile.TiffFile(path) as tf:
                tag = tf.pages[0].tags.get("XResolution")
                if tag is None:
                    return None
                num, den = tag.value
                if num == 0:
                    return None
                return den / num
        except Exception:
            return None

    @staticmethod
    def _nice_length(max_len: float) -> float:
        """Largest 1/2/5 x 10^k that is <= max_len. Same rounding
        figures/method_meanstd.py's add_scale_bar uses, so bar lengths
        read as familiar round numbers instead of odd fractions."""
        if max_len <= 0:
            return 0.0
        base = 10 ** math.floor(math.log10(max_len))
        for m in (5, 2, 1):
            if m * base <= max_len:
                return m * base
        return base

    @staticmethod
    def _proportional_pt(frac_of_width: float, img_w_px: int, dpi: int) -> float:
        """Convert a size given as a fraction of the image's pixel width into
        matplotlib points, so markers/lines/text render at the same
        proportion of the image regardless of the frame's resolution or the
        chosen export DPI -- a fixed point size instead looks tiny on a
        large frame and oversized on a small crop, and shrinks relative to
        the image as DPI increases (since points are a fixed physical size)."""
        return max(0.1, frac_of_width * img_w_px / dpi * 72.0)

    @staticmethod
    def _trail_color(u: float) -> tuple[float, float, float]:
        """Yellow (u=0, oldest) -> purple (u=1, newest) trail gradient --
        matches PGImagePanel._trail_color (gui.py), the same coloring
        set_trajectories uses for the on-screen live trajectory trail, so
        exported tracks read identically to what's shown in the viewer."""
        yellow = (255, 255, 0)
        purple = (160, 0, 255)
        r = yellow[0] + (purple[0] - yellow[0]) * u
        g = yellow[1] + (purple[1] - yellow[1]) * u
        b = yellow[2] + (purple[2] - yellow[2]) * u
        return r / 255.0, g / 255.0, b / 255.0

    def _draw_scale_bar(self, ax, src_w_px: int, um_per_px: float | None, dpi: int) -> None:
        """White scale bar in the bottom-left, sized to ~1/4 of the image
        width and rounded to a nice number -- in microns if the source
        TIFF's resolution tag is known, else in pixels. Axes-fraction
        positioned, and line width/font size derived from the image width
        (see _proportional_pt), so it looks right at any DPI/output size."""
        if um_per_px:
            um = self._nice_length(0.25 * src_w_px * um_per_px)
            length_px, label = um / um_per_px, f"{um:g} µm"
        else:
            length_px = self._nice_length(0.25 * src_w_px)
            label = f"{length_px:g} px"
        x0, y0 = 0.05, 0.045
        lw = self._proportional_pt(0.006, src_w_px, dpi)
        fontsize = self._proportional_pt(0.035, src_w_px, dpi)
        ax.plot(
            [x0, x0 + length_px / src_w_px], [y0, y0], transform=ax.transAxes,
            color="white", lw=lw, solid_capstyle="butt", clip_on=False,
        )
        ax.text(
            x0, y0 + 0.02, label, transform=ax.transAxes, color="white",
            fontsize=fontsize, ha="left", va="bottom",
        )

    def _points_overlay_for_export(self, idx: int):
        """Return the point/trajectory overlay shown in the Tracked Points
        view at frame idx, styled to match _render_view's on-screen colors,
        or None if there's nothing to draw. Tracked points (once tracking
        has run) are pure green, matching set_points_colored's (0,255,0)
        there; before tracking, falls back to the same neon-green (57,255,20)
        preview points/crosses draw_points_preview shows."""
        tracks = self.viz_data.get("tracks", None)
        if tracks is not None:
            tr = np.asarray(tracks)
            if tr.ndim == 4:
                tr = tr[0]
            if tr.ndim == 3 and idx < tr.shape[0]:
                overlay = {
                    "points": tr[idx],
                    "point_color": (0.0, 1.0, 0.0, 1.0),
                    "point_size": 8,
                }
                if self.show_tracks_chk.isChecked() and idx > 0:
                    # set_trajectories needs tr[t-1] for the first segment (t=1) -- array must start at frame 0.
                    overlay["trajectories"] = tr[: idx + 1]
                    overlay["current_frame"] = idx
                return overlay

        pts_active = None
        if (
            self.base_grid_pts is not None
            and self.base_grid_active is not None
            and np.any(self.base_grid_active)
        ):
            pts_active = self.base_grid_pts[self.base_grid_active]
        crosses = np.asarray(self.custom_grid_pts, dtype=np.float32) if self.custom_grid_pts else None
        if pts_active is None and crosses is None:
            return None

        neon_green = (57 / 255, 255 / 255, 20 / 255, 1.0)
        return {
            "points": pts_active,
            "point_color": neon_green,
            "point_size": 8,
            "crosses": crosses,
            "cross_color": neon_green,
        }

    def _save_frame_image(
        self, img_u8: np.ndarray, out_path: str, dpi: int, um_per_px: float | None,
        draw_scale_bar: bool = True, overlay: dict | None = None,
    ) -> None:
        """Save a uint8 (H,W) or (H,W,3) array as an image at the given DPI,
        optionally with a scale bar and a points/trajectories overlay (see
        _points_overlay_for_export). PNG/TIFF are rasterized at exactly the
        source pixel size; PDF/SVG keep the scale bar and overlay markers as
        real editable vector elements (matches the pdf.fonttype/svg.fonttype
        convention used throughout figures/*.py) around an embedded raster
        of the frame itself."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["svg.fonttype"] = "none"

        h, w = img_u8.shape[:2]
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        cmap = "gray" if img_u8.ndim == 2 else None
        ax.imshow(img_u8, cmap=cmap, vmin=0, vmax=255)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        if overlay:
            traj = overlay.get("trajectories")
            cf = overlay.get("current_frame", 0)
            if traj is not None and traj.shape[0] > 1 and cf > 0:
                # Mirrors PGImagePanel.set_trajectories's loop exactly, so the same yellow->purple age gradient appears.
                lw = self._proportional_pt(0.003, w, dpi)
                T = traj.shape[0]
                start_t, end_t = 1, min(cf, T - 1)
                den = max(1, end_t - start_t + 1)
                for t in range(start_t, end_t + 1):
                    u = (t - start_t) / den
                    color = self._trail_color(u)
                    p1, p2 = traj[t - 1], traj[t]  # (N, 2) each
                    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
                    if not np.any(finite):
                        continue
                    xs = np.stack([p1[finite, 0], p2[finite, 0], np.full(finite.sum(), np.nan)], axis=1).ravel()
                    ys = np.stack([p1[finite, 1], p2[finite, 1], np.full(finite.sum(), np.nan)], axis=1).ravel()
                    ax.plot(xs, ys, color=color, lw=lw, zorder=2)

            point_d = self._proportional_pt(0.018, w, dpi)
            pts = overlay.get("points")
            if pts is not None and len(pts) > 0:
                p = np.asarray(pts)
                ax.scatter(
                    p[:, 0], p[:, 1], s=point_d ** 2,
                    color=overlay.get("point_color", (0.0, 1.0, 0.0, 1.0)),
                    edgecolors="none", zorder=3,
                )

            crosses = overlay.get("crosses")
            if crosses is not None and len(crosses) > 0:
                c = np.asarray(crosses)
                ax.scatter(
                    c[:, 0], c[:, 1], s=point_d ** 2, marker="x",
                    color=overlay.get("cross_color", (0.0, 1.0, 0.0, 1.0)), zorder=3,
                )

        if draw_scale_bar:
            self._draw_scale_bar(ax, w, um_per_px, dpi)

        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    def export_frame_dialog(self) -> None:
        """Ask which panel/DPI/format to use, then save the currently
        displayed frame -- PNG/TIFF (raster) or PDF/SVG (vector) -- at that
        DPI, with an optional scale bar (microns if the source TIFF's
        resolution tag is known, else pixels)."""
        if self.video_data is None:
            self.status_label.setText("Load a video before exporting a frame.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Export Frame")
        layout = QVBoxLayout(dlg)

        total = int(self.video_data.shape[0])
        layout.addWidget(QLabel(f"Frame {self.current_frame} / {total - 1}"))

        layout.addWidget(QLabel("View:"))
        view_combo = QComboBox()
        view_combo.addItems(
            [
                f"Left panel ({self.view_left_combo.currentText()})",
                f"Right panel ({self.view_right_combo.currentText()})",
            ]
        )
        layout.addWidget(view_combo)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI:"))
        dpi_spin = QSpinBox()
        dpi_spin.setRange(50, 2400)
        dpi_spin.setValue(300)
        dpi_row.addWidget(dpi_spin)
        layout.addLayout(dpi_row)

        scale_bar_chk = QCheckBox("Scale bar")
        scale_bar_chk.setChecked(True)
        um_per_px = self._read_micron_per_pixel()
        scale_bar_chk.setToolTip(
            "In microns (from the source TIFF's resolution tag)."
            if um_per_px else
            "No pixel-size metadata found for this source -- shown in pixels instead."
        )
        layout.addWidget(scale_bar_chk)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Export...")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Export canceled.")
            return

        which = (
            self.view_left_combo.currentText()
            if view_combo.currentIndex() == 0
            else self.view_right_combo.currentText()
        )
        dpi = dpi_spin.value()
        want_scale_bar = scale_bar_chk.isChecked()

        frame = self._frame_array_for_export(which, self.current_frame)
        if frame is None:
            self.status_label.setText(f"Nothing to export for '{which}' at frame {self.current_frame}.")
            return
        overlay = self._points_overlay_for_export(self.current_frame) if which == "Tracked Points" else None

        base = (
            os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "frame"
        )
        suggested = f"{base}_{which.replace(' ', '_')}_f{self.current_frame:04d}.png"

        export_filters = [
            ("PNG Image (*.png)", ".png"),
            ("TIFF Image (*.tif *.tiff)", ".tif"),
            ("PDF vector (*.pdf)", ".pdf"),
            ("SVG vector (*.svg)", ".svg"),
        ]
        out_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Frame", suggested, ";;".join(f for f, _ in export_filters),
        )
        if not out_path:
            self.status_label.setText("Export canceled.")
            return

        # Some Qt/WM combos don't rewrite the typed extension when switching the filter dropdown -- force it.
        wanted_ext = dict(export_filters).get(selected_filter)
        if wanted_ext:
            valid_exts = (".tif", ".tiff") if wanted_ext == ".tif" else (wanted_ext,)
            if os.path.splitext(out_path)[1].lower() not in valid_exts:
                out_path = os.path.splitext(out_path)[0] + wanted_ext

        try:
            img_u8 = self._frame_to_uint8(frame, already_display_ready=(which == "Deformation Field"))
            self._save_frame_image(
                img_u8, out_path, dpi=dpi, um_per_px=um_per_px,
                draw_scale_bar=want_scale_bar, overlay=overlay,
            )
            self.status_label.setText(
                f"Exported {which} frame {self.current_frame} to {out_path} at {dpi} DPI."
            )
        except Exception as err:
            self.status_label.setText(f"Export error: {err}")

    # ------------------------------------------------------------------
    # .coreg session serialisation
    # ------------------------------------------------------------------

    def _collect_session_config(self) -> dict:
        """Collect all scalar/dict configuration into a JSON-serialisable dict."""
        return {
            "version": 1,
            "video_path": self.video_path,
            "frame_start": self._frame_start,
            "frame_end": self._frame_end,
            "fps": self.fps,
            "current_filter": self.current_filter,
            "preset_index": self.preset_combo.currentIndex(),
            # point-preset parameters
            "num_points": self.num_points_spin.value(),
            "min_dist": self.min_dist_spin.value(),
            "match_min_score": self.match_min_score_spin.value(),
            # grid parameters
            "grid_rows": self.grid_rows_ctrl.value(),
            "grid_cols": self.grid_cols_ctrl.value(),
            "grid_spacing": self.grid_spacing_spin.value(),
            "grid_offset_x": self.grid_offset_x_spin.value(),
            "grid_offset_y": self.grid_offset_y_spin.value(),
            # displacement field
            "df_sigma": self.df_sigma_spin.value(),
            # greedy-motion parameters
            "gm_sigma": self.gm_sigma_spin.value(),
            "gm_gamma": self.gm_gamma_spin.value(),
            "gm_bg_weight": self.gm_bg_weight_spin.value(),
            "gm_peak_min_dist": self.gm_peak_min_dist_spin.value(),
            "gm_peak_thr": self.gm_peak_thr_spin.value(),
            "gm_peak_frac": self.gm_peak_frac_spin.value(),
            "gm_cand_factor": self.gm_cand_factor_spin.value(),
            "gm_samp_dist": self.gm_samp_dist_spin.value(),
            "gm_seed": self.gm_seed_spin.value(),
            # method-specific parameter dicts
            "ants_params": self.ants_params,
            "normcorre_params": {
                k: (list(v) if isinstance(v, (tuple, list)) else v)
                for k, v in self.normcorre_params.items()
            },
            "pyflowreg_params": self.pyflowreg_params,
            # UI state
            "force_cpu": self.force_cpu_chk.isChecked(),
            "show_tracks": self.show_tracks_chk.isChecked(),
            "view_left": self.view_left_combo.currentText(),
            "view_right": self.view_right_combo.currentText(),
        }

    def _restore_session_config(self, cfg: dict) -> None:
        """Apply a config dict (from _collect_session_config) back to the UI."""
        if cfg.get("video_path"):
            self.video_path = cfg["video_path"]
        self._frame_start = cfg.get("frame_start", 0)
        self._frame_end = cfg.get("frame_end", None)
        self.fps = cfg.get("fps", 30)

        filt = cfg.get("current_filter", "None")
        self.current_filter = filt
        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentText(filt)
        self.filter_combo.blockSignals(False)

        idx = cfg.get("preset_index", 0)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

        self.num_points_spin.setValue(cfg.get("num_points", 50))
        self.min_dist_spin.setValue(cfg.get("min_dist", 10))
        self.match_min_score_spin.setValue(cfg.get("match_min_score", 0.20))

        self.grid_rows_ctrl.setValue(cfg.get("grid_rows", 4))
        self.grid_cols_ctrl.setValue(cfg.get("grid_cols", 4))
        self.grid_spacing_spin.setValue(cfg.get("grid_spacing", 0))
        self.grid_offset_x_spin.setValue(cfg.get("grid_offset_x", 0))
        self.grid_offset_y_spin.setValue(cfg.get("grid_offset_y", 0))

        self.df_sigma_spin.setValue(cfg.get("df_sigma", 5))

        self.gm_sigma_spin.setValue(cfg.get("gm_sigma", 1.8))
        self.gm_gamma_spin.setValue(cfg.get("gm_gamma", 0.5))
        self.gm_bg_weight_spin.setValue(cfg.get("gm_bg_weight", 0.015))
        self.gm_peak_min_dist_spin.setValue(cfg.get("gm_peak_min_dist", 3))
        self.gm_peak_thr_spin.setValue(cfg.get("gm_peak_thr", 0.06))
        self.gm_peak_frac_spin.setValue(cfg.get("gm_peak_frac", 0.8))
        self.gm_cand_factor_spin.setValue(cfg.get("gm_cand_factor", 5))
        self.gm_samp_dist_spin.setValue(cfg.get("gm_samp_dist", 8.0))
        self.gm_seed_spin.setValue(cfg.get("gm_seed", 0))

        self.force_cpu_chk.setChecked(cfg.get("force_cpu", False))
        self.show_tracks_chk.setChecked(cfg.get("show_tracks", True))

        left = cfg.get("view_left", "Tracked Points")
        self.view_left_combo.blockSignals(True)
        self.view_left_combo.setCurrentText(left)
        self.view_left_combo.blockSignals(False)

        right = cfg.get("view_right", "CoReg")
        self.view_right_combo.blockSignals(True)
        self.view_right_combo.setCurrentText(right)
        self.view_right_combo.blockSignals(False)

        if "ants_params" in cfg:
            self.ants_params.update(cfg["ants_params"])
        if "normcorre_params" in cfg:
            restored = {}
            for k, v in cfg["normcorre_params"].items():
                # tuples were serialised as lists
                if k in ("max_shifts", "strides", "overlaps") and isinstance(v, list):
                    v = tuple(v)
                restored[k] = v
            self.normcorre_params.update(restored)
        if "pyflowreg_params" in cfg:
            self.pyflowreg_params.update(cfg["pyflowreg_params"])

    def _write_coreg(self, path: str) -> None:
        """Write session config, points and tracks to *path* (.coreg zip archive)."""
        cfg = self._collect_session_config()

        def _arr(a):
            if a is None:
                return None
            if isinstance(a, torch.Tensor):
                return a.detach().cpu().numpy()
            return np.asarray(a)

        arrays: dict[str, np.ndarray] = {}

        if self.base_grid_pts is not None:
            arrays["points_base"] = _arr(self.base_grid_pts)

        if self.custom_grid_pts:
            stacked = [_arr(p) for p in self.custom_grid_pts if p is not None]
            if stacked:
                arrays["points_custom"] = np.stack(stacked, axis=0)

        if self.all_query_points_at_frame0 is not None:
            arrays["query_points"] = _arr(self.all_query_points_at_frame0)

        if self.pred_tracks is not None:
            arrays["tracks"] = _arr(self.pred_tracks)

        if self.pred_visibility is not None:
            arrays["visibility"] = _arr(self.pred_visibility)

        if self.warped_video is not None:
            arrays["warped_video"] = _arr(self.warped_video)

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.json", json.dumps(cfg, indent=2))
            for name, arr in arrays.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{name}.npy", buf.getvalue())

    def load_coreg(self) -> None:
        """Restore session configuration, points and tracks from a .coreg file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "CoReg Session (*.coreg)"
        )
        if not path:
            return

        try:
            with zipfile.ZipFile(path, "r") as zf:
                cfg = json.loads(zf.read("config.json"))

                def _load(name):
                    entry = f"{name}.npy"
                    if entry in zf.namelist():
                        return np.load(io.BytesIO(zf.read(entry)), allow_pickle=False)
                    return None

                base_pts  = _load("points_base")
                custom_raw = _load("points_custom")
                query_pts  = _load("query_points")
                tracks     = _load("tracks")
                visibility = _load("visibility")
                warped     = _load("warped_video")

            self._restore_session_config(cfg)

            # Reload video first -- load_data calls apply_point_preset, which would overwrite saved points.
            stored_path = cfg.get("video_path")
            if stored_path and os.path.isfile(stored_path):
                self.video_path = stored_path
                self.load_data()

            # Constrain to frames actually processed. tracks: (T, N, 2); warped: (T, H, W, C).
            n_frames = None
            if tracks is not None and tracks.ndim >= 2:
                n_frames = int(tracks.shape[0])
            elif warped is not None and warped.ndim >= 1:
                n_frames = int(warped.shape[0])

            if self._original_video_data is not None:
                total = int(self._original_video_data.shape[0])
                # load_data() resets _frame_start/_frame_end to 0/None, undoing what _restore_session_config just
                # restored -- re-read the saved slice from cfg instead of assuming frame 0.
                start = int(cfg.get("frame_start") or 0)
                end = cfg.get("frame_end")
                if n_frames is not None:
                    # Saved tracks/warp define exactly which frames were processed.
                    end = start + n_frames - 1
                # No tracks/warp array (session saved pre-tracking) -- slice must come from the saved frame range,
                # or it silently loads as the full video.
                if start > 0 or end is not None:
                    self._frame_start = max(0, min(start, total - 1))
                    self._frame_end = None if end is None else min(int(end), total - 1)
                    self._apply_frame_slice()  # updates video_data, slider, cache

            # Re-apply saved points/tracks after load_data() has finished
            if base_pts is not None:
                self.base_grid_pts = base_pts
                self.base_grid_active = np.ones(len(base_pts), dtype=bool)
                self.base_grid_tracked = np.zeros(len(base_pts), dtype=bool)
            if custom_raw is not None:
                self.custom_grid_pts = [custom_raw[i] for i in range(custom_raw.shape[0])]
                self.custom_grid_tracked = [False] * len(self.custom_grid_pts)
            if query_pts is not None:
                self.all_query_points_at_frame0 = query_pts
            if tracks is not None:
                self.pred_tracks = tracks
                self.viz_data["tracks"] = tracks
            if visibility is not None:
                self.pred_visibility = visibility
            if warped is not None:
                self.warped_video = warped
                self.viz_data["warped"] = warped

            self._update_points_label()
            self.update_plot()

            self.status_label.setText(f"Session loaded from {path}")

        except Exception as err:
            self.status_label.setText(f"Load session error: {err}")
