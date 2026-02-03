import os
import numpy as np
import torch
import cv2
import csv
import torch.nn.functional as F
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QCheckBox, QGridLayout, QComboBox, QSpinBox, QSizePolicy, QProgressBar, QDialog,
     QDoubleSpinBox,  QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPalette
import pyqtgraph as pg
from gui import ParameterDialog, PGImagePanel, MotionProfileWidget, ProcessingWorker, VideoLoadSettingsDialog

# Add current directory to path to allow imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import load_video, load_tiff, load_hdf5, probe_media_info, resize_video_tensor_to_wh
pg.setConfigOptions(imageAxisOrder='row-major')  # damit (H,W) wie numpy angezeigt wird
from utils.utils import video_ssim_score

class MotionCorrectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Correction GUI")
        screen = QApplication.primaryScreen()
        g = screen.availableGeometry()
        self._initial_zoom_done = False

        w = int(g.width() * 0.8)
        h = int(g.height() * 0.8)
        self.resize(w, h)
        self.move(g.left() + (g.width() - w)//2, g.top() + (g.height() - h)//2)

        self.setAcceptDrops(True)
        self._tracks_user_edited = False
        self.video_path = None
        self.video_data = None

        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.ants_video = None
        self.lddmm_video = None
        self.disp_field = None

        self.grid_rows = 4
        self.grid_cols = 4
        self.base_grid_pts = None
        self.base_grid_active = None
        self.base_grid_pts0 = None
        self.custom_grid_pts = []

        self._drag = None
        self._img_cids = {}

        self.edit_points_mode = False
        self.edit_tracks_mode = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0

        self.viz_data = {}

        self.worker = None
        self.current_filter = "None"

        self.ssim_score = None
        self.track_active = None
        self.selected_track_idx = None
        self.base_grid_tracked = None          # bool array same length as base_grid_pts
        self.custom_grid_tracked = []          # list[bool] same length as custom_grid_pts

        self.track_meta = []            # list[tuple[str,int]] size N_tracks
        self._pending_track_meta = []   # meta for the queries in the current run


        # ---- instant auto-recalc warping (debounced) ----
        self._recalc_timer = QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.timeout.connect(self._recalculate_warping_from_tracks)
        self._recalc_inflight = False
        self._pending_recalc = False

        # ---- debounce config: 5s after last edit ----
        self._debounce_ms = 5000
        self._edit_clock = QElapsedTimer()
        self._edit_clock.start()
        self._target_recalc_time_ms = 0  # absolute time in elapsed-ms domain
        
        # Incremental tracking state
        self.all_query_points_at_frame0 = None # (N, 2) matches pred_tracks

        self.init_ui()

    def _schedule_recalc_warping(self, delay_ms=None):
        if self.video_data is None or self.pred_tracks is None:
            return

        if delay_ms is None:
            delay_ms = self._debounce_ms

        # Setze "Deadline" = jetzt + delay
        now = self._edit_clock.elapsed()
        self._target_recalc_time_ms = now + int(max(0, delay_ms))

        # wenn gerade ein recalc läuft, nur merken: nach dem Ende erneut versuchen
        if self._recalc_inflight:
            self._pending_recalc = True
            return

        # Timer neu starten auf "restliche Zeit bis Deadline"
        remaining = max(0, self._target_recalc_time_ms - self._edit_clock.elapsed())
        self._recalc_timer.stop()
        self._recalc_timer.start(int(remaining))



    def _get_video_input_for_warping(self):
        """
        Fast-path: wir nehmen für instant recalc das RAW video_data (torch).
        (Falls du unbedingt den Filter auch fürs Warping willst, sag’s – dann cache ich filtered tensors.)
        """
        return self.video_data

    # ---------- display helpers ----------
    @staticmethod
    def flow_to_color(flow):
        if flow is None:
            return None
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        if mag.max() > 0:
            return (mag / mag.max() * 255.0).astype(np.uint8)
        return np.zeros_like(mag, dtype=np.uint8)

    def disp_scalar_u8(self, disp_hw, clip=None):
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
        if self.pred_tracks is None:
            return None
        if isinstance(self.pred_tracks, np.ndarray):
            return self.pred_tracks
        try:
            return self.pred_tracks.detach().cpu().numpy()
        except Exception:
            return np.asarray(self.pred_tracks)
    
   
    # ---------- UI ----------
    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- Top Bar Controls ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        second_bar = QHBoxLayout()
        second_bar.setSpacing(0)
        second_bar.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(top_bar)
        main_layout.addLayout(second_bar)

        # 1. Load File
        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.select_file)
        self.load_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        top_bar.addWidget(self.load_btn)

        # 2. Filter
        top_bar.addWidget(QLabel("Input Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Gaussian", "Sobel", "Canny"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        top_bar.addWidget(self.filter_combo)

        # 3a. Edit Points (grid/custom)
        self.edit_points_btn = QCheckBox("Edit Points")
        self.edit_points_btn.stateChanged.connect(self.on_edit_modes_changed)
        self.edit_points_btn.setEnabled(False)
        second_bar.addWidget(self.edit_points_btn)

        # 3b. Edit Tracks (post-registration)
        self.edit_tracks_btn = QCheckBox("Edit Tracks")
        self.edit_tracks_btn.stateChanged.connect(self.on_edit_modes_changed)
        self.edit_tracks_btn.setEnabled(False)  # enabled only when tracks exist
        second_bar.addWidget(self.edit_tracks_btn)

        # self.apply_all_tracks_chk = QCheckBox("Apply to all")
        # self.apply_all_tracks_chk.setEnabled(False)
        # top_bar.addWidget(self.apply_all_tracks_chk)

        # 4. Preset
        top_bar.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "None (no points)",
            "Regular grid",
            "Features (Shi–Tomasi)",
            "Activity map (motion)",
            "Segmentation centroids"
        ])
        self._last_preset_index = self.preset_combo.currentIndex()
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        top_bar.addWidget(self.preset_combo)

        # 5. Parameters
        self.settings_btn = QPushButton("⚙️ Parameters")
        self.settings_btn.clicked.connect(self.open_parameter_dialog)
        top_bar.addWidget(self.settings_btn)

        # 6. Run Registration
        self.process_btn = QPushButton("Registration")
        self.process_btn.clicked.connect(self.visualize)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        top_bar.addWidget(self.process_btn)

        # 7. Stop
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        top_bar.addWidget(self.stop_btn)

        # 8. Recalculate
        self.recalculate_btn = QPushButton("Recalculate Warping")
        self.recalculate_btn.clicked.connect(self.recalculate_warping)
        self.recalculate_btn.setEnabled(False)
        top_bar.addWidget(self.recalculate_btn)

        # 8. Save
        self.save_btn = QPushButton("💾 Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)  # Enabled when results exist
        top_bar.addWidget(self.save_btn)
        
        # Add filler to push buttons to left (optional)
        top_bar.addStretch(1)

        # --- Parameter Widgets (Detailed) ---
        # These are created here but added to the ParameterDialog instead of main window
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(10, 5000)
        self.num_points_spin.setValue(400)
        self.num_points_spin.valueChanged.connect(self.on_preset_params_changed)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(1, 100)
        self.min_dist_spin.setValue(10)
        self.min_dist_spin.valueChanged.connect(self.on_preset_params_changed)

        self.grid_rows_ctrl = QSpinBox()
        self.grid_rows_ctrl.setRange(2, 100)
        self.grid_rows_ctrl.setValue(int(getattr(self, "grid_rows", 4)))
        self.grid_rows_ctrl.valueChanged.connect(self.on_preset_params_changed)

        self.grid_cols_ctrl = QSpinBox()
        self.grid_cols_ctrl.setRange(2, 100)
        self.grid_cols_ctrl.setValue(int(getattr(self, "grid_cols", 4)))
        self.grid_cols_ctrl.valueChanged.connect(self.on_preset_params_changed)

        # self.apply_preset_btn = QPushButton("Apply to Grid")
        # self.apply_preset_btn.clicked.connect(self.apply_point_preset)

        self.df_sigma_spin = QSpinBox()
        self.df_sigma_spin.setRange(0, 20)
        self.df_sigma_spin.setValue(3)
        self.df_sigma_spin.valueChanged.connect(lambda *_: self.update_plot())
        
        
        self.rigid_motion_chk = QCheckBox("Rigid motion (apply track to grid)")
        self.rigid_motion_chk.setToolTip("When checked, click a track to apply its motion to a regular grid.")
        
        self.lbl_N = QLabel("Number of Points:")
        self.lbl_minDist = QLabel("Min Distance:")
        self.lbl_rows = QLabel("Rows:")
        self.lbl_cols = QLabel("Cols:")
          
        self.sync_view_chk = QCheckBox("Sync view")
        self.sync_view_chk.stateChanged.connect(self.on_sync_view_toggled)

        self.clear_points_btn = QPushButton("Clear Custom Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)

        self.undo_btn = QPushButton("Reset to Preset")
        self.undo_btn.clicked.connect(self.undo_all_changes)
        self.undo_btn.setEnabled(False)
        
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_all_zoom)


        # --- Initialize Dialog ---
        self.param_dialog = ParameterDialog(self)
        pd_layout = self.param_dialog.layout

        # Add widgets to dialog layout
        
        # Parameters section
        pd_layout.addWidget(self.lbl_rows)
        pd_layout.addWidget(self.grid_rows_ctrl)
        pd_layout.addWidget(self.lbl_cols)
        pd_layout.addWidget(self.grid_cols_ctrl)
        
        pd_layout.addWidget(self.lbl_N)
        pd_layout.addWidget(self.num_points_spin)
        pd_layout.addWidget(self.lbl_minDist)
        pd_layout.addWidget(self.min_dist_spin)
        
        
        self.rotate_btn = QPushButton("Rotate Input 90°")
        self.rotate_btn.clicked.connect(self.rotate_input)

        second_bar.addWidget(self.sync_view_chk)   
        pd_layout.addWidget(self.reset_zoom_btn)
        pd_layout.addWidget(self.rotate_btn)
        pd_layout.addWidget(self.clear_points_btn)
        #pd_layout.addWidget(self.undo_btn)

        # Move close button to bottom
        pd_layout.removeWidget(self.param_dialog.close_btn)
        pd_layout.addWidget(self.param_dialog.close_btn)

         
        self.points_label = QLabel("Points: 0")
        self.status_label = QLabel("Ready - Drag & drop video or use Load button")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.points_label) # Put points label on main window or dialog? User didn't specify. Keeping on main seems useful.
        
        # --- Two-panel viewer (left/right) with dropdown selectors ---
        viewer_widget = QWidget(central_widget)
        viewer_layout = QGridLayout(viewer_widget)
        viewer_layout.setContentsMargins(6, 6, 6, 6)
        viewer_layout.setHorizontalSpacing(8)
        viewer_layout.setVerticalSpacing(6)
        viewer_layout.setColumnStretch(0, 1)
        viewer_layout.setColumnStretch(1, 1)
        viewer_layout.setRowStretch(1, 1)

        # Left selector
        self.view_left_combo = QComboBox()
        self.view_left_combo.addItems(["Input", "Tracked Points", "Corrected", "Deformation Field"]) #, "Corrected (ANTs)", "Corrected (LDDMM)"
        self.view_left_combo.setCurrentText("Tracked Points")
        self.view_left_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        # Right selector
        self.view_right_combo = QComboBox()
        self.view_right_combo.addItems(["Input", "Tracked Points", "Corrected", "Deformation Field"]) #, "Corrected (ANTs)", "Corrected (LDDMM)"
        self.view_right_combo.setCurrentText("Corrected")
        self.view_right_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        # Panels
        self.view_left = PGImagePanel()
        self.view_right = PGImagePanel()
        self.view_left.clicked_outside_image.connect(self._exit_edit_modes_if_enabled)
        self.view_right.clicked_outside_image.connect(self._exit_edit_modes_if_enabled)
        self.view_left._parent_gui = self
        self.view_right._parent_gui = self

        viewer_layout.addWidget(self.view_left_combo, 1, 0)
        viewer_layout.addWidget(self.view_right_combo, 1, 1)
        viewer_layout.addWidget(self.view_left, 2, 0)
        viewer_layout.addWidget(self.view_right, 2, 1)

        main_layout.addWidget(viewer_widget, stretch=1)
        playback_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet("""
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
        """)

        self.step_back_btn = QPushButton("◀")
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setEnabled(False)

        self.step_fwd_btn = QPushButton("▶")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        self.step_fwd_btn.setEnabled(False)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.on_speed_change)

        self.speed_label = QLabel("1.00x")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setEnabled(False)

        self.frame_label = QLabel("0/0")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.step_back_btn)
        playback_layout.addWidget(self.step_fwd_btn)

        # Speed control with label on top
        speed_layout = QVBoxLayout()
        speed_header_layout = QHBoxLayout() # To put Speed and 1.00x next to each other
        speed_header_layout.addWidget(QLabel("Speed"))
        speed_header_layout.addWidget(self.speed_label)
        speed_layout.addLayout(speed_header_layout)
        speed_layout.addWidget(self.speed_slider)
        playback_layout.addLayout(speed_layout)

        # Frames control with label on top
        frames_layout = QVBoxLayout()
        frames_header_layout = QHBoxLayout() # To put Frames and 0/0 next to each other
        frames_header_layout.addWidget(QLabel("Frames"))
        frames_header_layout.addWidget(self.frame_label)
        frames_layout.addLayout(frames_header_layout)
        
        self.motion_plot = MotionProfileWidget()
        self.motion_plot.frame_changed.connect(self.slider.setValue)
        frames_layout.addWidget(self.motion_plot)
        
        frames_layout.addWidget(self.slider)
        playback_layout.addLayout(frames_layout, stretch=1)

        # playback_layout.addWidget(self.progress_bar) # Removed as requested

        main_layout.addLayout(playback_layout)

        self._update_preset_controls()
        self.status_label.setText("UI loaded. Click 'Load File' or drag & drop a file.")

    def _exit_edit_modes_if_enabled(self):
        if not (self.edit_points_btn.isChecked() or self.edit_tracks_btn.isChecked()):
            return

        # turn both off
        self.edit_points_btn.blockSignals(True)
        self.edit_tracks_btn.blockSignals(True)
        self.edit_points_btn.setChecked(False)
        self.edit_tracks_btn.setChecked(False)
        self.edit_points_btn.blockSignals(False)
        self.edit_tracks_btn.blockSignals(False)

        # update internal flags + disconnect handlers etc.
        self.edit_points_mode = False
        self.edit_tracks_mode = False
        self.toggle_edit_mode()

        self.status_label.setText("Edit mode off (clicked outside image).")

    def _range_str(self, a, name=""):
        if a is None:
            return f"{name}: None"
        x = np.asarray(a)
        finite = np.isfinite(x)
        if not finite.any():
            return f"{name}: no finite values"
        mn = float(np.min(x[finite]))
        mx = float(np.max(x[finite]))
        return f"{name}: dtype={x.dtype}, min={mn:.3f}, max={mx:.3f}"

    def _get_minmax(self, a):
        x = np.asarray(a)
        finite = np.isfinite(x)
        if not finite.any():
            return None, None
        return float(np.min(x[finite])), float(np.max(x[finite]))

    def on_sync_view_toggled(self, state):
        master_vb = self.view_left.getViewBox()
        vb_r = self.view_right.getViewBox()

        if state == Qt.CheckState.Checked.value:
            vb_r.setXLink(master_vb)
            vb_r.setYLink(master_vb)
            self.status_label.setText("View sync: ON")
        else:
            vb_r.setXLink(None)
            vb_r.setYLink(None)
            self.status_label.setText("View sync: OFF")

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
        is_activity = preset.startswith("Activity")
        is_seg = preset.startswith("Segmentation")

        show_rows_cols = is_grid and not is_none
        show_n = (is_features or is_activity or is_seg) and not is_none
        show_mindist = (is_features or is_activity) and not is_none

        self._set_visible([self.lbl_rows, self.grid_rows_ctrl, self.lbl_cols, self.grid_cols_ctrl], show_rows_cols)
        self._set_visible([self.lbl_N, self.num_points_spin], show_n)
        self._set_visible([self.lbl_minDist, self.min_dist_spin], show_mindist)

        # self.apply_preset_btn.setVisible(True)
        # self.apply_preset_btn.setEnabled(True)

    def on_preset_changed(self, *_):
        self._update_preset_controls()
        preset = self.preset_combo.currentText()

        # Wenn noch kein Video geladen ODER "None" gewählt wurde: einfach direkt anwenden (replace)
        if self.video_data is None or preset.startswith("None"):
            self.apply_point_preset(mode="replace")
            self._last_preset_index = self.preset_combo.currentIndex()
            return

        # Dialog: ersetzen oder hinzufügen?
        box = QMessageBox(self)
        box.setWindowTitle("Preset ändern")
        box.setText(
            "Soll das neue Preset die bisherigen Punkte ersetzen oder hinzugefügt werden?\n\n"
            "• Ersetzen: löscht alle Punkte, Tracks und Resultate.\n"
            "• Hinzufügen: fügt Punkte hinzu (Tracks/Resultate werden zurückgesetzt)."
        )

        btn_replace = box.addButton("Ersetzen", QMessageBox.ButtonRole.AcceptRole)
        btn_add     = box.addButton("Hinzufügen", QMessageBox.ButtonRole.ActionRole)
        btn_cancel  = box.addButton("Abbrechen", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_replace)

        box.exec()
        clicked = box.clickedButton()

        # Cancel -> Preset-Auswahl zurückrollen
        if clicked == btn_cancel:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(self._last_preset_index)
            self.preset_combo.blockSignals(False)
            return

        mode = "replace" if clicked == btn_replace else "add"

        # Immer auf Frame 0 springen und anzeigen
        self._pause_if_playing()
        self.current_frame = 0
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_frame_label()

        self.apply_point_preset(mode=mode)
        self._last_preset_index = self.preset_combo.currentIndex()

    def _invalidate_filtered_video_cache(self):
        for k in list(self.viz_data.keys()):
            if k.startswith("filtered_video::"):
                self.viz_data.pop(k, None)

    def on_filter_changed(self, text):
        self.current_filter = text
        self._invalidate_filtered_video_cache()
        self.update_plot()

        # Only offer to re-track if we have a video + at least one point
        if self.video_data is None:
            return

        n_base = int(self.base_grid_active.sum()) if self.base_grid_active is not None else 0
        n_custom = len(self.custom_grid_pts) if self.custom_grid_pts else 0
        if (n_base + n_custom) == 0:
            return

        # Popup: ask whether to start registration now
        box = QMessageBox(self)
        box.setWindowTitle("Filter changed")
        box.setText(
            "The input filter changed.\n\n"
            "Do you want to start registration now and re-track ALL points using the filtered video?"
        )
        btn_yes = box.addButton("Start registration", QMessageBox.ButtonRole.AcceptRole)
        btn_no  = box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_yes)

        box.exec()
        clicked = box.clickedButton()

        if clicked == btn_yes:
            self.status_label.setText(f"Filter = {self.current_filter}. Re-tracking all points…")
            self._track_all_points_now(method="CoTracker")
        else:
            self.status_label.setText(f"Filter = {self.current_filter}. (Tracking not restarted)")


    def on_preset_params_changed(self):
        if self.video_data is not None:
            # re-apply current preset in-place (replace) ONLY if the preset is grid/features/etc.
            # and only when user expects it.
            self.apply_point_preset(mode="replace")

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
                    self.status_label.setText("Tracking is already running. Stop it first to restart.")
                    return
            except Exception:
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

    def _clip_border(self, pts, H, W, border=0):
        if pts is None or len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        good = (pts[:, 0] >= border) & (pts[:, 0] < (W - border)) & (pts[:, 1] >= border) & (pts[:, 1] < (H - border))
        return pts[good]

    def _preset_features(self, ref_u8_filt, num_points=400, min_dist=10):
        H, W = ref_u8_filt.shape[:2]
        _, m = cv2.threshold(ref_u8_filt, 0, 255, cv2.THRESH_OTSU)
        m = cv2.medianBlur(m, 5)

        pts = cv2.goodFeaturesToTrack(
            ref_u8_filt,
            maxCorners=int(num_points),
            qualityLevel=0.01,
            minDistance=float(min_dist),
            blockSize=7,
            useHarrisDetector=False,
            mask=m
        )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 2).astype(np.float32)
        return self._clip_border(pts, H, W, border=0)

    def _preset_activity(self, gray_video, num_points=200, min_dist=15):
        H, W = gray_video.shape[1], gray_video.shape[2]
        T = gray_video.shape[0]
        if T < 2:
            return np.zeros((0, 2), dtype=np.float32)

        diffs = np.abs(gray_video[1:] - gray_video[:-1])
        act = diffs.mean(axis=0)
        act_u8 = cv2.normalize(act, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        act_u8 = cv2.GaussianBlur(act_u8, (0, 0), 1.0)

        _, m = cv2.threshold(act_u8, 0, 255, cv2.THRESH_OTSU)
        m = cv2.medianBlur(m, 5)

        pts = cv2.goodFeaturesToTrack(
            act_u8,
            maxCorners=int(num_points),
            qualityLevel=0.01,
            minDistance=float(min_dist),
            blockSize=7,
            useHarrisDetector=False,
            mask=m
        )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 2).astype(np.float32)
        return self._clip_border(pts, H, W, border=0)

    def _preset_segmentation_centroids(self, ref_u8_filt, num_points=200):
        H, W = ref_u8_filt.shape[:2]
        _, bw = cv2.threshold(ref_u8_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.medianBlur(bw, 5)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num_labels <= 1:
            return np.zeros((0, 2), dtype=np.float32)

        areas = stats[1:, cv2.CC_STAT_AREA]
        cent = centroids[1:].astype(np.float32)

        order = np.argsort(-areas)
        cent = cent[order][:int(num_points)]
        cent = self._clip_border(cent, H, W, border=0)
        return cent

    def _preset_regular_grid(self, H, W, rows, cols):
        xs = np.linspace(5, W -5, cols, dtype=np.float32) #0, H-1
        ys = np.linspace(5, H -5, rows, dtype=np.float32) #10, H - 10
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)

    def apply_point_preset(self, mode="replace"):
        if self.video_data is None:
            return

        preset = self.preset_combo.currentText()

        # Wenn None gewählt wurde: Add macht keinen Sinn -> als Replace behandeln
        if preset.startswith("None"):
            mode = "replace"

        # Bei Replace: alles (inkl. Punkte) platt machen
        if mode == "replace":
            self._reset_results_and_tracks()
            self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            self.base_grid_active = np.zeros((0,), dtype=bool)
            self.base_grid_pts0 = self.base_grid_pts.copy()
            self.custom_grid_pts = []

        # Bei Add: Tracks/Resultate zurücksetzen (Punkte bleiben!)
        if mode == "add":
            self._reset_results_and_tracks()
            if self.base_grid_pts is None:
                self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            if self.base_grid_active is None:
                self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)
            if self.base_grid_pts0 is None:
                self.base_grid_pts0 = np.asarray(self.base_grid_pts, np.float32).copy()

        # --- None (no points) ---
        if preset.startswith("None"):
            self._update_points_label()
            self.update_grid_display_pre_registration()
            self.status_label.setText("Preset applied: None (no points)")
            self.process_btn.setEnabled(True)
            self.update_plot()
            return

        # --- wie bisher: Punkte für Preset berechnen ---
        gray = self._get_gray_video_np()
        H, W = gray.shape[1], gray.shape[2]
        ref_u8 = self._reference_image(gray)
        ref_u8_filt = self.apply_filter(ref_u8, self.current_filter)

        if preset.startswith("Regular grid"):
            try:
                rows = self.grid_rows_ctrl.value()
                cols = self.grid_cols_ctrl.value()
            except Exception:
                rows, cols = self.grid_rows, self.grid_cols

            rows = max(2, rows) 
            cols = max(2, cols)
            self.grid_rows, self.grid_cols = rows, cols
            pts_new = self._preset_regular_grid(H, W, rows, cols)

            # min_dist als "Dedup"-Heuristik
            min_dist = max(5, int(self.min_dist_spin.value()) if self.min_dist_spin is not None else 10)

        else:
            num_points = int(self.num_points_spin.value())
            min_dist = int(self.min_dist_spin.value())

            if preset.startswith("Features"):
                pts_new = self._preset_features(ref_u8_filt, num_points=num_points, min_dist=min_dist)
            elif preset.startswith("Activity"):
                pts_new = self._preset_activity(gray, num_points=num_points, min_dist=min_dist)
            else:
                pts_new = self._preset_segmentation_centroids(ref_u8_filt, num_points=num_points)

        pts_new = np.asarray(pts_new, dtype=np.float32).reshape(-1, 2)

        # ---- APPLY pts_new to internal state ----
        if mode == "replace":
            self.base_grid_pts = pts_new.astype(np.float32)
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)
            self.base_grid_pts0 = self.base_grid_pts.copy()

        elif mode == "add":
            # add as base points, keep existing
            if self.base_grid_pts is None:
                self.base_grid_pts = np.zeros((0, 2), dtype=np.float32)
            old = np.asarray(self.base_grid_pts, dtype=np.float32).reshape(-1, 2)

            # optional: remove near-duplicates vs existing points
            pts_new_f = self._filter_new_points(pts_new, min_dist=min_dist)

            self.base_grid_pts = np.vstack([old, pts_new_f]).astype(np.float32)

            if self.base_grid_active is None:
                self.base_grid_active = np.ones((len(old),), dtype=bool)
            # newly added are active
            self.base_grid_active = np.concatenate(
                [self.base_grid_active.astype(bool), np.ones((len(pts_new_f),), dtype=bool)]
            )

            if self.base_grid_pts0 is None:
                self.base_grid_pts0 = old.copy()

        # keep tracked flags aligned with base points length
        if self.base_grid_tracked is None or len(self.base_grid_tracked) != len(self.base_grid_pts):
            self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)

        # enable registration if we have any points
        self.process_btn.setEnabled(len(self.base_grid_pts) > 0 or len(self.custom_grid_pts) > 0)

        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.update_plot()
        self.status_label.setText(f"Preset applied: {preset} ({mode})")


        # after base_grid_pts/base_grid_active updated:
        if mode == "replace":
            self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
            self.custom_grid_tracked = []
        elif mode == "add":
            # extend tracked flags for new base points
            if self.base_grid_tracked is None:
                self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
            else:
                n_need = len(self.base_grid_pts) - len(self.base_grid_tracked)
                if n_need > 0:
                    self.base_grid_tracked = np.concatenate([self.base_grid_tracked, np.zeros((n_need,), bool)])

        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.update_plot()

    # ---------- file loading ----------
    def select_file(self):
        file_dialog = QFileDialog(self, "Select Video File")
        file_dialog.setNameFilters([
            "Video files (*.avi *.mp4 *.tiff *.tif *.h5 *.hdf5)",
            "All files (*.*)"
        ])
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
        self._initial_zoom_done = False
        self.viz_data = {}

        if not keep_video:
            self.video_data = None
            self.total_frames = 0
            self.current_frame = 0

        # Results
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None
        self.ants_video = None
        self.lddmm_video = None

        # Clear panels (NEW UI)
        if hasattr(self, "view_left"):
            self.view_left.set_image(None)
        if hasattr(self, "view_right"):
            self.view_right.set_image(None)

        # Motion plot
        if hasattr(self, 'motion_plot'):
            self.motion_plot.set_current_frame(0)
            self.motion_plot.set_deformation_data([])
            if not keep_video:
                self.motion_plot.set_data([])

        # UI controls reset
        if not keep_video:
            self.slider.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.step_fwd_btn.setEnabled(False)
            self.step_back_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.edit_points_btn.setEnabled(False)
            self.clear_points_btn.setEnabled(False)
            self.frame_label.setText("0/0")
            self.edit_tracks_btn.setEnabled(False)
            self.edit_tracks_btn.setChecked(False)


    def _reset_results_and_tracks(self):
        """
        Clear anything that depends on tracking/warping results, but keep:
        - loaded video_data
        - base/custom point presets (those are handled by apply_point_preset)
        """
        # Tracking outputs
        self.pred_tracks = None
        self.pred_visibility = None
        self.pred_confidence = None
        self.grid_pts = None

        # Warping / registration outputs
        self.warped_video = None
        self.disp_field = None
        self.ants_video = None
        self.lddmm_video = None
        self.ssim_score = None

        # Per-track UI state
        self.track_active = None
        self.selected_track_idx = None

        # Viz cache for derived views
        if hasattr(self, "viz_data") and isinstance(self.viz_data, dict):
            # keep input cache if you want (faster), remove everything else
            inp = self.viz_data.get("input", None)
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
            self.save_btn.setEnabled(False)

        # stop playback if running
        if hasattr(self, "timer"):
            self.timer.stop()
        self.is_playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.setText("▶")

        # refresh view
        try:
            self.update_plot()
        except Exception:
            pass


    def _gather_existing_points(self):
        pts = []
        if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
            if self.base_grid_active is None:
                pts.append(np.asarray(self.base_grid_pts, np.float32))
            else:
                a = np.asarray(self.base_grid_active, bool)
                if a.any():
                    pts.append(np.asarray(self.base_grid_pts[a], np.float32))

        if self.custom_grid_pts:
            pts.append(np.asarray(self.custom_grid_pts, np.float32))

        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(pts).astype(np.float32)


    def _filter_new_points(self, new_pts, min_dist):
        new_pts = np.asarray(new_pts, dtype=np.float32)
        if new_pts.size == 0:
            return new_pts.reshape(0, 2)

        exist = self._gather_existing_points()
        if exist.shape[0] == 0:
            return new_pts

        thr2 = (float(min_dist) / 2.0) ** 2
        keep = []
        for p in new_pts:
            d2 = np.sum((exist - p) ** 2, axis=1)
            if d2.size == 0 or float(np.min(d2)) > thr2:
                keep.append(p)

        return np.asarray(keep, dtype=np.float32).reshape(-1, 2)



    def _retrack_all_points_with_current_filter(self):
        if self.video_data is None:
            return

        n_base = int(self.base_grid_pts.shape[0]) if self.base_grid_pts is not None else 0
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
            except Exception:
                pass
            self.worker = None

        # --- Wichtig: "Corrected/DF" ungültig machen, weil Input jetzt anders ist ---
        self.warped_video = None
        self.disp_field = None
        self.ants_video = None
        self.lddmm_video = None
        self.ssim_score = None
        if isinstance(self.viz_data, dict):
            self.viz_data.pop("warped", None)

        # --- Alle Punkte als pending markieren ---
        if self.base_grid_pts is None:
            self.base_grid_pts = np.zeros((0, 2), np.float32)
        if self.base_grid_active is None:
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
        self.custom_grid_tracked = [False] * (len(self.custom_grid_pts) if self.custom_grid_pts else 0)

        # Mapping reset
        self.track_meta = []
        self._pending_track_meta = []

        # Tracks zurücksetzen (aber NICHT den Filter / Input cache killen)
        self.pred_tracks = None
        self.pred_visibility = None
        self.pred_confidence = None
        self.track_active = None
        self.selected_track_idx = None

        # Keine Auto-Warp-Flags setzen
        self._recalc_inflight = False
        self._pending_recalc = False

        self.status_label.setText(f"Filter changed → re-tracking all points ({self.current_filter})…")

        # Jetzt neu tracken (nutzt _get_video_input_for_tracking -> gefiltert)
        self.visualize(method="CoTracker")


    def load_data(self, load_settings=None):
        if not self.video_path:
            return

        if load_settings is None:
            load_settings = {"target_w": None, "target_h": None, "max_frames": None}

        target_w = load_settings.get("target_w", None)   # None => keep original
        target_h = load_settings.get("target_h", None)   # None => keep original
        max_frames = load_settings.get("max_frames", None)  # None => all frames

        self.clear_all_data()
        ext = os.path.splitext(self.video_path)[1].lower()
        self.status_label.setText("Loading data...")
        QApplication.processEvents()

        try:
            # -----------------------------
            # 1) LOAD (as torch tensor)
            # -----------------------------
            if ext in ['.avi', '.mp4']:
                # load_video returns torch in your pipeline (T,C,H,W) likely
                video_t = load_video(self.video_path, resize=False, max_frames=max_frames)

                # enforce torch
                if not isinstance(video_t, torch.Tensor):
                    video_t = torch.from_numpy(np.asarray(video_t))

                # normalize to (T,C,H,W) if needed
                if video_t.ndim == 3:  # (T,H,W)
                    video_t = video_t.unsqueeze(1)
                elif video_t.ndim == 4 and video_t.shape[1] not in (1, 3) and video_t.shape[-1] in (1, 3):
                    # (T,H,W,C) -> (T,C,H,W)
                    video_t = video_t.permute(0, 3, 1, 2)

            elif ext in ['.tiff', '.tif']:
                video_t = load_tiff(self.video_path, resize=False)

                if not isinstance(video_t, torch.Tensor):
                    video_t = torch.from_numpy(np.asarray(video_t))

                if video_t.ndim == 3:  # (T,H,W)
                    video_t = video_t.unsqueeze(1)
                elif video_t.ndim == 4 and video_t.shape[1] not in (1, 3) and video_t.shape[-1] in (1, 3):
                    video_t = video_t.permute(0, 3, 1, 2)

                if max_frames is not None:
                    video_t = video_t[:int(max_frames)]

            elif ext in ['.h5', '.hdf5']:
                frames_arg = int(max_frames) if max_frames is not None else None

                if frames_arg is None:
                    # try load all
                    try:
                        video_np = load_hdf5(self.video_path, "/data", frames=None)
                    except Exception:
                        # fallback to prior behavior
                        video_np = load_hdf5(self.video_path, "/data", frames=None)
                else:
                    video_np = load_hdf5(self.video_path, "/data", frames=frames_arg)

                if isinstance(video_np, torch.Tensor):
                    video_t = video_np
                else:
                    video_t = torch.from_numpy(np.asarray(video_np))

                if video_t.ndim == 3:  # (T,H,W)
                    video_t = video_t.unsqueeze(1)

                # your old behavior: repeat grayscale to 3ch
                if video_t.shape[1] == 1:
                    video_t = video_t.repeat(1, 3, 1, 1)

                video_t = video_t.float()

            else:
                self.status_label.setText(f"Unsupported file extension: {ext}")
                return

            # -----------------------------
            # 2) RESIZE by target W/H
            # -----------------------------
            if (target_w is not None) and (target_h is not None):
                video_t = resize_video_tensor_to_wh(video_t, target_w=int(target_w), target_h=int(target_h))

            self.video_data = video_t

            # -----------------------------
            # 3) Continue with existing UI logic
            # -----------------------------
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

            self.update_plot()
            QTimer.singleShot(100, lambda: self.reset_all_zoom())

        except Exception as e:
            self.status_label.setText(f"Error loading data: {e}")
            self.video_data = None
            return

        # -----------------------------
        # 4) Your correlation code (unchanged)
        # -----------------------------
        try:
            gray_full = self._get_gray_video_np()  # (T, H, W)

            T, H, W = gray_full.shape
            target_dim = 64
            scale = min(target_dim / H, target_dim / W)
            if scale < 1.0:
                dsize = (int(W * scale), int(H * scale))
                gray_small = []
                for i in range(T):
                    gray_small.append(cv2.resize(gray_full[i], dsize, interpolation=cv2.INTER_AREA))
                gray_small = np.array(gray_small)
            else:
                gray_small = gray_full

            flat = gray_small.reshape(T, -1).astype(np.float32)
            means = flat.mean(axis=1, keepdims=True)
            stds = flat.std(axis=1, keepdims=True) + 1e-6
            flat_norm = (flat - means) / stds

            if T > 1:
                diffs = flat_norm[1:] - flat_norm[:-1]
                l2_norms = np.linalg.norm(diffs, axis=1)

                max_val = l2_norms.max()
                if max_val > 1e-6:
                    scores = l2_norms / max_val
                else:
                    scores = np.zeros_like(l2_norms)

                scores = np.concatenate([[0.0], scores])
                self.motion_plot.set_data(scores)
            else:
                self.motion_plot.set_data([0.0])

        except Exception as e:
            print(f"Error calculating correlation: {e}")
            self.motion_plot.set_data([])



    def toggle_edit_mode(self, *args):
        # Always disconnect first to avoid double connection
        for panel in [self.view_left, self.view_right]:
            try:
                panel.pressed.disconnect(self.on_image_press)
                panel.moved.disconnect(self.on_image_move)
                panel.released.disconnect(self.on_image_release)
            except Exception:
                pass

        if self.edit_points_mode or self.edit_tracks_mode:
            for panel in [self.view_left, self.view_right]:
                panel.pressed.connect(self.on_image_press)
                panel.moved.connect(self.on_image_move)
                panel.released.connect(self.on_image_release)

            self._pause_if_playing()
            self.status_label.setText(
                "Edit mode: " +
                ("Points " if self.edit_points_mode else "") +
                ("Tracks" if self.edit_tracks_mode else "")
            )
        else:
            self._img_cids.clear()
            self._drag = None
            self.status_label.setText(f"Loaded: {os.path.basename(self.video_path) if self.video_path else 'Ready'}")

        if self.pred_tracks is not None:
            self.update_plot()
        else:
            self.update_grid_display_pre_registration()


    def _nearest_point(self, pts, x, y):
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

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
        if not (self.edit_points_mode or self.edit_tracks_mode):
            return

        x_f, y_f = float(x), float(y)
        btn = int(button_value)

        # Only react on left-click for "clicked elsewhere" behavior
        if btn != 1:
            # keep your existing right-click logic etc.
            pass
        if not (self.edit_points_mode or self.edit_tracks_mode):
            return

        # --- NEW: only allow point editing on frame 0 ---
        if self.edit_points_mode and self.current_frame != 0:
            self.status_label.setText("Edit Points works only on frame 0.")
            return

        if not (self.edit_points_mode or self.edit_tracks_mode):
            return
        
        # If we already have a drag in progress, ignore this press (prevents duplicate events)
        if self._drag is not None:
            return
        
        # Find which panel sent the signal to update its specific selection box
        source = self.sender() if hasattr(self, 'sender') else self.lbl_grid
        
        # normalize button
        btn = int(button_value)  # already 1/2/3 from PGImagePanel

        btn = int(button_value)
        x_f, y_f = float(x), float(y)
        tol = 15.0

        # RIGHT CLICK
        if btn == 3:
            # --- Track toggling in Edit-Tracks ---
            if self.edit_tracks_mode and self.pred_tracks is not None:
                tracks = np.asarray(self.pred_tracks)
                if tracks.ndim == 4 and tracks.shape[0] == 1:
                    tracks = tracks[0]
                current_pts = tracks[self.current_frame]
                tidx, tdist = self._nearest_point(current_pts, x_f, y_f)
                if tdist < tol:
                    self._tracks_user_edited = True
                    self.selected_track_idx = tidx
                    if self.track_active is not None and len(self.track_active) == current_pts.shape[0]:
                        self.track_active[tidx] = ~self.track_active[tidx]
                    self.update_plot()
                    self._schedule_recalc_warping()
                return

            # --- Pre-registration delete only in Edit-Points ---
            if not self.edit_points_mode:
                return

            tol_del = 20.0

            # Find nearest custom point
            cidx, cdist = None, 1e9
            if len(self.custom_grid_pts) > 0:
                cpts = np.array(self.custom_grid_pts, dtype=np.float32)
                cidx, cdist = self._nearest_point(cpts, x_f, y_f)

            # Find nearest active base point
            bidx, bdist = None, 1e9
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                if self.base_grid_active is None:
                    bidx, bdist = self._nearest_point(self.base_grid_pts, x_f, y_f)
                else:
                    active_idxs = np.where(self.base_grid_active)[0]
                    if len(active_idxs) > 0:
                        ptsA = self.base_grid_pts[active_idxs]
                        loc, bdist = self._nearest_point(ptsA, x_f, y_f)
                        bidx = int(active_idxs[loc])

            # Prefer deleting whichever is closer
            delete_custom = (cidx is not None and cdist < tol_del and cdist <= bdist)
            delete_base   = (bidx is not None and bdist < tol_del and bdist < cdist)

            if delete_custom:
                # If this custom point has a track, delete it too
                tidx = self._find_track_index_for_point("custom", cidx)
                if tidx is not None:
                    self._delete_track_idx(tidx)

                # remove point itself + tracked flag
                self.custom_grid_pts.pop(cidx)
                if self.custom_grid_tracked is not None and cidx < len(self.custom_grid_tracked):
                    self.custom_grid_tracked.pop(cidx)

                # IMPORTANT: custom indices shift -> fix track_meta custom indices > cidx
                if hasattr(self, "track_meta") and self.track_meta is not None:
                    fixed = []
                    for (k, i) in self.track_meta:
                        if k == "custom" and i > cidx:
                            fixed.append((k, i - 1))
                        else:
                            fixed.append((k, i))
                    self.track_meta = fixed

                self._update_points_label()
                self.update_plot()
                return

            if delete_base:
                # If this base point has a track, delete it too
                tidx = self._find_track_index_for_point("base", bidx)
                if tidx is not None:
                    self._delete_track_idx(tidx)

                # remove base point itself
                self._delete_base_point(bidx)  # you already have this method

                # IMPORTANT: base indices shift -> fix track_meta base indices > bidx
                if hasattr(self, "track_meta") and self.track_meta is not None:
                    fixed = []
                    for (k, i) in self.track_meta:
                        if k == "base" and i > bidx:
                            fixed.append((k, i - 1))
                        else:
                            fixed.append((k, i))
                    self.track_meta = fixed

                # keep base_grid_tracked aligned if you physically delete base points
                if self.base_grid_tracked is not None and bidx < len(self.base_grid_tracked):
                    self.base_grid_tracked = np.delete(self.base_grid_tracked, bidx)

                self._update_points_label()
                self.update_plot()
                return

            return

        # LEFT CLICK
        if btn != 1:
            return

        # Drag tracks ONLY in Edit-Tracks mode
        if self.edit_tracks_mode and self.pred_tracks is not None:
            tracks = self.pred_tracks
            if isinstance(tracks, torch.Tensor):
                tracks = tracks.detach().cpu().numpy()
            if tracks.ndim == 4:
                tracks = tracks[0]

            current_pts = tracks[self.current_frame]
            tidx, tdist = self._nearest_point(current_pts, x_f, y_f)

            if tdist < 15.0:
                self._drag = ("track", tidx)
                return

        bidx, bdist = None, 1e9
        if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
            idxs = np.where(self.base_grid_active)[0]
            ptsA = self.base_grid_pts[idxs]
            loc_idx, bdist = self._nearest_point(ptsA, x_f, y_f)
            bidx = int(idxs[loc_idx])

        cidx, cdist = None, 1e9
        if len(self.custom_grid_pts) > 0:
            cpts = np.array(self.custom_grid_pts, dtype=np.float32)
            cidx, cdist = self._nearest_point(cpts, x_f, y_f)

        if bidx is not None and bdist < tol and bdist <= cdist:
            self._drag = ("base", bidx)
            self.base_grid_pts[bidx] = (x_f, y_f)
            self.update_grid_display_pre_registration()
            return

        if cidx is not None and cdist < tol:
            self._drag = ("custom", cidx)
            self.custom_grid_pts[cidx] = (int(x_f), int(y_f))
            self.update_grid_display_pre_registration()
            return

        # add new custom point
        self.custom_grid_pts.append((int(x_f), int(y_f)))
        self.custom_grid_tracked.append(False)
        self._update_points_label()
        self.update_grid_display_pre_registration()
    
    def on_image_move(self, x, y):
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

                if 'tracks' in self.viz_data:
                    self.viz_data['tracks'][self.current_frame, idx] = (float(x), float(y))

                self.update_plot()

                # <<< ADD THIS >>>
                self._schedule_recalc_warping(delay_ms=3000)

            return

        if kind == "base":
            self.base_grid_pts[idx] = (float(x), float(y))
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()


    def on_image_release(self):
        # enable recalc button only if tracks were edited
        if self._drag is not None and self._drag[0] == "track":
            self.recalculate_btn.setEnabled(True)
            self.status_label.setText("Track modified. Auto-warping in 3s…")
            self._schedule_recalc_warping(delay_ms=3000)

        # IMPORTANT: always clear drag state
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
        self.custom_grid_tracked = []       
        self.update_grid_display_pre_registration()

    def reset_all_zoom(self):
        for panel in [self.view_left, self.view_right]:
            panel.reset_view()
        self.status_label.setText("View reset on both panels")


    # ---------- visualization ----------
    def _get_current_tracks(self):
        tr = self.viz_data.get('tracks', None)
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

    def _calc_levels(self, img):
        if img is None: return None
        a = np.asarray(img)
        if a.dtype.kind == 'f':
            finite = np.isfinite(a)
            if finite.any():
                # User request: take strict min and max values (not percentiles)
                lo = float(np.nanmin(a[finite]))
                hi = float(np.nanmax(a[finite]))
                if hi <= lo: hi = lo + 1.0
                return (lo, hi)
        else:
            # Integer types: min and max directly
            lo = float(np.min(a))
            hi = float(np.max(a))
            if hi <= lo: hi = lo + 1.0
            return (lo, hi)
        return None

    def update_plot(self):
        """Render current frame into the two selected panels."""
        if self.video_data is None:
            return

        # Cache input as THWC float32 for display
        if 'input' not in self.viz_data:
            # video_data is (T,C,H,W)
            self.viz_data['input'] = (
                self.video_data.permute(0, 2, 3, 1)
                .detach().cpu().numpy().astype(np.float32)
            )

        T = self.viz_data['input'].shape[0]
        self.current_frame = int(np.clip(self.current_frame, 0, T - 1))
        idx = self.current_frame

        # Build input display frame (your filter returns uint8 grayscale usually)
        input_frame = self.viz_data['input'][idx]  # (H,W,C)
        fv = self._get_filtered_video_for_filter(self.current_filter)
        filtered = fv[idx] if fv is not None else self.apply_filter(input_frame, "None")


        # Levels based on filtered image (optional; PGImagePanel can handle None)
        levels = self._calc_levels(filtered)

        left_choice = self.view_left_combo.currentText()
        right_choice = self.view_right_combo.currentText()

        self._render_view(self.view_left, left_choice, idx, filtered, levels)
        self._render_view(self.view_right, right_choice, idx, filtered, levels)

        if not getattr(self, "_initial_zoom_done", False):
            self.view_left.reset_view()
            self.view_right.reset_view()
            self._initial_zoom_done = True

        # cursor in motion plot
        if hasattr(self, "motion_plot"):
            self.motion_plot.set_current_frame(self.current_frame)

    def _render_view(self, panel, which, idx, filtered_frame, levels=None):
        which = (which or "").strip()

        def clear_overlays():
            panel.points_item.setData([], [])
            panel.cross_item.setData([], [])
            if hasattr(panel, "pending_item"):
                panel.pending_item.setData([], [])
            panel.set_trajectories(None, 0)

        # ---------- helpers ----------
        def draw_points_preview():
            ptsA = None
            crosses = None

            if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
                ptsA = self.base_grid_pts[self.base_grid_active]

            if self.custom_grid_pts:
                crosses = np.asarray(self.custom_grid_pts, dtype=np.float32)

            panel.set_points(pts_xy=ptsA, crosses_xy=crosses)
            panel.set_trajectories(None, 0)  # NEVER draw tracks here

        # ---------- INPUT ----------
        if which == "Input":
            panel.set_image(filtered_frame, auto_range=False, levels=levels)
            clear_overlays()
            return

        # ---------- TRACKED POINTS (ONLY place to draw tracks) ----------
        if which == "Tracked Points":
            panel.set_image(filtered_frame, auto_range=False, levels=levels)

            tracks = self.viz_data.get("tracks", None)
            if tracks is None:
                # no tracks yet -> show preset points only
                draw_points_preview()
                return

            tr = np.asarray(tracks)
            if tr.ndim == 4:
                tr = tr[0]  # (T,N,2)
            if tr.ndim != 3 or idx >= tr.shape[0]:
                draw_points_preview()
                return

            ptsA = tr[idx]
            N = ptsA.shape[0]
            # vis = self._get_visibility_frame(idx, N)
            pts_show = ptsA
            colors = None

            N = pts_show.shape[0]
            colors = np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (N, 1))
            panel.set_points_colored(pts_show, colors_rgba=colors, size=8)

            
            # ensure view includes outside points too (use full ptsA, not just pts_show)
            if not (self.edit_points_mode or self.edit_tracks_mode):
                panel.fit_view_to_image_and_points(ptsA, margin=20)

            # draw trajectories ONLY here
            panel.set_trajectories(tr, idx, trail=15)

            # ---- draw pending (untracked) points so they don't "disappear" ----
            pending_pts = []

            # pending base = active & not tracked
            if self.base_grid_pts is not None and self.base_grid_active is not None and self.base_grid_tracked is not None:
                pend_mask = self.base_grid_active.astype(bool) & (~np.asarray(self.base_grid_tracked, bool))
                if np.any(pend_mask):
                    pending_pts.append(self.base_grid_pts[pend_mask].astype(np.float32))

            # pending custom = not tracked
            if self.custom_grid_pts and self.custom_grid_tracked is not None:
                pend_custom = [p for p, trk in zip(self.custom_grid_pts, self.custom_grid_tracked) if not trk]
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
        if which == "Corrected":
            if self.viz_data.get("warped", None) is not None:
                wf = self.viz_data["warped"][idx]
                panel.set_image(wf, auto_range=False, levels=self._calc_levels(wf))
            else:
                panel.set_image(None)
            clear_overlays()

            return

        # ---------- DEFORMATION FIELD ----------
        if which == "Deformation Field":
            if self.disp_field is not None:
                disp_frame = self.disp_field[idx] if idx < len(self.disp_field) else self.disp_field[0]
                disp_preview = self._make_disp_preview(disp_frame)
                panel.set_image(disp_preview, auto_range=False, levels=self._calc_levels(disp_preview))
            else:
                panel.set_image(None)
            clear_overlays()

            return

        # ---------- Corrected (ANTs) ----------
        if which == "Corrected (ANTs)":
            if self.ants_video is not None:
                panel.set_image(self.ants_video[idx], auto_range=False, levels=self._calc_levels(self.ants_video[idx]))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # ---------- Corrected (LDDMM) ----------
        if which == "Corrected (LDDMM)":
            if self.lddmm_video is not None:
                panel.set_image(self.lddmm_video[idx], auto_range=False, levels=self._calc_levels(self.lddmm_video[idx]))
            else:
                panel.set_image(None)
            clear_overlays()
            return

        # fallback
        panel.set_image(filtered_frame, auto_range=False, levels=levels)
        panel.set_points(pts_xy=None, crosses_xy=None)
        all_pts = []
        if ptsA is not None and len(ptsA) > 0:
            all_pts.append(np.asarray(ptsA, dtype=np.float32))
        # if crosses is not None and len(crosses) > 0:
        #     all_pts.append(np.asarray(crosses, dtype=np.float32))
        P = np.vstack(all_pts) if all_pts else None

        panel.fit_view_to_image_and_points(P, margin=20)
        panel.set_trajectories(None, 0)
        panel.points_item.setData([], [])
        panel.cross_item.setData([], [])
        if hasattr(panel, "pending_item"):
            panel.pending_item.setData([], [])

    
    def visualize(self, method="CoTracker"):
        if self.video_data is None:
            return

        self._initial_zoom_done = False
        QTimer.singleShot(0, self.reset_all_zoom)

        # Ensure edit modes are inactive while processing
        if self.edit_points_btn.isChecked():
            self.edit_points_btn.setChecked(False)
        if self.edit_tracks_btn.isChecked():
            self.edit_tracks_btn.setChecked(False)
        
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("▶")

        if 'input' not in self.viz_data:
            self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        if self.warped_video is not None:
            self.viz_data['warped'] = self.warped_video

        # UI lock
        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.rotate_btn.setEnabled(False)

        video_input = self._get_video_input_for_tracking()


        # -------------------------------
        # Ensure point lists/flags exist
        # -------------------------------
        if self.base_grid_pts is None:
            self.base_grid_pts = np.zeros((0, 2), np.float32)

        if self.base_grid_active is None:
            self.base_grid_active = np.ones((len(self.base_grid_pts),), dtype=bool)

        if self.base_grid_tracked is None or len(self.base_grid_tracked) != len(self.base_grid_pts):
            tmp = np.zeros((len(self.base_grid_pts),), dtype=bool)
            if self.base_grid_tracked is not None:
                m = min(len(tmp), len(self.base_grid_tracked))
                tmp[:m] = self.base_grid_tracked[:m]
            self.base_grid_tracked = tmp

        if self.custom_grid_tracked is None:
            self.custom_grid_tracked = []

        if len(self.custom_grid_tracked) != len(self.custom_grid_pts):
            if len(self.custom_grid_tracked) < len(self.custom_grid_pts):
                self.custom_grid_tracked = list(self.custom_grid_tracked) + [False] * (len(self.custom_grid_pts) - len(self.custom_grid_tracked))
            else:
                self.custom_grid_tracked = list(self.custom_grid_tracked[:len(self.custom_grid_pts)])

        # -------------------------------
        # Build pending queries (ONLY untracked points)
        # -------------------------------
        active_mask = self.base_grid_active.astype(bool)
        pending_base_mask = active_mask & (~self.base_grid_tracked)
        pending_base_idxs = np.where(pending_base_mask)[0]
        pending_base_pts = self.base_grid_pts[pending_base_idxs].astype(np.float32) if len(pending_base_idxs) else np.zeros((0, 2), np.float32)

        pending_custom_idxs = [i for i, trk in enumerate(self.custom_grid_tracked) if not trk]
        pending_custom_pts = (np.asarray([self.custom_grid_pts[i] for i in pending_custom_idxs], dtype=np.float32)
                            if len(pending_custom_idxs) else np.zeros((0, 2), np.float32))

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

            new_queries = torch.from_numpy(np.vstack(all_new)).unsqueeze(0)  # (1,N_new,2)

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
        if new_queries is None and existing_data is None and method not in ["ANTs", "LDDMM"]:
            self.status_label.setText("No points to track.")
            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.settings_btn.setEnabled(True)
            self.rotate_btn.setEnabled(True)
            return

        # -------------------------------
        # Start worker
        # -------------------------------
        self.worker = ProcessingWorker(
            video_input,
            parent=self,
            existing_data=existing_data,
            method=method,
            queries=new_queries
        )

        # your worker expects new_queries in custom_grid_pts sometimes:
        if new_queries is not None:
            self.worker.custom_grid_pts = new_queries

        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.start()

        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)


    def recalculate_warping(self):
        # Backwards-compatible entry point (falls irgendwo noch aufgerufen)
        self._schedule_recalc_warping(delay_ms=0)

    def _recalculate_warping_from_tracks(self):
        if self.video_data is None or self.pred_tracks is None:
            return

        # schon am Rechnen? dann pending markieren
        if self._recalc_inflight:
            self._pending_recalc = True
            return

        # wenn ein anderer worker läuft, vermeiden wir Overlap
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self._pending_recalc = True
                    return
            except Exception:
                pass

        self._recalc_inflight = True
        self._pending_recalc = False

        self.status_label.setText("Auto: recalculating warping...")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.rotate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        video_input = self._get_video_input_for_warping()

        # tracks -> torch (T,N,2)
        tracks = self.pred_tracks
        if isinstance(tracks, torch.Tensor):
            tr_np = tracks.detach().cpu().numpy()
        else:
            tr_np = np.asarray(tracks)

        if tr_np.ndim == 4:
            tr_np = tr_np[0]

        if self.track_active is not None and len(self.track_active) == tr_np.shape[1]:
            tr_np_use = tr_np[:, self.track_active, :]
        else:
            tr_np_use = tr_np

        tracks_t = torch.from_numpy(np.asarray(tr_np_use, dtype=np.float32))

        # dummy base grid tensor (worker needs it), content doesn't matter for warping path
        base_active = self.base_grid_pts[self.base_grid_active].astype(np.float32) if (self.base_grid_pts is not None and self.base_grid_active is not None) else np.zeros((0,2), np.float32)
        base_grid_tensor = torch.from_numpy(base_active).unsqueeze(0)

        # stop existing worker cleanly
        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass

        self.worker = ProcessingWorker(video_input, base_grid_tensor, tracks=tracks_t, parent=self, filter_name=None)
        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.start()

    def _ensure_track_meta_len(self, N: int):
        """Ensure self.track_meta exists and has length N."""
        if not hasattr(self, "track_meta") or self.track_meta is None:
            self.track_meta = []
        if len(self.track_meta) < N:
            self.track_meta.extend([("unknown", -1)] * (N - len(self.track_meta)))
        elif len(self.track_meta) > N:
            self.track_meta = self.track_meta[:N]


    def _find_track_index_for_point(self, kind: str, idx: int):
        """
        Returns track index tidx such that track_meta[tidx] == (kind, idx).
        kind in {"base","custom"}.
        """
        if not hasattr(self, "track_meta") or self.track_meta is None:
            return None
        target = (str(kind), int(idx))
        for tidx, meta in enumerate(self.track_meta):
            if meta == target:
                return tidx
        return None
    
    def _delete_track_idx(self, tidx: int):
        """Remove one track (point) across all frames from pred_tracks/visibility/confidence and UI state."""
        if self.pred_tracks is None:
            return

        tr = np.asarray(self.pred_tracks)
        if tr.ndim == 4 and tr.shape[0] == 1:
            tr = tr[0]  # (T,N,2)

        if tr.ndim != 3:
            return

        T, N, _ = tr.shape
        if tidx < 0 or tidx >= N:
            return

        # --- remove from pred_tracks ---
        tr = np.delete(tr, tidx, axis=1)
        self.pred_tracks = tr
        self.viz_data["tracks"] = tr

        # --- remove from visibility ---
        if self.pred_visibility is not None:
            vis = np.asarray(self.pred_visibility)
            if vis.ndim == 3 and vis.shape[0] == 1:
                vis = vis[0]
            if vis.ndim == 3 and vis.shape[-1] == 1:
                vis = vis[..., 0]
            if vis.ndim == 2 and vis.shape[1] > tidx:
                self.pred_visibility = np.delete(vis, tidx, axis=1)

        # --- remove from confidence (optional) ---
        if getattr(self, "pred_confidence", None) is not None:
            conf = np.asarray(self.pred_confidence)
            if conf.ndim == 3 and conf.shape[0] == 1:
                conf = conf[0]
            if conf.ndim == 2 and conf.shape[1] > tidx:
                self.pred_confidence = np.delete(conf, tidx, axis=1)

        # --- remove from track_active ---
        if self.track_active is not None and len(self.track_active) > tidx:
            self.track_active = np.delete(self.track_active, tidx)
        else:
            self.track_active = np.ones((tr.shape[1],), dtype=bool)

        # --- CRITICAL: remove from track_meta to keep mapping aligned ---
        if hasattr(self, "track_meta") and self.track_meta is not None:
            if tidx < len(self.track_meta):
                self.track_meta.pop(tidx)

        # --- selection fix ---
        if self.selected_track_idx is not None:
            if self.selected_track_idx == tidx:
                self.selected_track_idx = None
            elif self.selected_track_idx > tidx:
                self.selected_track_idx -= 1




    def on_tracks_ready(self, pred_tracks, pred_visibility, pred_confidence, grid_pts):
        # pred_tracks comes from worker and is the source of truth here
        tr = np.asarray(pred_tracks)

        # normalize shape -> (T, N, 2)
        if tr.ndim == 4:          # (B,T,N,2)
            tr = tr[0]
        if tr.ndim != 3 or tr.shape[-1] != 2:
            raise ValueError(f"Unexpected tracks shape: {tr.shape}")

        T, N, _ = tr.shape

        # store
        self.pred_tracks = tr
        
        # Normalize visibility to numpy (T,N)
        vis = pred_visibility
        try:
            if isinstance(vis, torch.Tensor):
                vis = vis.detach().cpu().numpy()
            vis = np.asarray(vis) if vis is not None else None
        except Exception:
            vis = None

        if vis is not None:
            if vis.ndim == 3:
                vis = vis[0]
            self.pred_visibility = vis
        else:
            self.pred_visibility = None

        # Normalize confidence to numpy (T,N)
        conf = pred_confidence
        try:
            if isinstance(conf, torch.Tensor):
                conf = conf.detach().cpu().numpy()
            conf = np.asarray(conf)
            if conf.ndim == 3: # (B,T,N)
                conf = conf[0]
            self.pred_confidence = conf
        except Exception:
            self.pred_confidence = None

        # init track_active from confidence (mean visibility)
        self.track_active = np.ones((N,), dtype=bool)

        # grid points
        if grid_pts is not None:
            self.grid_pts = grid_pts
        
        # warped empty
        if self.warped_video is None:
            self.status_label.setText(
                "Tracking updated for new filter. Please re-run Registration for corrected output."
            )

        # initialize per-track UI state
        self.track_active = np.ones((N,), dtype=bool)
        self.selected_track_idx = None

        # for display
        self.viz_data["tracks"] = tr

        # make sure frame UI won't go out of range if tracker returns fewer frames
        self.total_frames = int(self.video_data.shape[0]) if self.video_data is not None else T
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
        self.edit_points_mode = self.edit_points_btn.isChecked()
        self.edit_tracks_mode = self.edit_tracks_btn.isChecked()

        # --- NEW: Edit Points only on frame 0 ---
        if self.edit_points_mode:
            self._pause_if_playing()
            if self.current_frame != 0:
                self.current_frame = 0
                self.slider.blockSignals(True)
                self.slider.setValue(0)
                self.slider.blockSignals(False)
                self.update_frame_label()
                self.update_plot()
                self.status_label.setText("Edit Points is only available on frame 0 (jumped to frame 0).")

        if self.edit_tracks_mode:
            self.view_left_combo.blockSignals(True)
            self.view_right_combo.blockSignals(True)

            self.view_left_combo.setCurrentText("Tracked Points")
            self.view_right_combo.setCurrentText("Corrected")

            self.view_left_combo.blockSignals(False)
            self.view_right_combo.blockSignals(False)

            self.update_plot()

        self.toggle_edit_mode()

    # def _delete_track_for_point(self, kind: str, idx: int):
    #     if self.pred_tracks is None or not self.track_meta:
    #         return
    #     for t, meta in enumerate(self.track_meta):
    #         if meta == (kind, int(idx)):
    #             self._delete_track_idx(t)
    #             # keep meta consistent with track deletion
    #             self.track_meta.pop(t)
    #             return



    def on_processing_finished(self, pred_tracks, pred_visibility, pred_confidence, grid_pts, warped_video, disp_field):
        if pred_tracks is None and warped_video is not None:
            method = getattr(self.worker, "method", "ANTs")
            if method == "LDDMM":
                self.lddmm_video = self._to_thwc(warped_video)
                self.status_label.setText("LDDMM registration finished")
            else:
                self.ants_video = self._to_thwc(warped_video)
                self.status_label.setText("ANTs registration finished")

            self.update_plot()

            # Cleanup worker + UI unlock
            if self.worker is not None:
                try:
                    self.worker.quit()
                    self.worker.wait(200)
                except Exception:
                    pass
            self.worker = None

            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.settings_btn.setEnabled(True)
            self.rotate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(100)

            QTimer.singleShot(0, self.reset_all_zoom)
            return

        # ---------------------------------------
        # 2) Normalize tracks to numpy (T,N,2)
        # ---------------------------------------
        try:
            if isinstance(pred_tracks, torch.Tensor):
                pred_tracks = pred_tracks.detach().cpu().numpy()
        except Exception:
            pass

        new_tr = np.asarray(pred_tracks) if pred_tracks is not None else None
        if new_tr is None:
            # nothing usable; treat as error-ish
            self.status_label.setText("Processing finished but no tracks returned.")
            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.settings_btn.setEnabled(True)
            self.rotate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.worker = None
            self._recalc_inflight = False
            return

        # remove batch dim if present
        if new_tr.ndim == 4 and new_tr.shape[0] == 1:
            new_tr = new_tr[0]
        if new_tr.ndim != 3 or new_tr.shape[-1] != 2:
            self.status_label.setText(f"Unexpected tracks shape: {getattr(new_tr, 'shape', None)}")
            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.settings_btn.setEnabled(True)
            self.rotate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.worker = None
            self._recalc_inflight = False
            return

        T, N, _ = new_tr.shape

        # ---------------------------------------
        # 3) Normalize visibility/confidence (T,N)
        # ---------------------------------------
        new_vis = None
        if pred_visibility is not None:
            try:
                if isinstance(pred_visibility, torch.Tensor):
                    pred_visibility = pred_visibility.detach().cpu().numpy()
                v = np.asarray(pred_visibility)
                if v.ndim == 3 and v.shape[0] == 1:
                    v = v[0]
                if v.ndim == 3 and v.shape[-1] == 1:
                    v = v[..., 0]
                if v.ndim == 2:
                    new_vis = v
            except Exception:
                new_vis = None

        new_conf = None
        if pred_confidence is not None:
            try:
                if isinstance(pred_confidence, torch.Tensor):
                    pred_confidence = pred_confidence.detach().cpu().numpy()
                c = np.asarray(pred_confidence)
                if c.ndim == 3 and c.shape[0] == 1:
                    c = c[0]
                if c.ndim == 3 and c.shape[-1] == 1:
                    c = c[..., 0]
                if c.ndim == 2:
                    new_conf = c
            except Exception:
                new_conf = None

        # store
        self.pred_tracks = new_tr
        self.pred_visibility = new_vis
        self.pred_confidence = new_conf

        # ---------------------------------------
        # 4) Mark points tracked for this run
        # ---------------------------------------
        # base
        if hasattr(self, "_pending_base_idxs_last_run"):
            if self.base_grid_tracked is None:
                self.base_grid_tracked = np.zeros((len(self.base_grid_pts),), dtype=bool)
            for bi in self._pending_base_idxs_last_run:
                bi = int(bi)
                if 0 <= bi < len(self.base_grid_tracked):
                    self.base_grid_tracked[bi] = True
            del self._pending_base_idxs_last_run

        # custom
        if hasattr(self, "_pending_custom_idxs_last_run"):
            if self.custom_grid_tracked is None:
                self.custom_grid_tracked = [False] * len(self.custom_grid_pts)
            if len(self.custom_grid_tracked) < len(self.custom_grid_pts):
                self.custom_grid_tracked += [False] * (len(self.custom_grid_pts) - len(self.custom_grid_tracked))

            for ci in self._pending_custom_idxs_last_run:
                ci = int(ci)
                if 0 <= ci < len(self.custom_grid_tracked):
                    self.custom_grid_tracked[ci] = True
            del self._pending_custom_idxs_last_run

        # ---------------------------------------
        # 5) Maintain track_meta mapping (append only newly tracked points)
        # ---------------------------------------
        # track_meta maps each track index to ("base", base_idx) or ("custom", custom_idx)
        # It must stay aligned with pred_tracks N.
        if not hasattr(self, "track_meta") or self.track_meta is None:
            self.track_meta = []

        # If we already had tracks before, ensure meta length for those old tracks.
        # The worker merges as [old, new], and we build _pending_track_meta in visualize().
        pending_meta = getattr(self, "_pending_track_meta", None)

        if pending_meta is None:
            pending_meta = []

        pending_meta = list(pending_meta)

        # Old track count = N - len(pending_meta) (clamped)
        oldN_est = max(0, N - len(pending_meta))

        # Ensure track_meta covers old tracks
        if len(self.track_meta) < oldN_est:
            self.track_meta.extend([("unknown", -1)] * (oldN_est - len(self.track_meta)))
        elif len(self.track_meta) > oldN_est and len(pending_meta) > 0:
            # If something drifted, clamp to old part before appending
            self.track_meta = self.track_meta[:oldN_est]

        # Append meta for newly tracked points (this run)
        if len(pending_meta) > 0:
            self.track_meta.extend(pending_meta)

        # Final clamp/pad to exactly N
        if len(self.track_meta) < N:
            self.track_meta.extend([("unknown", -1)] * (N - len(self.track_meta)))
        elif len(self.track_meta) > N:
            self.track_meta = self.track_meta[:N]

        # clear pending
        if hasattr(self, "_pending_track_meta"):
            self._pending_track_meta = []

        # ---------------------------------------
        # 6) Warped video + DF
        # ---------------------------------------
        self.warped_video = self._to_thwc(warped_video) if warped_video is not None else None
        
        # ---- RANGE CHECK: input vs corrected ----
        inp = self.viz_data.get("input", None)  # THWC float32
        if inp is not None and self.warped_video is not None:
            in_mn, in_mx = self._get_minmax(inp)
            w_mn, w_mx   = self._get_minmax(self.warped_video)

            msg = " | ".join([
                self._range_str(inp, "INPUT"),
                self._range_str(self.warped_video, "WARPED"),
            ])

            # If you want a strict "same range" check:
            same_range = (in_mn is not None and w_mn is not None and
                        abs(in_mn - w_mn) < 1e-3 and abs(in_mx - w_mx) < 1e-3)

            # OPTIONAL: enforce the same range by clamping warped to input min/max
            # (this keeps original scale but prevents overshoot)
            if in_mn is not None and in_mx is not None:
                #self.warped_video = np.clip(self.warped_video, in_mn, in_mx).astype(self.warped_video.dtype)
                self.warped_video = np.clip(self.warped_video, 0, 255).astype(np.uint8)
                self.viz_data["warped"] = self.warped_video

            self.status_label.setText(msg + (" | SAME" if same_range else " | DIFFERENT"))


        if self.warped_video is not None:
            self.viz_data["warped"] = self.warped_video

        # Displacement field
        self.disp_field = disp_field

        # tracks cache for rendering
        self.viz_data["tracks"] = self.pred_tracks

        # ---------------------------------------
        # 7) SSIM info (optional)
        # ---------------------------------------
        try:
            if self.warped_video is not None:
                self.ssim_score = video_ssim_score(self.warped_video, mode="ref_first")
                # self.status_label.setText(f"Warping updated | SSIM={self.ssim_score:.4f}")
                self.status_label.setText(f"Done")
            else:
                self.status_label.setText("Warping updated")
        except Exception as e:
            self.status_label.setText(f"Warping updated | SSIM failed: {e}")

        # ---------------------------------------
        # 8) Rebuild / update track_active based on confidence filter
        # ---------------------------------------
        # Keep length consistent with N
        if self.track_active is None or len(self.track_active) != N:
            self.track_active = np.ones((N,), dtype=bool)



        # ---------------------------------------
        # 9) UI unlock + worker cleanup
        # ---------------------------------------
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.rotate_btn.setEnabled(True)
        self.edit_tracks_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        if self.worker is not None:
            try:
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass
        self.worker = None

        # mark recalc done
        self._recalc_inflight = False

        # Refresh viewports
        self.update_plot()

        # Fit both viewports after results arrive
        QTimer.singleShot(50, lambda: self.view_left.reset_view())
        QTimer.singleShot(50, lambda: self.view_right.reset_view())

        # If a recalc was requested while inflight, schedule it after debounce deadline
        if getattr(self, "_pending_recalc", False):
            self._pending_recalc = False
            remaining = max(0, getattr(self, "_target_recalc_time_ms", 0) - self._edit_clock.elapsed())
            self._schedule_recalc_warping(delay_ms=remaining)

        self._tracks_user_edited = False

    def on_processing_error(self, error_msg):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Processing Error", error_msg)

        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.rotate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        if self.worker is not None:
            try:
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass
        self.worker = None
        self._recalc_inflight = False

    def on_progress(self, value):
        self.progress_bar.setValue(int(value))

    def stop_processing(self):
        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(500)
            except Exception:
                pass
        self.worker = None
        self.stop_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.rotate_btn.setEnabled(True)
        self.status_label.setText("Stopped")
        self.progress_bar.setValue(0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- PLAY Logic of video ----------
    def _pause_if_playing(self):
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")

    def step_forward(self):
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.slider.setValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def step_backward(self):
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame - 1) % self.total_frames
        self.slider.setValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def toggle_playback(self):
        # hard guard
        if self.video_data is None:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            return

        # ensure total_frames is correct
        try:
            self.total_frames = int(self.video_data.shape[0])
        except Exception:
            self.total_frames = 0

        if self.total_frames <= 0:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            return

        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
        else:
            speed_factor = self.speed_slider.value() / 100.0
            interval = max(1, int(50 / speed_factor))
            self.timer.start(interval)
            self.is_playing = True
            self.play_btn.setText("⏸")

    def on_timer(self):
        if self.video_data is None:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            return

        # keep frames in sync
        self.total_frames = int(self.video_data.shape[0]) if hasattr(self.video_data, "shape") else 0
        if self.total_frames <= 0:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
            return

        self.current_frame = (self.current_frame + 1) % self.total_frames
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

    def on_speed_change(self, value):
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.2f}x")
        if self.is_playing:
            interval = max(1, int(50 / speed))
            self.timer.start(interval)

    # ----------Visualization adjustment of input data----------
    def _show_load_settings_dialog(self):
        if not self.video_path:
            return None

        frames, w, h = probe_media_info(self.video_path)
        dlg = VideoLoadSettingsDialog(self, frames, w, h)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.result_settings()
        return None

    def _get_filtered_video_for_filter(self, filt_name: str):
        if self.video_data is None:
            return None

        key = f"filtered_video::{(filt_name or 'None')}"
        if key in self.viz_data:
            return self.viz_data[key]

        # build from cached input THWC if present, else from video_data
        if "input" not in self.viz_data:
            self.viz_data["input"] = (
                self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
            )

        inp = self.viz_data["input"]  # (T,H,W,C) float32
        T = inp.shape[0]
        out = []
        for t in range(T):
            out.append(self.apply_filter(inp[t], filt_name))  # -> (H,W) uint8
        out = np.stack(out, axis=0).astype(np.uint8)  # (T,H,W)

        self.viz_data[key] = out
        return out
    
    def apply_filter(self, frame, filt: str):
        if frame is None:
            return None

        f = np.asarray(frame)

        if f.ndim == 3 and f.shape[2] == 3:
            gray = f[..., 0]
        else:
            gray = f

        g = gray.astype(np.float32)
        mn, mx = float(np.nanmin(g)), float(np.nanmax(g))
        if mx > mn:
            g = (g - mn) / (mx - mn) * 255.0
        gray_u8 = np.clip(g, 0, 255).astype(np.uint8)

        filt = (filt or "None").lower()

        if filt in ("none", "raw"):
            return gray_u8
        if filt in ("gaus", "gaussian"):
            return cv2.GaussianBlur(gray_u8, (0, 0), 1.2)
        if filt in ("sobel",):
            sx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(sx, sy)
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if filt in ("edge", "canny"):
            return cv2.Canny(gray_u8, 50, 150)

        return gray_u8

    def _update_points_label(self):
        n_base = int(self.base_grid_active.sum()) if self.base_grid_active is not None else 0
        n_custom = len(self.custom_grid_pts) if self.custom_grid_pts else 0
        self.points_label.setText(f"Points: {n_base + n_custom} ") #(base {n_base} + custom {n_custom})
    

    def rotate_input(self, checked=False):
        if self.video_data is None:
            return
        try:          
            # Let's do 90 deg clockwise (standard "rotate")
            self.video_data = torch.rot90(self.video_data, k=-1, dims=[2, 3])
            
            # Update T, H, W (though T doesn't change)
            self.total_frames = int(self.video_data.shape[0])
            
            # Clear everything that depends on coordinates
            self.clear_all_data(keep_video=True) 
            
            # Clear manual points too
            self.custom_points = []
            self.custom_crosses = []
            self.points_label.setText(f"Points: 0")

            self.apply_point_preset()

            self.status_label.setText("Input video rotated 90° clockwise")
            self.update_plot()
            self.reset_all_zoom()

        except Exception as e:
            self.status_label.setText(f"Error rotating: {e}")

    def _get_video_input_for_tracking(self):
        """
        Returns a torch Tensor shaped (T,C,H,W) to feed the tracker.
        If a filter is selected, we build/cached a filtered uint8 video and convert to torch.
        """
        # No filter -> keep original fast path
        filt = (self.current_filter or "None").lower()
        if filt in ("none", "raw"):
            return self.video_data

        # Build/cached filtered video as (T,H,W) uint8
        fv = self._get_filtered_video_for_filter(self.current_filter)  # you already added/plan this
        if fv is None:
            return self.video_data

        # Convert to torch (T,1,H,W) float
        t = torch.from_numpy(fv).unsqueeze(1).float()

        # Many trackers expect 3 channels -> repeat to (T,3,H,W)
        t = t.repeat(1, 3, 1, 1)

        # Match normalization of your current pipeline (best-effort)
        # If your video_data is float in [0,1], normalize filtered the same way
        try:
            vd = self.video_data
            if isinstance(vd, torch.Tensor) and vd.dtype.is_floating_point:
                mx = float(vd.detach().max().cpu())
                if mx <= 1.5:   # heuristic: treat as [0,1]
                    t = t / 255.0
        except Exception:
            pass

        return t



    #---------- drag&drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.avi', '.mp4', '.tiff', '.tif', '.h5', '.hdf5']:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.avi', '.mp4', '.tiff', '.tif', '.h5', '.hdf5']:
                    self.video_path = file_path
                    self.status_label.setText(f"Selected: {os.path.basename(self.video_path)}")

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


    #-----Save Logic-----
    def save_results(self):
        if self.warped_video is None:
             return
        
        # Default filename based on input video name
        default_name = "output.mp4"
        if self.video_path:
             base = os.path.splitext(os.path.basename(self.video_path))[0]
             default_name = f"{base}_warped.mp4"

        path, filter_str = QFileDialog.getSaveFileName(
            self, 
            "Save Warped Video", 
            default_name, 
            "MP4 Video (*.mp4);;TIFF Stack (*.tiff *.tif);;HDF5 File (*.h5 *.hdf5)"
        )
        if not path:
             return
             
        try:
            # Determine format from extension
            ext = os.path.splitext(path)[1].lower()
            vid = np.asarray(self.warped_video)
            
            if ext in ['.tiff', '.tif']:
                from utils.utils import save_video_tiff
                save_video_tiff(vid, path)
            elif ext in ['.h5', '.hdf5']:
                from utils.utils import save_video_hdf5
                save_video_hdf5(vid, path)
            else:
                # Default to MP4
                from utils.mc_cotracker import save_and_display_video
                # Ensure path ends with mp4 if not present (though dialog usually handles it)
                if not ext:
                    path += ".mp4"
                save_and_display_video(vid, path, fps=30)

            
            # Also save tracks?
            tracks_path = os.path.splitext(path)[0] + "_tracks.npy"
            csv_path = os.path.splitext(path)[0] + "_tracks.csv"
            
            if self.viz_data.get('tracks') is not None:
                tracks = self.viz_data['tracks'] # (T, N, 2)
                np.save(tracks_path, tracks)
                
                # Export to CSV
                try:
    
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Frame", "PointID", "X", "Y"])
                        
                        T, N, _ = tracks.shape
                        for t in range(T):
                            for n in range(N):
                                x, y = tracks[t, n]
                                writer.writerow([t, n, x, y])
                                 
                    self.status_label.setText(f"Saved to {path}, {tracks_path}, and {csv_path}")
                except Exception as e:
                    print(f"Error saving CSV: {e}")
                    self.status_label.setText(f"Saved to {path} and {tracks_path} (CSV failed)")

            else:
                 self.status_label.setText(f"Saved to {path}")
                 
        except Exception as e:
            self.status_label.setText(f"Error saving: {e}")
            print(e)
    


