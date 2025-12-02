import sys
import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QCheckBox, QGridLayout, QComboBox, QSpinBox, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPalette

import cv2

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5


class ProcessingWorker(QThread):
    tracks_ready = pyqtSignal(object, object, object)  # tracks_np, visibility, grid
    progress = pyqtSignal(int)  # percent complete
    finished = pyqtSignal(object, object, object, object, object)  # tracks_np, visibility, grid, warped_full, disp_field
    error = pyqtSignal(str)

    def __init__(self, video_data, base_grid_pts, custom_grid_pts=None, parent=None):
        super().__init__(parent)
        self.video_data = video_data
        self.base_grid_pts = base_grid_pts              # (1, K, 2) torch
        self.custom_grid_pts = custom_grid_pts          # (M, 2) torch or None
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            grid_xy = self.base_grid_pts
            if self.custom_grid_pts is not None:
                custom_xy = self.custom_grid_pts
                if isinstance(custom_xy, np.ndarray):
                    custom_xy = torch.from_numpy(custom_xy)
                custom_xy = custom_xy.to(grid_xy.device).float()
                if custom_xy.ndim == 2:
                    custom_xy = custom_xy.unsqueeze(0)  # (1,M,2)
                grid_xy = torch.cat([grid_xy, custom_xy], dim=1)  # (1,K+M,2)

            t = torch.zeros((*grid_xy.shape[:-1], 1), device=grid_xy.device, dtype=grid_xy.dtype)
            queries = torch.cat([t, grid_xy], dim=-1)

            pred_tracks, pred_visibility = cotrack_model(self.video_data, queries=queries)

            tracks_raw = pred_tracks.detach().cpu().numpy()   # could be (B,T,N,2) or (T,N,2)

            # GUI wants (T,N,2)
            tracks_gui = tracks_raw[0] if tracks_raw.ndim == 4 else tracks_raw

            # warping wants (B,T,N,2)
            tracks_for_warp = tracks_raw if tracks_raw.ndim == 4 else tracks_raw[None, ...]

            try:
                vis_raw = pred_visibility.detach().cpu().numpy()
            except Exception:
                vis_raw = np.asarray(pred_visibility)

            vis_gui = vis_raw[0] if vis_raw.ndim == 3 else vis_raw      # (T,N) expected for GUI
           
            self.tracks_ready.emit(tracks_gui, vis_gui, grid_xy)
            self.progress.emit(25)

            warped_video, disp_field = warping(tracks_for_warp, self.video_data)
            warped_np = np.asarray(warped_video)

            # If warping returns batched outputs, squeeze them for GUI:
            if warped_np.ndim >= 4 and warped_np.shape[0] == 1:
                warped_np = warped_np[0]
            try:
                disp_arr = disp_field.detach().cpu().numpy() if isinstance(disp_field, torch.Tensor) else np.asarray(disp_field)
                if disp_arr.ndim >= 4 and disp_arr.shape[0] == 1:
                    disp_arr = disp_arr[0]
                disp_field = disp_arr
            except Exception:
                pass

            self.progress.emit(100)
            self.finished.emit(tracks_gui, vis_gui, grid_xy, warped_np, disp_field)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Zoom/pan QLabel as a viewport. Zoom=1 fits image to panel. No window resizing side-effects."""
    pressed = pyqtSignal(int, int, int)  # x, y, button_value
    moved = pyqtSignal(int, int)
    released = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(10, 10)

        self.zoom_factor = 1.0
        self.pan_offset = [0.0, 0.0]
        self._base_image = None
        self._overlay_data = {}

        self._fit_scale = 1.0
        self._is_panning = False
        self._pan_start = None

        self._viewport_pixmap = QPixmap()

        # Alt + Left drag to pan (keeps left-drag free for editing points)
        self.left_pan_modifier = Qt.KeyboardModifier.AltModifier

    def setPixmap(self, pixmap, overlay_data=None):
        self._base_image = pixmap if isinstance(pixmap, QPixmap) else QPixmap(pixmap)
        self._overlay_data = overlay_data if overlay_data else {}
        self._update_fit_scale()
        self._redraw_viewport()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_fit_scale()
        self._redraw_viewport()

    def _update_fit_scale(self):
        if self._base_image is None or self._base_image.isNull():
            self._fit_scale = 1.0
            return
        bw, bh = self._base_image.width(), self._base_image.height()
        if bw <= 0 or bh <= 0 or self.width() <= 0 or self.height() <= 0:
            self._fit_scale = 1.0
            return
        self._fit_scale = min(self.width() / bw, self.height() / bh)

    def _draw_position_and_scale(self):
        if self._base_image is None or self._base_image.isNull():
            return None

        bw, bh = self._base_image.width(), self._base_image.height()
        scale = self._fit_scale * self.zoom_factor
        sw = bw * scale
        sh = bh * scale

        wx, wy = self.width(), self.height()
        x0 = (wx - sw) * 0.5 + self.pan_offset[0]
        y0 = (wy - sh) * 0.5 + self.pan_offset[1]
        return (x0, y0, sw, sh, scale)

    def _redraw_viewport(self):
        if self.width() <= 0 or self.height() <= 0:
            return

        self._viewport_pixmap = QPixmap(self.width(), self.height())
        bg = self.palette().color(QPalette.ColorRole.Window)
        self._viewport_pixmap.fill(bg)

        if self._base_image is None or self._base_image.isNull():
            super().setPixmap(self._viewport_pixmap)
            return

        geom = self._draw_position_and_scale()
        if geom is None:
            super().setPixmap(self._viewport_pixmap)
            return

        x0, y0, sw, sh, scale = geom

        painter = QPainter(self._viewport_pixmap)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        scaled = self._base_image.scaled(
            int(round(sw)), int(round(sh)),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(int(round(x0)), int(round(y0)), scaled)

        self._draw_overlays(painter, x0, y0, scale)

        painter.end()
        super().setPixmap(self._viewport_pixmap)

    def _img_to_widget(self, x_img, y_img, x0, y0, scale):
        xw = x0 + x_img * scale
        yw = y0 + y_img * scale
        return int(round(xw)), int(round(yw))

    def _draw_overlays(self, painter, x0, y0, scale):
        if not self._overlay_data:
            return

        if 'points' in self._overlay_data:
            for pts, color, radius in self._overlay_data['points']:
                if pts is None or len(pts) == 0:
                    continue
                qcolor = QColor(color[0], color[1], color[2])
                painter.setBrush(qcolor)
                painter.setPen(qcolor)
                r = max(1, int(round(radius * scale)))
                for pt in pts:
                    xw, yw = self._img_to_widget(pt[0], pt[1], x0, y0, scale)
                    painter.drawEllipse(xw - r, yw - r, 2 * r, 2 * r)

        if 'crosses' in self._overlay_data:
            for pts, color, size in self._overlay_data['crosses']:
                if pts is None or len(pts) == 0:
                    continue
                qcolor = QColor(color[0], color[1], color[2])
                painter.setPen(qcolor)
                s = max(2, int(round(size * scale)))
                for pt in pts:
                    xw, yw = self._img_to_widget(pt[0], pt[1], x0, y0, scale)
                    painter.drawLine(xw - s, yw - s, xw + s, yw + s)
                    painter.drawLine(xw - s, yw + s, xw + s, yw - s)

        if 'trajectories' in self._overlay_data:
            tracks, current_frame, trail_length, color = self._overlay_data['trajectories']
            tracks = np.asarray(tracks)

            if tracks.ndim == 4:
                tracks = tracks[0]  # (T,N,2)

            T = tracks.shape[0]
            if T < 2:
                return

            cf = int(np.clip(current_frame, 0, T - 1))

            # retrospective segment index t means: draw (t-1) -> t
            start_t = max(1, cf - int(trail_length))
            end_t   = cf

            # only draw if the point actually moved (in image pixels)
            eps_px = 0.5
            eps2 = eps_px * eps_px

            blue   = (255,255,0)
            purple = (160, 0, 255)

            den = max(1, end_t - start_t)  # for ramp within the visible trail

            for point_idx in range(tracks.shape[1]):
                for t in range(start_t, end_t + 1):
                    pt1 = tracks[t - 1, point_idx]  # previous frame
                    pt2 = tracks[t,     point_idx]  # current frame
                    if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
                        continue

                    dx = float(pt2[0] - pt1[0])
                    dy = float(pt2[1] - pt1[1])
                    if (dx*dx + dy*dy) < eps2:
                        continue  # no movement -> don't draw

                    u = (t - start_t) / den  # 0..1 along the retrospective trail

                    r = int(round(blue[0]   + (purple[0] - blue[0]) * u))
                    g = int(round(blue[1]   + (purple[1] - blue[1]) * u))
                    b = int(round(blue[2]   + (purple[2] - blue[2]) * u))

                    qcolor = QColor(r, g, b, 255)
                    pen = painter.pen()
                    pen.setColor(qcolor)
                    pen.setWidth(max(1, int(round(2 * scale))))
                    painter.setPen(pen)

                    x1, y1 = self._img_to_widget(pt1[0], pt1[1], x0, y0, scale)
                    x2, y2 = self._img_to_widget(pt2[0], pt2[1], x0, y0, scale)
                    painter.drawLine(x1, y1, x2, y2)

    def _map_to_image(self, pos):
        if self._base_image is None or self._base_image.isNull():
            return int(pos.x()), int(pos.y())

        geom = self._draw_position_and_scale()
        if geom is None:
            return int(pos.x()), int(pos.y())

        x0, y0, sw, sh, scale = geom

        x_img = (pos.x() - x0) / scale
        y_img = (pos.y() - y0) / scale

        x_img = max(0.0, min(self._base_image.width() - 1.0, x_img))
        y_img = max(0.0, min(self._base_image.height() - 1.0, y_img))
        return int(round(x_img)), int(round(y_img))

    def wheelEvent(self, event):
        if self._base_image is None or self._base_image.isNull():
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        old_zoom = self.zoom_factor
        self.zoom_factor *= (1.1 if delta > 0 else 1 / 1.1)
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))

        mouse = event.position()
        geom = self._draw_position_and_scale()
        if geom is not None and self.zoom_factor != old_zoom:
            x0, y0, sw, sh, old_scale = geom
            x_img = (mouse.x() - x0) / old_scale
            y_img = (mouse.y() - y0) / old_scale

            new_scale = self._fit_scale * self.zoom_factor
            new_sw = self._base_image.width() * new_scale
            new_sh = self._base_image.height() * new_scale

            new_x0 = mouse.x() - x_img * new_scale
            new_y0 = mouse.y() - y_img * new_scale

            wx, wy = self.width(), self.height()
            centered_x0 = (wx - new_sw) * 0.5
            centered_y0 = (wy - new_sh) * 0.5
            self.pan_offset[0] = new_x0 - centered_x0
            self.pan_offset[1] = new_y0 - centered_y0

        self._redraw_viewport()
        event.accept()

    def mousePressEvent(self, event):
        is_left_pan = (event.button() == Qt.MouseButton.LeftButton and
                       (event.modifiers() & self.left_pan_modifier))
        is_mid_pan = (event.button() == Qt.MouseButton.MiddleButton)

        if is_mid_pan or is_left_pan:
            self._is_panning = True
            self._pan_start = [event.position().x(), event.position().y()]
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        x, y = self._map_to_image(event.position())
        self.pressed.emit(x, y, event.button().value)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._is_panning and self._pan_start is not None:
            dx = event.position().x() - self._pan_start[0]
            dy = event.position().y() - self._pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self._pan_start = [event.position().x(), event.position().y()]
            self._redraw_viewport()
            event.accept()
            return

        x, y = self._map_to_image(event.position())
        self.moved.emit(x, y)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._is_panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.released.emit()
        event.accept()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_offset = [0.0, 0.0]
        self._redraw_viewport()


class MotionCorrectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Correction GUI")
        screen = QApplication.primaryScreen()
        g = screen.availableGeometry()

        w = min(1400, int(g.width() * 0.9))
        h = min(900,  int(g.height() * 0.9))
        self.resize(w, h)
        self.move(g.left() + (g.width() - w)//2, g.top() + (g.height() - h)//2)

        self.setAcceptDrops(True)

        self.video_path = None
        self.video_data = None

        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None

        self.grid_rows = 4
        self.grid_cols = 4
        self.base_grid_pts = None          # (N,2) float32
        self.base_grid_active = None       # (N,) bool
        self.base_grid_pts0 = None         # backup
        self.custom_grid_pts = []          # list[(x,y)]

        self._drag = None
        self._img_cids = {}

        self.edit_mode = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0

        self.viz_data = {}

        self.worker = None
        self.current_filter = "None"

        self.init_ui()

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

    def _array_to_qpixmap(self, arr):
        if arr is None:
            return QPixmap()

        a = np.asarray(arr)
        a = np.ascontiguousarray(a)

        if a.ndim == 3 and a.shape[2] == 3:
            if np.issubdtype(a.dtype, np.floating):
                lo = np.percentile(a, 1.0)
                hi = np.percentile(a, 99.0)
                if hi <= lo:
                    hi = lo + 1.0
                a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            elif a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)

            h, w, _ = a.shape
            qimg = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())

        if a.ndim == 2:
            if a.dtype == np.uint16:
                h, w = a.shape
                qimg = QImage(a.data, w, h, 2 * w, QImage.Format.Format_Grayscale16)
                return QPixmap.fromImage(qimg.copy())

            if np.issubdtype(a.dtype, np.floating):
                lo = np.percentile(a, 1.0)
                hi = np.percentile(a, 99.0)
                if hi <= lo:
                    hi = lo + 1.0
                a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            elif a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)

            h, w = a.shape
            qimg = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8)
            return QPixmap.fromImage(qimg.copy())

        gray = a.mean(axis=-1)
        return self._array_to_qpixmap(gray)

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
        self.points_label.setText(f"Points: {n_base + n_custom} (base {n_base} + custom {n_custom})")

    def _set_input_image(self, frame):
        bg = self.apply_filter(frame, self.current_filter)
        pix = self._array_to_qpixmap(bg)
        self.lbl_input.setPixmap(pix, overlay_data={})

    def _update_label_with_overlay(self, label, arr, ptsA=None, custom_pts=None, tracks=None, show_trajectories=False):
        pix = self._array_to_qpixmap(arr)
        overlay_data = {}

        points_list = []
        if ptsA is not None:
            points_list.append((ptsA, (255, 255, 0), 4))
        if tracks is not None:
            points_list.append((tracks, (255, 255, 0), 2))
        if points_list:
            overlay_data['points'] = points_list

        if custom_pts is not None and len(custom_pts) > 0:
            overlay_data['crosses'] = [(custom_pts, (255, 255, 0), 6)]

        if show_trajectories and 'tracks' in self.viz_data and self.viz_data['tracks'] is not None:
            full_tracks = self.viz_data['tracks']  # (T,N,2)
            overlay_data['trajectories'] = (full_tracks, self.current_frame, 15, (0, 255, 255))

        label.setPixmap(pix, overlay_data)

    # ---------- UI ----------
    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        controls = QWidget(central_widget)
        controls_grid = QGridLayout(controls)
        controls_grid.setContentsMargins(6, 6, 6, 6)
        controls_grid.setHorizontalSpacing(8)
        controls_grid.setVerticalSpacing(4)

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.select_file)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Regular grid",
            "Features (Shi–Tomasi)",
            "Activity map (motion)",
            "Segmentation centroids",
        ])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Gaussian", "Sobel", "Canny edge"])
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)

        self.lbl_filter = QLabel("Filter:")
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(10, 5000)
        self.num_points_spin.setValue(400)
        self.num_points_spin.valueChanged.connect(self.on_preset_params_changed)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(1, 100)
        self.min_dist_spin.setValue(10)
        self.min_dist_spin.valueChanged.connect(self.on_preset_params_changed)

        self.apply_preset_btn = QPushButton("Apply Preset")
        self.apply_preset_btn.clicked.connect(self.apply_point_preset)

        self.grid_rows_ctrl = QLineEdit(str(getattr(self, "grid_rows", 4)))
        self.grid_cols_ctrl = QLineEdit(str(getattr(self, "grid_cols", 4)))

        self.process_btn = QPushButton("Run Registration")
        self.process_btn.clicked.connect(self.run_processing)
        self.process_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        self.edit_points_btn = QCheckBox("Edit Points")
        self.edit_points_btn.stateChanged.connect(self.toggle_edit_mode)
        self.edit_points_btn.setEnabled(False)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)

        self.undo_btn = QPushButton("Undo All")
        self.undo_btn.clicked.connect(self.undo_all_changes)
        self.undo_btn.setEnabled(False)

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_all_zoom)

        self.points_label = QLabel("Points: 0")
        self.status_label = QLabel("Ready - Drag & drop video or use Load button")
        self.status_label.setWordWrap(True)

        self.lbl_N = QLabel("N:")
        self.lbl_minDist = QLabel("MinDist:")
        self.lbl_rows = QLabel("Rows:")
        self.lbl_cols = QLabel("Cols:")

        c = 0
        controls_grid.addWidget(self.load_btn, 0, c); c += 1
        controls_grid.addWidget(QLabel("Preset:"), 0, c); c += 1
        controls_grid.addWidget(self.preset_combo, 0, c); c += 1
        controls_grid.addWidget(self.lbl_N, 0, c); c += 1
        controls_grid.addWidget(self.num_points_spin, 0, c); c += 1
        controls_grid.addWidget(self.lbl_minDist, 0, c); c += 1
        controls_grid.addWidget(self.min_dist_spin, 0, c); c += 1
        controls_grid.addWidget(self.lbl_filter, 0, c); c += 1
        controls_grid.addWidget(self.filter_combo, 0, c); c += 1
        controls_grid.addWidget(self.lbl_rows, 0, c); c += 1
        controls_grid.addWidget(self.grid_rows_ctrl, 0, c); c += 1
        controls_grid.addWidget(self.lbl_cols, 0, c); c += 1
        controls_grid.addWidget(self.grid_cols_ctrl, 0, c); c += 1
        controls_grid.addWidget(self.apply_preset_btn, 0, c); c += 1
        controls_grid.addWidget(self.process_btn, 0, c); c += 1
        controls_grid.addWidget(self.stop_btn, 0, c); c += 1
        ncols = c

        c = 0
        controls_grid.addWidget(self.edit_points_btn, 1, c); c += 1
        controls_grid.addWidget(self.clear_points_btn, 1, c); c += 1
        controls_grid.addWidget(self.undo_btn, 1, c); c += 1
        controls_grid.addWidget(self.reset_zoom_btn, 1, c); c += 1
        controls_grid.addWidget(self.points_label, 1, c); c += 1

        controls_grid.addWidget(self.status_label, 2, 0, 1, ncols)
        main_layout.addWidget(controls)

        self.grid_widget = QWidget(central_widget)
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)
        self.grid_layout.setHorizontalSpacing(6)
        self.grid_layout.setVerticalSpacing(6)
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

        self.lbl_input_title = QLabel("Input")
        self.lbl_input = ImageLabel()

        self.lbl_grid_title = QLabel("Tracked Points")
        self.lbl_grid = ImageLabel()

        self.lbl_warped_title = QLabel("Corrected Video")
        self.lbl_warped = ImageLabel()

        self.lbl_tracks_title = QLabel("Deformation Field")
        self.lbl_tracks = ImageLabel()

        panels = [
            (self.lbl_input_title, self.lbl_input),
            (self.lbl_grid_title, self.lbl_grid),
            (self.lbl_warped_title, self.lbl_warped),
            (self.lbl_tracks_title, self.lbl_tracks),
        ]
        for i, (title, label) in enumerate(panels):
            r, col = divmod(i, 2)
            vbox = QVBoxLayout()
            vbox.addWidget(title)
            vbox.addWidget(label)
            self.grid_layout.addLayout(vbox, r, col)

        main_layout.addWidget(self.grid_widget, stretch=1)

        playback_layout = QHBoxLayout()

        self.play_btn = QPushButton("⏯️")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)

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
        playback_layout.addWidget(self.speed_slider)
        playback_layout.addWidget(self.speed_label)
        playback_layout.addWidget(self.slider, stretch=1)
        playback_layout.addWidget(self.frame_label)
        playback_layout.addWidget(self.progress_bar)

        main_layout.addLayout(playback_layout)

        self._update_preset_controls()
        self.status_label.setText("UI loaded. Click 'Load File' or drag & drop a file.")

    # ---------- preset controls ----------
    def _set_visible(self, widgets, visible: bool):
        for w in widgets:
            w.setVisible(visible)

    def _update_preset_controls(self):
        preset = self.preset_combo.currentText()
        is_grid = preset.startswith("Regular grid")
        is_features = preset.startswith("Features")
        is_activity = preset.startswith("Activity")
        is_seg = preset.startswith("Segmentation")

        show_rows_cols = is_grid
        show_n = is_features or is_activity or is_seg
        show_mindist = is_features or is_activity

        self._set_visible([self.lbl_rows, self.grid_rows_ctrl, self.lbl_cols, self.grid_cols_ctrl], show_rows_cols)
        self._set_visible([self.lbl_N, self.num_points_spin], show_n)
        self._set_visible([self.lbl_minDist, self.min_dist_spin], show_mindist)

        self.apply_preset_btn.setVisible(True)
        self.apply_preset_btn.setEnabled(True)

    def on_preset_changed(self, *_):
        self._update_preset_controls()
        if self.video_data is not None:
            self.apply_point_preset()

    def on_filter_changed(self):
        self.current_filter = self.filter_combo.currentText()
        self.update_plot()

    def on_preset_params_changed(self):
        if self.video_data is not None:
            self.apply_point_preset()

    # ---------- presets ----------
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

    def _clip_border(self, pts, H, W, border=10):
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
        return self._clip_border(pts, H, W, border=10)

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
        return self._clip_border(pts, H, W, border=10)

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
        cent = self._clip_border(cent, H, W, border=10)
        return cent

    def _preset_regular_grid(self, H, W, rows, cols):
        xs = np.linspace(0, W - 1, cols, dtype=np.float32)
        ys = np.linspace(0, H - 1, rows, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)

    def apply_point_preset(self):
        if self.video_data is None:
            return

        gray = self._get_gray_video_np()
        H, W = gray.shape[1], gray.shape[2]
        ref_u8 = self._reference_image(gray)
        ref_u8_filt = self.apply_filter(ref_u8, self.current_filter)

        preset = self.preset_combo.currentText()

        if preset.startswith("Regular grid"):
            try:
                rows = int(self.grid_rows_ctrl.text().strip())
                cols = int(self.grid_cols_ctrl.text().strip())
            except ValueError:
                rows, cols = self.grid_rows, self.grid_cols

            rows = max(2, rows)
            cols = max(2, cols)
            self.grid_rows, self.grid_cols = rows, cols
            pts = self._preset_regular_grid(H, W, rows, cols)
        else:
            num_points = int(self.num_points_spin.value())
            min_dist = int(self.min_dist_spin.value())

            if preset.startswith("Features"):
                pts = self._preset_features(ref_u8_filt, num_points=num_points, min_dist=min_dist)
            elif preset.startswith("Activity"):
                pts = self._preset_activity(gray, num_points=num_points, min_dist=min_dist)
            else:
                pts = self._preset_segmentation_centroids(ref_u8_filt, num_points=num_points)

        self.base_grid_pts = pts.astype(np.float32)
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.base_grid_pts0 = self.base_grid_pts.copy()

        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.status_label.setText(f"Preset applied: {preset} ({len(self.base_grid_pts)} pts)")

    # ---------- file loading ----------
    def clear_all_data(self):
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None

        self.viz_data.clear()
        self.custom_grid_pts = []

        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
        self.current_frame = 0

        self.lbl_input.clear()
        self.lbl_grid.clear()
        self.lbl_warped.clear()
        self.lbl_tracks.clear()

    def select_file(self):
        file_dialog = QFileDialog(self, "Select Video File")
        file_dialog.setNameFilters([
            "Video files (*.avi *.mp4 *.tiff *.tif *.h5)",
            "All files (*.*)"
        ])
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.video_path = selected_files[0]
                self.status_label.setText(f"Loading: {os.path.basename(self.video_path)}")
                self.load_data()

    def load_data(self):
        if not self.video_path:
            return

        self.clear_all_data()
        ext = os.path.splitext(self.video_path)[1].lower()
        self.status_label.setText("Loading data...")
        QApplication.processEvents()

        try:
            if ext in ['.avi', '.mp4']:
                self.video_data = load_video(self.video_path, resize=False, max_frames=50)
            elif ext in ['.tiff', '.tif']:
                video_np = load_tiff(self.video_path, frames=50)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            elif ext in ['.h5', '.hdf5']:
                video_np = load_hdf5(self.video_path, "/data", frames=50)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            else:
                self.status_label.setText(f"Unsupported file extension: {ext}")
                return

            self.apply_point_preset()
            self.undo_btn.setEnabled(True)

            self.status_label.setText(f"Loaded: {os.path.basename(self.video_path)}")

            self.total_frames = int(self.video_data.shape[0])
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)

            self.current_frame = 0
            self.update_frame_label()

            self.process_btn.setEnabled(True)
            self.edit_points_btn.setEnabled(True)
            self.clear_points_btn.setEnabled(True)

            self.play_btn.setEnabled(True)
            self.step_back_btn.setEnabled(True)
            self.step_fwd_btn.setEnabled(True)

            self.visualize()

        except Exception as e:
            self.status_label.setText(f"Error loading data: {e}")
            self.video_data = None

    # ---------- editing ----------
    def toggle_edit_mode(self):
        self.edit_mode = self.edit_points_btn.isChecked()
        if self.edit_mode:
            self.status_label.setText("Edit Mode: drag to move | right-click base toggles delete | right-click custom deletes")
            if not self._img_cids:
                self.lbl_grid.pressed.connect(self.on_image_press)
                self.lbl_grid.moved.connect(self.on_image_move)
                self.lbl_grid.released.connect(self.on_image_release)
                self._img_cids = {"press": True, "move": True, "release": True}
        else:
            try:
                if self._img_cids.get('press'):
                    self.lbl_grid.pressed.disconnect(self.on_image_press)
                if self._img_cids.get('move'):
                    self.lbl_grid.moved.disconnect(self.on_image_move)
                if self._img_cids.get('release'):
                    self.lbl_grid.released.disconnect(self.on_image_release)
            except Exception:
                pass
            self._img_cids.clear()
            self._drag = None
            self.status_label.setText(f"Loaded: {os.path.basename(self.video_path) if self.video_path else 'Ready'}")

        self.update_grid_display_pre_registration()

    def _nearest_point(self, pts, x, y):
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def on_image_press(self, x, y, button_value):
        if not self.edit_mode:
            return

        # normalize button
        btn = 1
        if button_value == Qt.MouseButton.RightButton.value:
            btn = 3
        elif button_value == Qt.MouseButton.MiddleButton.value:
            btn = 2

        x_f, y_f = float(x), float(y)

        # Right click: delete custom or toggle base
        if btn == 3:
            cidx, cdist = None, 1e9
            if len(self.custom_grid_pts) > 0:
                cpts = np.array(self.custom_grid_pts, dtype=np.float32)
                cidx, cdist = self._nearest_point(cpts, x_f, y_f)

            bidx, bdist = None, 1e9
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                bidx, bdist = self._nearest_point(self.base_grid_pts, x_f, y_f)

            if cidx is not None and cdist < 20 and cdist <= bdist:
                self.custom_grid_pts.pop(cidx)
                self._update_points_label()
                self.update_grid_display_pre_registration()
                return

            if bidx is not None and bdist < 20:
                self.base_grid_active[bidx] = ~self.base_grid_active[bidx]
                self._update_points_label()
                self.update_grid_display_pre_registration()
            return

        # Left click: drag nearest active base/custom or add custom
        if btn != 1:
            return

        bidx, bdist = None, 1e9
        if self.base_grid_pts is not None and np.any(self.base_grid_active):
            idxs = np.where(self.base_grid_active)[0]
            ptsA = self.base_grid_pts[idxs]
            loc_idx, bdist = self._nearest_point(ptsA, x_f, y_f)
            bidx = int(idxs[loc_idx])

        cidx, cdist = None, 1e9
        if len(self.custom_grid_pts) > 0:
            cpts = np.array(self.custom_grid_pts, dtype=np.float32)
            cidx, cdist = self._nearest_point(cpts, x_f, y_f)

        tol = 15.0
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

        self.custom_grid_pts.append((int(x_f), int(y_f)))
        self._update_points_label()
        self.update_grid_display_pre_registration()

    def on_image_move(self, x, y):
        if not self.edit_mode or self._drag is None:
            return
        kind, idx = self._drag
        if kind == "base":
            self.base_grid_pts[idx] = (float(x), float(y))
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()

    def on_image_release(self):
        self._drag = None

    def clear_points(self):
        self.custom_grid_pts = []
        self._update_points_label()
        self.update_grid_display_pre_registration()

    def undo_all_changes(self):
        if self.base_grid_pts0 is None:
            return
        self.base_grid_pts = self.base_grid_pts0.copy()
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.custom_grid_pts = []
        self._update_points_label()
        self.update_grid_display_pre_registration()

    def reset_all_zoom(self):
        for label in [self.lbl_input, self.lbl_grid, self.lbl_warped, self.lbl_tracks]:
            label.reset_zoom()
        self.status_label.setText("Zoom reset on all panels")

    # ---------- visualization ----------
    def _get_current_tracks(self):
        tr = self.viz_data.get('tracks', None)
        if tr is None or not hasattr(tr, "ndim") or tr.ndim != 3 or tr.shape[0] == 0:
            return None
        idx = min(self.current_frame, tr.shape[0] - 1)
        return tr[idx]

    def update_grid_display_pre_registration(self):
        if self.video_data is None:
            return
        frame0 = self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        bg = self.apply_filter(frame0, self.current_filter)

        ptsA = None
        if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
            ptsA = self.base_grid_pts[self.base_grid_active]

        custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if self.custom_grid_pts else None
        self._update_label_with_overlay(self.lbl_grid, bg, ptsA=ptsA, custom_pts=custom_pts)

    def visualize(self):
        if self.video_data is None:
            return

        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("▶")

        if 'input' not in self.viz_data:
            self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        if self.warped_video is not None:
            self.viz_data['warped'] = self.warped_video

        tracks_np = self._tracks_to_numpy()
        if tracks_np is not None:
            tracks_np = np.asarray(tracks_np)
            if tracks_np.ndim == 4:          # (B,T,N,2)
                tracks_np = tracks_np[0]     # -> (T,N,2)
            self.viz_data['tracks'] = tracks_np

        if self.disp_field is not None:
            df = self.disp_field
            if isinstance(df, torch.Tensor):
                df = df.detach().cpu().numpy()
            self.viz_data['disp_field'] = np.asarray(df)

        self.update_plot()

    def update_plot(self):
        if 'input' not in self.viz_data:
            return

        input_frame = self.viz_data['input'][self.current_frame]

        # Panel 1: Input (filtered, no overlays)
        self._set_input_image(input_frame)

        # Panel 2: Tracked points over the SAME filtered background
        bg = self.apply_filter(input_frame, self.current_filter)
        if not self.edit_mode and 'tracks' in self.viz_data:
            current_tracks = self._get_current_tracks()
            self._update_label_with_overlay(self.lbl_grid, bg, tracks=current_tracks, show_trajectories=True)
        else:
            self.update_grid_display_pre_registration()

        # Panel 3: Warped (filtered)
        warped = self.viz_data.get('warped', None)
        if warped is not None:
            warp_idx = min(self.current_frame, warped.shape[0] - 1)
            warped_frame = warped[warp_idx]
            warped_bg = self.apply_filter(warped_frame, self.current_filter)
            self._update_label_with_overlay(self.lbl_warped, warped_bg)
        else:
            self.lbl_warped.clear()

        # Panel 4: Deformation field
        disp_stack = self.viz_data.get('disp_field', None)
        if disp_stack is not None:
            d_idx = min(self.current_frame, disp_stack.shape[0] - 1)
            if disp_stack.ndim == 3:
                img = self.disp_scalar_u8(disp_stack[d_idx])
                self._update_label_with_overlay(self.lbl_tracks, img)
            elif disp_stack.ndim == 4 and disp_stack.shape[-1] == 2:
                img = self.flow_to_color(disp_stack[d_idx])
                self._update_label_with_overlay(self.lbl_tracks, img)
        else:
            self.lbl_tracks.clear()

    # ---------- processing ----------
    def run_processing(self):
        if self.video_data is None:
            return

        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        base_active = self.base_grid_pts[self.base_grid_active].astype(np.float32)
        base_grid_tensor = torch.from_numpy(base_active).unsqueeze(0)

        custom_pts_tensor = None
        if len(self.custom_grid_pts) > 0:
            pts_array = np.array(self.custom_grid_pts, dtype=np.float32)
            custom_pts_tensor = torch.from_numpy(pts_array)

        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass

        self.worker = ProcessingWorker(self.video_data, base_grid_tensor, custom_pts_tensor, parent=self)
        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    def on_tracks_ready(self, pred_tracks, pred_visibility, grid_pts):
        pred_tracks = np.asarray(pred_tracks)
        if pred_tracks.ndim == 4:      # (B,T,N,2)
            pred_tracks = pred_tracks[0]

        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        self.visualize()

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video, disp_field):
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        self.warped_video = warped_video
        self.disp_field = disp_field

        self.status_label.setText("Processing complete")
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        try:
            self.worker.quit()
            self.worker.wait(200)
        except Exception:
            pass
        self.worker = None

        self.visualize()

    def on_processing_error(self, error_msg):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Processing Error", error_msg)

        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        if self.worker is not None:
            try:
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass
        self.worker = None

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
        self.status_label.setText("Stopped")
        self.progress_bar.setValue(0)

    # ---------- playback ----------
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
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.slider.setValue(self.current_frame)
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

    # ---------- drag&drop ----------
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
                    self.status_label.setText(f"Loading: {os.path.basename(self.video_path)}")
                    self.load_data()
                    event.acceptProposedAction()
                    return

    def closeEvent(self, event):
        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(500)
            except Exception:
                pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MotionCorrectionGUI()
    gui.show()
    sys.exit(app.exec())
