
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('QtAgg')
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    # Fallback for older Matplotlib that may not include backend_qtagg
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except Exception as e:
        raise ImportError(
            "Matplotlib backend import error. Your Matplotlib version may not support PyQt6. "
            "Please upgrade matplotlib (>=3.7) or install PyQt5/PySide2."
        ) from e
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QCheckBox, QGridLayout, QComboBox, QSpinBox, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPalette

import cv2
# QIcon import removed

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5
from cotracker.models.core.model_utils import get_points_on_a_grid


class ProcessingWorker(QThread):
    tracks_ready = pyqtSignal(object, object, object)  # tracks_np, visibility, grid
    progress = pyqtSignal(int)  # percent complete
    finished = pyqtSignal(object, object, object, object, object)  # tracks_np, visibility, grid, warped_full, disp_field
    error = pyqtSignal(str)

    def __init__(self, video_data, base_grid_pts, custom_grid_pts=None, low_res_factor=2, 
                 cached_tracks=None, cached_base_pts=None, cached_custom_pts=None, parent=None):
        super().__init__(parent)
        self.video_data = video_data
        self.base_grid_pts = base_grid_pts
        self.custom_grid_pts = custom_grid_pts
        self.low_res_factor = low_res_factor
        self.cached_tracks = cached_tracks
        self.cached_base_pts = cached_base_pts
        self.cached_custom_pts = cached_custom_pts
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # Determine which points are new/changed
            base_pts_np = self.base_grid_pts[0].cpu().numpy()  # (N_base, 2)
            custom_pts_list = self.custom_grid_pts if self.custom_grid_pts is not None else []
            
            # Check if we can use cached tracks
            use_cache = (self.cached_tracks is not None and 
                        self.cached_base_pts is not None)
            
            if use_cache:
                # Find which points are unchanged
                cached_base_np = self.cached_base_pts
                cached_custom_list = self.cached_custom_pts if self.cached_custom_pts else []
                
                # Identify new/changed points
                new_base_mask = np.ones(len(base_pts_np), dtype=bool)
                if len(cached_base_np) == len(base_pts_np):
                    # Check if base points are identical
                    diffs = np.linalg.norm(base_pts_np - cached_base_np, axis=1)
                    new_base_mask = diffs > 1e-3  # Threshold for "changed"
                
                # Check custom points
                new_custom_indices = []
                for i, pt in enumerate(custom_pts_list):
                    if pt not in cached_custom_list:
                        new_custom_indices.append(i)
                
                n_new_base = new_base_mask.sum()
                n_new_custom = len(new_custom_indices)
                
                print(f"Incremental tracking: {n_new_base} new/changed base points, {n_new_custom} new custom points")
                
                if n_new_base == 0 and n_new_custom == 0:
                    # No changes - use cached tracks entirely
                    print("No changes detected, using cached tracks")
                    tracks_np = self.cached_tracks
                    vis_np = np.ones((tracks_np.shape[0], tracks_np.shape[1], tracks_np.shape[2]), dtype=bool)
                    grid_xy = self.base_grid_pts
                    if len(custom_pts_list) > 0:
                        custom_xy = torch.from_numpy(np.array(custom_pts_list, dtype=np.float32)).unsqueeze(0)
                        grid_xy = torch.cat([grid_xy, custom_xy.to(grid_xy.device)], dim=1)
                    
                    self.tracks_ready.emit(tracks_np, vis_np, grid_xy)
                    self.progress.emit(10)
                    
                    # Skip to warping
                    import time
                    t2 = time.time()
                    warped_video, disp_field = warping(tracks_np, self.video_data)
                    t3 = time.time()
                    print(f"warping time: {t3 - t2:.3f}s")
                    warped_np = np.asarray(warped_video)
                    self.progress.emit(100)
                    self.finished.emit(tracks_np, vis_np, grid_xy, warped_np, disp_field)
                    return
            
            # Build queries for new/changed points only or all points if no cache
            grid_xy = self.base_grid_pts
            if self.custom_grid_pts is not None and len(self.custom_grid_pts) > 0:
                custom_xy = self.custom_grid_pts
                if isinstance(custom_xy, np.ndarray):
                    custom_xy = torch.from_numpy(custom_xy)
                custom_xy = custom_xy.to(grid_xy.device).float()
                if custom_xy.ndim == 2:
                    custom_xy = custom_xy.unsqueeze(0)
                grid_xy = torch.cat([grid_xy, custom_xy], dim=1)

            t = torch.zeros((*grid_xy.shape[:-1], 1), device=grid_xy.device, dtype=grid_xy.dtype)
            queries = torch.cat([t, grid_xy], dim=-1)

            import time
            t0 = time.time()
            pred_tracks, pred_visibility = cotrack_model(self.video_data, queries=queries)
            t1 = time.time()
            print(f"cotrack_model time: {t1 - t0:.3f}s")

            # Send tracks to GUI right away
            tracks_np = pred_tracks.detach().cpu().numpy()
            # Convert visibility to numpy if tensor
            try:
                vis_np = pred_visibility.detach().cpu().numpy()
            except Exception:
                vis_np = pred_visibility
            self.tracks_ready.emit(tracks_np, vis_np, grid_xy)
            self.progress.emit(10)

            # Preview generation removed
            self.progress.emit(50)

            # Full-resolution warp
            t2 = time.time()
            warped_video, disp_field = warping(tracks_np, self.video_data)
            t3 = time.time()
            print(f"warping time: {t3 - t2:.3f}s")

            # Keep raw warped result dtype as returned by warping() - do not convert to uint8 here
            warped_np = np.asarray(warped_video)

            self.progress.emit(100)
            self.finished.emit(tracks_np, pred_visibility, grid_xy, warped_np, disp_field)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Zoom/pan QLabel as a viewport. Zoom=1 fits image to panel. No window resizing side-effects."""
    pressed = pyqtSignal(int, int, int)  # x, y, button
    moved = pyqtSignal(int, int)
    released = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Let layouts resize this widget
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Don't force a large min size; keep it small so the window can shrink too
        self.setMinimumSize(10, 10)

        self.zoom_factor = 1.0              # relative zoom; 1.0 = fit-to-panel
        self.pan_offset = [0.0, 0.0]        # in widget pixels (viewport pan)
        self._base_image = None             # QPixmap
        self._overlay_data = {}

        self._fit_scale = 1.0              # computed from panel size / image size
        self._is_panning = False
        self._pan_start = None

        self._viewport_pixmap = QPixmap()

        # Optional: Alt + Left drag to pan (keeps left-drag free for editing points)
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

        # Use theme/palette background instead of hard black
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
            if tracks is None or len(tracks) == 0:
                return
            start_frame = max(0, current_frame - trail_length)

            for point_idx in range(tracks.shape[1]):
                for t in range(start_frame, current_frame):
                    pt1 = tracks[t, point_idx]
                    pt2 = tracks[t + 1, point_idx]
                    if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
                        continue

                    alpha = int(255 * (t - start_frame + 1) / (current_frame - start_frame + 1))
                    alpha = max(50, min(255, alpha))
                    qcolor = QColor(color[0], color[1], color[2], alpha)

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

        # zoom towards cursor
        mouse = event.position()
        geom = self._draw_position_and_scale()
        if geom is not None and self.zoom_factor != old_zoom:
            x0, y0, sw, sh, old_scale = geom
            x_img = (mouse.x() - x0) / old_scale
            y_img = (mouse.y() - y0) / old_scale

            # new scale after zoom
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
        
        # Enable drag and drop
        self.setAcceptDrops(True)

        self.video_path = None
        self.video_data = None
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None

        self.grid_rows = 4
        self.grid_cols = 4
        self.base_grid_pts = None
        self.base_grid_active = None
        self.base_grid_pts0 = None

        self._drag = None
        self._mpl_cids = {}
        self._img_cids = {}

        self.base_scatter = None
        self.custom_scatter = None

        self.custom_grid_pts = []
        self.edit_mode = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.playback_speed = 1.0

        self.viz_data = {}
        self.viz_artists = {}
        
        # Tracking cache for incremental updates
        self.cached_tracks = None  # Full tracks array from last run
        self.cached_visibility = None
        self.cached_base_pts = None  # Base grid points from last run
        self.cached_custom_pts = None  # Custom points from last run

        self.init_ui()
        self.worker = None

    @staticmethod
    def flow_to_color(flow):
        # flow: (H, W, 2)
        if flow is None:
            return None
        # Calculate magnitude
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Normalize to 0-255 range for display
        # We use a dynamic range based on the max flow in this frame, 
        # or a fixed range if preferred. Here we use dynamic range for visibility.
        if mag.max() > 0:
            mag_norm = (mag / mag.max() * 255.0).astype(np.uint8)
        else:
            mag_norm = mag.astype(np.uint8)
            
        return mag_norm

    def _tracks_to_numpy(self):
        if self.pred_tracks is None:
            return None
        if isinstance(self.pred_tracks, np.ndarray):
            return self.pred_tracks
        try:
            return self.pred_tracks.detach().cpu().numpy()
        except Exception:
            return np.asarray(self.pred_tracks)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # -----------------------------
        # Create ALL controls FIRST
        # -----------------------------
        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.select_file)

        # Preset dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Regular grid",
            "Features (Shi–Tomasi)",
            "Activity map (motion)",
            "Segmentation centroids",
        ])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)

        # Preset params (non-grid)
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

        # Grid-only params (enabled only when "Regular grid" is selected)
        self.grid_rows_ctrl = QLineEdit(str(getattr(self, "grid_rows", 4)))
        self.grid_cols_ctrl = QLineEdit(str(getattr(self, "grid_cols", 4)))

        # Processing
        self.process_btn = QPushButton("Run Registration")
        self.process_btn.clicked.connect(self.run_processing)
        self.process_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        # Editing
        self.edit_points_btn = QCheckBox("Edit Points")
        self.edit_points_btn.stateChanged.connect(self.toggle_edit_mode)
        self.edit_points_btn.setEnabled(False)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)

        self.undo_btn = QPushButton("Undo All")
        self.undo_btn.clicked.connect(self.undo_all_changes)
        elif a.ndim == 3 and a.shape[2] == 3:
            h, w, _ = a.shape
            # Ensure format is RGB
            qimg = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())
            return pix
        else:
            # Unknown shape, convert to grayscale
            gray = a.mean(axis=-1).astype(np.uint8)
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
            return QPixmap.fromImage(qimg.copy())

    def _draw_points_on_pixmap(self, pixmap, points, color=(255, 0, 0), radius=3):
        """Return a new QPixmap with the points drawn on top."""
        if points is None or len(points) == 0:
            return pixmap
        p = QPixmap(pixmap)
        painter = QPainter(p)
        qcolor = QColor(color[0], color[1], color[2])
        painter.setBrush(qcolor)
        painter.setPen(qcolor)
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        return p

    def _draw_cross_on_pixmap(self, pixmap, points, color=(255, 255, 0), size=6):
        if points is None or len(points) == 0:
            return pixmap
        p = QPixmap(pixmap)
        painter = QPainter(p)
        qcolor = QColor(color[0], color[1], color[2])
        painter.setPen(qcolor)
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            painter.drawLine(x - size, y - size, x + size, y + size)
            painter.drawLine(x - size, y + size, x + size, y - size)
        painter.end()
        return p

    def _draw_trajectories_on_pixmap(self, pixmap, tracks, current_frame, trail_length=10, color=(0, 255, 255)):
        """Draw trajectories (trails) for tracked points.
        
        Args:
            pixmap: QPixmap to draw on
            tracks: (T, N, 2) array of track positions
            current_frame: current frame index
            trail_length: number of frames to show in trail
            color: RGB tuple for trajectory color
        """
        if tracks is None or len(tracks) == 0:
            return pixmap
        
        p = QPixmap(pixmap)
        painter = QPainter(p)
        
        # Draw trajectories with fading alpha
        start_frame = max(0, current_frame - trail_length)
        
        for point_idx in range(tracks.shape[1]):
            # Draw line segments for this point's trajectory
            for t in range(start_frame, current_frame):
                if t + 1 > current_frame:
                    break
                    
                pt1 = tracks[t, point_idx]
                pt2 = tracks[t + 1, point_idx]
                
                # Skip if points are invalid
                if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
                    continue
                
                # Calculate alpha based on how recent the segment is
                alpha = int(255 * (t - start_frame + 1) / (current_frame - start_frame + 1))
                alpha = max(50, min(255, alpha))  # Clamp between 50-255
                
                qcolor = QColor(color[0], color[1], color[2], alpha)
                pen = painter.pen()
                pen.setColor(qcolor)
                pen.setWidth(2)
                painter.setPen(pen)
                
                painter.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
        
        painter.end()
        return p

    def _update_label_with_overlay(self, label, arr, ptsA=None, custom_pts=None, tracks=None, show_trajectories=False):
        pix = self._array_to_qpixmap(arr)
        
        # Build overlay data dictionary
        overlay_data = {}
        
        # Collect points to draw
        points_list = []
        if ptsA is not None:
            points_list.append((ptsA, (255, 255, 0), 4))
        if tracks is not None:
            points_list.append((tracks, (255, 255, 0), 2))
        if points_list:
            overlay_data['points'] = points_list
        
        # Collect crosses to draw
        if custom_pts is not None and len(custom_pts) > 0:
            overlay_data['crosses'] = [(custom_pts, (255, 255, 0), 6)]
        
        # Add trajectory data if requested
        if show_trajectories and hasattr(self, 'viz_data') and 'tracks' in self.viz_data:
            full_tracks = self.viz_data['tracks']  # (T, N, 2)
            overlay_data['trajectories'] = (full_tracks, self.current_frame, 15, (0, 255, 255))
        
        # Set pixmap with overlay data
        label.setPixmap(pix, overlay_data)

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
                self.status_label.setWordWrap(True)
                self.status_label.setMaximumHeight(40)   # tweak 30–60
                self.load_data()

    def show_input_frame_only(self):
        frame = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        if 'input' not in self.viz_data:
            self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # Update input label
        self._update_label_with_overlay(self.lbl_input, frame)
        # Clear other panels
        self.lbl_grid.clear()
        self.lbl_warped.clear()
        self.lbl_tracks.clear()

    def on_grid_size_enter(self):
        self.on_apply_grid_size()

    def on_apply_grid_size(self):
        try:
            r = int(self.grid_rows_ctrl.text().strip())
            c = int(self.grid_cols_ctrl.text().strip())
        except ValueError:
            self.status_label.setText("Rows and Cols must be integers.")
            self.grid_rows_ctrl.setText(str(self.grid_rows))
            self.grid_cols_ctrl.setText(str(self.grid_cols))
            return
        if r < 2 or c < 2:
            self.status_label.setText("Rows and Cols must be >= 2.")
            self.grid_rows_ctrl.setText(str(self.grid_rows))
            self.grid_cols_ctrl.setText(str(self.grid_cols))
            return
        self.grid_rows = r
        self.grid_cols = c
        if self.video_data is not None:
            self.reset_base_grid_from_size()
            self.status_label.setText(f"Grid updated: {self.grid_rows}x{self.grid_cols}")

    def _generate_grid_points(self):
        """Generate (1, N, 2) tensor of grid points based on rows/cols."""
        if self.video_data is None:
            return None
        _, _, H, W = self.video_data.shape
        xs = np.linspace(0, W - 1, self.grid_cols)
        ys = np.linspace(0, H - 1, self.grid_rows)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
        return torch.from_numpy(points).float().unsqueeze(0)

    def reset_base_grid_from_size(self):
        """Rebuild editable base grid using current self.grid_rows/cols. Clears edits."""
        grid_xy = self._generate_grid_points()
        if grid_xy is None:
            return
        self.base_grid_pts = grid_xy[0].detach().cpu().numpy().astype(np.float32)  # (N,2)
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.base_grid_pts0 = self.base_grid_pts.copy()

        # Clear custom points too (usually what users expect when changing grid size)
        self.custom_grid_pts = []
        self.points_label.setText(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()


    def on_preset_params_changed(self):
        # light: regenerate points when params change (if data loaded)
        if self.video_data is not None:
            self.apply_point_preset()

    def _get_gray_video_np(self):
        """Return grayscale video as (T,H,W) float32 numpy."""
        vd = self.video_data
        if "torch" in str(type(vd)):
            vd = vd.detach().cpu().numpy()
        vd = np.asarray(vd)

        # expected: (T,C,H,W) with C=3 in your loader
        if vd.ndim == 4 and vd.shape[1] in (1, 3):
            gray = vd[:, 0].astype(np.float32)  # take channel 0
        elif vd.ndim == 3:
            gray = vd.astype(np.float32)
        else:
            raise ValueError(f"Unsupported video_data shape for grayscale: {vd.shape}")

        return gray

    def _reference_image(self, gray_video, n=10):
        """Denoised reference image from first n frames."""
        n = int(min(max(n, 1), gray_video.shape[0]))
        ref = gray_video[:n].mean(axis=0)
        ref = cv2.GaussianBlur(ref, (0, 0), 1.0)
        # normalize to uint8 for robust feature/threshold ops
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # CLAHE helps across datasets
        ref_u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ref_u8)
        return ref_u8

    def _clip_border(self, pts, H, W, border=10):
        if pts is None or len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        good = (pts[:, 0] >= border) & (pts[:, 0] < (W - border)) & (pts[:, 1] >= border) & (pts[:, 1] < (H - border))
        return pts[good]

    def _preset_features(self, ref_u8, num_points=400, min_dist=10):
        """Shi–Tomasi features on reference image."""
        H, W = ref_u8.shape[:2]

        # foreground-ish mask (optional but helps a lot)
        _, m = cv2.threshold(ref_u8, 0, 255, cv2.THRESH_OTSU)
        m = cv2.medianBlur(m, 5)

        pts = cv2.goodFeaturesToTrack(
            ref_u8,
            maxCorners=int(num_points),
            qualityLevel=0.01,
            minDistance=float(min_dist),
            blockSize=7,
            useHarrisDetector=False,
            mask=m
        )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 2).astype(np.float32)  # (x,y)
        return self._clip_border(pts, H, W, border=10)

    def _preset_activity(self, gray_video, num_points=400, min_dist=10):
        """Activity map (mean abs diff) -> features."""
        H, W = gray_video.shape[1], gray_video.shape[2]
        T = gray_video.shape[0]
        if T < 2:
            return np.zeros((0, 2), dtype=np.float32)

        diffs = np.abs(gray_video[1:] - gray_video[:-1])
        act = diffs.mean(axis=0)  # (H,W) float
        act_u8 = cv2.normalize(act, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        act_u8 = cv2.GaussianBlur(act_u8, (0, 0), 1.0)

        # mask to avoid pure background (optional)
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

    def _preset_segmentation_centroids(self, ref_u8, num_points=400):
        """Otsu segmentation -> connected components centroids."""
        H, W = ref_u8.shape[:2]
        _, bw = cv2.threshold(ref_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # clean up a bit
        bw = cv2.medianBlur(bw, 5)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

        # skip background label 0
        if num_labels <= 1:
            return np.zeros((0, 2), dtype=np.float32)

        areas = stats[1:, cv2.CC_STAT_AREA]
        cent = centroids[1:].astype(np.float32)  # (K,2) in (x,y)

        # take largest components up to num_points
        order = np.argsort(-areas)
        cent = cent[order][:int(num_points)]
        cent = self._clip_border(cent, H, W, border=10)
        return cent

    def _preset_regular_grid(self, H, W, rows, cols):
        xs = np.linspace(0, W - 1, cols, dtype=np.float32)
        ys = np.linspace(0, H - 1, rows, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # (N,2) (x,y)
        return pts

    def apply_point_preset(self):
        if self.video_data is None:
            return

        gray = self._get_gray_video_np()      # (T,H,W)
        H, W = gray.shape[1], gray.shape[2]
        ref_u8 = self._reference_image(gray)

        preset = self.preset_combo.currentText()

        if preset.startswith("Regular grid"):
            try:
                rows = int(self.grid_rows_ctrl.text().strip())
                cols = int(self.grid_cols_ctrl.text().strip())
            except ValueError:
                rows, cols = self.grid_rows, self.grid_cols

            rows = max(2, rows); cols = max(2, cols)
            self.grid_rows, self.grid_cols = rows, cols

            pts = self._preset_regular_grid(H, W, rows, cols)

        else:
            num_points = int(self.num_points_spin.value())
            min_dist = int(self.min_dist_spin.value())

            if preset.startswith("Features"):
                pts = self._preset_features(ref_u8, num_points=num_points, min_dist=min_dist)
            elif preset.startswith("Activity"):
                pts = self._preset_activity(gray, num_points=num_points, min_dist=min_dist)
            else:
                pts = self._preset_segmentation_centroids(ref_u8, num_points=num_points)

        self.base_grid_pts = pts.astype(np.float32)
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.base_grid_pts0 = self.base_grid_pts.copy()

        self.points_label.setText(f"Points: {int(self.base_grid_active.sum())}")
        self.update_grid_display_pre_registration()
        self.status_label.setText(f"Preset applied: {preset} ({len(self.base_grid_pts)} pts)")

    def on_visualize_grid_pts(self):
        if self.video_data is None:
            return
        self.update_grid_display_pre_registration()


    def update_grid_display_pre_registration(self):
        if self.video_data is None:
            return
        frame0 = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        # Base grid: only active points
        ptsA = None
        if self.base_grid_pts is not None and self.base_grid_active is not None:
            active = self.base_grid_active
            if np.any(active):
                ptsA = self.base_grid_pts[active]
        custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if len(self.custom_grid_pts) > 0 else None
        self._update_label_with_overlay(self.lbl_grid, frame0, ptsA=ptsA, custom_pts=custom_pts)

    def clear_all_data(self):
        """Clear all stored processing results and cached data."""
        # Clear processing results
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None
        
        # Clear cached tracking data
        self.cached_tracks = None
        self.cached_visibility = None
        self.cached_base_pts = None
        self.cached_custom_pts = None
        
        # Clear visualization data
        self.viz_data.clear()
        self.viz_artists.clear()
        
        # Clear custom points
        self.custom_grid_pts = []
        
        # Reset playback
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
        self.current_frame = 0
        
        # Clear all visualization panels
        self.lbl_input.clear()
        self.lbl_grid.clear()
        self.lbl_warped.clear()
        self.lbl_tracks.clear()
        
        print("All stored data cleared")

    def load_data(self):
        if not self.video_path:
            return
        
        # Clear all previous data before loading new video
        self.clear_all_data()
        ext = os.path.splitext(self.video_path)[1].lower()
        print(f"Loading data from {self.video_path}...")
        self.status_label.setText("Loading data...")
        QApplication.processEvents()
        try:
            if ext in ['.avi', '.mp4']:
                self.video_data = load_video(self.video_path, resize=False)
            elif ext in ['.tiff', '.tif']:
                video_np = load_tiff(self.video_path, frames=20)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            elif ext in ['.h5', '.hdf5']:
                video_np = load_hdf5(self.video_path, "/data", frames=20)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            else:
                self.status_label.setText(f"Unsupported file extension: {ext}")
                return
            # grid_xy = self._generate_grid_points()
            # self.base_grid_pts = grid_xy[0].detach().cpu().numpy().astype(np.float32)
            # self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
            # self.base_grid_pts0 = self.base_grid_pts.copy()
            # self.undo_btn.setEnabled(True)

            # Generate initial point preset from dropdown (instead of regular grid)
            self.apply_point_preset()
            self.undo_btn.setEnabled(True)
            print(f"Data loaded. Shape: {self.video_data.shape}")
            self.status_label.setText(f"Loaded: {os.path.basename(self.video_path)}")
            self.show_input_frame_only()
            self.process_btn.setEnabled(True)
            self.edit_points_btn.setEnabled(True)
            self.clear_points_btn.setEnabled(True)
            self.total_frames = self.video_data.shape[0]
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.current_frame = 0
            self.step_back_btn.setEnabled(True)
            self.step_fwd_btn.setEnabled(True)
            self.update_frame_label()
            self.visualize()
            # self.on_apply_grid_size()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.status_label.setText(f"Error loading data: {e}")
            self.video_data = None



    def toggle_edit_mode(self):
        self.edit_mode = self.edit_points_btn.isChecked()
        if self.edit_mode:
            self.status_label.setText("Edit Mode: drag to move | right-click base toggles delete | right-click custom deletes")
            if not self._img_cids:
                # connect our QLabel-based image signals
                try:
                    self.lbl_grid.pressed.connect(self.on_image_press)
                    self.lbl_grid.moved.connect(self.on_image_move)
                    self.lbl_grid.released.connect(self.on_image_release)
                    self._img_cids['press'] = True
                    self._img_cids['move'] = True
                    self._img_cids['release'] = True
                except Exception:
                    pass
        else:
            # disconnect image signals
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

    def clear_points(self):
        self.custom_grid_pts = []
        self.points_label.setText(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display()

    def undo_all_changes(self):
        if self.base_grid_pts0 is None:
            return
        self.base_grid_pts = self.base_grid_pts0.copy()
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.custom_grid_pts = []
        self.points_label.setText(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()

    def reset_all_zoom(self):
        """Reset zoom and pan on all visualization panels."""
        for label in [self.lbl_input, self.lbl_grid, self.lbl_warped, self.lbl_tracks]:
            label.zoom_factor = 1.0
            label.pan_offset = [0, 0]
            label.reset_zoom()  
        self.status_label.setText("Zoom reset on all panels")

    def on_canvas_click(self, event):
        # Kept for backward compatibility with Matplotlib events; with Qt conversion we use on_image_press/move/release
        if not self.edit_mode:
            return
        if self.video_data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            self.custom_grid_pts.append((x, y))
            self.points_label.setText(f"Points: {len(self.base_grid_active)}")
            self.update_grid_display_pre_registration()

        elif event.button == 3:
            if len(self.custom_grid_pts) > 0:
                x, y = event.xdata, event.ydata
                distances = [((px - x)**2 + (py - y)**2)**0.5 for px, py in self.custom_grid_pts]
                min_idx = distances.index(min(distances))
                if distances[min_idx] < 20:
                    self.custom_grid_pts.pop(min_idx)
                    self.points_label.setText(f"Points: {len(self.base_grid_active)}")
                    self.update_grid_display_pre_registration()

    def _nearest_point(self, pts, x, y):
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    # --- Qt image handlers (connected to ImageLabel) ---
    def on_image_press(self, x, y, button):
    #print(f"DEBUG: on_image_press x={x}, y={y}, button={button}, edit_mode={self.edit_mode}")
        # Map button enum to numeric values similar to Matplotlib: left=1, middle=2, right=3
        btn = 1
        try:
            val = int(button)
            if val == Qt.MouseButton.RightButton.value:
                btn = 3
            elif val == Qt.MouseButton.MiddleButton.value:
                btn = 2
            elif val == Qt.MouseButton.LeftButton.value:
                btn = 1
            else:
                btn = 1
        except Exception:
            # fallback
            btn = 1
        # Now fake a Matplotlib-style event
        event = type('ev', (), {})()
        event.xdata, event.ydata, event.button = float(x), float(y), btn
        # Reuse logic from on_canvas_press with adapted event
        # Right click for deletion/toggle
        if not self.edit_mode:
            return
        if event.button == 3:
            if len(self.custom_grid_pts) > 0:
                cpts = np.array(self.custom_grid_pts, dtype=np.float32)
                cidx, cdist = self._nearest_point(cpts, event.xdata, event.ydata)
            else:
                cidx, cdist = None, 1e9
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                bidx, bdist = self._nearest_point(self.base_grid_pts, event.xdata, event.ydata)
            else:
                bidx, bdist = None, 1e9
            if cdist < 20 and cdist <= bdist and cidx is not None:
                self.custom_grid_pts.pop(cidx)
                self.points_label.setText(f"Points: {len(self.base_grid_active)}")
                self.update_grid_display_pre_registration()
                return
            if bdist < 20 and bidx is not None:
                self.base_grid_active[bidx] = ~self.base_grid_active[bidx]
                self.update_grid_display_pre_registration()
                return
            return
        # Left click: nearest base/custom or add custom
        if event.button != 1:
            return
        if self.base_grid_pts is not None and np.any(self.base_grid_active):
            idxs = np.where(self.base_grid_active)[0]
            ptsA = self.base_grid_pts[idxs]
            loc_idx, bdist = self._nearest_point(ptsA, event.xdata, event.ydata)
            bidx = int(idxs[loc_idx])
        else:
            bidx, bdist = None, 1e9
        if len(self.custom_grid_pts) > 0:
            cpts = np.array(self.custom_grid_pts, dtype=np.float32)
            cidx, cdist = self._nearest_point(cpts, event.xdata, event.ydata)
        else:
            cidx, cdist = None, 1e9
        tol = 15.0
        if bdist < tol and bdist <= cdist:
            self._drag = ("base", bidx)
            self.base_grid_pts[bidx] = (event.xdata, event.ydata)
            self.update_grid_display_pre_registration()
            return
        if cdist < tol:
            self._drag = ("custom", cidx)
            self.custom_grid_pts[cidx] = (int(event.xdata), int(event.ydata))
            self.update_grid_display_pre_registration()
            return
        self.custom_grid_pts.append((int(event.xdata), int(event.ydata)))
        self.points_label.setText(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()

    def on_image_move(self, x, y):
        if not self.edit_mode or self._drag is None:
            return
        kind, idx = self._drag
        if kind == "base":
            self.base_grid_pts[idx] = (x, y)
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()

    def on_image_release(self):
        self._drag = None

    def on_canvas_press(self, event):
        # Deprecated: For Matplotlib event handlers only
        if not self.edit_mode:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        tol = 15.0

        # Right click: delete custom OR toggle delete/restore base
        if event.button == 3:
            # custom first
            if len(self.custom_grid_pts) > 0:
                cpts = np.array(self.custom_grid_pts, dtype=np.float32)
                cidx, cdist = self._nearest_point(cpts, x, y)
            else:
                cidx, cdist = None, 1e9

            # base
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                bidx, bdist = self._nearest_point(self.base_grid_pts, x, y)
            else:
                bidx, bdist = None, 1e9

            if cdist < tol and cdist <= bdist:
                self.custom_grid_pts.pop(cidx)
                self.points_label.setText(f"Points: {len(self.base_grid_active)}")
                self.update_grid_display_pre_registration()
                return

            if bdist < tol:
                self.base_grid_active[bidx] = ~self.base_grid_active[bidx]  # toggle delete/restore
                self.update_grid_display_pre_registration()
                return

            return

        # Left click: drag nearest ACTIVE base/custom; else add custom
        if event.button != 1:
            return

        # nearest ACTIVE base
        if self.base_grid_pts is not None and np.any(self.base_grid_active):
            idxs = np.where(self.base_grid_active)[0]
            ptsA = self.base_grid_pts[idxs]
            loc_idx, bdist = self._nearest_point(ptsA, x, y)
            bidx = int(idxs[loc_idx])
        else:
            bidx, bdist = None, 1e9

        # nearest custom
        if len(self.custom_grid_pts) > 0:
            cpts = np.array(self.custom_grid_pts, dtype=np.float32)
            cidx, cdist = self._nearest_point(cpts, x, y)
        else:
            cidx, cdist = None, 1e9

        if bdist < tol and bdist <= cdist:
            self._drag = ("base", bidx)
            self.base_grid_pts[bidx] = (x, y)
            self.update_grid_display_pre_registration()
            return

        if cdist < tol:
            self._drag = ("custom", cidx)
            self.custom_grid_pts[cidx] = (int(x), int(y))
            self.update_grid_display_pre_registration()
            return

        # add custom
        self.custom_grid_pts.append((int(x), int(y)))
        self.points_label.setText(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()

    def on_canvas_move(self, event):
        # Deprecated: for Matplotlib
        if not self.edit_mode or self._drag is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        kind, idx = self._drag

        if kind == "base":
            self.base_grid_pts[idx] = (x, y)
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()

    def on_canvas_release(self, event):
        # Deprecated
        self._drag = None


    def update_grid_display(self):
        if self.video_data is None:
            return
        frame0 = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        title = "Tracked Points (Frame 0) — click to add/remove" if self.edit_mode else "Tracked Points (Frame 0)"
        self.lbl_grid_title.setText(title)
        init_xy = None
        if self.pred_tracks is not None:
            tracks_np = self._tracks_to_numpy()
            if tracks_np is not None:
                init_xy = tracks_np[0, 0]
        ptsA = None
        if self.base_grid_pts is not None and self.base_grid_active is not None:
            active = self.base_grid_active
            if np.any(active):
                ptsA = self.base_grid_pts[active]
        custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if len(self.custom_grid_pts) > 0 else None
        # Show grid with base points, custom points, and initial tracks overlay
        self._update_label_with_overlay(self.lbl_grid, frame0, ptsA=ptsA, custom_pts=custom_pts, tracks=init_xy)

    def run_processing(self):
        if self.video_data is None:
            return

        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Base grid points: only ACTIVE ones go into queries
        base_active = self.base_grid_pts[self.base_grid_active].astype(np.float32)  # (K,2)
        base_grid_tensor = torch.from_numpy(base_active).unsqueeze(0)  # (1,K,2)

        custom_pts_tensor = None
        if len(self.custom_grid_pts) > 0:
            pts_array = np.array(self.custom_grid_pts, dtype=np.float32)           # (M,2)
            custom_pts_tensor = torch.from_numpy(pts_array)

        # Ensure any existing worker is stopped
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(100)
            except Exception:
                pass

        # Pass cached tracking data for incremental updates
        self.worker = ProcessingWorker(
            self.video_data, base_grid_tensor, custom_pts_tensor, 
            low_res_factor=2,
            cached_tracks=self.cached_tracks,
            cached_base_pts=self.cached_base_pts,
            cached_custom_pts=self.cached_custom_pts,
            parent=self
        )
        # Connect signals
        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
        # enable stop and reset progress
        self.stop_btn.setEnabled(True)
        try:
            self.progress_bar.setValue(0)
        except Exception:
            pass

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video, disp_field):
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        # Worker thread emits raw warped array; set attributes accordingly
        if warped_video is None:
            self.warped_video = None
        else:
            self.warped_video = warped_video
        # Save per-frame displacement field if provided
        try:
            self.disp_field = disp_field
            self.viz_data['disp_field'] = disp_field
        except Exception:
            self.disp_field = None
        
        # Cache the tracking results for incremental updates
        self.cached_tracks = pred_tracks.copy() if isinstance(pred_tracks, np.ndarray) else pred_tracks
        self.cached_visibility = pred_visibility
        self.cached_base_pts = self.base_grid_pts[self.base_grid_active].copy()
        self.cached_custom_pts = self.custom_grid_pts.copy() if self.custom_grid_pts else []
        
        self.status_label.setText("Processing complete")
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        try:
            self.progress_bar.setValue(100)
        except Exception:
            pass
        # Release worker
        try:
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.quit()
                self.worker.wait(100)
        except Exception:
            pass
        self.worker = None
        # Update total_frames if warped is shorter than input video
        if self.warped_video is not None:
            warp_T = self.warped_video.shape[0]
            if warp_T < self.total_frames:
                print(f"Warped frames shorter ({warp_T}) than original ({self.total_frames}); will display last warped frame for remaining indices")
        self.visualize()

    def on_processing_error(self, error_msg):
        self.status_label.setText("Error during processing")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        try:
            self.progress_bar.setValue(0)
        except Exception:
            pass
        # ensure worker is cleared
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.quit()
                self.worker.wait(100)
            except Exception:
                pass
        self.worker = None

    def on_tracks_ready(self, pred_tracks, pred_visibility, grid_pts):
        """Light-weight handler to display predicted tracks as soon as they are computed.
        This avoids waiting for the full warping to finish."""
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        # Update grid/track visualization on the GUI
        self.update_grid_display()

    def on_progress(self, value):
        try:
            self.progress_bar.setValue(int(value))
        except Exception:
            pass

    def stop_processing(self):
        # Stop the current worker and reset GUI state
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(500)
            except Exception as e:
                print("Error stopping worker:", e)
        self.stop_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.status_label.setText("Stopped")
        try:
            self.progress_bar.setValue(0)
        except Exception:
            pass



    def toggle_playback(self):
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
        else:
            # Calculate interval based on speed (base is 50ms for 20fps)
            speed_factor = self.speed_slider.value() / 100.0
            interval = int(50 / speed_factor)
            self.timer.start(interval)
            self.is_playing = True
            self.play_btn.setText("⏸")

    def on_timer(self):
        self.next_frame()

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
        
        self.slider.setValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def on_slider_change(self, value):
        if not self.is_playing:  # Only update if not auto-playing
            self.current_frame = self.slider.value()
            self.update_plot()
            self.update_frame_label()

    def update_frame_label(self):
        self.frame_label.setText(f"{self.current_frame}/{self.total_frames}")

    def on_speed_change(self, value):
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.2f}x")
        # Update timer interval if currently playing
        if self.is_playing:
            interval = int(50 / speed)
            self.timer.start(interval)

    def closeEvent(self, event):
        # Stop worker thread if running
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(500)
            except Exception:
                pass
        super().closeEvent(event)
    
    def dragEnterEvent(self, event):
        """Accept drag events if they contain file URLs."""
        if event.mimeData().hasUrls():
            # Check if any of the URLs are video files
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.avi', '.mp4', '.tiff', '.tif', '.h5', '.hdf5']:
                    event.acceptProposedAction()
                    return
    
    def dropEvent(self, event):
        """Handle dropped files."""
        if event.mimeData().hasUrls():
            # Get the first valid video file
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.avi', '.mp4', '.tiff', '.tif', '.h5', '.hdf5']:
                    self.video_path = file_path
                    self.status_label.setText(f"Loading: {os.path.basename(self.video_path)}")
                    self.load_data()
                    event.acceptProposedAction()
                    return
    
    def visualize(self):
        if self.video_data is None:
            return

        print("Preparing visualization...")
        
        # Stop existing animation if any
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("▶")
        
        full_viz = self.warped_video is not None

        # Prepare data - cache as numpy arrays for faster access
        if 'input' not in self.viz_data:
            # Apply filter to each frame before caching
            filtered = []
            for frame in self.video_data.permute(0, 2, 3, 1).cpu().numpy():
                filtered.append(self.apply_filter(frame, self.current_filter))
            self.viz_data['input'] = np.stack(filtered, axis=0).astype(np.float32)
        
        if full_viz:
            self.viz_data['warped'] = self.warped_video
            
            grid_pts_np = self.grid_pts.cpu().numpy()
            # grid_pts format is (N, 3) where columns are [frame_idx, x, y] or (N, 2) where columns are [x, y]
            if grid_pts_np.shape[1] == 3:
                self.viz_data['grid_x'] = grid_pts_np[:, 1]
                self.viz_data['grid_y'] = grid_pts_np[:, 2]
            else:  # shape[1] == 2
                self.viz_data['grid_x'] = grid_pts_np[:, 0]
                self.viz_data['grid_y'] = grid_pts_np[:, 1]
            
            tracks_np = self._tracks_to_numpy()
            if tracks_np is not None:
                self.viz_data['tracks'] = tracks_np[0]
        
        # 1. Input Video - update label
        input_frame = self.viz_data['input'][self.current_frame]
        self._update_label_with_overlay(self.lbl_input, input_frame)
        
        if full_viz:
            # 2. Grid Points (Input + Tracks at current frame)
            if not self.edit_mode:
                current_tracks = self.viz_data['tracks'][min(self.current_frame, self.viz_data['tracks'].shape[0]-1)] if 'tracks' in self.viz_data else None
                self._update_label_with_overlay(self.lbl_grid, self.viz_data['input'][self.current_frame], tracks=current_tracks, show_trajectories=True)
            else:
                self.update_grid_display_pre_registration()
            
            # 3. Corrected Video - ensure color consistency with input frames
            if full_viz:
                warped = self.viz_data['warped']
            else:
                warped = None
            
            if warped is not None:
                warp_T = warped.shape[0]
                warp_idx = min(self.current_frame, warp_T - 1)
                warped_frame = warped[warp_idx]
                self._update_label_with_overlay(self.lbl_warped, warped_frame)
            
            # 4. Deformation Field
            if 'disp_field' in self.viz_data and self.viz_data['disp_field'] is not None:
                disp_stack = self.viz_data['disp_field']
                # disp_stack should be (T, H, W, 2)
                if disp_stack.ndim == 4:
                    d_idx = min(self.current_frame, disp_stack.shape[0]-1)
                    flow_img = self.flow_to_color(disp_stack[d_idx])
                    self._update_label_with_overlay(self.lbl_tracks, flow_img)
        else:
            # No visualization data yet - keep default labels
            pass

    def update_plot(self):
        if 'input' not in self.viz_data:
            return
        # Update Input label
        input_frame = self.viz_data['input'][self.current_frame]
        self._update_label_with_overlay(self.lbl_input, input_frame)

        # Update warped or preview
        # Update warped
        if 'warped' in self.viz_data:
            warped = self.viz_data['warped']
            warp_idx = min(self.current_frame, warped.shape[0]-1)
            warped_frame = warped[warp_idx]
            self._update_label_with_overlay(self.lbl_warped, warped_frame)

        # Update Moving Grid Points (Panel 2)
        if not self.edit_mode and 'tracks' in self.viz_data:
            current_tracks = self.viz_data['tracks'][min(self.current_frame, self.viz_data['tracks'].shape[0]-1)]
            # Show on top of current input frame
            self._update_label_with_overlay(self.lbl_grid, input_frame, tracks=current_tracks, show_trajectories=True)

        # Update Displacement Field (Panel 4)
        if 'disp_field' in self.viz_data and self.viz_data['disp_field'] is not None:
            disp_stack = self.viz_data['disp_field']
            if disp_stack.ndim == 4:
                d_idx = min(self.current_frame, disp_stack.shape[0]-1)
                flow_img = self.flow_to_color(disp_stack[d_idx])
                self._update_label_with_overlay(self.lbl_tracks, flow_img)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MotionCorrectionGUI()
    gui.show()
    sys.exit(app.exec())

