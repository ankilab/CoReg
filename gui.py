import sys
import os
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QCheckBox, QGridLayout, QComboBox, QSpinBox, QSizePolicy, QProgressBar, QDialog,
    QGraphicsRectItem, QScrollArea, QFormLayout, QGroupBox, QMessageBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPalette
import pyqtgraph as pg

import cv2

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5
pg.setConfigOptions(imageAxisOrder='row-major')  # damit (H,W) wie numpy angezeigt wird
from utils.utils import video_ssim_score


class ProcessingWorker(QThread):
    tracks_ready = pyqtSignal(object, object, object)  # tracks_np, visibility, grid
    progress = pyqtSignal(int)  # percent complete
    status_update = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object, object)  # tracks_np, visibility, grid, warped_full, disp_field
    error = pyqtSignal(str)

    def __init__(self, video_data, base_grid_pts=None, custom_grid_pts=None, df_sigma=3.0, tracks=None, parent=None, existing_data=None):
        super().__init__(parent)
        self.video_data = video_data
        self.base_grid_pts = base_grid_pts
        self.custom_grid_pts = custom_grid_pts
        self.df_sigma = float(df_sigma)
        self.tracks = tracks
        self.existing_data = existing_data # tuple of (existing_tracks, existing_visibility, target_indices_for_new, total_N)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            video = self.video_data.to(device) if isinstance(self.video_data, torch.Tensor) else self.video_data

            # 1. Determine which points to track
            if self.existing_data is not None:
                # Incremental mode
                existing_tracks, existing_vis, new_indices, total_N = self.existing_data
                T = video.shape[0] if video.ndim == 4 else video.shape[1] # video is (T,C,H,W) or (B,T,C,H,W)
                
                # Prepare merged buffers for results
                # existing_tracks is (T, N_old, 2)
                # existing_vis is (T, N_old)
                # total_N = N_old + len(new_indices)
                merged_tracks = np.zeros((T, total_N, 2), dtype=np.float32)
                merged_vis = np.zeros((T, total_N), dtype=np.float32)
                
                # Fill existing data
                old_mask = np.ones(total_N, dtype=bool)
                old_mask[new_indices] = False
                merged_tracks[:, old_mask] = existing_tracks
                merged_vis[:, old_mask] = existing_vis
                
                if len(new_indices) > 0:
                    # Collect queries for ONLY new points
                    grid_xy = self.base_grid_pts.to(device) if self.base_grid_pts is not None else None
                    if self.custom_grid_pts is not None:
                        custom_xy = self.custom_grid_pts
                        if isinstance(custom_xy, np.ndarray):
                            custom_xy = torch.from_numpy(custom_xy)
                        custom_xy = custom_xy.to(device).float()
                        if custom_xy.ndim == 2:
                            custom_xy = custom_xy.unsqueeze(0) # (1, M, 2)
                        
                        if grid_xy is not None:
                            grid_xy = torch.cat([grid_xy, custom_xy], dim=1)
                        else:
                            grid_xy = custom_xy
                    
                    # Track NEW queries
                    t_q = torch.zeros((*grid_xy.shape[:-1], 1), device=grid_xy.device, dtype=grid_xy.dtype)
                    queries = torch.cat([t_q, grid_xy], dim=-1)
                    
                    self.status_update.emit(f"Tracking {len(new_indices)} new points...")
                    self.progress.emit(0)
                    
                    pred_tracks, pred_visibility = cotrack_model(video, queries=queries)
                    
                    new_tracks_raw = pred_tracks.detach().cpu().numpy()
                    new_tracks_gui = new_tracks_raw[0] if new_tracks_raw.ndim == 4 else new_tracks_raw
                    
                    try:
                        new_vis_raw = pred_visibility.detach().cpu().numpy()
                    except Exception:
                        new_vis_raw = np.asarray(pred_visibility)
                    new_vis_gui = new_vis_raw[0] if new_vis_raw.ndim == 3 else new_vis_raw
                    
                    # Merge NEW into merged buffers
                    merged_tracks[:, new_indices] = new_tracks_gui
                    merged_vis[:, new_indices] = new_vis_gui
                
                tracks_gui = merged_tracks
                vis_gui = merged_vis
                tracks_for_warp = torch.from_numpy(merged_tracks).unsqueeze(0) # (1, T, N, 2)
                
                # grid_xy for return (optional, used in finishing)
                # We can just return None or something sensible
                grid_xy_ret = None 

            else:
                # Normal mode (all points or manual re-warp)
                grid_xy = self.base_grid_pts.to(device) if self.base_grid_pts is not None else None
                if self.custom_grid_pts is not None:
                    custom_xy = self.custom_grid_pts
                    if isinstance(custom_xy, np.ndarray):
                        custom_xy = torch.from_numpy(custom_xy)
                    custom_xy = custom_xy.to(device).float()
                    if custom_xy.ndim == 2:
                        custom_xy = custom_xy.unsqueeze(0)  # (1,M,2)
                    
                    if grid_xy is not None:
                        grid_xy = torch.cat([grid_xy, custom_xy], dim=1)  # (1,K+M,2)
                    else:
                        grid_xy = custom_xy

                vis_gui = None
                if self.tracks is None:
                    if grid_xy is None:
                         raise ValueError("No points to track and no existing tracks provided")
                         
                    t_q = torch.zeros((*grid_xy.shape[:-1], 1), device=grid_xy.device, dtype=grid_xy.dtype)
                    queries = torch.cat([t_q, grid_xy], dim=-1)

                    self.status_update.emit("Tracking running...")
                    self.progress.emit(0)

                    pred_tracks, pred_visibility = cotrack_model(video, queries=queries)
                    
                    tracks_raw = pred_tracks.detach().cpu().numpy()   # could be (B,T,N,2) or (T,N,2)
                    tracks_gui = tracks_raw[0] if tracks_raw.ndim == 4 else tracks_raw
                    tracks_for_warp = tracks_raw if tracks_raw.ndim == 4 else tracks_raw[None, ...]

                    try:
                        vis_raw = pred_visibility.detach().cpu().numpy()
                    except Exception:
                        vis_raw = np.asarray(pred_visibility)
                    vis_gui = vis_raw[0] if vis_raw.ndim == 3 else vis_raw      # (T,N) expected for GUI
                    
                    self.tracks_ready.emit(tracks_gui, vis_gui, grid_xy)
                else:
                    self.status_update.emit("Using existing tracks...")
                    self.progress.emit(10)
                    tracks_gui = self.tracks
                    tracks_for_warp = tracks_gui if tracks_gui.ndim == 4 else tracks_gui[None, ...]
                    if tracks_for_warp.ndim == 3: # (T,N,2) -> (1,T,N,2) if missed
                         tracks_for_warp = tracks_for_warp[None, ...]
                
                grid_xy_ret = grid_xy

            # 2. Warping
            self.progress.emit(70)
            self.status_update.emit("Warping / Displacement optimization ...")

            if not isinstance(tracks_for_warp, torch.Tensor):
                tracks_for_warp = torch.from_numpy(tracks_for_warp)
            tracks_for_warp = tracks_for_warp.to(device).float()

            def warp_cb(frac):
                self.progress.emit(int(70 + frac * 30))

            warped_video, disp_field = warping(tracks_for_warp, video, smoothing=self.df_sigma, callback=warp_cb)
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
            self.status_update.emit("Done")
            self.finished.emit(tracks_gui, vis_gui, grid_xy_ret, warped_np, disp_field)
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

class PGImagePanel(pg.PlotWidget):
    pressed  = pyqtSignal(float, float, int)   # x,y,button (1/2/3)
    moved    = pyqtSignal(float, float)
    released = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(50, 50)

        self.setBackground(None)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showAxis('left', False)
        self.showAxis('bottom', False)

        # Set size policy to expand and fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Disable native_1to1 to allow full image viewing
        self.native_1to1 = False
        self._img_w = None
        self._img_h = None

        self.vb = self.getViewBox()
        self.vb.setDefaultPadding(0.0)
        
        # Lock aspect ratio to prevent stretching
        # The image will fit in the panel with black borders if needed
        self.vb.setAspectLocked(True)
        
        # Invert Y axis for image coordinates
        self.vb.invertY(True)
        
        # We'll control mouse handling dynamically based on edit mode
        self.vb.setMouseEnabled(x=True, y=True)
        
        # Track if we're in a custom drag operation
        self._custom_drag_active = False
        
        # Store reference to parent GUI (will be set later)
        self._parent_gui = None

        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)

        # keep full-resolution rendering (no auto decimation)
        try:
            self.img_item.setAutoDownsample(False)
        except Exception:
            self.img_item.setOpts(autoDownsample=False)

        self.points_item = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(57, 255, 20, 200))
        self.cross_item  = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(57, 255, 20), symbol='o')
        self.traj_item   = pg.PlotDataItem(pen=pg.mkPen(57, 255, 20, 180, width=2))

        self.addItem(self.points_item)
        self.addItem(self.cross_item)
        self.max_trail = 30  
        self.traj_items = []
        for _ in range(self.max_trail):
            it = pg.PlotDataItem()
            self.traj_items.append(it)
            self.addItem(it)
        

        self.scene().sigMouseMoved.disconnect(self._on_mouse_moved) if hasattr(self, '_on_mouse_moved') else None
        self.scene().sigMouseClicked.disconnect(self._on_mouse_clicked) if hasattr(self, '_on_mouse_clicked') else None
        
        self.last_mouse_pos = None
        self._possible_click = False

    def _trail_color(self, u):
        # u in [0..1]
        yellow = (255, 255, 0)
        purple = (160, 0, 255)
        r = int(round(yellow[0] + (purple[0] - yellow[0]) * u))
        g = int(round(yellow[1] + (purple[1] - yellow[1]) * u))
        b = int(round(yellow[2] + (purple[2] - yellow[2]) * u))
        return r, g, b

    def set_image(self, img_np, auto_range=True, levels=None):
        if img_np is None:
            self.img_item.clear()
            self.points_item.setData([], [])
            self.cross_item.setData([], [])
            for it in self.traj_items:
                it.setData([], [])
            self._img_w, self._img_h = None, None
            return

        a = np.asarray(img_np)

        # ---- make display-safe for pyqtgraph ----
        if a.dtype.kind == "f":
            # grayscale float -> provide explicit levels
            if a.ndim == 2:
                if levels is not None:
                    self.img_item.setImage(a, levels=levels, autoLevels=False)
                else:
                    finite = np.isfinite(a)
                    if finite.any():
                        lo = float(np.nanpercentile(a[finite], 1))
                        hi = float(np.nanpercentile(a[finite], 99))
                        if hi <= lo:
                            hi = lo + 1.0
                        self.img_item.setImage(a, levels=(lo, hi), autoLevels=False)
                    else:
                        self.img_item.setImage(a)
            else:
                # RGB/RGBA float -> convert to uint8
                finite = np.isfinite(a)
                if finite.any():
                    if levels is not None:
                        lo, hi = levels
                    else:
                        lo = float(np.nanpercentile(a[finite], 1))
                        hi = float(np.nanpercentile(a[finite], 99))
                    
                    if hi <= lo:
                        hi = lo + 1.0
                    a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                else:
                    a = np.zeros_like(a, dtype=np.uint8)
                self.img_item.setImage(a, autoLevels=False)
        else:
            if a.dtype == np.bool_:
                a = (a.astype(np.uint8) * 255)
            
            if levels is not None:
                self.img_item.setImage(a, levels=levels, autoLevels=False)
            else:
                self.img_item.setImage(a, autoLevels=True)

        self._img_w = a.shape[1]
        self._img_h = a.shape[0]

        if auto_range:
            self.vb.autoRange()

    def _map_to_image(self, pos):
        """Map QPointF pos to image coordinates."""
        if self._img_w is None:
            return pos.x(), pos.y()
        
        # PlotWidget coordinates -> ViewBox coordinates
        scene_pos = self.mapToScene(pos.toPoint() if hasattr(pos, 'toPoint') else pos)
        view_pos = self.vb.mapSceneToView(scene_pos)
        
        x = max(0, min(self._img_w - 1, view_pos.x()))
        y = max(0, min(self._img_h - 1, view_pos.y()))
        return x, y

    def set_points(self, pts_xy=None, crosses_xy=None):
        if pts_xy is None or len(pts_xy) == 0:
            self.points_item.setData([], [])
        else:
            p = np.asarray(pts_xy, dtype=float)
            self.points_item.setData(p[:, 0], p[:, 1])

        if crosses_xy is None or len(crosses_xy) == 0:
            self.cross_item.setData([], [])
        else:
            c = np.asarray(crosses_xy, dtype=float)
            self.cross_item.setData(c[:, 0], c[:, 1])

    def set_trajectories(self, tracks_TN2, current_frame, trail=15):
        # Clear everything if nothing to draw
        if tracks_TN2 is None:
            for it in self.traj_items:
                it.setData([], [])
            return

        tr = np.asarray(tracks_TN2)
        if tr.ndim == 4:
            tr = tr[0]
        if tr.ndim != 3 or tr.shape[0] < 2:
            for it in self.traj_items:
                it.setData([], [])
            return

        T, N, _ = tr.shape
        cf = int(np.clip(current_frame, 0, T - 1))

        # At frame 0, don't draw any trajectories (no previous frame exists)
        if cf == 0:
            for it in self.traj_items:
                it.setData([], [])
            return

        # From frame 1 onwards, draw trajectories
        L = int(min(max(1, trail), self.max_trail))
        start_t = max(1, cf - L + 1)  # Start from frame 1 at minimum
        end_t = cf

        # movement threshold
        eps_px = 0.5
        eps2 = eps_px * eps_px

        den = max(1, end_t - start_t + 1)

        # For each segment in the trail
        # We draw segments from (t-1) -> t for t in [start_t, end_t]
        segment_idx = 0
        for t in range(start_t, end_t + 1):
            if segment_idx >= self.max_trail:
                break
                
            u = (t - start_t) / den  # 0..1 color gradient
            r, g, b = self._trail_color(u)
            pen = pg.mkPen(r, g, b, 255, width=2)

            xs, ys = [], []
            for i in range(N):
                p1 = tr[t - 1, i]  # previous frame
                p2 = tr[t, i]      # current frame
                if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                    continue
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                if (dx * dx + dy * dy) < eps2:
                    continue

                xs.extend([float(p1[0]), float(p2[0]), np.nan])
                ys.extend([float(p1[1]), float(p2[1]), np.nan])

            self.traj_items[segment_idx].setData(xs, ys, pen=pen)
            segment_idx += 1

        # Clear remaining trajectory items
        for k in range(segment_idx, self.max_trail):
            self.traj_items[k].setData([], [])


    def _map_scene_to_image(self, scene_pos):
        if scene_pos is None:
            return None
        p = self.vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _qt_btn_to_int(self, btn):
        if btn == Qt.MouseButton.LeftButton:
            return 1
        if btn == Qt.MouseButton.MiddleButton:
            return 2
        if btn == Qt.MouseButton.RightButton:
            return 3
        return 0

    def _is_over_point(self, scene_pos):
        # We need to map scene pos to view pos (image coords), then check distance to points
        xy = self._map_scene_to_image(scene_pos)
        if xy is None:
            return False
        
        x, y = xy
        # Check scatter points
        pts_x, pts_y = self.points_item.getData() # arrays
        if len(pts_x) > 0:
            # We want visual distance in screen pixels approx.
            # But here we measure in image pixels.
            # 15 pixels screen distance approx? 
            # We can use mapFromViewToScene to check screen pixel dist but that's expensive for all points.
            # Instead, assume reasonable buffer in image space depending on zoom?
            # Actually, user interaction feels best if fixed screen tolerance.
            
            # Map all points to scene? No.
            # Let's use simple image space distance. If zoomed in, tolerance in pixels is smaller physically.
            # Wait, if I zoom in, 10 px on screen matches 0.1 px on image.
            # So I should use mapSceneToView to get the size of 1 screen pixel.
            p0 = self.vb.mapSceneToView(pg.Point(0,0))
            p1 = self.vb.mapSceneToView(pg.Point(10,0)) # 10 screen pixels
            tol = abs(p1.x() - p0.x()) # Tolerance in image units
            
            d2 = (pts_x - x)**2 + (pts_y - y)**2
            if np.min(d2) < (tol*tol):
                return True

        # Check crosses
        c_x, c_y = self.cross_item.getData()
        if len(c_x) > 0:
             p0 = self.vb.mapSceneToView(pg.Point(0,0))
             p1 = self.vb.mapSceneToView(pg.Point(10,0))
             tol = abs(p1.x() - p0.x())
             
             d2 = (c_x - x)**2 + (c_y - y)**2
             if np.min(d2) < (tol*tol):
                 return True
                 
        return False

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # Check if over a point - if so, capture event for Editing (Dragging)
            if self._is_over_point(ev.position()):
                pos = self._map_scene_to_image(ev.position())
                if pos:
                    x, y = pos
                    self.pressed.emit(x, y, 1)
                    ev.accept()
                    return # Don't pass to super (no Pan)
            else:
                # Not over point: prepare for possible Add Point (on release) or Pan (on drag)
                self._possible_click = True
                self.last_mouse_pos = ev.position()
        
        elif ev.button() == Qt.MouseButton.RightButton:
             # Right click logic (delete etc)
             # Emit pressed immediately
             pos = self._map_scene_to_image(ev.position())
             if pos:
                x, y = pos
                self.pressed.emit(x, y, 3)
                ev.accept()
                return

        super().mousePressEvent(ev) # Standard Pan/Zoom

    def mouseMoveEvent(self, ev):
        # Update movement tracking for click detection
        if self._possible_click:
            dist = (ev.position() - self.last_mouse_pos).manhattanLength()
            if dist > 5: # Threshold for drag
                self._possible_click = False

        pos = self._map_scene_to_image(ev.position())
        if pos:
            self.moved.emit(pos[0], pos[1])
            
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._possible_click: # Was a clean click (no drag) on empty space
                pos = self._map_scene_to_image(ev.position())
                if pos:
                     self.pressed.emit(pos[0], pos[1], 1)
                self._possible_click = False
        
        self.released.emit()
        super().mouseReleaseEvent(ev)

    def reset_view(self):
        if getattr(self, "native_1to1", False):
            self._apply_native_1to1(keep_center=False)
        else:
            self.vb.autoRange()

    def set_points_colored(self, pts_xy, colors_rgba=None, size=8):       
        if pts_xy is None or len(pts_xy) == 0:
            self.points_item.setData([], [])
            return

        p = np.asarray(pts_xy, dtype=float)
        N = p.shape[0]

        if colors_rgba is None:
            # Default highlight color if no specific colors provided
            spots = [{'pos': (p[i,0], p[i,1]), 'size': size, 'brush': pg.mkBrush(0,255,0,255)} for i in range(N)]
        else:
            c = np.asarray(colors_rgba, dtype=np.uint8)
            spots = []
            for i in range(N):
                r,g,b,a = map(int, c[i])
                spots.append({'pos': (p[i,0], p[i,1]), 'size': size, 'brush': pg.mkBrush(r,g,b,a)})

        self.points_item.setData(spots=spots)


class MotionProfileWidget(QWidget):
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.motion_scores = []      # Cyan line
        self.deformation_scores = [] # Magenta line
        self.current_frame = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, scores):
        """
        scores: list or array of correlation values (L2 norm of frame diff).
                Length T.
        """
        self.motion_scores = scores
        self.update()

    def set_deformation_data(self, scores):
        """
        scores: list or array of avg L2 norm of deformation field.
                Length T.
        """
        self.deformation_scores = scores
        self.update()

    def set_current_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.update()

    def mousePressEvent(self, event):
        self._handle_mouse(event)

    def mouseMoveEvent(self, event):
        self._handle_mouse(event)

    def _handle_mouse(self, event):
        # Allow interaction if EITHER plot exists
        count_m = len(self.motion_scores) if self.motion_scores is not None else 0
        count_d = len(self.deformation_scores) if self.deformation_scores is not None else 0
        count = max(count_m, count_d)

        if count < 1:
            return

        x = event.position().x()
        w = self.width()
        
        frac = max(0.0, min(1.0, x / w))
        frame_idx = int(round(frac * (count - 1)))
        
        self.frame_changed.emit(frame_idx)

    def paintEvent(self, event):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QPolygonF, QPen
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            w = self.width()
            h = self.height()

            # Fill background
            painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

            # Helper to draw a score array
            def draw_line(scores, color):
                if scores is None or len(scores) < 2:
                    return
                
                count = len(scores)
                points = []
                for i, val in enumerate(scores):
                    x = (i / max(1, count - 1)) * w
                    try:
                        v = float(val)
                        if not np.isfinite(v): v = 0.0
                    except: v = 0.0
                    val_c = max(0.0, min(1.0, v))
                    y = (1.0 - val_c) * h
                    points.append(QPointF(x, y))
                
                poly = QPolygonF(points)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawPolyline(poly)

            # Draw Motion (Cyan)
            draw_line(self.motion_scores, QColor(0, 200, 255))

            # # Draw Deformation (Magenta)
            # draw_line(self.deformation_scores, QColor(255, 0, 255))

            # Draw current frame cursor
            # If score length is T-1 or T, we map current_frame to x in same domain.
            # If we have T frames, slider goes 0..T-1.
            # scores usually has length T-1 (diffs).
            # We match x scale to T-1.
            # If current_frame is T-1 (last frame), x is w.
            
            # Draw current frame cursor
            count_m = len(self.motion_scores) if self.motion_scores is not None else 0
            count_d = len(self.deformation_scores) if self.deformation_scores is not None else 0
            count = max(count_m, count_d)

            if count > 0:
                cx = (self.current_frame / max(1, count - 1)) * w
                painter.setPen(QPen(QColor(57, 255, 20), 1))
                painter.drawLine(int(cx), 0, int(cx), h)
        finally:
            painter.end()


class ParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.setModal(False)

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        self.layout = QVBoxLayout(content)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.hide)
        outer.addWidget(self.close_btn)


class MotionCorrectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Correction GUI")
        screen = QApplication.primaryScreen()
        g = screen.availableGeometry()

        w = int(g.width() * 0.8)
        h = int(g.height() * 0.8)
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
    
    def _get_visibility_frame(self, frame_idx: int, N: int):
        """
        Returns visibility for a given frame as (N,) float32 in [0,1], or None if unavailable.
        Handles common shapes: (T,N), (T,N,1), (B,T,N), (B,T,N,1)
        """
        if self.pred_visibility is None:
            return None

        v = self.pred_visibility
        try:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.asarray(v)
        except Exception:
            return None

        # normalize to remove batch / trailing singleton dims
        # possible shapes:
        # (T,N) / (T,N,1) / (1,T,N) / (1,T,N,1)
        if v.ndim == 4 and v.shape[0] == 1:
            v = v[0]  # -> (T,N,1) or (T,N)
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]  # -> (T,N)
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]  # -> (T,N)

        if v.ndim != 2:
            return None

        t = int(np.clip(frame_idx, 0, v.shape[0] - 1))
        vis = v[t]

        # vis should be (N,) but guard mismatches
        if vis.shape[0] != N:
            # if model returned more/less points, clip/pad safely
            M = min(N, vis.shape[0])
            out = np.zeros((N,), dtype=np.float32)
            out[:M] = vis[:M].astype(np.float32)
            vis = out

        vis = np.asarray(vis, dtype=np.float32)
        vis = np.clip(vis, 0.0, 1.0)
        return vis


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
        self.update_plot()

    def _update_panel(self, panel: PGImagePanel, img, pts=None, crosses=None, tracks=None, show_traj=False):
        panel.set_image(img, auto_range=False)
        panel.set_points(pts_xy=pts, crosses_xy=crosses)
        if show_traj and tracks is not None:
            panel.set_trajectories(tracks, self.current_frame, trail=15)
        else:
            panel.set_trajectories(None, 0)

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
        main_layout.addLayout(top_bar)

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
        top_bar.addWidget(self.edit_points_btn)

        # 3b. Edit Tracks (post-registration)
        self.edit_tracks_btn = QCheckBox("Edit Tracks")
        self.edit_tracks_btn.stateChanged.connect(self.on_edit_modes_changed)
        self.edit_tracks_btn.setEnabled(False)  # enabled only when tracks exist
        top_bar.addWidget(self.edit_tracks_btn)

        self.apply_all_tracks_chk = QCheckBox("Apply to all")
        self.apply_all_tracks_chk.setEnabled(False)
        top_bar.addWidget(self.apply_all_tracks_chk)

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


        # 4b. Expand preset after motion analysis
        top_bar.addWidget(QLabel("Expand:"))
        self.expand_mode_combo = QComboBox()
        self.expand_mode_combo.addItems([
            "Heuristic",
            "Algorithmus",
            "Uniform motion"
        ])
        top_bar.addWidget(self.expand_mode_combo)

        self.expand_preset_btn = QPushButton("➕ Expand Preset")
        self.expand_preset_btn.clicked.connect(self.expand_preset_after_motion)
        self.expand_preset_btn.setEnabled(False)  # enabled after first motion analysis
        top_bar.addWidget(self.expand_preset_btn)


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


        # 8. Save
        self.save_btn = QPushButton("💾 Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)  # Enabled when results exist
        top_bar.addWidget(self.save_btn)
        
        # Add filler to push buttons to left (optional)
        top_bar.addStretch(1)

        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.0, 1.0)
        self.conf_thresh_spin.setSingleStep(0.05)
        self.conf_thresh_spin.setValue(0.5)
        self.conf_thresh_spin.valueChanged.connect(self._apply_confidence_filter)

        self.auto_conf_chk = QCheckBox("Auto-disable low confidence tracks")
        self.auto_conf_chk.setChecked(True)
        self.auto_conf_chk.stateChanged.connect(lambda *_: self._apply_confidence_filter())



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

        self.grid_rows_ctrl = QLineEdit(str(getattr(self, "grid_rows", 4)))
        self.grid_cols_ctrl = QLineEdit(str(getattr(self, "grid_cols", 4)))

        self.apply_preset_btn = QPushButton("Apply to Grid")
        self.apply_preset_btn.clicked.connect(self.apply_point_preset)

        self.df_sigma_spin = QSpinBox()
        self.df_sigma_spin.setRange(0, 20)
        self.df_sigma_spin.setValue(3)
        self.df_sigma_spin.valueChanged.connect(lambda *_: self.update_plot())
        
        self.lbl_N = QLabel("Number of Points:")
        self.lbl_minDist = QLabel("Min Distance:")
        self.lbl_rows = QLabel("Rows:")
        self.lbl_cols = QLabel("Cols:")
        self.lbl_sigma = QLabel("DF Smoothing (σ):")
        
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
        
        pd_layout.addWidget(self.apply_preset_btn)
        
        pd_layout.addWidget(QLabel("---"))
        
        pd_layout.addWidget(self.lbl_sigma)
        pd_layout.addWidget(self.df_sigma_spin)
        
        pd_layout.addWidget(QLabel("---"))
        
        self.rotate_btn = QPushButton("Rotate Input 90°")
        self.rotate_btn.clicked.connect(self.rotate_input)

        pd_layout.addWidget(self.sync_view_chk)
        pd_layout.addWidget(self.reset_zoom_btn)
        pd_layout.addWidget(self.rotate_btn)
        pd_layout.addWidget(self.clear_points_btn)
        pd_layout.addWidget(self.undo_btn)

        # Move close button to bottom
        pd_layout.removeWidget(self.param_dialog.close_btn)
        pd_layout.addWidget(self.param_dialog.close_btn)

        pd_layout.addWidget(QLabel("Confidence threshold (visibility):"))
        pd_layout.addWidget(self.conf_thresh_spin)
        pd_layout.addWidget(self.auto_conf_chk)

        
        self.points_label = QLabel("Points: 0")
        self.status_label = QLabel("Ready - Drag & drop video or use Load button")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.points_label) # Put points label on main window or dialog? User didn't specify. Keeping on main seems useful.
        

        views_widget = QWidget(central_widget)
        views_layout = QHBoxLayout(views_widget)
        views_layout.setContentsMargins(6, 6, 6, 6)
        views_layout.setSpacing(6)

        self.view_items = ["Input", "Tracked Points", "Corrected Video", "Deformation Field"]

        # Left viewport
        left_box = QVBoxLayout()
        self.view_left_combo = QComboBox()
        self.view_left_combo.addItems(self.view_items)
        self.view_left_combo.setCurrentText("Tracked Points")
        self.view_left_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        self.view_left = PGImagePanel()
        self.view_left._parent_gui = self

        left_box.addWidget(self.view_left_combo)
        left_box.addWidget(self.view_left, stretch=1)

        # Right viewport
        right_box = QVBoxLayout()
        self.view_right_combo = QComboBox()
        self.view_right_combo.addItems(self.view_items)
        self.view_right_combo.setCurrentText("Corrected Video")
        self.view_right_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        self.view_right = PGImagePanel()
        self.view_right._parent_gui = self

        right_box.addWidget(self.view_right_combo)
        right_box.addWidget(self.view_right, stretch=1)

        views_layout.addLayout(left_box, stretch=1)
        views_layout.addLayout(right_box, stretch=1)

        main_layout.addWidget(views_widget, stretch=1)

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

        self.apply_preset_btn.setVisible(True)
        self.apply_preset_btn.setEnabled(True)

    def on_preset_changed(self, *_):
        self._update_preset_controls()

        # Wenn noch kein Video geladen: einfach direkt anwenden (replace)
        if self.video_data is None:
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

    def on_filter_changed(self, text):
        self.current_filter = text
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
        xs = np.linspace(0, W - 1, cols, dtype=np.float32) #0, H-1
        ys = np.linspace(0, H - 1, rows, dtype=np.float32) #10, H - 10
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
            self.process_btn.setEnabled(False)
            self.update_plot()
            return

        # --- wie bisher: Punkte für Preset berechnen ---
        gray = self._get_gray_video_np()
        H, W = gray.shape[1], gray.shape[2]
        ref_u8 = self._reference_image(gray)
        ref_u8_filt = self.apply_filter(ref_u8, self.current_filter)

        if preset.startswith("Regular grid"):
            try:
                rows = int(self.grid_rows_ctrl.text().strip())
                cols = int(self.grid_cols_ctrl.text().strip())
            except ValueError:
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

        # --- APPLY: replace vs add ---
        if mode == "replace":
            self.base_grid_pts = pts_new.astype(np.float32)
            self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
            self.base_grid_pts0 = self.base_grid_pts.copy()

        else:  # mode == "add"
            pts_add = self._filter_new_points(pts_new, min_dist=min_dist)
            if pts_add.shape[0] > 0:
                self.base_grid_pts = np.vstack([np.asarray(self.base_grid_pts, np.float32), pts_add]).astype(np.float32)
                self.base_grid_active = np.concatenate([np.asarray(self.base_grid_active, bool),
                                                        np.ones((pts_add.shape[0],), dtype=bool)])
                # Reset-to-preset soll den "neuen Basiszustand" enthalten:
                self.base_grid_pts0 = self.base_grid_pts.copy()

        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.update_plot()

        has_points = ((self.base_grid_pts is not None and len(self.base_grid_pts) > 0) or (len(self.custom_grid_pts) > 0))
        self.process_btn.setEnabled(has_points)

        if mode == "replace":
            self.status_label.setText(f"Preset replaced: {preset} ({len(self.base_grid_pts)} pts)")
        else:
            self.status_label.setText(f"Preset added: {preset} (+{len(pts_add)} pts, total {len(self.base_grid_pts)})")

    def _reset_results_and_tracks(self):
        """Clear tracking/warping results because point set changed."""
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None
        self.ssim_score = None

        if isinstance(self.viz_data, dict):
            self.viz_data.pop("tracks", None)
            self.viz_data.pop("warped", None)

        self.track_active = None
        self.selected_track_idx = None

        self.edit_tracks_btn.setEnabled(False)
        self.edit_tracks_btn.setChecked(False)
        self.save_btn.setEnabled(False)


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

    # ---------- file loading ----------
    def clear_all_data(self, keep_video=False):
        self._initial_zoom_done = False
        self.viz_data = {}
        if not keep_video:
            self.video_data = None
            self.total_frames = 0
            self.current_frame = 0
        
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.disp_field = None
        
        self.view_left.set_image(None)
        self.view_right.set_image(None)

        if hasattr(self, 'motion_plot'):
             self.motion_plot.set_current_frame(0)
             self.motion_plot.set_deformation_data([]) # Reset deformation line
             # If we rotate, we might want to keep motion scores? 
             # Or re-calc. If we keep video, we might want to keep motion profile?
             # But if we don't recalculate, it's fine.
             if not keep_video:
                 self.motion_plot.set_data([])

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


        # ---------- expand preset after motion analysis ----------
    def _get_existing_points_stack(self):
        exist = []
        if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
            exist.append(self.base_grid_pts[self.base_grid_active].astype(np.float32))
        if self.custom_grid_pts:
            exist.append(np.asarray(self.custom_grid_pts, dtype=np.float32))
        if len(exist) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(exist)

    def _add_points_unique(self, new_pts_xy, min_dist_px=10):
        """Append new points as custom points, avoid duplicates by min_dist."""
        if new_pts_xy is None or len(new_pts_xy) == 0:
            return 0

        new_pts = np.asarray(new_pts_xy, dtype=np.float32).reshape(-1, 2)
        exist = self._get_existing_points_stack()

        keep = []
        md2 = float(min_dist_px) * float(min_dist_px)

        for p in new_pts:
            if not (np.isfinite(p).all()):
                continue
            if exist.shape[0] > 0:
                d2 = np.sum((exist - p[None, :]) ** 2, axis=1)
                if np.min(d2) <= md2:
                    continue
            keep.append((float(p[0]), float(p[1])))
            # update exist incrementally (so we also keep min-dist among new pts)
            exist = np.vstack([exist, np.asarray([[p[0], p[1]]], dtype=np.float32)])

        for x, y in keep:
            self.custom_grid_pts.append((x, y))
        return len(keep)

    def _activity_map_from_motion_peaks(self, gray_video, top_k=8):
        """
        Heuristic: use frames with highest global motion score, average abs-diffs there.
        gray_video: (T,H,W) float32
        """
        T = gray_video.shape[0]
        if T < 2:
            return None

        scores = np.asarray(getattr(self.motion_plot, "motion_scores", []), dtype=np.float32)
        if scores.size != T:
            # fallback: compute simple global diff score (as you already do)
            diffs = np.abs(gray_video[1:] - gray_video[:-1]).mean(axis=(1,2))
            scores = np.concatenate([[0.0], diffs.astype(np.float32)], axis=0)

        # pick top_k frames (excluding frame 0 ideally)
        idxs = np.argsort(-scores)
        idxs = [int(i) for i in idxs if i > 0]
        idxs = idxs[:max(1, min(top_k, len(idxs)))]

        # accumulate abs diffs around those peaks (t-1 -> t)
        acc = np.zeros_like(gray_video[0], dtype=np.float32)
        for t in idxs:
            acc += np.abs(gray_video[t] - gray_video[t - 1]).astype(np.float32)
        acc /= float(len(idxs))

        # smooth + normalize to u8
        acc = cv2.GaussianBlur(acc, (0, 0), 1.2)
        acc_u8 = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return acc_u8

    def _flow_magnitude_map(self, gray_video, downscale_to=128):
        """
        Algorithmus: dense optical flow magnitude accumulated across time.
        Produces a single HxW uint8 map.
        """
        T, H, W = gray_video.shape
        if T < 2:
            return None

        # downscale for speed
        scale = min(downscale_to / float(H), downscale_to / float(W), 1.0)
        if scale < 1.0:
            dsize = (max(8, int(W * scale)), max(8, int(H * scale)))
        else:
            dsize = (W, H)

        acc = np.zeros((dsize[1], dsize[0]), dtype=np.float32)

        prev = cv2.resize(gray_video[0].astype(np.float32), dsize, interpolation=cv2.INTER_AREA)
        prev_u8 = cv2.normalize(prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        for t in range(1, T):
            cur = cv2.resize(gray_video[t].astype(np.float32), dsize, interpolation=cv2.INTER_AREA)
            cur_u8 = cv2.normalize(cur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                prev_u8, cur_u8, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
            acc += mag

            prev_u8 = cur_u8

        acc /= float(max(1, T - 1))
        acc = cv2.GaussianBlur(acc, (0, 0), 1.0)
        acc_u8_small = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # upscale back to full-res
        acc_u8 = cv2.resize(acc_u8_small, (W, H), interpolation=cv2.INTER_CUBIC)
        return acc_u8

    def _uniform_motion_points(self, gray_video, extra_points=300, min_dist=10):
        """
        Uniform motion: estimate global translation via phase correlation
        and place an additional shifted grid (uniform coverage).
        """
        T, H, W = gray_video.shape
        if T < 2:
            return np.zeros((0,2), dtype=np.float32)

        # estimate average translation from frame0 -> others
        ref = gray_video[0].astype(np.float32)
        ref = cv2.GaussianBlur(ref, (0, 0), 1.0)

        shifts = []
        for t in range(1, T):
            frm = gray_video[t].astype(np.float32)
            frm = cv2.GaussianBlur(frm, (0, 0), 1.0)
            try:
                (dx, dy), _ = cv2.phaseCorrelate(ref, frm)
                if np.isfinite(dx) and np.isfinite(dy):
                    shifts.append((dx, dy))
            except Exception:
                pass

        if len(shifts) == 0:
            dx, dy = 0.0, 0.0
        else:
            dx = float(np.median([s[0] for s in shifts]))
            dy = float(np.median([s[1] for s in shifts]))

        # build a grid roughly matching extra_points density
        n = int(max(50, extra_points))
        aspect = W / float(max(1, H))
        cols = int(np.sqrt(n * aspect))
        rows = int(np.sqrt(n / max(1e-6, aspect)))
        cols = max(2, cols)
        rows = max(2, rows)

        base = self._preset_regular_grid(H, W, rows, cols)  # (K,2)
        shifted = base.copy()
        shifted[:, 0] = np.clip(shifted[:, 0] + dx, 0, W - 1)
        shifted[:, 1] = np.clip(shifted[:, 1] + dy, 0, H - 1)

        # optional: also add a half-shift for coverage
        half = base.copy()
        half[:, 0] = np.clip(half[:, 0] + 0.5 * dx, 0, W - 1)
        half[:, 1] = np.clip(half[:, 1] + 0.5 * dy, 0, H - 1)

        out = np.vstack([shifted, half]).astype(np.float32)
        return self._clip_border(out, H, W, border=10)

    def expand_preset_after_motion(self):
        if self.video_data is None:
            return

        mode = self.expand_mode_combo.currentText().strip().lower()
        gray = self._get_gray_video_np()  # (T,H,W) float32
        H, W = gray.shape[1], gray.shape[2]

        N = int(self.num_points_spin.value())  # use same knob as presets
        min_dist = int(self.min_dist_spin.value())

        new_pts = np.zeros((0,2), dtype=np.float32)

        try:
            if "heur" in mode:
                act_u8 = self._activity_map_from_motion_peaks(gray, top_k=8)
                if act_u8 is None:
                    self.status_label.setText("Expand (heuristic): not enough frames.")
                    return

                # pick features on activity map
                p = cv2.goodFeaturesToTrack(
                    act_u8,
                    maxCorners=max(50, N),
                    qualityLevel=0.01,
                    minDistance=float(min_dist),
                    blockSize=7,
                    useHarrisDetector=False
                )
                if p is not None:
                    new_pts = p.reshape(-1,2).astype(np.float32)
                    new_pts = self._clip_border(new_pts, H, W, border=10)

            elif "algo" in mode:
                mag_u8 = self._flow_magnitude_map(gray, downscale_to=128)
                if mag_u8 is None:
                    self.status_label.setText("Expand (algorithmus): not enough frames.")
                    return

                # detect features where flow magnitude is high
                p = cv2.goodFeaturesToTrack(
                    mag_u8,
                    maxCorners=max(50, N),
                    qualityLevel=0.01,
                    minDistance=float(min_dist),
                    blockSize=7,
                    useHarrisDetector=False
                )
                if p is not None:
                    new_pts = p.reshape(-1,2).astype(np.float32)
                    new_pts = self._clip_border(new_pts, H, W, border=10)

            else:
                # uniform motion
                new_pts = self._uniform_motion_points(gray, extra_points=max(100, N), min_dist=min_dist)

        except Exception as e:
            self.status_label.setText(f"Expand failed: {e}")
            return

        added = self._add_points_unique(new_pts, min_dist_px=min_dist)
        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.status_label.setText(f"Expand Preset: {self.expand_mode_combo.currentText()} -> +{added} points")


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
        num_frames = 30
        try:
            if ext in ['.avi', '.mp4']:
                self.video_data = load_video(self.video_path, resize=False, max_frames=num_frames)
            elif ext in ['.tiff', '.tif']:
                self.video_data = load_tiff(self.video_path, frames=num_frames)
            elif ext in ['.h5', '.hdf5']:
                self.video_data = load_hdf5(self.video_path, "/data", frames=num_frames, resize=False)
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
            self.edit_tracks_btn.setEnabled(False)
            self.edit_tracks_btn.setChecked(False)
            self.clear_points_btn.setEnabled(True)

            self.play_btn.setEnabled(True)
            self.step_fwd_btn.setEnabled(True)
            self.step_back_btn.setEnabled(True)

            self.update_plot()
            QTimer.singleShot(100, lambda: self.reset_all_zoom())

        except Exception as e:
            self.status_label.setText(f"Error loading data: {e}")
            self.video_data = None
            return

        # Calculate correlation scores for the loaded video
        try:
            # video_data is (T, C, H, W)
            # Use grayscale for correlation
            # We want to be fast, so downsample
            gray_full = self._get_gray_video_np() # (T, H, W)
            
            # Downsample for speed (e.g., 64x64 or 128x128)
            # We just need global correlation trend.
            T, H, W = gray_full.shape
            
            target_dim = 64
            scale = min(target_dim/H, target_dim/W)
            if scale < 1.0:
                dsize = (int(W*scale), int(H*scale))
                # Use cv2 for resizing frames
                gray_small = []
                for i in range(T):
                    gray_small.append(cv2.resize(gray_full[i], dsize, interpolation=cv2.INTER_AREA))
                gray_small = np.array(gray_small) # (T, h, w)
            else:
                gray_small = gray_full

            # L2 Norm of Normalized Difference (effectively correlation distance)
            # Normalize each frame: (x - mean) / std
            flat = gray_small.reshape(T, -1).astype(np.float32)
            means = flat.mean(axis=1, keepdims=True)
            stds = flat.std(axis=1, keepdims=True) + 1e-6
            flat_norm = (flat - means) / stds
            
            # L2 Norm of difference from the first frame
            if T > 1:
                # diffs[t] = frame[t] - frame[0]
                diffs = flat_norm - flat_norm[0]
                
                # L2 norm (Euclidean distance)
                l2_norms = np.linalg.norm(diffs, axis=1)
                
                # Normalize to 0..1 for visualization
                max_val = l2_norms.max()
                if max_val > 1e-6:
                    scores = l2_norms / max_val
                else:
                    scores = np.zeros_like(l2_norms)
                
                self.motion_plot.set_data(scores)

                # Enable expand button once we have first motion analysis
                try:
                    if hasattr(self, "expand_preset_btn"):
                        self.expand_preset_btn.setEnabled(True)
                except Exception:
                    pass
             
            else:
                self.motion_plot.set_data([0.0])


        except Exception as e:
             print(f"Error calculating correlation: {e}")
             self.motion_plot.set_data([])


    def on_edit_modes_changed(self, *args):
        sender = self.sender()

        # Mutual exclusion
        if sender is self.edit_points_btn and self.edit_points_btn.isChecked():
            self.edit_tracks_btn.blockSignals(True)
            self.edit_tracks_btn.setChecked(False)
            self.edit_tracks_btn.blockSignals(False)

        if sender is self.edit_tracks_btn and self.edit_tracks_btn.isChecked():
            self.edit_points_btn.blockSignals(True)
            self.edit_points_btn.setChecked(False)
            self.edit_points_btn.blockSignals(False)

        # ✅ Flags (wichtig!)
        self.edit_points_mode = self.edit_points_btn.isChecked()
        self.edit_tracks_mode = self.edit_tracks_btn.isChecked()
        self.apply_all_tracks_chk.setEnabled(self.edit_tracks_mode)

        # ✅ ENTER Edit Points => springe zu Frame 0
        entering_edit_points = (sender is self.edit_points_btn and self.edit_points_mode)
        if entering_edit_points and self.video_data is not None:
            self._pause_if_playing()

            self.current_frame = 0
            self.slider.blockSignals(True)
            self.slider.setValue(0)
            self.slider.blockSignals(False)

            self.update_frame_label()
            self.update_plot()

        # Always disconnect first to avoid double connections
        for panel in [self.view_left, self.view_right]:
            try:
                panel.pressed.disconnect(self.on_image_press)
                panel.moved.disconnect(self.on_image_move)
                panel.released.disconnect(self.on_image_release)
            except Exception:
                pass

        # Connect only if any edit mode active
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

        # refresh overlays
        if self.pred_tracks is not None:
            self.update_plot()
        else:
            self.update_grid_display_pre_registration()

    def _delete_base_point(self, bidx: int):
        if self.base_grid_pts is None or len(self.base_grid_pts) == 0:
            return

        self.base_grid_pts = np.delete(self.base_grid_pts, bidx, axis=0)

        if self.base_grid_active is not None and len(self.base_grid_active) > bidx:
            self.base_grid_active = np.delete(self.base_grid_active, bidx, axis=0)

        # Wenn "Reset to Preset" NICHT wieder erscheinen lassen soll:
        if self.base_grid_pts0 is not None and len(self.base_grid_pts0) > bidx:
            self.base_grid_pts0 = np.delete(self.base_grid_pts0, bidx, axis=0)

    def _fill_box_with_points(self, x1, y1, x2, y2):
        if self.video_data is None:
            return
            
        N = self.num_points_spin.value()
        min_dist = self.min_dist_spin.value()
        preset = self.preset_combo.currentText()
        
        # We need to generate points for the WHOLE image then clip? 
        # Or generate for the sub-region?
        # Regular grid: easy to generate for sub-region.
        # Features: detect on ROI?
        
        # Let's try to detect on ROI if possible for features, or Clip global.
        # Clipping global is safer for consistency.
        
        # But wait, User might want to ADD points to this area, effectively increasing density locally?
        # If we just clip global grid, we get the same points that might already be there (or inactive).
        # "Box Fill" implies adding content.
        
        # Let's implement generation on the ROI.
        
        W = int(x2 - x1)
        H = int(y2 - y1)
        if W < 5 or H < 5: return
        
        # Prepare ROI image
        gray_full = self._get_gray_video_np()
        ref = self._reference_image(gray_full) # (H_full, W_full)
        
        ix1, iy1 = int(max(0, x1)), int(max(0, y1))
        ix2, iy2 = int(min(ref.shape[1], x2)), int(min(ref.shape[0], y2))
        
        if ix2 <= ix1 or iy2 <= iy1: return
        
        roi = ref[iy1:iy2, ix1:ix2]
        
        new_pts = []
        
        if preset.startswith("Regular grid"):
            rows = int(self.grid_rows_ctrl.text())
            cols = int(self.grid_cols_ctrl.text())
            # For box fill, rows/cols might mean density? 
            # Or just use N? Regular grid usually ignores N.
            # Let's map rows/cols to density or just use fixed step?
            # Existing get_grid_points uses rows/cols for the whole image.
            # Maybe just try to fill with similar density?
            # Let's use min_dist to define a grid?
            step = max(10, min_dist)
            
            # Create grid in ROI
            gy, gx = np.mgrid[step//2:H:step, step//2:W:step]
            # flatten
            gp = np.vstack([gx.ravel(), gy.ravel()]).T
            # Offset back to global
            gp[:, 0] += x1
            gp[:, 1] += y1
            new_pts = gp

        elif preset.startswith("Features"):
            # Detect features in ROI
            # N needs to be scaled to ROI area vs Full area?
            # Or just use N? If user wants to fill box, maybe they want N points in that box?
            # Let's try using N/4 (heuristic) or just N if they typed it.
            # Usually N is total. Let's use min_dist mainly.
            
            p = cv2.goodFeaturesToTrack(roi, maxCorners=N, qualityLevel=0.01, minDistance=min_dist)
            if p is not None:
                p = p.reshape(-1, 2)
                p[:, 0] += x1
                p[:, 1] += y1
                new_pts = p
                
        elif preset.startswith("Activity"):
             # Activity map on ROI
             # We need activity map for the whole video? expensive.
             # We might have `self.base_grid_pts` assuming it was generated from Activity?
             # Let's assume re-running activity on ROI is okay.
             
             # Need activity map for ROI
             # Extract ROI from video?
             # video (T, H, W)
             if gray_full.ndim == 3:
                 vid_roi = gray_full[:, iy1:iy2, ix1:ix2]
                 diff = np.abs(np.diff(vid_roi.astype(np.float32), axis=0))
                 activity = np.mean(diff, axis=0) # (H_roi, W_roi)
                 
                 # Weighted sampling
                 # ... re-implement simple sampling or reuse utils?
                 # For simplicity, just use pure random weighted by activity?
                 # Or just peak picking.
                 # Let's use nice utility if available, else simple grid rejection.
                 
                 # Simple rejection sampling
                 act_flat = activity.flatten()
                 probs = act_flat / (act_flat.sum() + 1e-6)
                 
                 # choice
                 n_gen = min(N, len(act_flat))
                 indices = np.random.choice(len(act_flat), size=n_gen, p=probs, replace=False)
                 y_g, x_g = np.unravel_index(indices, activity.shape)
                 pts = np.vstack([x_g, y_g]).T
                 pts[:, 0] += x1
                 pts[:, 1] += y1
                 new_pts = pts

        elif preset.startswith("Segmentation"):
             pass # Not implemented for box fill easily without full seg
        
        
        # Add non-duplicate points
        if len(new_pts) > 0:
            # Filter duplicates against existing custom + active base
            # Simple distance check
            
            # Current points
            exist = []
            if len(self.custom_grid_pts) > 0:
                exist.append(np.array(self.custom_grid_pts))
            
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                exist.append(self.base_grid_pts[self.base_grid_active])
                
            if len(exist) > 0:
                exist_cat = np.vstack(exist)
                
                # Check distances
                # Brute force or kdtree. Brute force fine for N < 1000
                keep = []
                for p in new_pts:
                    # dist to all exist
                    ds = np.sum((exist_cat - p)**2, axis=1)
                    if np.min(ds) > (min_dist/2.0)**2:
                        keep.append(p)
                
                new_pts = np.array(keep)
            
            if len(new_pts) > 0:
                for p in new_pts:
                    self.custom_grid_pts.append((float(p[0]), float(p[1])))
                
                self._update_points_label()
                self.update_grid_display_pre_registration()

    def _nearest_point(self, pts, x, y):
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def on_image_press(self, x, y, button_value):
        if not (self.edit_points_mode or self.edit_tracks_mode):
            return

        if self._drag is not None:
            return

        btn = int(button_value)
        x_f, y_f = float(x), float(y)
        tol = 15.0

        # RIGHT CLICK
        if btn == 3:
            # --- Track toggling nur in Edit-Tracks ---
            if self.edit_tracks_mode and self.pred_tracks is not None:
                tracks = np.asarray(self.pred_tracks)
                if tracks.ndim == 4:
                    tracks = tracks[0]
                current_pts = tracks[self.current_frame]
                tidx, tdist = self._nearest_point(current_pts, x_f, y_f)
                if tdist < tol:
                    self.selected_track_idx = tidx
                    if self.track_active is not None and len(self.track_active) == current_pts.shape[0]:
                        self.track_active[tidx] = ~self.track_active[tidx]
                    self.update_plot()
                    self._schedule_recalc_warping()
                return

            # --- Pre-registration löschen nur in Edit-Points ---
            if not self.edit_points_mode:
                return

            tol = 20.0

            # 1) Custom löschen (wie bisher)
            cidx, cdist = None, 1e9
            if len(self.custom_grid_pts) > 0:
                cpts = np.array(self.custom_grid_pts, dtype=np.float32)
                cidx, cdist = self._nearest_point(cpts, x_f, y_f)

            # 2) Base löschen ABER nur unter den AKTIVEN (sichtbaren) Punkten suchen
            bidx, bdist = None, 1e9
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                if self.base_grid_active is None:
                    # fallback: alle
                    bidx, bdist = self._nearest_point(self.base_grid_pts, x_f, y_f)
                else:
                    active_idxs = np.where(self.base_grid_active)[0]
                    if len(active_idxs) > 0:
                        ptsA = self.base_grid_pts[active_idxs]
                        loc, bdist = self._nearest_point(ptsA, x_f, y_f)
                        bidx = int(active_idxs[loc])

            if cidx is not None and cdist < tol and cdist <= bdist:
                if getattr(self, "_query_base_indices", None) is not None and self.pred_tracks is not None:
                    baseN = len(self._query_base_indices)
                    tidx = baseN + int(cidx)
                    self._delete_track_idx(tidx)
                    self._query_custom_count = max(0, self._query_custom_count - 1)

                self.custom_grid_pts.pop(cidx)
                self._update_points_label()
                self.update_plot()
                return

            if bidx is not None and bdist < tol:
                # 1) passenden Track löschen (falls vorhanden + mapping existiert)
                if getattr(self, "_query_base_indices", None) is not None and self.pred_tracks is not None:
                    hits = np.where(self._query_base_indices == bidx)[0]
                    if len(hits) > 0:
                        tidx = int(hits[0])
                        self._delete_track_idx(tidx)
                        # mapping: entfernten base-index raus, und alle >bidx um 1 runter (weil base-grid schrumpft)
                        mask = self._query_base_indices != bidx
                        self._query_base_indices = self._query_base_indices[mask]
                        self._query_base_indices[self._query_base_indices > bidx] -= 1
                    else:
                        # trotzdem base index list schrumpft:
                        self._query_base_indices[self._query_base_indices > bidx] -= 1

                # 2) base punkt wirklich löschen
                self._delete_base_point(bidx)

                self._update_points_label()
                self.update_plot()
                return

            return

        # LEFT CLICK
        if btn != 1:
            return

        # Prefer tracks if they exist: select + drag
        if btn == 1 and self.edit_tracks_mode and self.pred_tracks is not None:
            if self.pred_tracks is not None:
                tracks = np.asarray(self.pred_tracks)
                if tracks.ndim == 4:
                    tracks = tracks[0]
                if tracks.ndim == 3 and tracks.shape[0] > 0:
                    current_pts = tracks[self.current_frame]
                    tidx, tdist = self._nearest_point(current_pts, x_f, y_f)

                    if tdist < tol:
                        self.selected_track_idx = tidx
                        self._drag = ("track", tidx)
                        if self.apply_all_tracks_chk.isChecked():
                            self._drag_start_pos = (x_f, y_f)
                            self._drag_start_tracks = self.pred_tracks.copy()
                        self.update_plot()
                        return

        # Otherwise base/custom editing (pre-registration)
        if not self.edit_points_mode:
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
        self._update_points_label()
        self.update_grid_display_pre_registration()


    def on_image_move(self, x, y):
        if not (self.edit_points_mode or self.edit_tracks_mode) or self._drag is None:
            return

        kind, idx = self._drag

        if kind == "track":
            if not self.edit_tracks_mode:
                return

            if self.pred_tracks is not None:
                # pred_tracks ist bei dir i.d.R. numpy (T,N,2)
                if self.apply_all_tracks_chk.isChecked() and hasattr(self, "_drag_start_tracks"):
                    dx = float(x) - self._drag_start_pos[0]
                    dy = float(y) - self._drag_start_pos[1]
                    # Apply shift only to the current frame
                    self.pred_tracks[self.current_frame] = self._drag_start_tracks[self.current_frame] + np.array([dx, dy], dtype=np.float32)
                    self.viz_data['tracks'] = self.pred_tracks
                else:
                    self.pred_tracks[self.current_frame, idx, 0] = float(x)
                    self.pred_tracks[self.current_frame, idx, 1] = float(y)

                    if 'tracks' in self.viz_data and self.viz_data['tracks'] is not None:
                        self.viz_data['tracks'][self.current_frame, idx] = (float(x), float(y))

                self.update_plot()
                self._schedule_recalc_warping()
            return

        # base/custom drag nur im Edit-Points
        if not self.edit_points_mode:
            return

        if kind == "base":
            self.base_grid_pts[idx] = (float(x), float(y))
            self.update_grid_display_pre_registration()
        elif kind == "custom":
            self.custom_grid_pts[idx] = (int(x), int(y))
            self.update_grid_display_pre_registration()


    def on_image_release(self):
        if self._drag is not None and self._drag[0] == "track" and self.edit_tracks_mode:
            self._schedule_recalc_warping()
        self._drag = None
        self._drag_start_tracks = None
        self._drag_start_pos = None

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
        """Display the current frame without starting any processing (2-panel UI only)."""
        if self.video_data is None:
            return

        # Ensure viz_data has input frames (THWC float32)
        if 'input' not in self.viz_data:
            self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        # Clamp current frame index
        idx = int(np.clip(self.current_frame, 0, len(self.viz_data['input']) - 1))

        # Input frame (THWC) -> filtered display frame (usually HxW uint8)
        input_frame = self.viz_data['input'][idx]
        filtered_frame = self.apply_filter(input_frame, self.current_filter)

        # Use consistent contrast reference for "Input/Tracked/Corrected"
        levels = self._calc_levels(filtered_frame)

        left_choice = self.view_left_combo.currentText()
        right_choice = self.view_right_combo.currentText()

        # Render both viewports according to dropdown selection
        self._render_view(self.view_left, left_choice, idx, filtered_frame, levels)
        self._render_view(self.view_right, right_choice, idx, filtered_frame, levels)

        # Update motion plot cursor
        if hasattr(self, 'motion_plot'):
            self.motion_plot.set_current_frame(self.current_frame)

        # --- ensure first display always starts "fit to view" ---
        # (only once per loaded/cleared dataset)
        if not hasattr(self, "_initial_zoom_done"):
            self._initial_zoom_done = False

        if not self._initial_zoom_done:
            self._initial_zoom_done = True
            # 0ms: after current paint/event cycle
            QTimer.singleShot(0, self.reset_all_zoom)
            # small fallback if layout still settling
            QTimer.singleShot(150, self.reset_all_zoom)

    def _render_view(self, panel: PGImagePanel, which: str, idx: int, filtered_frame, levels):
        which = (which or "").strip()

        def _draw_pre_registration():
            ptsA = None
            crosses = None
            if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
                ptsA = self.base_grid_pts[self.base_grid_active]
            if self.custom_grid_pts:
                crosses = np.asarray(self.custom_grid_pts, dtype=np.float32)

            panel.set_points(pts_xy=ptsA, crosses_xy=crosses)
            panel.set_trajectories(None, 0)

        def _draw_tracks():
            tracks = self.viz_data.get("tracks", None)
            if tracks is None:
                return False

            tracks = np.asarray(tracks)
            if tracks.ndim == 4:
                tracks = tracks[0]
            if tracks.ndim != 3 or idx >= tracks.shape[0]:
                return False

            ptsA = tracks[idx]

            if self.pred_visibility is not None:
                vis = np.asarray(self.pred_visibility)
                if vis.ndim == 3:
                    vis = vis[0]
                v = vis[idx]  # (N,)

                # optional: combine with track_active
                mask = np.ones((ptsA.shape[0],), dtype=bool)
                if self.track_active is not None and len(self.track_active) == ptsA.shape[0]:
                    mask &= self.track_active

                # per-frame visibility threshold (e.g. 0.2)
                vthr = float(self.conf_thresh_spin.value())
                mask &= (v >= vthr)

                pts_show = ptsA[mask]

                # alpha proportional to visibility for displayed points
                v_show = v[mask]
                alpha = np.clip((v_show * 255).astype(np.uint8), 30, 255)
                colors = np.stack([np.full_like(alpha, 0),
                                np.full_like(alpha, 255),
                                np.full_like(alpha, 0),
                                alpha], axis=1)

                panel.set_points_colored(pts_show, colors_rgba=colors, size=8)
            else:
                # Visibility -> per-point color intensity
                N = ptsA.shape[0]
                vis = self._get_visibility_frame(idx, N)

                colors = None
                if vis is not None:
                    # green intensity + alpha scale with visibility
                    g = (vis * 255.0).astype(np.uint8)
                    a = (vis * 255.0).astype(np.uint8)

                    zeros = np.zeros_like(g, dtype=np.uint8)
                    colors = np.stack([zeros, g, zeros, a], axis=1)  # (N,4) RGBA

                panel.set_points_colored(ptsA, colors_rgba=colors, size=8)

            # custom points als crosses zusätzlich
            if self.custom_grid_pts:
                c = np.asarray(self.custom_grid_pts, dtype=np.float32)
                panel.cross_item.setData(c[:, 0], c[:, 1])
            else:
                panel.cross_item.setData([], [])

            return True

        # ---------------- views ----------------

        if which == "Input":
            panel.set_image(filtered_frame, auto_range=False, levels=levels)
            panel.set_points(None, None)
            panel.set_trajectories(None, 0)
            return

        if which == "Tracked Points":
            panel.set_image(filtered_frame, auto_range=False, levels=levels)
            panel.cross_item.setData([], [])

            tracks = self.viz_data.get('tracks', None)
            if tracks is not None:
                tr = np.asarray(tracks)
                if tr.ndim == 4:
                    tr = tr[0]
                if tr.ndim == 3 and idx < tr.shape[0]:
                    ptsA = tr[idx]  # (N,2)

                    N = ptsA.shape[0]
                    vis = self._get_visibility_frame(idx, N)

                    # --- mask: track_active + visibility threshold ---
                    mask = np.ones((N,), dtype=bool)

                    if self.track_active is not None and len(self.track_active) == N:
                        mask &= self.track_active

                    vthr = float(self.conf_thresh_spin.value()) if hasattr(self, "conf_thresh_spin") else 0.0
                    if vis is not None:
                        mask &= (vis >= vthr)

                    pts_show = ptsA[mask]

                    # --- color intensity from visibility ---
                    colors = None
                    if vis is not None:
                        v_show = vis[mask]
                        a = np.clip((v_show * 255.0), 30, 255).astype(np.uint8)  # alpha by visibility
                        g = np.clip((v_show * 255.0), 0, 255).astype(np.uint8)   # green by visibility

                        zeros = np.zeros_like(a, dtype=np.uint8)
                        colors = np.stack([zeros, g, zeros, a], axis=1)  # (M,4) RGBA

                    panel.set_points_colored(pts_show, colors_rgba=colors, size=8)
                    panel.set_trajectories(tr, idx, trail=15)

                    # optional crosses in edit mode
                    if self.edit_points_mode and self.custom_grid_pts:
                        c = np.asarray(self.custom_grid_pts, dtype=np.float32)
                        panel.cross_item.setData(c[:, 0], c[:, 1])
                    else:
                        panel.cross_item.setData([], [])

                    return

            # No tracks yet -> pre-registration
            ptsA = None
            if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
                ptsA = self.base_grid_pts[self.base_grid_active]
            crosses = np.array(self.custom_grid_pts, dtype=np.float32) if self.custom_grid_pts else None
            panel.set_points(pts_xy=ptsA, crosses_xy=crosses)
            panel.set_trajectories(None, 0)
            return


        if which == "Corrected Video":
            if self.viz_data.get("warped", None) is not None:
                wf = self.viz_data["warped"][idx]
                panel.set_image(wf, auto_range=False, levels=levels)
            else:
                panel.set_image(None)

            panel.set_points(None, None)
            panel.set_trajectories(None, 0)
            return

        if which == "Deformation Field":
            if self.disp_field is not None:
                disp_frame = self.disp_field[idx] if len(self.disp_field) > idx else self.disp_field[0]
                disp_preview = self._make_disp_preview(disp_frame)
                panel.set_image(disp_preview, auto_range=False)
            else:
                panel.set_image(None)

            panel.set_points(None, None)
            panel.set_trajectories(None, 0)
            return

    
    def visualize(self):
        if self.video_data is None:
            return

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

        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        # Prepare video input (apply filter if selected)
        video_input = self.video_data
        if self.current_filter and self.current_filter.lower() not in ("none", "raw"):
            try:
                self.status_label.setText(f"Applying filter '{self.current_filter}' to input frames...")
                QApplication.processEvents()

                vd = self.video_data
                if isinstance(vd, torch.Tensor):
                    vd = vd.detach().cpu().numpy()
                
                # vd is likely (T, C, H, W)
                filtered_frames = []
                for i in range(len(vd)):
                    frame = vd[i]
                    # If (C, H, W), transpose to (H, W, C) for apply_filter
                    if frame.ndim == 3 and frame.shape[0] in [1, 3]: 
                         frame = np.transpose(frame, (1, 2, 0))
                    
                    filt = self.apply_filter(frame, self.current_filter) # returns (H,W) uint8
                    
                    # Convert back to (3, H, W) float32 tensor
                    ft = torch.from_numpy(filt).float()
                    ft = ft.unsqueeze(0).repeat(3, 1, 1)
                    filtered_frames.append(ft)
                
                video_input = torch.stack(filtered_frames, dim=0)

            except Exception as e:
                print(f"Error filtering input: {e}")
                video_input = self.video_data

        # Prepare queries
        new_base_idxs = np.where(self.base_grid_active)[0] if self.base_grid_active is not None else []
        new_base_pts = self.base_grid_pts[new_base_idxs].astype(np.float32) if len(new_base_idxs) > 0 else np.zeros((0,2), np.float32)
        
        new_custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if len(self.custom_grid_pts) > 0 else np.zeros((0,2), np.float32)
        
        # All NEW queries
        # We cat them. The worker will handle them.
        new_queries = None
        if len(new_base_pts) > 0 or len(new_custom_pts) > 0:
            all_new = []
            if len(new_base_pts) > 0: all_new.append(new_base_pts)
            if len(new_custom_pts) > 0: all_new.append(new_custom_pts)
            new_queries = torch.from_numpy(np.vstack(all_new)).unsqueeze(0) # (1, N_new, 2)

        # Existing tracks
        existing_data = None
        if self.pred_tracks is not None:
            # Current tracks (T, N_old, 2)
            tr = self.pred_tracks
            vis = self.pred_visibility
            if vis is None:
                vis = np.ones((tr.shape[0], tr.shape[1]), dtype=np.float32)
            
            # total_N = N_old + N_new
            N_old = tr.shape[1]
            N_new = new_queries.shape[1] if new_queries is not None else 0
            total_N = N_old + N_new
            
            # Target indices for NEW points (append to the end)
            new_indices = list(range(N_old, total_N))
            existing_data = (tr, vis, new_indices, total_N)

        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass

        # Mapping for finishing
        self._last_run_new_base_idxs = new_base_idxs
        self._last_run_new_custom_count = len(self.custom_grid_pts)

        self.worker = ProcessingWorker(video_input, parent=self, existing_data=existing_data)
        # If we have new queries, pass them
        if new_queries is not None:
            self.worker.custom_grid_pts = new_queries
        elif existing_data is None:
            # Fallback if somehow called with nothing (should be blocked by UI but just in case)
            self.status_label.setText("No points to track")
            self.process_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            return

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

        self.worker = ProcessingWorker(video_input, base_grid_tensor, tracks=tracks_t, parent=self)
        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.start()

    def _delete_track_idx(self, tidx: int):
        if self.pred_tracks is None:
            return

        tr = np.asarray(self.pred_tracks)
        if tr.ndim == 4:
            tr = tr[0]
        if tr.ndim != 3:
            return

        if tidx < 0 or tidx >= tr.shape[1]:
            return

        # delete track across all frames
        tr = np.delete(tr, tidx, axis=1)
        self.pred_tracks = tr
        self.viz_data["tracks"] = tr

        # visibility (optional)
        if self.pred_visibility is not None:
            try:
                vis = np.asarray(self.pred_visibility)
                if vis.ndim == 2 and vis.shape[1] > tidx:
                    self.pred_visibility = np.delete(vis, tidx, axis=1)
            except Exception:
                pass

        # track_active
        if self.track_active is not None and len(self.track_active) > tidx:
            self.track_active = np.delete(self.track_active, tidx)

        # selection fix
        if self.selected_track_idx is not None:
            if self.selected_track_idx == tidx:
                self.selected_track_idx = None
            elif self.selected_track_idx > tidx:
                self.selected_track_idx -= 1

    def on_tracks_ready(self, pred_tracks, pred_visibility, grid_pts):
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
        self.pred_visibility = pred_visibility
        # Normalize visibility to numpy (T,N)
        vis = pred_visibility
        try:
            if isinstance(vis, torch.Tensor):
                vis = vis.detach().cpu().numpy()
            vis = np.asarray(vis)
        except Exception:
            vis = None

        if vis is not None:
            if vis.ndim == 3:  # (B,T,N)
                vis = vis[0]
            self.pred_visibility = vis  # (T,N)
        else:
            self.pred_visibility = None

        # --- Cleanup query points (incremental tracking) ---
        if hasattr(self, "_last_run_new_base_idxs"):
            if len(self._last_run_new_base_idxs) > 0:
                self.base_grid_active[self._last_run_new_base_idxs] = False
                del self._last_run_new_base_idxs
        
        if hasattr(self, "_last_run_new_custom_count"):
            # Remove the first N_new custom points (assuming they were added to the end of tracking)
            # Actually, visualize adds ALL current customs to queries and then they are tracked.
            # So we should clear all custom points that were tracked.
            self.custom_grid_pts = self.custom_grid_pts[self._last_run_new_custom_count:]
            del self._last_run_new_custom_count

        # init track_active from confidence (mean visibility)
        self.track_active = np.ones((N,), dtype=bool)
        self._apply_confidence_filter()

        # grid points
        if grid_pts is not None:
            self.grid_pts = grid_pts

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
    
    def _apply_confidence_filter(self):
        if self.pred_tracks is None:
            return
        if self.pred_visibility is None:
            return

        if not hasattr(self, "auto_conf_chk") or not self.auto_conf_chk.isChecked():
            # keep user's manual toggles
            self.update_plot()
            return

        thr = float(self.conf_thresh_spin.value()) if hasattr(self, "conf_thresh_spin") else 0.5
        vis = np.asarray(self.pred_visibility)  # (T,N)

        # robust mean confidence per track
        conf = np.nanmean(vis, axis=0)
        conf = np.where(np.isfinite(conf), conf, 0.0)

        self.track_active = conf >= thr
        self.update_plot()

        # If you want warping to update automatically when confidence changes:
        self._schedule_recalc_warping(delay_ms=0)

    def _to_thwc(self, vid):
        v = np.asarray(vid)
        # (T,3,H,W) -> (T,H,W,3)
        if v.ndim == 4 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
            v = np.transpose(v, (0, 2, 3, 1))
        return v

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video, disp_field):
        # tracks konsistent als numpy speichern (wichtig fürs Editieren)
        try:
            if isinstance(pred_tracks, torch.Tensor):
                pred_tracks = pred_tracks.detach().cpu().numpy()
            pred_tracks = np.asarray(pred_tracks)
            if pred_tracks.ndim == 4:
                pred_tracks = pred_tracks[0]
        except Exception:
            pass

        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        if grid_pts is not None:
            self.grid_pts = grid_pts

        # --- Cleanup query points (incremental tracking) ---
        if hasattr(self, "_last_run_new_base_idxs"):
            if len(self._last_run_new_base_idxs) > 0:
                self.base_grid_active[self._last_run_new_base_idxs] = False
                del self._last_run_new_base_idxs
        
        if hasattr(self, "_last_run_new_custom_count"):
            self.custom_grid_pts = self.custom_grid_pts[self._last_run_new_custom_count:]
            del self._last_run_new_custom_count
            self._update_points_label()

        # Warped video for GUI display in THWC
        self.warped_video = self._to_thwc(warped_video)
        self.viz_data['warped'] = self.warped_video

        # SSIM info (optional)
        try:
            self.ssim_score = video_ssim_score(self.warped_video, mode="consecutive")
            self.status_label.setText(f"Warping updated | SSIM={self.ssim_score:.4f}")
        except Exception as e:
            self.status_label.setText(f"Warping updated | SSIM failed: {e}")

        # Displacement field
        self.disp_field = disp_field

        # Populate viz_data with tracks in (T,N,2) for overlay
        tracks_np = self._tracks_to_numpy()
        if tracks_np is not None:
            tracks_np = np.asarray(tracks_np)
            if tracks_np.ndim == 4:
                tracks_np = tracks_np[0]
            self.viz_data['tracks'] = tracks_np

        # UI state
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        # Cleanup worker
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

        if self._pending_recalc:
            self._pending_recalc = False
            # nicht sofort starten – sondern bis zur letzten Deadline warten
            remaining = max(0, self._target_recalc_time_ms - self._edit_clock.elapsed())
            self._schedule_recalc_warping(delay_ms=remaining)
        
        self.edit_tracks_btn.setEnabled(True)

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

    def rotate_input(self, checked=False):
        if self.video_data is None:
            return

        try:
            # --- helpers (clockwise 90°) ---
            def rot90_cw_points_xy(p_xy, H, W):
                """
                p_xy: (...,2) in image coords (x right, y down)
                CW mapping:
                x' = H - 1 - y
                y' = x
                """
                p = np.asarray(p_xy, dtype=np.float32)
                out = p.copy()
                out[..., 0] = (H - 1) - p[..., 1]
                out[..., 1] = p[..., 0]
                return out

            def rot90_cw_list_of_tuples(pts_list, H, W):
                if not pts_list:
                    return []
                arr = np.asarray(pts_list, dtype=np.float32)
                arr2 = rot90_cw_points_xy(arr, H, W)
                return [(float(x), float(y)) for x, y in arr2]

            def rot90_cw_frames_thwc(frames_thwc):
                # frames: (T,H,W,C) -> (T,W,H,C)
                v = np.asarray(frames_thwc)
                return np.rot90(v, k=-1, axes=(1, 2))

            def rot90_cw_tracks_tn2(tracks_tn2, H, W):
                tr = np.asarray(tracks_tn2, dtype=np.float32)
                return rot90_cw_points_xy(tr, H, W)

            def rot90_cw_disp_field(df, H, W):
                """
                df can be:
                (T,H,W,2) vector field (u,v) in x/y pixel coords
                (T,H,W)   scalar
                (H,W,2) or (H,W)
                We rotate spatially CW and rotate vectors:
                dx' = -dy  => u' = -v
                dy' =  dx  => v' =  u
                """
                a = np.asarray(df)
                if a.ndim == 4 and a.shape[-1] == 2:
                    # (T,H,W,2)
                    a_rot = np.rot90(a, k=-1, axes=(1, 2))  # -> (T,W,H,2)
                    u = a_rot[..., 0].copy()
                    v = a_rot[..., 1].copy()
                    a_rot[..., 0] = -v
                    a_rot[..., 1] =  u
                    return a_rot

                if a.ndim == 3:
                    # (T,H,W) scalar
                    return np.rot90(a, k=-1, axes=(1, 2))

                if a.ndim == 3 and a.shape[-1] == 2:
                    # (H,W,2) (rare)
                    a_rot = np.rot90(a, k=-1, axes=(0, 1))
                    u = a_rot[..., 0].copy()
                    v = a_rot[..., 1].copy()
                    a_rot[..., 0] = -v
                    a_rot[..., 1] =  u
                    return a_rot

                if a.ndim == 2:
                    # (H,W) scalar
                    return np.rot90(a, k=-1, axes=(0, 1))

                return a  # fallback

            # --- old dims (before rotation) ---
            H_old = int(self.video_data.shape[2])
            W_old = int(self.video_data.shape[3])

            # --- rotate the core video tensor (T,C,H,W) clockwise ---
            # torch.rot90 k=-1 -> 90° clockwise
            self.video_data = torch.rot90(self.video_data, k=-1, dims=[2, 3])

            # --- rotate base grid + backup + custom points (keep everything) ---
            if self.base_grid_pts is not None and len(self.base_grid_pts) > 0:
                self.base_grid_pts = rot90_cw_points_xy(self.base_grid_pts, H_old, W_old)

            if self.base_grid_pts0 is not None and len(self.base_grid_pts0) > 0:
                self.base_grid_pts0 = rot90_cw_points_xy(self.base_grid_pts0, H_old, W_old)

            if self.custom_grid_pts is not None and len(self.custom_grid_pts) > 0:
                self.custom_grid_pts = rot90_cw_list_of_tuples(self.custom_grid_pts, H_old, W_old)

            # --- rotate tracks (pred + viz) ---
            if self.pred_tracks is not None:
                tr = np.asarray(self.pred_tracks)
                if tr.ndim == 4:
                    tr = tr[0]
                if tr.ndim == 3 and tr.shape[-1] == 2:
                    tr = rot90_cw_tracks_tn2(tr, H_old, W_old)
                    self.pred_tracks = tr
                    self.viz_data["tracks"] = tr

            # --- rotate grid_pts (if present) ---
            if self.grid_pts is not None:
                gp = self.grid_pts
                try:
                    if isinstance(gp, torch.Tensor):
                        gp_np = gp.detach().cpu().numpy()
                    else:
                        gp_np = np.asarray(gp)
                    if gp_np.ndim == 3 and gp_np.shape[-1] == 2:  # (1,N,2)
                        gp_np = rot90_cw_points_xy(gp_np, H_old, W_old)
                    elif gp_np.ndim == 2 and gp_np.shape[-1] == 2:  # (N,2)
                        gp_np = rot90_cw_points_xy(gp_np, H_old, W_old)
                    self.grid_pts = gp_np
                except Exception:
                    pass

            # --- rotate any cached frames currently displayed ---
            if "input" in self.viz_data and self.viz_data["input"] is not None:
                self.viz_data["input"] = rot90_cw_frames_thwc(self.viz_data["input"])

            if "warped" in self.viz_data and self.viz_data["warped"] is not None:
                self.viz_data["warped"] = rot90_cw_frames_thwc(self.viz_data["warped"])
                self.warped_video = self.viz_data["warped"]

            # --- rotate displacement field ---
            if self.disp_field is not None:
                self.disp_field = rot90_cw_disp_field(self.disp_field, H_old, W_old)

            # --- update frame/UI bookkeeping ---
            self.total_frames = int(self.video_data.shape[0])
            self.current_frame = int(np.clip(self.current_frame, 0, self.total_frames - 1))
            self.slider.setMaximum(max(0, self.total_frames - 1))
            self.slider.setValue(self.current_frame)
            self.update_frame_label()

            # --- refresh UI ---
            self._update_points_label()
            self.update_plot()
            self.reset_all_zoom()
            self.status_label.setText("Rotated 90° clockwise (video + overlays + points kept)")

        except Exception as e:
            self.status_label.setText(f"Error rotating: {e}")


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

    def open_parameter_dialog(self):
        self.param_dialog.show()
        self.param_dialog.raise_()
        self.param_dialog.activateWindow()

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
                from mc_cotracker import save_and_display_video
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
                     import csv
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
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MotionCorrectionGUI()
    gui.show()
    sys.exit(app.exec())
