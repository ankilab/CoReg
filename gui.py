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
import pyqtgraph as pg

import cv2

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5
pg.setConfigOptions(imageAxisOrder='row-major')  # damit (H,W) wie numpy angezeigt wird

class ProcessingWorker(QThread):
    tracks_ready = pyqtSignal(object, object, object)  # tracks_np, visibility, grid
    progress = pyqtSignal(int)  # percent complete
    status_update = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object, object)  # tracks_np, visibility, grid, warped_full, disp_field
    error = pyqtSignal(str)

    def __init__(self, video_data, base_grid_pts, custom_grid_pts=None, df_sigma=3.0,parent=None):
        super().__init__(parent)
        self.video_data = video_data
        self.base_grid_pts = base_grid_pts
        self.custom_grid_pts = custom_grid_pts
        self.df_sigma = float(df_sigma)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            video = self.video_data.to(device) if isinstance(self.video_data, torch.Tensor) else self.video_data

            grid_xy = self.base_grid_pts.to(device)
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

            self.status_update.emit("Tracking running...")
            self.progress.emit(0)

            pred_tracks, pred_visibility = cotrack_model(video, queries=queries)

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
            
            self.progress.emit(70)
            self.status_update.emit("Warping / Displacement optimization ...")

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

class PGImagePanel(pg.PlotWidget):
    pressed  = pyqtSignal(float, float, int)   # x,y,button (1/2/3)
    moved    = pyqtSignal(float, float)
    released = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

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
        
        # Enable mouse interaction
        self.vb.setMouseEnabled(x=True, y=True)
        
        # Lock aspect ratio to prevent stretching
        # The image will fit in the panel with black borders if needed
        self.vb.setAspectLocked(True)
        
        # Invert Y axis for image coordinates
        self.vb.invertY(True)

        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)

        # keep full-resolution rendering (no auto decimation)
        try:
            self.img_item.setAutoDownsample(False)
        except Exception:
            self.img_item.setOpts(autoDownsample=False)

        self.points_item = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(255, 255, 0, 200))
        self.cross_item  = pg.ScatterPlotItem(size=10, pen=pg.mkPen(255, 255, 0), brush=None, symbol='x')
        self.traj_item   = pg.PlotDataItem(pen=pg.mkPen(255, 255, 0, 180, width=2))

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

    def set_image(self, img_np, auto_range=True):
        if img_np is None:
            self.img_item.clear()
            self.points_item.setData([], [])
            self.cross_item.setData([], [])
            for it in self.traj_items:
                it.setData([], [])
            self._img_w = None
            self._img_h = None
            return

        a = np.asarray(img_np)

        # ---- make display-safe for pyqtgraph ----
        if a.dtype.kind == "f":
            # grayscale float -> provide explicit levels
            if a.ndim == 2:
                finite = np.isfinite(a)
                if finite.any():
                    lo = float(np.nanpercentile(a[finite], 1))
                    hi = float(np.nanpercentile(a[finite], 99))
                    if hi <= lo:
                        hi = lo + 1.0
                else:
                    lo, hi = 0.0, 1.0

                self.img_item.setImage(a, levels=(lo, hi), autoLevels=False)
            else:
                # RGB/RGBA float -> convert to uint8 (no levels needed)
                finite = np.isfinite(a)
                if finite.any():
                    lo = float(np.nanpercentile(a[finite], 1))
                    hi = float(np.nanpercentile(a[finite], 99))
                    if hi <= lo:
                        hi = lo + 1.0
                    a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                else:
                    a = np.zeros_like(a, dtype=np.uint8)
                self.img_item.setImage(a, autoLevels=False)

        else:
            # integer/bool types are fine; keep autoLevels for uint16 etc if you want
            if a.dtype == np.bool_:
                a = (a.astype(np.uint8) * 255)
            self.img_item.setImage(a, autoLevels=True)

        # ---- rest of your original sizing / autorange ----
        H, W = a.shape[:2]
        size_changed = (self._img_w != W or self._img_h != H)
        self._img_w = W
        self._img_h = H

        if size_changed:
            self.img_item.setRect(0, 0, W, H)
            self.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            self.setMinimumSize(200, 200)

        if auto_range:
            QTimer.singleShot(50, lambda: self.vb.autoRange())

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


class MotionCorrectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Correction GUI")
        screen = QApplication.primaryScreen()
        g = screen.availableGeometry()

        # Start with a reasonable default size, but allow free resizing
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
        self.lbl_input.set_image(bg)
        self.lbl_input.set_points(None, None)
        self.lbl_input.set_trajectories(None, 0)

    # def _update_label_with_overlay(self, label, arr, ptsA=None, custom_pts=None, tracks=None, show_trajectories=False):
    #     pix = self._array_to_qpixmap(arr)
    #     overlay_data = {}

    #     points_list = []
    #     if ptsA is not None:
    #         points_list.append((ptsA, (255, 255, 0), 4))
    #     if tracks is not None:
    #         points_list.append((tracks, (255, 255, 0), 2))
    #     if points_list:
    #         overlay_data['points'] = points_list

    #     if custom_pts is not None and len(custom_pts) > 0:
    #         overlay_data['crosses'] = [(custom_pts, (255, 255, 0), 6)]

    #     if show_trajectories and 'tracks' in self.viz_data and self.viz_data['tracks'] is not None:
    #         full_tracks = self.viz_data['tracks']  # (T,N,2)
    #         overlay_data['trajectories'] = (full_tracks, self.current_frame, 15, (0, 255, 255))

    #     #label.setPixmap(pix, overlay_data)
    #     panel.set_image(pix, overlay_data)
    #     panel.set_points(pix, overlay_data)

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

        controls = QWidget(central_widget)
        controls_grid = QGridLayout(controls)
        controls_grid.setContentsMargins(6, 6, 6, 6)
        controls_grid.setHorizontalSpacing(3)  # Reduced from 8 to minimize gaps
        controls_grid.setVerticalSpacing(4)

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.select_file)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Regular grid",
            "Features (Shi–Tomasi)",
            "Activity map (motion)",
            "Segmentation centroids"
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
        self.process_btn.clicked.connect(self.visualize)
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

        self.sync_view_chk = QCheckBox("Sync view")
        self.sync_view_chk.stateChanged.connect(self.on_sync_view_toggled)

        self.df_sigma_spin = QSpinBox()
        self.df_sigma_spin.setRange(0, 20)
        self.df_sigma_spin.setValue(3)
        self.df_sigma_spin.valueChanged.connect(lambda *_: self.update_plot())

        self.points_label = QLabel("Points: 0")
        self.status_label = QLabel("Ready - Drag & drop video or use Load button")
        self.status_label.setWordWrap(True)

        self.lbl_N = QLabel("N:")
        self.lbl_minDist = QLabel("MinDist:")
        self.lbl_rows = QLabel("Rows:")
        self.lbl_cols = QLabel("Cols:")

        # Row 0: Load File, Sync View, DF Gaussian, Filter
        c = 0
        controls_grid.addWidget(self.load_btn, 0, c); c += 1
        controls_grid.addWidget(self.sync_view_chk, 0, c); c += 1
        controls_grid.addWidget(QLabel("DF σ:"), 0, c); c += 1
        controls_grid.addWidget(self.df_sigma_spin, 0, c); c += 1
        controls_grid.addWidget(self.lbl_filter, 0, c); c += 1
        controls_grid.addWidget(self.filter_combo, 0, c); c += 1
        controls_grid.addWidget(self.reset_zoom_btn, 0, c); c += 1
        ncols = max(c, 0)

        # Row 1: Preset, Apply Preset, Run Registration
        c = 0
        controls_grid.addWidget(QLabel("Preset:"), 1, c); c += 1
        controls_grid.addWidget(self.preset_combo, 1, c); c += 1
        controls_grid.addWidget(self.apply_preset_btn, 1, c); c += 1
        controls_grid.addWidget(self.process_btn, 1, c); c += 1
        controls_grid.addWidget(self.stop_btn, 1, c); c += 1
  
        
        # Fixed positions for parameter controls
        # These controls occupy the same grid positions but are toggled with visibility
        param_col = c  # Start of parameter section (column 5)
        # Grid controls (rows/cols) - shown for "Regular grid" preset
        controls_grid.addWidget(self.lbl_rows, 1, param_col, 1, 1)
        controls_grid.addWidget(self.grid_rows_ctrl, 1, param_col + 1, 1, 1)
        controls_grid.addWidget(self.lbl_cols, 1, param_col + 2, 1, 1)
        controls_grid.addWidget(self.grid_cols_ctrl, 1, param_col + 3, 1, 1)
        # Feature controls (N/minDist) - shown for other presets, occupy same positions
        controls_grid.addWidget(self.lbl_N, 1, param_col, 1, 1)
        controls_grid.addWidget(self.num_points_spin, 1, param_col + 1, 1, 1)
        controls_grid.addWidget(self.lbl_minDist, 1, param_col + 2, 1, 1)
        controls_grid.addWidget(self.min_dist_spin, 1, param_col + 3, 1, 1)
        
        ncols = max(ncols, param_col + 4)

        # Row 2: Point editing controls
        c = 0
        controls_grid.addWidget(self.edit_points_btn, 2, c); c += 1
        controls_grid.addWidget(self.clear_points_btn, 2, c); c += 1
        controls_grid.addWidget(self.undo_btn, 2, c); c += 1
        controls_grid.addWidget(self.points_label, 2, c); c += 1
        ncols = max(ncols, c)

        # Row 3: Status label (spans all columns)
        controls_grid.addWidget(self.status_label, 3, 0, 1, ncols)
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
        #self.lbl_input = ImageLabel()
        self.lbl_grid_title = QLabel("Tracked Points")
        #self.lbl_grid = ImageLabel()
        self.lbl_warped_title = QLabel("Corrected Video")
        #self.lbl_warped = ImageLabel()
        self.lbl_tracks_title = QLabel("Deformation Field")
        #self.lbl_tracks = ImageLabel()

        self.lbl_input = PGImagePanel()
        self.lbl_grid  = PGImagePanel()
        self.lbl_warped= PGImagePanel()
        self.lbl_tracks= PGImagePanel()

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
            vbox.addWidget(label, stretch=1)  # Give the panel all available space
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


    def on_sync_view_toggled(self, state):
        panels = [self.lbl_input, self.lbl_grid, self.lbl_warped, self.lbl_tracks]
        master_vb = panels[0].getViewBox()

        if state == Qt.CheckState.Checked.value:
            for p in panels[1:]:
                vb = p.getViewBox()
                vb.setXLink(master_vb)
                vb.setYLink(master_vb)
            self.status_label.setText("View sync: ON")
        else:
            for p in panels[1:]:
                vb = p.getViewBox()
                vb.setXLink(None)
                vb.setYLink(None)
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
        xs = np.linspace(10, W - 10, cols, dtype=np.float32) #0, H-1
        ys = np.linspace(10, H - 10, rows, dtype=np.float32)
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

        self.lbl_input.set_image(None)
        self.lbl_grid.set_image(None)
        self.lbl_warped.set_image(None)
        self.lbl_tracks.set_image(None)

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
                self.video_data = load_video(self.video_path, resize=False, max_frames=20)
            elif ext in ['.tiff', '.tif']:
                video_np = load_tiff(self.video_path, frames=20)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            elif ext in ['.h5', '.hdf5']:
                video_np = load_hdf5(self.video_path, "/data", frames=20)
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
            self.step_fwd_btn.setEnabled(True)

            self.update_plot()
            QTimer.singleShot(100, lambda: self.reset_all_zoom())

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
        btn = int(button_value)  # already 1/2/3 from PGImagePanel

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
        if self.base_grid_active is not None:
            self.base_grid_active[:] = False
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
        for panel in [self.lbl_input, self.lbl_grid, self.lbl_warped, self.lbl_tracks]:
            panel.reset_view()
        self.status_label.setText("View reset on all panels")

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

        frame0 = self.viz_data['input'][0] if 'input' in self.viz_data else \
                self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy()[0]

        bg = self.apply_filter(frame0, self.current_filter)

        ptsA = None
        if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
            ptsA = self.base_grid_pts[self.base_grid_active]

        custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if self.custom_grid_pts else None

        self.lbl_grid.set_image(bg, auto_range=False)
        self.lbl_grid.set_points(pts_xy=ptsA, crosses_xy=custom_pts)
        self.lbl_grid.set_trajectories(None, 0)

    def update_plot(self):
        """Display the current frame without starting any processing."""
        if self.video_data is None:
            return

        # Ensure viz_data has input frames
        if 'input' not in self.viz_data:
            self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        # Get current frame index
        idx = min(self.current_frame, len(self.viz_data['input']) - 1)
        
        # Display input frame with filter
        input_frame = self.viz_data['input'][idx]
        filtered_frame = self.apply_filter(input_frame, self.current_filter)
        self.lbl_input.set_image(filtered_frame, auto_range=False)
        
        # Display tracked points panel
        ptsA = None
        custom_pts = None
        
        if 'tracks' in self.viz_data and self.viz_data['tracks'] is not None:
            # Get current frame's tracked positions
            tracks = self.viz_data['tracks']  # (T, N, 2)
            if tracks.ndim == 3 and idx < tracks.shape[0]:
                current_positions = tracks[idx]  # (N, 2) - all points at current frame
                ptsA = current_positions
        else:
            # No tracks yet, show initial grid positions
            if self.base_grid_pts is not None and self.base_grid_active is not None and np.any(self.base_grid_active):
                ptsA = self.base_grid_pts[self.base_grid_active]
            custom_pts = np.array(self.custom_grid_pts, dtype=np.float32) if self.custom_grid_pts else None
        
        self.lbl_grid.set_image(filtered_frame, auto_range=False)
        self.lbl_grid.set_points(pts_xy=ptsA, crosses_xy=custom_pts)
        
        # Display trajectories if available
        if 'tracks' in self.viz_data and self.viz_data['tracks'] is not None:
            self.lbl_grid.set_trajectories(self.viz_data['tracks'], idx, trail=15)
        else:
            self.lbl_grid.set_trajectories(None, 0)
        
        # Display warped video if available
        if 'warped' in self.viz_data and self.viz_data['warped'] is not None:
            warped_frame = self.viz_data['warped'][idx]
            self.lbl_warped.set_image(warped_frame, auto_range=False)
        else:
            self.lbl_warped.set_image(None)
        
        # Display displacement field if available (with smoothing for display)
        if self.disp_field is not None:
            disp_frame = self.disp_field[idx] if len(self.disp_field) > idx else self.disp_field[0]
            disp_preview = self._make_disp_preview(disp_frame)  # Apply smoothing here
            self.lbl_tracks.set_image(disp_preview, auto_range=False)
        else:
            self.lbl_tracks.set_image(None)


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

        self.worker = ProcessingWorker(video_input, base_grid_tensor, custom_pts_tensor, parent=self)
        self.worker.tracks_ready.connect(self.on_tracks_ready)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.status_update.connect(self.status_label.setText)
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
        
        # Populate viz_data with tracks so update_plot() can display them
        tracks_np = self._tracks_to_numpy()
        if tracks_np is not None:
            tracks_np = np.asarray(tracks_np)
            if tracks_np.ndim == 4:  # (B,T,N,2)
                tracks_np = tracks_np[0]  # -> (T,N,2)
            self.viz_data['tracks'] = tracks_np
        
        # Update display with tracked points, don't restart processing
        self.update_plot()

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video, disp_field):
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        self.warped_video = warped_video
        self.disp_field = disp_field

        # Populate viz_data with results so update_plot() can display them
        if self.warped_video is not None:
            self.viz_data['warped'] = self.warped_video
        
        tracks_np = self._tracks_to_numpy()
        if tracks_np is not None:
            tracks_np = np.asarray(tracks_np)
            if tracks_np.ndim == 4:  # (B,T,N,2)
                tracks_np = tracks_np[0]  # -> (T,N,2)
            self.viz_data['tracks'] = tracks_np

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

        # Update display with final results, don't restart processing
        self.update_plot()
        
        # Ensure new panels are fitted correctly
        QTimer.singleShot(100, lambda: self.lbl_warped.reset_view())
        QTimer.singleShot(100, lambda: self.lbl_tracks.reset_view())

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
