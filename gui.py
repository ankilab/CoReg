import sys
import os
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QCheckBox, QGridLayout, QComboBox, QSpinBox, QSizePolicy, QProgressBar, QDialog,
    QGraphicsRectItem, QDoubleSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPalette
import pyqtgraph as pg

import cv2
try:
    import ants
except ImportError:
    ants = None
try:
    import utils.torch_lddmm as torch_lddmm
except ImportError:
    torch_lddmm = None

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils.mc_cotracker as mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from utils.mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5
pg.setConfigOptions(imageAxisOrder='row-major')  # damit (H,W) wie numpy angezeigt wird
from utils.utils import video_ssim_score



class ProcessingWorker(QThread):
    # tracks, visibility, confidence, grid
    tracks_ready = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    # tracks_np, visibility, confidence, grid, warped_full, disp_field
    finished = pyqtSignal(object, object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(
        self,
        video_data,
        base_grid_pts=None,
        custom_grid_pts=None,
        df_sigma=3.0,
        tracks=None,
        parent=None,
        method="CoTracker",
        filter_name=None,
        existing_data=None,
        queries=None,
    ):
        super().__init__(parent)

        self.video_data = video_data
        self.base_grid_pts = base_grid_pts
        self.custom_grid_pts = custom_grid_pts
        self.df_sigma = float(df_sigma)
        self.tracks = tracks

        self.method = method
        self.filter_name = filter_name
        self.existing_data = existing_data  # currently not merged here; stored for future if you need it
        self.queries = queries  # tensor (1,N,2) or None

        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        tracks_gui = None
        vis_gui = None
        conf_gui = None
        grid_xy_ret = None

        try:
            # ---------- ANTs ----------
            if self.method == "ANTs":
                if ants is None:
                    raise ImportError("ANTs is not installed. Please install it with 'pip install antspyx'")

                self.status_update.emit("Running ANTs registration...")
                vd = self.video_data
                if isinstance(vd, torch.Tensor):
                    vd = vd.detach().cpu().numpy()

                if vd.ndim == 4 and vd.shape[1] in [1, 3]:
                    vd = np.transpose(vd, (0, 2, 3, 1))  # (T,H,W,C)

                T = vd.shape[0]
                results = []

                ref_frame = vd[0]
                if ref_frame.ndim == 3 and ref_frame.shape[-1] == 3:
                    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
                fixed_image = ants.from_numpy(ref_frame)

                for i in range(T):
                    if not self._is_running:
                        return
                    self.progress.emit(int(i / max(1, T) * 100))

                    if i == 0:
                        first = ref_frame
                        if vd[0].ndim == 3 and vd[0].shape[-1] == 3:
                            first = np.stack([first] * 3, axis=-1)
                        results.append(first)
                        continue

                    moving_frame = vd[i]
                    if moving_frame.ndim == 3 and moving_frame.shape[-1] == 3:
                        moving_frame = cv2.cvtColor(moving_frame, cv2.COLOR_RGB2GRAY)
                    moving_image = ants.from_numpy(moving_frame)

                    reg = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyNOnly")
                    warped = reg["warpedmovout"].numpy()

                    if vd[i].ndim == 3 and vd[i].shape[-1] == 3:
                        warped = np.stack([warped] * 3, axis=-1)
                    results.append(warped)

                warped_np = np.stack(results, axis=0)
                self.progress.emit(100)
                self.status_update.emit("Done")
                self.finished.emit(None, None, None, None, warped_np, None)
                return

            # ---------- LDDMM ----------
            if self.method == "LDDMM":
                if torch_lddmm is None:
                    raise ImportError("torch_lddmm is not available. Please ensure utils/torch_lddmm.py exists.")

                self.status_update.emit("Running LDDMM registration...")
                vd = self.video_data
                if isinstance(vd, torch.Tensor):
                    vd = vd.detach().cpu().numpy()

                if vd.ndim == 4 and vd.shape[1] in [1, 3]:
                    vd = np.transpose(vd, (0, 2, 3, 1))  # (T,H,W,C)

                T = vd.shape[0]
                results = []

                ref_frame = vd[0]
                if ref_frame.ndim == 3 and ref_frame.shape[-1] == 3:
                    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)

                for i in range(T):
                    if not self._is_running:
                        return
                    self.progress.emit(int(i / max(1, T) * 100))

                    if i == 0:
                        first = ref_frame
                        if vd[0].ndim == 3 and vd[0].shape[-1] == 3:
                            first = np.stack([first] * 3, axis=-1)
                        results.append(first)
                        continue

                    moving_frame = vd[i]
                    if moving_frame.ndim == 3 and moving_frame.shape[-1] == 3:
                        moving_frame = cv2.cvtColor(moving_frame, cv2.COLOR_RGB2GRAY)

                    lddmm = torch_lddmm.LDDMM(
                        template=[moving_frame],
                        target=[ref_frame],
                        do_affine=0,
                        do_lddmm=1,
                        a=7,
                        nt=5,
                        niter=200,
                        epsilon=4e0,
                        sigma=20.0,
                        sigmaR=40.0,
                        optimizer="gdr",
                        gpu_number=0,
                        verbose=False,
                    )
                    lddmm.run()
                    warped = lddmm.outputDeformedTemplate()[0]

                    if vd[i].ndim == 3 and vd[i].shape[-1] == 3:
                        warped = np.stack([warped] * 3, axis=-1)

                    results.append(warped)

                warped_np = np.stack(results, axis=0)
                self.progress.emit(100)
                self.status_update.emit("Done")
                self.finished.emit(None, None, None, None, warped_np, None)
                return

            # ---------- CoTracker + warping ----------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            video = self.video_data.to(device) if isinstance(self.video_data, torch.Tensor) else self.video_data

            # Build queries tensor (1,N,2)
            queries = None
            if self.tracks is None:
                if self.queries is not None:
                    queries = self.queries
                elif self.custom_grid_pts is not None:
                    queries = self.custom_grid_pts

                if queries is None:
                    # fallback: try base_grid_pts
                    if self.base_grid_pts is not None:
                        q = self.base_grid_pts
                        if isinstance(q, np.ndarray):
                            q = torch.from_numpy(q)
                        if q.ndim == 2:
                            q = q.unsqueeze(0)
                        queries = q
                    else:
                        raise ValueError("No query points provided for tracking.")

                if isinstance(queries, np.ndarray):
                    queries = torch.from_numpy(queries)
                if queries.ndim == 2:
                    queries = queries.unsqueeze(0)
                queries = queries.to(device).float()

                self.status_update.emit("Tracking running...")
                self.progress.emit(0)
                

                if queries is not None:
                    if isinstance(queries, np.ndarray):
                        queries = torch.from_numpy(queries)

                    if queries.ndim == 2:          # (N,2) -> (1,N,2)
                        queries = queries.unsqueeze(0)

                    if queries.shape[-1] == 2:
                        B, N, _ = queries.shape
                        t0 = torch.zeros((B, N, 1), device=queries.device, dtype=queries.dtype)
                        queries = torch.cat([t0, queries], dim=-1)  # -> (B,N,3) = (t,x,y)


                pred_tracks, pred_visibility = cotrack_model(video, queries=queries)

                tracks_raw = pred_tracks.detach().cpu().numpy()
                tracks_gui = tracks_raw[0] if tracks_raw.ndim == 4 else tracks_raw
                tracks_for_warp = tracks_raw if tracks_raw.ndim == 4 else tracks_raw[None, ...]

                vis_raw = pred_visibility.detach().cpu().numpy() if isinstance(pred_visibility, torch.Tensor) else np.asarray(pred_visibility)
                vis_gui = vis_raw[0] if vis_raw.ndim == 3 else vis_raw
                conf_gui = None

                # ----- NEW: merge with existing tracks if provided -----
                if self.existing_data is not None:
                    old_tr, old_vis, old_conf = self.existing_data

                    old_tr = np.asarray(old_tr)  # (T, N_old, 2)
                    if old_tr.ndim == 4: old_tr = old_tr[0]
                    new_tr = tracks_gui          # (T, N_new, 2)

                    Tm = min(old_tr.shape[0], new_tr.shape[0])
                    old_tr = old_tr[:Tm]
                    new_tr = new_tr[:Tm]
                    tracks_gui = np.concatenate([old_tr, new_tr], axis=1)

                    # visibility merge (optional)
                    if vis_gui is not None and old_vis is not None:
                        ov = np.asarray(old_vis)
                        if ov.ndim == 3: ov = ov[0]
                        ov = ov[:Tm]
                        nv = np.asarray(vis_gui)[:Tm]
                        vis_gui = np.concatenate([ov, nv], axis=1)
                    else:
                        vis_gui = None

                    # confidence currently unused in your UI; keep None
                    conf_gui = None

                # for warping, always warp with merged tracks
                tracks_for_warp = tracks_gui[None, ...]  # (1,T,N,2)
                self.tracks_ready.emit(tracks_gui, vis_gui, conf_gui, grid_xy_ret)


            else:
                self.status_update.emit("Using existing tracks...")
                self.progress.emit(10)
                tracks_gui = self.tracks.detach().cpu().numpy() if isinstance(self.tracks, torch.Tensor) else np.asarray(self.tracks)
                tracks_for_warp = tracks_gui if tracks_gui.ndim == 4 else tracks_gui[None, ...]

            self.progress.emit(70)
            self.status_update.emit("Warping / Displacement optimization ...")

            if not isinstance(tracks_for_warp, torch.Tensor):
                tracks_for_warp = torch.from_numpy(tracks_for_warp)
            tracks_for_warp = tracks_for_warp.to(device).float()

            def warp_cb(frac):
                if not self._is_running:
                    return
                self.progress.emit(int(70 + frac * 30))

            if not self._is_running:
                return

            warped_video, disp_field = warping(tracks_for_warp, video, smoothing=self.df_sigma, callback=warp_cb)

            if not self._is_running:
                return

            warped_np = np.asarray(warped_video)
            if warped_np.ndim >= 4 and warped_np.shape[0] == 1:
                warped_np = warped_np[0]

            disp_arr = disp_field.detach().cpu().numpy() if isinstance(disp_field, torch.Tensor) else np.asarray(disp_field)
            if disp_arr.ndim >= 4 and disp_arr.shape[0] == 1:
                disp_arr = disp_arr[0]

            self.progress.emit(100)
            self.status_update.emit("Done")
            self.finished.emit(tracks_gui, vis_gui, conf_gui, grid_xy_ret, warped_np, disp_arr)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    # def mousePressEvent(self, event):
    #     is_left_pan = (event.button() == Qt.MouseButton.LeftButton and
    #                    (event.modifiers() & self.left_pan_modifier))
    #     is_mid_pan = (event.button() == Qt.MouseButton.MiddleButton)

    #     if is_mid_pan or is_left_pan:
    #         self._is_panning = True
    #         self._pan_start = [event.position().x(), event.position().y()]
    #         self.setCursor(Qt.CursorShape.ClosedHandCursor)
    #         event.accept()
    #         return

    #     x, y = self._map_to_image(event.position())
    #     self.pressed.emit(x, y, event.button().value)
    #     event.accept()

    # def mouseMoveEvent(self, event):
    #     if self._is_panning and self._pan_start is not None:
    #         dx = event.position().x() - self._pan_start[0]
    #         dy = event.position().y() - self._pan_start[1]
    #         self.pan_offset[0] += dx
    #         self.pan_offset[1] += dy
    #         self._pan_start = [event.position().x(), event.position().y()]
    #         self._redraw_viewport()
    #         event.accept()
    #         return

    #     x, y = self._map_to_image(event.position())
    #     self.moved.emit(x, y)
    #     event.accept()

    # def mouseReleaseEvent(self, event):
    #     if self._is_panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
    #         self._is_panning = False
    #         self._pan_start = None
    #         self.setCursor(Qt.CursorShape.ArrowCursor)
    #     self.released.emit()
    #     event.accept()

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
        self.pending_item = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 80, 80, 220), symbol='x')
        self.addItem(self.pending_item)

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
            self.pending_item.setData([], [])
            for it in self.traj_items:
                it.setData([], [])
            self._img_w, self._img_h = None, None
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

    def mousePressEvent(self, event):
        # Check if Box Fill mode is active first
        box_fill_active = False  # Box Fill feature removed
        
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if anyone is connected to our pressed signal
            if self.receivers(self.pressed) > 0:
                x, y = self._map_to_image(event.position())
                self.pressed.emit(float(x), float(y), 1)
                
                # If Box Fill is active, completely block the event from going to ViewBox
                if box_fill_active:
                    self._custom_drag_active = True
                    event.accept()
                    # Do NOT call super() - this prevents ViewBox from seeing the event
                    return
                else:
                    # Allow panning when Box Fill is off
                    self._custom_drag_active = False
        
        if event.button() == Qt.MouseButton.RightButton:
            x, y = self._map_to_image(event.position())
            self.pressed.emit(float(x), float(y), 3)
            
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Always emit the moved signal for tracking
        x, y = self._map_to_image(event.position())
        self.moved.emit(float(x), float(y))
        
        # If we're in a custom drag, don't pass to super (prevents panning)
        if self._custom_drag_active:
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.released.emit()
        self._custom_drag_active = False
        super().mouseReleaseEvent(event)


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

        self.grid_rows_ctrl = QSpinBox()
        self.grid_rows_ctrl.setRange(2, 100)
        self.grid_rows_ctrl.setValue(int(getattr(self, "grid_rows", 4)))
        self.grid_rows_ctrl.valueChanged.connect(self.on_preset_params_changed)

        self.grid_cols_ctrl = QSpinBox()
        self.grid_cols_ctrl.setRange(2, 100)
        self.grid_cols_ctrl.setValue(int(getattr(self, "grid_cols", 4)))
        self.grid_cols_ctrl.valueChanged.connect(self.on_preset_params_changed)

        self.apply_preset_btn = QPushButton("Apply to Grid")
        self.apply_preset_btn.clicked.connect(self.apply_point_preset)

        self.df_sigma_spin = QSpinBox()
        self.df_sigma_spin.setRange(0, 20)
        self.df_sigma_spin.setValue(3)
        self.df_sigma_spin.valueChanged.connect(lambda *_: self.update_plot())
        
        # Expand preset after motion analysis (moved to Parameters)
        self.expand_mode_combo = QComboBox()
        self.expand_mode_combo.addItems([
            "Heuristic",
            "Algorithm"
        ])
        self.expand_preset_btn = QPushButton("➕ Expand Preset")
        self.expand_preset_btn.clicked.connect(self.expand_preset_after_motion)
        self.expand_preset_btn.setEnabled(False)  # enabled after first motion analysis
        
        self.rigid_motion_chk = QCheckBox("Rigid motion (apply track to grid)")
        self.rigid_motion_chk.setToolTip("When checked, click a track to apply its motion to a regular grid.")
        
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
        pd_layout.addWidget(QLabel("Expand Preset:"))
        pd_layout.addWidget(self.expand_mode_combo)
        pd_layout.addWidget(self.expand_preset_btn)
        pd_layout.addWidget(self.rigid_motion_chk)
        
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
        self.view_left_combo.addItems(["Input", "Tracked Points", "Corrected", "Deformation Field", "Corrected (ANTs)", "Corrected (LDDMM)"])
        self.view_left_combo.setCurrentText("Tracked Points")
        self.view_left_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        # Right selector
        self.view_right_combo = QComboBox()
        self.view_right_combo.addItems(["Input", "Tracked Points", "Corrected", "Deformation Field", "Corrected (ANTs)", "Corrected (LDDMM)"])
        self.view_right_combo.setCurrentText("Corrected")
        self.view_right_combo.currentTextChanged.connect(lambda *_: self.update_plot())

        # Panels
        self.view_left = PGImagePanel()
        self.view_right = PGImagePanel()
        self.view_left._parent_gui = self
        self.view_right._parent_gui = self

        # Place selectors above each panel
        viewer_layout.addWidget(QLabel("Left view:"), 0, 0)
        viewer_layout.addWidget(QLabel("Right view:"), 0, 1)
        viewer_layout.addWidget(self.view_left_combo, 1, 0)
        viewer_layout.addWidget(self.view_right_combo, 1, 1)
        viewer_layout.addWidget(self.view_left, 2, 0)
        viewer_layout.addWidget(self.view_right, 2, 1)

        main_layout.addWidget(viewer_widget, stretch=1)

        ## Old four panel approach
        # self.grid_widget = QWidget(central_widget)
        # self.grid_layout = QGridLayout(self.grid_widget)
        # self.grid_layout.setContentsMargins(6, 6, 6, 6)
        # self.grid_layout.setHorizontalSpacing(6)
        # self.grid_layout.setVerticalSpacing(6)
        # self.grid_layout.setRowStretch(0, 1)
        # self.grid_layout.setRowStretch(1, 1)
        # self.grid_layout.setColumnStretch(0, 1)
        # self.grid_layout.setColumnStretch(1, 1)

        # self.lbl_input_title = QLabel("Input")
        # #self.lbl_input = ImageLabel()
        # self.lbl_grid_title = QLabel("Tracked Points")
        # #self.lbl_grid = ImageLabel()
        # self.lbl_warped_title = QLabel("Corrected Video")
        # #self.lbl_warped = ImageLabel()
        # self.lbl_tracks_title = QLabel("Deformation Field")
        # #self.lbl_tracks = ImageLabel()

        # self.lbl_input = PGImagePanel()
        # self.lbl_grid  = PGImagePanel()
        # self.lbl_warped= PGImagePanel()
        # self.lbl_tracks= PGImagePanel()
        
        # # Set parent GUI reference so panels can check Box Fill mode
        # for panel in [self.lbl_input, self.lbl_grid, self.lbl_warped, self.lbl_tracks]:
        #     panel._parent_gui = self

        # panels = [
        #     (self.lbl_input_title, self.lbl_input),
        #     (self.lbl_grid_title, self.lbl_grid),
        #     (self.lbl_warped_title, self.lbl_warped),
        #     (self.lbl_tracks_title, self.lbl_tracks),
        # ]
        # for i, (title, label) in enumerate(panels):
        #     r, col = divmod(i, 2)
        #     vbox = QVBoxLayout()
        #     vbox.addWidget(title)
        #     vbox.addWidget(label, stretch=1)  # Give the panel all available space
        #     self.grid_layout.addLayout(vbox, r, col)

        # main_layout.addWidget(self.grid_widget, stretch=1)

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
        xs = np.linspace(-10, W + 10, cols, dtype=np.float32) #0, H-1
        ys = np.linspace(-10, H + 10, rows, dtype=np.float32) #10, H - 10
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
    def clear_all_data(self, keep_video=False):
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
        
        self.lbl_grid.set_image(None)
        self.lbl_warped.set_image(None)
        self.lbl_tracks.set_image(None)

        if hasattr(self, 'motion_plot'):
             self.motion_plot.set_current_frame(0)
             self.motion_plot.set_deformation_data([]) # Reset deformation line
             # If we rotate, we might want to keep motion scores? 
             # Or re-calc. If we keep video, we might want to keep motion profile?
             # But if we don't recalculate, it's fine.
             if not keep_video:
                 self.motion_plot.set_data([])

        if not keep_video:
            self.lbl_input.set_image(None)
            self.slider.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.step_fwd_btn.setEnabled(False)
            self.step_back_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.edit_points_btn.setEnabled(False)
            self.clear_points_btn.setEnabled(False)
            self.frame_label.setText("0/0")

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
        if hasattr(self, "expand_preset_btn"):
            self.expand_preset_btn.setEnabled(False)

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
            if self.rigid_motion_chk.isChecked():
                # Rigid motion mode
                ref_idx = getattr(self, "selected_track_idx", None)
                if ref_idx is None:
                    self.status_label.setText("Please select a track first (click a point while Rigid motion is checked).")
                    return
                
                self._apply_rigid_motion_from_track(ref_idx)
                return

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
                self.status_label.setText("Heuristic or Algorithm required for expand.")
                return

        except Exception as e:
            self.status_label.setText(f"Expand failed: {e}")
            return

        added = self._add_points_unique(new_pts, min_dist_px=min_dist)
        self._update_points_label()
        self.update_grid_display_pre_registration()
        self.status_label.setText(f"Expand Preset: {self.expand_mode_combo.currentText()} -> +{added} points")
        
        # Automatically trigger tracking and warping for the new set
        if added > 0:
            QTimer.singleShot(100, self.visualize)


    def _apply_rigid_motion_from_track(self, tidx):
        if self.video_data is None or self.pred_tracks is None:
            return

        tracks_np = np.asarray(self.pred_tracks)
        if tracks_np.ndim == 4:
            tracks_np = tracks_np[0]
            
        if tidx >= tracks_np.shape[1]:
            return

        self.status_label.setText(f"Applying rigid motion from track {tidx} to all points...")
        QApplication.processEvents()

        # 1. Get reference motion
        ref_traj = tracks_np[:, tidx] # (T, 2)
        delta = ref_traj - ref_traj[0] # (T, 2) relative displacement

        # 2. Get existing initial points (BEFORE we arguably add a grid)
        # We want to keep everything that was there.
        # Initial positions for current tracks are tracks_np[0]
        pts0_existing = tracks_np[0].copy()
        
        # 3. Generate a regular grid (as requested earlier to "apply it to a grid of points")
        gray = self._get_gray_video_np()
        H, W = gray.shape[1], gray.shape[2]
        pts0_grid = self._preset_regular_grid(H, W, self.grid_rows, self.grid_cols)
        
        # 4. Merge old and new grid (avoiding exact duplicates at start frame)
        # We can use our _add_points_unique logic if we want, but let's just cat them for now 
        # as the user said "dont remove previous".
        final_pts0 = np.concatenate([pts0_existing, pts0_grid], axis=0)
        
        T = tracks_np.shape[0]
        N_total = final_pts0.shape[0]

        # 5. Create synthetic tracks for EVERYTHING
        synthetic = final_pts0[None, :, :] + delta[:, None, :] # (T, N_total, 2)
        
        # 6. Update state
        self.pred_tracks = synthetic.astype(np.float32)
        self.pred_visibility = np.ones((T, N_total), dtype=np.float32)
        self.pred_confidence = np.ones((T, N_total), dtype=np.float32)
        
        # Update point sets to match the new tracks
        # We split them back into base and custom? 
        # Actually, simpler to just keep the first N_old points as base/custom as they were
        # and add the grid to base.
        
        # To keep it consistent: update base_grid_pts and add to it
        N_old = pts0_existing.shape[0]
        self.base_grid_pts = final_pts0.astype(np.float32)
        self.base_grid_pts0 = final_pts0.copy()
        self.base_grid_active = np.ones((N_total,), dtype=bool)
        # custom_grid_pts: if they were part of pts0_existing, they are now "base" effectively?
        # That's fine for rigid motion fix.
        self.custom_grid_pts = [] 
        
        self.viz_data['tracks'] = self.pred_tracks
        self.selected_track_idx = None
        
        self.status_label.setText(f"Applied rigid motion (track {tidx}) to {N_total} points (held previous).")
        self._update_points_label()
        self.update_plot()
        self._schedule_recalc_warping(delay_ms=0)

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
                self.video_data = load_video(self.video_path, resize=False, max_frames=30)
            elif ext in ['.tiff', '.tif']:
                video_np = load_tiff(self.video_path, frames=30) # Load all (up to sensible limit) or modify util
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            elif ext in ['.h5', '.hdf5']:
                video_np = load_hdf5(self.video_path, "/data", frames=30)
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
             
             # Difference between t and t+1
             if T > 1:
                 # diffs[t] = frame[t+1] - frame[t]
                 diffs = flat_norm[1:] - flat_norm[:-1]
                 
                 # L2 norm per frame
                 # Norm(u - v) where |u|=|v|=Constant roughly
                 l2_norms = np.linalg.norm(diffs, axis=1)
                 
                 # Normalize to 0..1 for visualization
                 # 0 = No motion, 1 = Max motion
                 max_val = l2_norms.max()
                 if max_val > 1e-6:
                     scores = l2_norms / max_val
                 else:
                     scores = np.zeros_like(l2_norms)
                 
                 # Pad to length T
                 # User wants score[0]=0, score[t]=diff(t, t-1)
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
                self._schedule_recalc_warping(delay_ms=2000)

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
            self.status_label.setText("Track modified. Auto-warping in 2s…")
            self._schedule_recalc_warping(delay_ms=2000)

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
        filtered = self.apply_filter(input_frame, self.current_filter)

        # Levels based on filtered image (optional; PGImagePanel can handle None)
        levels = self._calc_levels(filtered)

        left_choice = self.view_left_combo.currentText()
        right_choice = self.view_right_combo.currentText()

        self._render_view(self.view_left, left_choice, idx, filtered, levels)
        self._render_view(self.view_right, right_choice, idx, filtered, levels)

        # cursor in motion plot
        if hasattr(self, "motion_plot"):
            self.motion_plot.set_current_frame(self.current_frame)

    def _render_view(self, panel, which, idx, filtered_frame, levels=None):
        which = (which or "").strip()

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
            vis = self._get_visibility_frame(idx, N)

            mask = np.ones((N,), dtype=bool)

            if self.track_active is not None and len(self.track_active) == N:
                mask &= self.track_active

            vthr = float(self.conf_thresh_spin.value()) if hasattr(self, "conf_thresh_spin") else 0.0
            if vis is not None:
                mask &= (vis >= vthr)

            pts_show = ptsA[mask]

            # optional coloring by visibility
            colors = None
            if vis is not None:
                v_show = vis[mask]
                a = np.clip(v_show * 255.0, 30, 255).astype(np.uint8)
                g = np.clip(v_show * 255.0, 0, 255).astype(np.uint8)
                r = np.zeros_like(a, dtype=np.uint8)
                b = np.zeros_like(a, dtype=np.uint8)
                colors = np.stack([r, g, b, a], axis=1)

            panel.set_points_colored(pts_show, colors_rgba=colors, size=8)

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

            return

        # ---------- DEFORMATION FIELD ----------
        if which == "Deformation Field":
            if self.disp_field is not None:
                disp_frame = self.disp_field[idx] if idx < len(self.disp_field) else self.disp_field[0]
                disp_preview = self._make_disp_preview(disp_frame)
                panel.set_image(disp_preview, auto_range=False, levels=self._calc_levels(disp_preview))
            else:
                panel.set_image(None)


            return

        # ---------- Corrected (ANTs) ----------
        if which == "Corrected (ANTs)":
            if self.ants_video is not None:
                panel.set_image(self.ants_video[idx], auto_range=False, levels=self._calc_levels(self.ants_video[idx]))
            else:
                panel.set_image(None)

            return

        # ---------- Corrected (LDDMM) ----------
        if which == "Corrected (LDDMM)":
            if self.lddmm_video is not None:
                panel.set_image(self.lddmm_video[idx], auto_range=False, levels=self._calc_levels(self.lddmm_video[idx]))
            else:
                panel.set_image(None)

            return

        # fallback
        panel.set_image(filtered_frame, auto_range=False, levels=levels)
        panel.set_points(pts_xy=None, crosses_xy=None)
        panel.set_trajectories(None, 0)
        panel.points_item.setData([], [])
        panel.cross_item.setData([], [])
        if hasattr(panel, "pending_item"):
            panel.pending_item.setData([], [])

    
    def visualize(self, method="CoTracker"):
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

        # UI lock
        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.rotate_btn.setEnabled(False)

        video_input = self.video_data

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

        # --- refresh ---
        try:
            self._apply_confidence_filter()
        except Exception:
            self.update_plot()



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

        # If no visibility, just keep all tracks active
        if self.pred_visibility is None:
            tr = np.asarray(self.pred_tracks)
            if tr.ndim == 4:
                tr = tr[0]
            N = tr.shape[1]
            if self.track_active is None or len(self.track_active) != N:
                self.track_active = np.ones((N,), dtype=bool)
            self.update_plot()
            return

        if not hasattr(self, "auto_conf_chk") or not self.auto_conf_chk.isChecked():
            self.update_plot()
            return

        thr = float(self.conf_thresh_spin.value()) if hasattr(self, "conf_thresh_spin") else 0.5

        vis = self.pred_visibility
        if isinstance(vis, torch.Tensor):
            vis = vis.detach().cpu().numpy()
        vis = np.asarray(vis)

        # normalize possible shapes
        if vis.ndim == 3 and vis.shape[0] == 1:
            vis = vis[0]
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]

        if vis.ndim != 2:
            # unexpected vis shape -> keep all active
            tr = np.asarray(self.pred_tracks)
            if tr.ndim == 4:
                tr = tr[0]
            self.track_active = np.ones((tr.shape[1],), dtype=bool)
            self.update_plot()
            return

        conf = np.nanmean(vis, axis=0)
        conf = np.where(np.isfinite(conf), conf, 0.0)

        self.track_active = conf >= thr
        self.update_plot()

        # optional: update warping immediately
        self._schedule_recalc_warping(delay_ms=0)

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

    def _delete_track_for_point(self, kind: str, idx: int):
        if self.pred_tracks is None or not self.track_meta:
            return
        for t, meta in enumerate(self.track_meta):
            if meta == (kind, int(idx)):
                self._delete_track_idx(t)
                # keep meta consistent with track deletion
                self.track_meta.pop(t)
                return



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
                self.ssim_score = video_ssim_score(self.warped_video, mode="consecutive")
                self.status_label.setText(f"Warping updated | SSIM={self.ssim_score:.4f}")
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

        # Apply auto confidence filter if enabled (this calls update_plot and may trigger recalc)
        try:
            self._apply_confidence_filter()
        except Exception:
            pass

        # ---------------------------------------
        # 9) UI unlock + worker cleanup
        # ---------------------------------------
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.rotate_btn.setEnabled(True)
        self.expand_preset_btn.setEnabled(True)
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


    def rotate_input(self, checked=False):
        if self.video_data is None:
            return

        try:
            # video_data: (T, C, H, W)
            # Rotate 90 degrees clockwise? or counter-clockwise?
            # torch.rot90(input, k, dims)
            # k=1 -> 90 deg counter-clockwise
            # k=-1 -> 90 deg clockwise
            
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
            
            # Need to refresh the preset points for new dimensions?
            self.apply_point_preset()

            self.status_label.setText("Input video rotated 90° clockwise")
            
            # Re-calculate motion profile?
            # The motion profile (global correlation) should be roughly invariant to rotation 
            # if we just rotate the whole image?
            # Actually, we resized to 64x64. If aspect ratio flips, resizing might change slightly.
            # Ideally we recalculate to be safe and consistent.
            # But let's just re-load data logic? No, load_data reloads from disk.
            # We can extract the motion calc logic to a helper or just re-run it.
            # For simplicity, let's leave motion profile as is for now, or clear it?
            # It's better to clear it to avoid confusion if aspect ratio change affects it.
            # But re-calc is better.
            
            # Let's just update the view
            self.update_plot()
            self.reset_all_zoom()

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
