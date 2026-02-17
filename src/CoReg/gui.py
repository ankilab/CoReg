import sys
import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QSpinBox,
    QSizePolicy,
    QDialog,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPointF
from PyQt6.QtGui import QPainter, QColor, QPolygonF, QPen
import pyqtgraph as pg

# Local imports
from utils.mc_cotracker import cotrack_model, warping

# Configure pyqtgraph to use row-major ordering (matching numpy)
pg.setConfigOptions(imageAxisOrder="row-major")

try:
    import ants
except ImportError:
    ants = None

try:
    import utils.torch_lddmm as torch_lddmm
except ImportError:
    torch_lddmm = None

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class ProcessingWorker(QThread):
    """
    Background worker that handles computationally intensive video registration.
    Supports three methods:
      - ANTs: Robust, intensity-based registration (requires antspyx).
      - LDDMM: Diffeomorphic registration (uses local torch implementation).
      - CoTracker: Point-based tracking combined with sparse-to-dense warping.
    """

    # Signals for UI updates
    tracks_ready = pyqtSignal(
        object, object, object, object
    )  # (tracks_np, visibility, confidence, grid)
    progress = pyqtSignal(int)  # Percent completion (0-100)
    status_update = pyqtSignal(str)  # Human-readable status message

    # final results: (tracks_np, visibility, confidence, grid, warped_full, disp_field)
    finished = pyqtSignal(object, object, object, object, object, object)
    error = pyqtSignal(str)  # Error message on failure

    def __init__(
        self,
        video_in,
        video_out=None,
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

        self.video_in = video_in
        self.video_out = video_out if video_out is not None else video_in
        self.base_grid_pts = base_grid_pts
        self.custom_grid_pts = custom_grid_pts
        self.df_sigma = float(df_sigma)
        self.tracks = tracks

        self.method = method
        self.filter_name = filter_name
        # Stored for future use if merging is needed in the background
        self.existing_data = existing_data
        self.queries = queries  # torch.Tensor of shape (1, N, 2) or None

        self._is_running = True

    def stop(self):
        """Safely terminate the background thread."""
        self._is_running = False

    def run(self):
        """Execute the selected registration method and emit results."""
        tracks_gui = None
        vis_gui = None
        conf_gui = None
        grid_xy_ret = None

        try:
            # ---------- ANTs REGISTRATION ----------
            # Uses symmetric normalization (SyN) to register frames to a reference (frame 0).
            if self.method == "ANTs":
                if ants is None:
                    raise ImportError(
                        "ANTs is not installed. Please install it with 'pip install antspyx'"
                    )

                self.status_update.emit("Running ANTs registration...")
                # Registration sources
                vd_reg = self.video_in
                if isinstance(vd_reg, torch.Tensor):
                    vd_reg = vd_reg.detach().cpu().numpy()
                if vd_reg.ndim == 4 and vd_reg.shape[1] in [1, 3]:
                    vd_reg = np.transpose(vd_reg, (0, 2, 3, 1))

                # Output sources (the data we actually warp)
                vd_out = self.video_out
                if isinstance(vd_out, torch.Tensor):
                    vd_out = vd_out.detach().cpu().numpy()
                if vd_out.ndim == 4 and vd_out.shape[1] in [1, 3]:
                    vd_out = np.transpose(vd_out, (0, 2, 3, 1))

                t = vd_reg.shape[0]
                results = []

                # Ensure ref_frame is 2D grayscale for ANTs
                ref_frame = vd_reg[0]
                if ref_frame.ndim == 3:
                    if ref_frame.shape[-1] == 3:
                        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
                    else:
                        ref_frame = ref_frame[:, :, 0]

                fixed_image = ants.from_numpy(ref_frame)

                for i in range(t):
                    if not self._is_running:
                        return
                    self.progress.emit(int(i / max(1, t) * 100))

                    if i == 0:
                        results.append(vd_out[0])
                        continue

                    moving_frame = vd_reg[i]
                    if moving_frame.ndim == 3:
                        if moving_frame.shape[-1] == 3:
                            moving_frame = cv2.cvtColor(
                                moving_frame, cv2.COLOR_RGB2GRAY
                            )
                        else:
                            moving_frame = moving_frame[:, :, 0]

                    moving_image = ants.from_numpy(moving_frame)

                    reg = ants.registration(
                        fixed=fixed_image,
                        moving=moving_image,
                        type_of_transform="SyNOnly",
                    )

                    # Apply transform to vd_out[i]
                    to_warp = vd_out[i]
                    if to_warp.ndim == 3 and to_warp.shape[-1] == 1:
                        # ANTs requires 2D input for 2D images
                        to_warp = to_warp[:, :, 0]

                    warped_image = ants.apply_transforms(
                        fixed=fixed_image,
                        moving=ants.from_numpy(to_warp),
                        transformlist=reg["fwdtransforms"],
                    )
                    warped = warped_image.numpy()

                    if vd_out[i].ndim == 3 and vd_out[i].shape[-1] == 1:
                        warped = warped[..., None]

                    results.append(warped)

                warped_np = np.stack(results, axis=0)
                self.progress.emit(100)
                self.status_update.emit("Done")
                self.finished.emit(None, None, None, None, warped_np, None)
                return

            # ---------- LDDMM REGISTRATION ----------
            # Large Deformation Diffeomorphic Metric Mapping. Uses a local torch-based implementation.
            if self.method == "LDDMM":
                if torch_lddmm is None:
                    raise ImportError(
                        "torch_lddmm is not available. Please ensure utils/torch_lddmm.py exists."
                    )

                self.status_update.emit("Running LDDMM registration...")
                # Registration sources
                vd_reg = self.video_in
                if isinstance(vd_reg, torch.Tensor):
                    vd_reg = vd_reg.detach().cpu().numpy()
                if vd_reg.ndim == 4 and vd_reg.shape[1] in [1, 3]:
                    vd_reg = np.transpose(vd_reg, (0, 2, 3, 1))

                # Output sources
                vd_out = self.video_out
                if isinstance(vd_out, torch.Tensor):
                    vd_out = vd_out.detach().cpu().numpy()
                if vd_out.ndim == 4 and vd_out.shape[1] in [1, 3]:
                    vd_out = np.transpose(vd_out, (0, 2, 3, 1))

                T = vd_reg.shape[0]
                results = []

                ref_frame = vd_reg[0]
                if ref_frame.ndim == 3:
                    if ref_frame.shape[-1] == 3:
                        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
                    else:
                        ref_frame = ref_frame[:, :, 0]

                for i in range(T):
                    if not self._is_running:
                        return
                    self.progress.emit(int(i / max(1, T) * 100))

                    if i == 0:
                        results.append(vd_out[0])
                        continue

                    moving_frame = vd_reg[i]
                    if moving_frame.ndim == 3:
                        if moving_frame.shape[-1] == 3:
                            moving_frame = cv2.cvtColor(
                                moving_frame, cv2.COLOR_RGB2GRAY
                            )
                        else:
                            moving_frame = moving_frame[:, :, 0]

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

                    # Apply displacement to vd_out[i]
                    disp = lddmm.outputDisplacement()
                    to_warp = vd_out[i]
                    if to_warp.ndim == 3 and to_warp.shape[-1] == 1:
                        # torch_lddmm.applyDisplacement usually wants (H,W) or (C,H,W)
                        to_warp = to_warp[:, :, 0]

                    warped = lddmm.applyDisplacement(to_warp, disp)

                    if vd_out[i].ndim == 3 and vd_out[i].shape[-1] == 1:
                        warped = warped[..., None]

                    results.append(warped)

                warped_np = np.stack(results, axis=0)
                self.progress.emit(100)
                self.status_update.emit("Done")
                self.finished.emit(None, None, None, None, warped_np, None)
                return

            # ---------- CoTracker + Warping ----------
            # 1. Track points using Facebook's CoTracker.
            # 2. Interpolate sparse point movements to a dense displacement field.
            # 3. Warp frames using the dense field.

            # Normalize inputs to device
            video_in = (
                self.video_in.to(DEFAULT_DEVICE)
                if isinstance(self.video_in, torch.Tensor)
                else self.video_in
            )
            video_out = (
                self.video_out.to(DEFAULT_DEVICE)
                if isinstance(self.video_out, torch.Tensor)
                else self.video_out
            )

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
                        raise ValueError("No new query points provided for tracking.")

                if isinstance(queries, np.ndarray):
                    queries = torch.from_numpy(queries)
                if queries.ndim == 2:
                    queries = queries.unsqueeze(0)
                queries = queries.to(DEFAULT_DEVICE).float()

                if DEFAULT_DEVICE == "cuda":
                    self.status_update.emit("Tracking running on GPU")
                else:
                    self.status_update.emit(
                        f"Tracking running on {DEFAULT_DEVICE}, you should switch to GPU for better performance if possible."
                    )

                self.progress.emit(0)

                if queries is not None:
                    if isinstance(queries, np.ndarray):
                        queries = torch.from_numpy(queries)

                    if queries.ndim == 2:  # (N, 2) -> (1, N, 2)
                        queries = queries.unsqueeze(0)

                    if queries.shape[-1] == 2:
                        b_size, n_pts, _ = queries.shape
                        t0 = torch.zeros(
                            (b_size, n_pts, 1),
                            device=DEFAULT_DEVICE,
                            dtype=queries.dtype,
                        )
                        queries = torch.cat([t0, queries], dim=-1)

                if isinstance(video_in, torch.Tensor) and video_in.shape[1] == 1:
                    video_in = video_in.repeat(1, 3, 1, 1)

                # Progress mapping: CoTracker reports stages (start,fmaps,iters,done) and dense_offsets
                # Add finer-grained status text using meta like chunk/num_chunks and it/iters
                def tracker_cb(p: float, stage: str, **meta):
                    if not self._is_running:
                        return
                    # Map model progress [0..1] to 0..70 in GUI
                    if stage == "dense_offsets":
                        total = int(meta.get("total", 1))
                        offset = int(meta.get("offset", 0))
                        frac = (offset + 1) / max(1, total)
                        val = int(min(70, max(0, round(frac * 70))))
                        self.progress.emit(val)
                        self.status_update.emit(
                            f"Tracking dense grid ({offset+1}/{total})..."
                        )
                        return

                    stage_text = {
                        "start": "Preparing tracking...",
                        "fmaps": "Computing features...",
                        "iters": "Refining tracks...",
                        "done": "Tracking done",
                    }.get(stage, f"Tracking: {stage}")

                    val = int(min(70, max(0, round(float(p) * 70))))
                    self.progress.emit(val)

                    # Enhance status with granular counters when available
                    if stage == "fmaps":
                        chunk = int(meta.get("chunk", 0))
                        num_chunks = int(meta.get("num_chunks", 0))
                        if num_chunks > 0:
                            stage_text = (
                                f"Computing features (chunk {chunk}/{num_chunks})..."
                            )
                    elif stage == "iters":
                        it = int(meta.get("it", 0))
                        iters = int(meta.get("iters", 0))
                        if iters > 0:
                            stage_text = f"Refining tracks (iter {it}/{iters})..."

                    self.status_update.emit(stage_text)

                pred_tracks, pred_visibility = cotrack_model(
                    video_in,
                    queries=queries,
                    device=DEFAULT_DEVICE,
                    progress_cb=tracker_cb,
                )

                tracks_raw = pred_tracks.detach().cpu().numpy()
                tracks_gui = tracks_raw[0] if tracks_raw.ndim == 4 else tracks_raw
                tracks_for_warp = (
                    tracks_raw if tracks_raw.ndim == 4 else tracks_raw[None, ...]
                )

                vis_raw = (
                    pred_visibility.detach().cpu().numpy()
                    if isinstance(pred_visibility, torch.Tensor)
                    else np.asarray(pred_visibility)
                )
                vis_gui = vis_raw[0] if vis_raw.ndim == 3 else vis_raw
                conf_gui = None

                # ----- NEW: merge with existing tracks if provided -----
                if self.existing_data is not None:
                    old_tr, old_vis, _ = self.existing_data

                    old_tr = np.asarray(old_tr)  # (T, N_old, 2)
                    if old_tr.ndim == 4:
                        old_tr = old_tr[0]
                    new_tr = tracks_gui  # (T, N_new, 2)

                    Tm = min(old_tr.shape[0], new_tr.shape[0])
                    old_tr = old_tr[:Tm]
                    new_tr = new_tr[:Tm]
                    tracks_gui = np.concatenate([old_tr, new_tr], axis=1)

                    # visibility merge (optional)
                    if vis_gui is not None and old_vis is not None:
                        ov = np.asarray(old_vis)
                        if ov.ndim == 3:
                            ov = ov[0]
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
                tracks_gui = (
                    self.tracks.detach().cpu().numpy()
                    if isinstance(self.tracks, torch.Tensor)
                    else np.asarray(self.tracks)
                )
                tracks_for_warp = (
                    tracks_gui if tracks_gui.ndim == 4 else tracks_gui[None, ...]
                )

            self.progress.emit(70)
            self.status_update.emit("Warping / Displacement optimization ...")

            if not isinstance(tracks_for_warp, torch.Tensor):
                tracks_for_warp = torch.from_numpy(tracks_for_warp)
            tracks_for_warp = tracks_for_warp.to(DEFAULT_DEVICE).float()

            def warp_cb(frac):
                if not self._is_running:
                    return
                self.progress.emit(int(70 + frac * 30))

            if not self._is_running:
                return

            warped_video, disp_field = warping(
                tracks_for_warp, video_out, smoothing=self.df_sigma, callback=warp_cb
            )

            if not self._is_running:
                return

            warped_np = np.asarray(warped_video)
            if warped_np.ndim >= 4 and warped_np.shape[0] == 1:
                warped_np = warped_np[0]

            disp_arr = (
                disp_field.detach().cpu().numpy()
                if isinstance(disp_field, torch.Tensor)
                else np.asarray(disp_field)
            )
            if disp_arr.ndim >= 4 and disp_arr.shape[0] == 1:
                disp_arr = disp_arr[0]

            self.progress.emit(100)
            self.status_update.emit("Done")
            self.finished.emit(
                tracks_gui, vis_gui, conf_gui, grid_xy_ret, warped_np, disp_arr
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(str(e))


class PGImagePanel(pg.GraphicsLayoutWidget):
    """
    A scientific image viewer based on pyqtgraph.
    Provides:
      - Interactive contrast adjustment (via HistogramLUTItem).
      - Coordinate mapping between screen, scene, and image units.
      - Overlay support for scatter points and motion trajectories.
      - Custom mouse handlers for point editing (add/delete/drag).
    """

    pressed = pyqtSignal(float, float, int)  # (x, y, button_id) - image coords
    moved = pyqtSignal(float, float)  # (x, y) - image coords
    released = pyqtSignal()
    clicked_outside_image = pyqtSignal()

    def __init__(self, parent=None, show_histogram=False):
        """
        Initialize the image panel.
        :param parent: Parent QWidget.
        :param show_histogram: Whether to include a HistogramLUTItem for contrast control.
        """
        super().__init__(parent)
        self.setMinimumSize(50, 50)
        self.setBackground(None)

        # Set size policy to expand and fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.show_histogram = show_histogram

        # Create PlotItem inside the layout
        self.p1 = self.addPlot()
        self.p1.setMenuEnabled(False)
        self.p1.hideAxis("left")
        self.p1.hideAxis("bottom")

        self.vb = self.p1.getViewBox()
        self.vb.setDefaultPadding(0.0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMouseEnabled(x=True, y=True)

        self._img_w = None
        self._img_h = None
        self._parent_gui = None
        self._custom_drag_active = False

        self.img_item = pg.ImageItem()
        self.p1.addItem(self.img_item)

        # Histograms
        self.hist = None
        if self.show_histogram:
            self.hist = pg.HistogramLUTItem()
            self.hist.setImageItem(self.img_item)
            self.addItem(self.hist)

        self.pending_item = pg.ScatterPlotItem(
            size=10, pen=None, brush=pg.mkBrush(57, 255, 20), symbol="o"
        )
        self.p1.addItem(self.pending_item)

        # keep full-resolution rendering (no auto decimation)
        try:
            self.img_item.setAutoDownsample(False)
        except Exception:
            self.img_item.setOpts(autoDownsample=False)

        self.points_item = pg.ScatterPlotItem(
            size=8, pen=None, brush=pg.mkBrush(57, 255, 20, 200)
        )
        self.cross_item = pg.ScatterPlotItem(
            size=10, pen=None, brush=pg.mkBrush(57, 255, 20), symbol="o"
        )
        self.traj_item = pg.PlotDataItem(pen=pg.mkPen(57, 255, 20, 180, width=2))

        self.p1.addItem(self.points_item)
        self.p1.addItem(self.cross_item)
        self.max_trail = 30
        self.traj_items = []
        for _ in range(self.max_trail):
            it = pg.PlotDataItem()
            self.traj_items.append(it)
            self.p1.addItem(it)

        self.img_item.setZValue(0)
        self.traj_item.setZValue(5)
        for it in self.traj_items:
            it.setZValue(5)

        self.points_item.setZValue(10)
        self.pending_item.setZValue(11)
        self.cross_item.setZValue(12)

        self.last_mouse_pos = None
        self._possible_click = False

    def fit_view_to_image_and_points(self, pts_xy=None, margin=20):
        """Expand view to show full image plus any points outside image bounds."""
        if self._img_w is None or self._img_h is None:
            return

        x0, x1 = 0.0, float(self._img_w - 1)
        y0, y1 = 0.0, float(self._img_h - 1)

        if pts_xy is not None and len(pts_xy) > 0:
            p = np.asarray(pts_xy, dtype=float)
            if np.isfinite(p).any():
                x0 = min(x0, float(np.nanmin(p[:, 0])))
                x1 = max(x1, float(np.nanmax(p[:, 0])))
                y0 = min(y0, float(np.nanmin(p[:, 1])))
                y1 = max(y1, float(np.nanmax(p[:, 1])))

        self.vb.setRange(
            xRange=(x0 - margin, x1 + margin),
            yRange=(y0 - margin, y1 + margin),
            padding=0.0,
        )

    def set_parent_gui(self, gui):
        """Set the controller/gui reference without exposing protected member."""
        self._parent_gui = gui

    def _trail_color(self, u):
        """
        Calculate a color for trajectory segments based on their age.
        :param u: Normalized time value [0, 1].
        :return: (r, g, b) tuple.
        """
        # u in [0..1]
        yellow = (255, 255, 0)
        purple = (160, 0, 255)
        r = int(round(yellow[0] + (purple[0] - yellow[0]) * u))
        g = int(round(yellow[1] + (purple[1] - yellow[1]) * u))
        b = int(round(yellow[2] + (purple[2] - yellow[2]) * u))
        return r, g, b

    def set_histogram_visible(self, visible):
        """Show or hide the HistogramLUTItem if it exists."""
        if self.hist:
            if visible:
                self.hist.show()
            else:
                self.hist.hide()

    def set_image(self, img_np, auto_range=True, levels=None):
        """
        Update the displayed image.
        :param img_np: Image data as a numpy array.
        :param auto_range: Whether to reset the ViewBox zoom.
        :param levels: Optional (min, max) display levels.
        """
        self._current_img = img_np
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
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[:, :, 0]

        # ---- make display-safe for pyqtgraph ----
        if a.dtype.kind == "f":
            # grayscale float
            if a.ndim == 2:
                # If explicit levels provided, use them and update histogram
                if levels is not None:
                    lo, hi = levels
                    self.img_item.setImage(a, levels=(lo, hi), autoLevels=False)
                    if self.hist:
                        self.hist.setLevels(lo, hi)
                elif self.hist and self.hist.isVisible():
                    # If histogram is visible, use ITS current levels
                    lo, hi = self.hist.getLevels()
                    self.img_item.setImage(a, levels=(lo, hi), autoLevels=False)
                else:
                    # Fallback to percentile calculation
                    finite = np.isfinite(a)
                    if finite.any():
                        lo = float(np.nanpercentile(a[finite], 1))
                        hi = float(np.nanpercentile(a[finite], 99))
                        if hi <= lo:
                            hi = lo + 1.0
                        self.img_item.setImage(a, levels=(lo, hi), autoLevels=False)
                        if self.hist:
                            self.hist.setLevels(lo, hi)
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
                if self.hist and levels is not None:
                    self.hist.setLevels(*levels)
        else:
            if a.dtype == np.bool_:
                a = a.astype(np.uint8) * 255
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
        scene_pos = self.mapToScene(pos.toPoint() if hasattr(pos, "toPoint") else pos)
        view_pos = self.vb.mapSceneToView(scene_pos)

        return float(view_pos.x()), float(view_pos.y())

    def set_points(self, pts_xy=None, crosses_xy=None):
        """
        Update overlay points.
        - pts_xy: Base grid or tracked points (drawn as circles).
        - crosses_xy: Custom/manual points (drawn as crosses).
        """
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
        """
        Draw point motion trajectories for the specified trail length.
        :param tracks_TN2: (T, N, 2) array of point tracks.
        :param current_frame: The frame index to render.
        :param trail: Number of previous frames to include in the trajectory.
        """
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
                p2 = tr[t, i]  # current frame
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
        """Convert graphics scene coordinates to image (view) coordinates."""
        if scene_pos is None:
            return None
        p = self.vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _qt_btn_to_int(self, btn):
        """Map Qt MouseButton enum to integer (1:Left, 2:Middle, 3:Right)."""
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
        pts_x, pts_y = self.points_item.getData()
        if len(pts_x) > 0:
            p0 = self.vb.mapSceneToView(pg.Point(0, 0))
            p1 = self.vb.mapSceneToView(pg.Point(10, 0))  # 10 screen pixels
            tol = abs(p1.x() - p0.x())  # Tolerance in image units

            d2 = (pts_x - x) ** 2 + (pts_y - y) ** 2
            if np.min(d2) < (tol * tol):
                return True

        # Check crosses
        c_x, c_y = self.cross_item.getData()
        if len(c_x) > 0:
            p0 = self.vb.mapSceneToView(pg.Point(0, 0))
            p1 = self.vb.mapSceneToView(pg.Point(10, 0))
            tol = abs(p1.x() - p0.x())

            d2 = (c_x - x) ** 2 + (c_y - y) ** 2
            if np.min(d2) < (tol * tol):
                return True

        return False

    def _img_shape_hw(self):
        """Return the current image dimensions as (H, W)."""
        img = getattr(self, "_current_img", None)
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim == 2:
            return int(a.shape[0]), int(a.shape[1])
        return int(a.shape[0]), int(a.shape[1])  # (H,W,C)

    def mousePressEvent(self, ev):
        """Custom mouse press handler for point editing and viewport panning."""
        btn = ev.button()

        # Map click to image coords (view coords)
        pos = self._map_scene_to_image(ev.position())
        if pos is None:
            super().mousePressEvent(ev)
            return

        x, y = pos

        # outside-image detection (for exiting edit modes)
        hw = self._img_shape_hw()
        if hw is None:
            # no image -> treat as outside
            self.clicked_outside_image.emit()
            super().mousePressEvent(ev)
            return

        H, W = hw
        outside = (x < 0) or (y < 0) or (x >= W) or (y >= H)
        if outside:
            self.clicked_outside_image.emit()
            # IMPORTANT: still pass to super so pan/zoom works in the black border too
            super().mousePressEvent(ev)
            return

        # ----- INSIDE IMAGE -----
        if btn == Qt.MouseButton.LeftButton:
            # INTERACTION LOGIC:
            # 1. If clicking directly over a point, we immediately start an "edit drag".
            #    This consumes the event to prevent the ViewBox from panning the image.
            if self._is_over_point(ev.position()):
                self.pressed.emit(float(x), float(y), 1)
                ev.accept()
                return

            # 2. If clicking on empty space, we WAIT to see if it's a drag or a click.
            #    If the mouse moves significantly (>5px) before release, it's a pan.
            #    If it stays still, it's a point-addition click (emitted in mouseReleaseEvent).
            self._possible_click = True
            self.last_mouse_pos = ev.position()
            # DO NOT emit pressed here (otherwise you "add" immediately and kill pan)

        elif btn == Qt.MouseButton.RightButton:
            # right click should trigger delete/toggle immediately, but still no pan
            self.pressed.emit(float(x), float(y), 3)
            ev.accept()
            return

        # Let pyqtgraph handle pan/zoom drag
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Custom mouse move handler for point dragging and hover previews."""
        # detect if user started dragging -> then it is NOT a click
        if self._possible_click and self.last_mouse_pos is not None:
            dist = (ev.position() - self.last_mouse_pos).manhattanLength()
            if dist > 5:
                self._possible_click = False

        pos = self._map_scene_to_image(ev.position())
        if pos:
            self.moved.emit(float(pos[0]), float(pos[1]))

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        """Custom mouse release handler to finalize point additions."""
        # if it was a clean left-click on empty space (no drag), emit pressed once here
        if ev.button() == Qt.MouseButton.LeftButton and self._possible_click:
            pos = self._map_scene_to_image(ev.position())
            if pos:
                x, y = pos
                hw = self._img_shape_hw()
                if hw is not None:
                    H, W = hw
                    if not ((x < 0) or (y < 0) or (x >= W) or (y >= H)):
                        self.pressed.emit(float(x), float(y), 1)

        self._possible_click = False
        self.last_mouse_pos = None

        self.released.emit()
        super().mouseReleaseEvent(ev)

    def reset_view(self):
        """Reset the ViewBox to show the full image."""
        if getattr(self, "native_1to1", False):
            self._apply_native_1to1(keep_center=False)
        else:
            self.vb.autoRange()

    def set_points_colored(self, pts_xy, colors_rgba=None, size=8):
        """Update tracked points with frame-specific colors (e.g., for visibility)."""
        if pts_xy is None or len(pts_xy) == 0:
            self.points_item.setData([], [])
            return

        p = np.asarray(pts_xy, dtype=float)
        N = p.shape[0]

        if colors_rgba is None:
            # Default highlight color if no specific colors provided
            spots = [
                {
                    "pos": (p[i, 0], p[i, 1]),
                    "size": size,
                    "brush": pg.mkBrush(0, 255, 0, 255),
                }
                for i in range(N)
            ]
        else:
            c = np.asarray(colors_rgba, dtype=np.uint8)
            spots = []
            for i in range(N):
                r, g, b, a = map(int, c[i])
                spots.append(
                    {
                        "pos": (p[i, 0], p[i, 1]),
                        "size": size,
                        "brush": pg.mkBrush(r, g, b, a),
                    }
                )

        self.points_item.setData(spots=spots)


class MotionProfileWidget(QWidget):
    """
    A custom widget that displays temporal motion and deformation scores.
    Rendered as a line plot with a frame-synced cursor.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialize the motion profile widget."""
        super().__init__(parent)
        self.setFixedHeight(50)
        self.motion_scores = []  # Cyan line
        self.deformation_scores = []  # Magenta line
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
        """Update the cursor position to the specified frame."""
        self.current_frame = frame_idx
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press to scrub through frames."""
        self._handle_mouse(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move to scrub through frames."""
        self._handle_mouse(event)

    def _handle_mouse(self, event):
        """Map mouse horizontal position to frame index and emit signal."""
        # Allow interaction if EITHER plot exists
        count_m = len(self.motion_scores) if self.motion_scores is not None else 0
        count_d = (
            len(self.deformation_scores) if self.deformation_scores is not None else 0
        )
        count = max(count_m, count_d)

        if count < 1:
            return

        x = event.position().x()
        w = self.width()

        frac = max(0.0, min(1.0, x / w))
        frame_idx = int(round(frac * (count - 1)))

        self.frame_changed.emit(frame_idx)

    def paintEvent(self, event):
        """Render the motion/deformation scores and frame cursor."""

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
                        if not np.isfinite(v):
                            v = 0.0
                    except ValueError:
                        v = 0.0
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

            # Draw current frame cursor
            count_m = len(self.motion_scores) if self.motion_scores is not None else 0
            count_d = (
                len(self.deformation_scores)
                if self.deformation_scores is not None
                else 0
            )
            count = max(count_m, count_d)

            if count > 0:
                cx = (self.current_frame / max(1, count - 1)) * w
                painter.setPen(QPen(QColor(57, 255, 20), 1))
                painter.drawLine(int(cx), 0, int(cx), h)
        finally:
            painter.end()


class ParameterDialog(QDialog):
    """A scrollable dialog for adjusting algorithm parameters."""

    def __init__(self, parent=None):
        """Initialize the parameter dialog."""
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.setModal(False)

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        # Prefer non-shadowing name to avoid confusion with QWidget.layout()
        self.content_layout = QVBoxLayout(content)
        # Back-compat for existing references
        self.layout = self.content_layout

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.hide)
        outer.addWidget(self.close_btn)


class VideoLoadSettingsDialog(QDialog):
    """
    Shows original video (frames, width, height) and allows user to load original,
    or choose smaller width/height + fewer frames.
    """

    def __init__(
        self, parent, orig_frames: int | None, orig_w: int | None, orig_h: int | None
    ):
        """
        Initialize the video load settings dialog.
        :param parent: Parent QWidget.
        :param orig_frames: Original number of frames.
        :param orig_w: Original width.
        :param orig_h: Original height.
        """
        super().__init__(parent)
        self.setWindowTitle("Load settings")

        self.orig_frames = orig_frames
        self.orig_w = orig_w
        self.orig_h = orig_h

        layout = QVBoxLayout(self)

        # --- info labels ---
        f_txt = str(orig_frames) if orig_frames is not None else "unknown"
        w_txt = str(orig_w) if orig_w is not None else "unknown"
        h_txt = str(orig_h) if orig_h is not None else "unknown"

        self.info_lbl = QLabel(
            f"Original:\n  Frames: {f_txt}\n  Size: {w_txt} × {h_txt}"
        )
        self.info_lbl.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.info_lbl)

        layout.addSpacing(8)

        # --- resize controls ---
        self.keep_original_size_chk = QCheckBox("Keep original size")
        self.keep_original_size_chk.setChecked(True)
        layout.addWidget(self.keep_original_size_chk)

        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("Target W:"))
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 100000)
        self.w_spin.setValue(int(orig_w) if orig_w else 512)
        self.w_spin.setEnabled(False)
        wh_row.addWidget(self.w_spin)

        wh_row.addSpacing(10)

        wh_row.addWidget(QLabel("Target H:"))
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 100000)
        self.h_spin.setValue(int(orig_h) if orig_h else 512)
        self.h_spin.setEnabled(False)
        wh_row.addWidget(self.h_spin)
        layout.addLayout(wh_row)

        # optional: quick hint
        self.hint_lbl = QLabel(
            "Tip: choose smaller W/H to reduce memory + speed up tracking."
        )
        self.hint_lbl.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.hint_lbl)

        layout.addSpacing(6)

        # --- frames controls ---
        self.load_all_frames_chk = QCheckBox("Load all frames")
        self.load_all_frames_chk.setChecked(True)
        layout.addWidget(self.load_all_frames_chk)

        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Max frames:"))
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 10_000_000)
        self.max_frames_spin.setValue(int(orig_frames) if orig_frames else 30)
        self.max_frames_spin.setEnabled(False)
        frames_row.addWidget(self.max_frames_spin)
        layout.addLayout(frames_row)

        # --- buttons (single load) ---
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        # wiring
        self.keep_original_size_chk.stateChanged.connect(self._update_enabled)
        self.load_all_frames_chk.stateChanged.connect(self._update_enabled)
        self.btn_load.clicked.connect(self._accept)
        self.btn_cancel.clicked.connect(self.reject)

        # outputs
        self._result = None  # dict or None

    def _update_enabled(self, *_):
        """Enable/disable spinboxes based on checkbox states."""
        self.w_spin.setEnabled(not self.keep_original_size_chk.isChecked())
        self.h_spin.setEnabled(not self.keep_original_size_chk.isChecked())
        self.max_frames_spin.setEnabled(not self.load_all_frames_chk.isChecked())

        # when "keep original" is checked, snap fields back to original (if known)
        if self.keep_original_size_chk.isChecked():
            if self.orig_w:
                self.w_spin.setValue(int(self.orig_w))
            if self.orig_h:
                self.h_spin.setValue(int(self.orig_h))

        if self.load_all_frames_chk.isChecked():
            if self.orig_frames:
                self.max_frames_spin.setValue(int(self.orig_frames))

    def _accept(self):
        """Validate settings and close the dialog with an accepted result."""
        # width/height: None means "use original"
        if self.keep_original_size_chk.isChecked():
            target_w = None
            target_h = None
        else:
            target_w = int(self.w_spin.value())
            target_h = int(self.h_spin.value())

        # max_frames: None means "all frames"
        max_frames = (
            None
            if self.load_all_frames_chk.isChecked()
            else int(self.max_frames_spin.value())
        )

        self._result = {
            "target_w": target_w,
            "target_h": target_h,
            "max_frames": max_frames,
        }
        self.accept()

    def result_settings(self):
        """Return the user-selected load parameters."""
        return self._result


if __name__ == "__main__":
    from controller import ControllerGUI

    app = QApplication(sys.argv)
    gui = ControllerGUI()
    gui.show()
    sys.exit(app.exec())
