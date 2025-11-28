import sys
import os
import wx
import threading
import numpy as np
import torch
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mc_cotracker
mc_cotracker.DEFAULT_DEVICE = 'cuda'
from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5
from cotracker.models.core.model_utils import get_points_on_a_grid


class ProcessingThread(threading.Thread):
    def __init__(self, parent, video_data, video_path, base_grid_pts, custom_grid_pts=None):
        super().__init__()
        self.parent = parent
        self.video_data = video_data
        self.video_path = video_path
        self.base_grid_pts = base_grid_pts          # (1,K,2)
        self.custom_grid_pts = custom_grid_pts      # (M,2) or None
        self.daemon = True

    def run(self):
        try:
            grid_xy = self.base_grid_pts  # (B,K,2)

            # append custom points
            if self.custom_grid_pts is not None and len(self.custom_grid_pts) > 0:
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

            # IMPORTANT: use the new non-uniform warping (no reshape)
            tracks_np = pred_tracks.detach().cpu().numpy()
            video_np = self.video_data.detach().cpu().numpy()
            warped_video = warping(tracks_np, video_np)  # uses your new scattered warping

            wx.CallAfter(self.parent.on_processing_finished, pred_tracks, pred_visibility, grid_xy, warped_video)
        except Exception as e:
            import traceback
            traceback.print_exc()
            wx.CallAfter(self.parent.on_processing_error, str(e))


class MotionCorrectionGUI(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Motion Correction GUI", size=(1000, 800))
        
        self.video_path = None
        self.video_data = None
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        
        # Editable base grid (old grid points)
        self.grid_size = 4
        self.base_grid_pts = None            # (Ngrid,2) float32 numpy
        self.base_grid_active = None         # (Ngrid,) bool numpy
        self.base_grid_pts0 = None           # backup copy for Undo All

        # Drag state
        self._drag = None                    # ("base"/"custom", idx)
        self._mpl_cids = {}                  # mpl connection ids

        self.base_scatter = None
        self.custom_scatter = None
        
        # Manual grid point editing
        self.custom_grid_pts = []  # List of (x, y) tuples
        self.edit_mode = False
        self.grid_scatter = None
        
        # Playback state
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.playback_speed = 1.0
        
        # Visualization data cache
        self.viz_data = {}
        self.viz_artists = {}
        
        self.init_ui()
        self.Centre()

    def init_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Controls layout
        # Controls layout
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.load_btn = wx.Button(panel, label="Load File")
        self.load_btn.Bind(wx.EVT_BUTTON, self.select_file)
        controls_sizer.Add(self.load_btn, 0, wx.ALL, 5)

        # Grid size input
        controls_sizer.Add(wx.StaticText(panel, label="Grid size:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.grid_size_ctrl = wx.TextCtrl(panel, value=str(self.grid_size),
                                        style=wx.TE_PROCESS_ENTER, size=(50, -1))
        self.grid_size_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_grid_size_enter)
        controls_sizer.Add(self.grid_size_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.apply_grid_btn = wx.Button(panel, label="Apply Grid")
        self.apply_grid_btn.Bind(wx.EVT_BUTTON, self.on_apply_grid_size)
        self.apply_grid_btn.Enable(True)  # enabled always; it will only regenerate if video loaded
        controls_sizer.Add(self.apply_grid_btn, 0, wx.ALL, 5)

        # self.viz_grid_btn = wx.Button(panel, label="Visualize Grid Pts")
        # self.viz_grid_btn.Bind(wx.EVT_BUTTON, self.on_visualize_grid_pts)
        # self.viz_grid_btn.Enable(False)
        # controls_sizer.Add(self.viz_grid_btn, 0, wx.ALL, 5)

        self.process_btn = wx.Button(panel, label="Run Registration")
        self.process_btn.Bind(wx.EVT_BUTTON, self.run_processing)
        self.process_btn.Enable(False)
        controls_sizer.Add(self.process_btn, 0, wx.ALL, 5)

        # (optional) keep edit/clear if you still want interactive point editing
        self.edit_points_btn = wx.ToggleButton(panel, label="Edit Points")
        self.edit_points_btn.Bind(wx.EVT_TOGGLEBUTTON, self.toggle_edit_mode)
        self.edit_points_btn.Enable(False)
        controls_sizer.Add(self.edit_points_btn, 0, wx.ALL, 5)

        self.clear_points_btn = wx.Button(panel, label="Clear Points")
        self.clear_points_btn.Bind(wx.EVT_BUTTON, self.clear_points)
        self.clear_points_btn.Enable(False)
        controls_sizer.Add(self.clear_points_btn, 0, wx.ALL, 5)

        self.undo_btn = wx.Button(panel, label="Undo All")
        self.undo_btn.Bind(wx.EVT_BUTTON, self.undo_all_changes)
        self.undo_btn.Enable(False)
        controls_sizer.Add(self.undo_btn, 0, wx.ALL, 5)

        self.points_label = wx.StaticText(panel, label="Points: ")
        controls_sizer.Add(self.points_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.status_label = wx.StaticText(panel, label="Ready")
        controls_sizer.Add(self.status_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        main_sizer.Add(controls_sizer, 0, wx.EXPAND)
        
        # Matplotlib Figure - no fixed figsize for dynamic resizing
        self.fig = Figure()
        self.canvas = FigureCanvas(panel, -1, self.fig)
        main_sizer.Add(self.canvas, 1, wx.EXPAND)
        
        # Playback Controls (Bottom)
        playback_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.play_btn = wx.Button(panel, label="⏯️")
        self.play_btn.Bind(wx.EVT_BUTTON, self.toggle_playback)
        self.play_btn.Enable(False)
        playback_sizer.Add(self.play_btn, 0, wx.ALL, 5)

        # NEW: step buttons next to each other
        self.step_back_btn = wx.Button(panel, label="◀", size=(40, -1))
        self.step_back_btn.Bind(wx.EVT_BUTTON, self.step_backward)
        self.step_back_btn.Enable(False)
        playback_sizer.Add(self.step_back_btn, 0, wx.ALL, 5)

        self.step_fwd_btn = wx.Button(panel, label="▶", size=(40, -1))
        self.step_fwd_btn.Bind(wx.EVT_BUTTON, self.step_forward)
        self.step_fwd_btn.Enable(False)
        playback_sizer.Add(self.step_fwd_btn, 0, wx.ALL, 5)

        # Slider after both step buttons
        self.slider = wx.Slider(panel, value=0, minValue=0, maxValue=100,
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider_change)
        self.slider.Enable(False)
        playback_sizer.Add(self.slider, 1, wx.ALL | wx.EXPAND, 5)

        self.frame_label = wx.StaticText(panel, label="0/0")
        playback_sizer.Add(self.frame_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        main_sizer.Add(playback_sizer, 0, wx.EXPAND)
        
        # Initialize subplots
        self.axs = self.fig.subplots(2, 2)
        self.axs[0, 0].set_title("Input Video")
        self.axs[0, 1].set_title("Grid Points (Frame 0)")
        self.axs[1, 0].set_title("Warped Result")
        self.axs[1, 1].set_title("Tracking Trajectories")
        
        for ax in self.axs.flat:
            ax.axis('off')
        
        panel.SetSizer(main_sizer)
        self.canvas.draw()

    def select_file(self, event):
        wildcard = "Video files (*.avi;*.mp4;*.tiff;*.tif;*.h5)|*.avi;*.mp4;*.tiff;*.tif;*.h5|All files (*.*)|*.*"
        dlg = wx.FileDialog(self, "Select Video File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        
        if dlg.ShowModal() == wx.ID_OK:
            self.video_path = dlg.GetPath()
            self.status_label.SetLabel(f"Loading: {os.path.basename(self.video_path)}")
            wx.CallAfter(self.load_data)
        
        dlg.Destroy()

    def show_input_frame_only(self):
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')

        self.axs[0, 0].set_title("Input Video")
        frame = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        self.viz_artists['im1'] = self.axs[0, 0].imshow(frame)

        self.axs[0, 1].set_title("Grid Points (Frame 0)")
        self.axs[1, 0].set_title("Warped Result")
        self.axs[1, 1].set_title("Tracking Trajectories")

        self.canvas.draw()

    def on_grid_size_enter(self, event):
        self.on_apply_grid_size(event)

    def on_apply_grid_size(self, event):
        # Parse and validate
        txt = self.grid_size_ctrl.GetValue().strip()
        try:
            gs = int(txt)
        except ValueError:
            wx.MessageBox("Grid size must be an integer (e.g. 4, 8, 16).", "Invalid Grid Size",
                        wx.OK | wx.ICON_WARNING)
            self.grid_size_ctrl.SetValue(str(self.grid_size))
            return

        if gs < 2:
            wx.MessageBox("Grid size must be >= 2.", "Invalid Grid Size", wx.OK | wx.ICON_WARNING)
            self.grid_size_ctrl.SetValue(str(self.grid_size))
            return

        self.grid_size = gs

        # If we already loaded a video, regenerate base grid immediately
        if self.video_data is not None:
            self.reset_base_grid_from_size()
            self.status_label.SetLabel(f"Grid updated: {self.grid_size}×{self.grid_size}")

    def reset_base_grid_from_size(self):
        """Rebuild editable base grid using current self.grid_size. Clears edits."""
        grid_xy = get_points_on_a_grid(self.grid_size, self.video_data.shape[-2:])  # (1, N, 2)
        self.base_grid_pts = grid_xy[0].detach().cpu().numpy().astype(np.float32)  # (N,2)
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.base_grid_pts0 = self.base_grid_pts.copy()

        # Clear custom points too (usually what users expect when changing grid size)
        self.custom_grid_pts = []
        self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()


    def on_visualize_grid_pts(self, event):
        if self.video_data is None:
            return
        self.update_grid_display_pre_registration()


    def update_grid_display_pre_registration(self):
        if self.video_data is None:
            return

        ax = self.axs[0, 1]
        ax.clear()
        ax.axis("off")
        ax.set_title("Grid Points (Frame 0) — Edit Mode" if self.edit_mode else "Grid Points (Frame 0)")

        frame0 = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        ax.imshow(frame0)

        # Base grid: only active points
        if self.base_grid_pts is not None and self.base_grid_active is not None:
            active = self.base_grid_active
            if np.any(active):
                ptsA = self.base_grid_pts[active]
                ax.scatter(ptsA[:, 0], ptsA[:, 1], c="yellow", s=20)

        # Custom points
        if len(self.custom_grid_pts) > 0:
            pts = np.array(self.custom_grid_pts, dtype=np.float32)
            ax.scatter(pts[:, 0], pts[:, 1], c="yellow", s=50, marker="x", linewidths=2)

        self.canvas.draw_idle()

    def load_data(self):
        if not self.video_path:
            return

        ext = os.path.splitext(self.video_path)[1].lower()
        print(f"Loading data from {self.video_path}...")
        self.status_label.SetLabel("Loading data...")
        wx.Yield()

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
                wx.MessageBox(f"Unsupported file extension: {ext}", "Error", wx.OK | wx.ICON_ERROR)
                self.status_label.SetLabel("Error loading data")
                return

            # Initialize editable base grid once (source of truth)
            grid_xy = get_points_on_a_grid(self.grid_size, self.video_data.shape[-2:])  # (1,N,2)
            self.base_grid_pts = grid_xy[0].detach().cpu().numpy().astype(np.float32)   # (N,2)
            self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
            self.base_grid_pts0 = self.base_grid_pts.copy()

            self.undo_btn.Enable(True)

            

            print(f"Data loaded. Shape: {self.video_data.shape}")
            self.status_label.SetLabel(f"Loaded: {os.path.basename(self.video_path)}")
            # self.viz_grid_btn.Enable(True)


            # Show first frame (input only) and wait for user to click "Visualize Grid Pts"
            self.show_input_frame_only()
            self.process_btn.Enable(True)
            self.edit_points_btn.Enable(True)
            self.clear_points_btn.Enable(True)
            
            # Setup playback
            self.total_frames = self.video_data.shape[0]
            self.slider.SetRange(0, self.total_frames - 1)
            self.slider.SetValue(0)
            self.slider.Enable(True)
            self.play_btn.Enable(True)
            self.current_frame = 0
            self.step_back_btn.Enable(True)
            self.step_fwd_btn.Enable(True)
            self.update_frame_label()
            
            # Show first frame
            self.visualize()

            # Ensure base grid matches textbox value
            self.on_apply_grid_size(None)  # parses textbox & regenerates grid

        except Exception as e:
            print(f"Error loading data: {e}")
            wx.MessageBox(f"Error loading data: {e}", "Error", wx.OK | wx.ICON_ERROR)
            self.status_label.SetLabel("Error loading data")
            self.video_data = None



    def toggle_edit_mode(self, event):
        self.edit_mode = self.edit_points_btn.GetValue()

        if self.edit_mode:
            self.status_label.SetLabel("Edit Mode: drag to move | right-click base toggles delete | right-click custom deletes")
            if not self._mpl_cids:
                self._mpl_cids["press"] = self.canvas.mpl_connect("button_press_event", self.on_canvas_press)
                self._mpl_cids["move"] = self.canvas.mpl_connect("motion_notify_event", self.on_canvas_move)
                self._mpl_cids["release"] = self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        else:
            for cid in self._mpl_cids.values():
                self.canvas.mpl_disconnect(cid)
            self._mpl_cids.clear()
            self._drag = None
            self.status_label.SetLabel(f"Loaded: {os.path.basename(self.video_path) if self.video_path else 'Ready'}")

        self.update_grid_display_pre_registration()

    def _pause_if_playing(self):
        if self.is_playing:
            self.timer.Stop()
            self.is_playing = False
            self.play_btn.SetLabel("▶")

    def step_forward(self, event):
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.slider.SetValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def step_backward(self, event):
        if self.video_data is None:
            return
        self._pause_if_playing()
        self.current_frame = (self.current_frame - 1) % self.total_frames
        self.slider.SetValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def clear_points(self, event):
        self.custom_grid_pts = []
        self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display()

    def undo_all_changes(self, event):
        if self.base_grid_pts0 is None:
            return
        self.base_grid_pts = self.base_grid_pts0.copy()
        self.base_grid_active = np.ones((self.base_grid_pts.shape[0],), dtype=bool)
        self.custom_grid_pts = []
        self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()

    def on_canvas_click(self, event):
        if not self.edit_mode or event.inaxes != self.axs[0, 1]:
            return
        if self.video_data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            self.custom_grid_pts.append((x, y))
            self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
            self.update_grid_display_pre_registration()

        elif event.button == 3:
            if len(self.custom_grid_pts) > 0:
                x, y = event.xdata, event.ydata
                distances = [((px - x)**2 + (py - y)**2)**0.5 for px, py in self.custom_grid_pts]
                min_idx = distances.index(min(distances))
                if distances[min_idx] < 20:
                    self.custom_grid_pts.pop(min_idx)
                    self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
                    self.update_grid_display_pre_registration()

    def _nearest_point(self, pts, x, y):
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def on_canvas_press(self, event):
        if not self.edit_mode or event.inaxes != self.axs[0, 1]:
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
                self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
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
        self.points_label.SetLabel(f"Points: {len(self.base_grid_active)}")
        self.update_grid_display_pre_registration()

    def on_canvas_move(self, event):
        if not self.edit_mode or self._drag is None or event.inaxes != self.axs[0, 1]:
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
        self._drag = None


    def update_grid_display(self):
        if self.video_data is None:
            return

        ax = self.axs[0, 1]
        ax.clear()
        ax.axis('off')
        ax.set_title("Tracked Points (Frame 0) — click to add/remove" if self.edit_mode else "Tracked Points (Frame 0)")

        frame0 = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        ax.imshow(frame0)

        # show tracked points (initial positions from pred_tracks on frame 0)
        if self.pred_tracks is not None:
            init_xy = self.pred_tracks[0, 0].detach().cpu().numpy()  # (N,2)
            ax.scatter(init_xy[:, 0], init_xy[:, 1], c='r', s=5)

        # show custom points
        if len(self.custom_grid_pts) > 0:
            pts = np.array(self.custom_grid_pts)
            ax.scatter(pts[:, 0], pts[:, 1], c='yellow', s=50, marker='x', linewidths=2)

        self.canvas.draw_idle()

    def run_processing(self, event):
        if self.video_data is None:
            return

        self.status_label.SetLabel("Processing... (This may take a while)")
        self.process_btn.Enable(False)
        self.load_btn.Enable(False)
        
        # Base grid points: only ACTIVE ones go into queries
        base_active = self.base_grid_pts[self.base_grid_active].astype(np.float32)  # (K,2)
        base_grid_tensor = torch.from_numpy(base_active).unsqueeze(0)  # (1,K,2)

        custom_pts_tensor = None
        if len(self.custom_grid_pts) > 0:
            pts_array = np.array(self.custom_grid_pts, dtype=np.float32)           # (M,2)
            custom_pts_tensor = torch.from_numpy(pts_array)

        thread = ProcessingThread(self, self.video_data, self.video_path,
                                base_grid_pts=base_grid_tensor,
                                custom_grid_pts=custom_pts_tensor)
        thread.start()

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video):
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        self.warped_video = warped_video
        
        self.status_label.SetLabel("Processing complete")
        self.process_btn.Enable(True)
        self.load_btn.Enable(True)
        
        self.visualize()

    def on_processing_error(self, error_msg):
        self.status_label.SetLabel("Error during processing")
        wx.MessageBox(error_msg, "Processing Error", wx.OK | wx.ICON_ERROR)
        self.process_btn.Enable(True)
        self.load_btn.Enable(True)

    def toggle_playback(self, event):
        if self.is_playing:
            self.timer.Stop()
            self.is_playing = False
            self.play_btn.SetLabel("▶")
        else:
            # Calculate interval based on speed (base is 50ms for 20fps)
            speed_factor = self.speed_slider.GetValue() / 100.0
            interval = int(50 / speed_factor)
            self.timer.Start(interval)
            self.is_playing = True
            self.play_btn.SetLabel("⏸")

    def on_timer(self, event):
        self.next_frame()

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
        
        self.slider.SetValue(self.current_frame)
        self.update_plot()
        self.update_frame_label()

    def on_slider_change(self, event):
        if not self.is_playing:  # Only update if not auto-playing
            self.current_frame = self.slider.GetValue()
            self.update_plot()
            self.update_frame_label()

    def update_frame_label(self):
        self.frame_label.SetLabel(f"{self.current_frame}/{self.total_frames}")

    def on_speed_change(self, event):
        speed = self.speed_slider.GetValue() / 100.0
        self.speed_label.SetLabel(f"{speed:.2f}x")
        # Update timer interval if currently playing
        if self.is_playing:
            interval = int(50 / speed)
            self.timer.Start(interval)

    def visualize(self):
        if self.video_data is None:
            return

        print("Preparing visualization...")
        
        # Stop existing animation if any
        self.timer.Stop()
        self.is_playing = False
        self.play_btn.SetLabel("▶")
        
        full_viz = self.warped_video is not None

        # Prepare data - cache as numpy arrays for faster access
        self.viz_data['input'] = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
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
            
            self.viz_data['tracks'] = self.pred_tracks[0].cpu().numpy()  # (T, G, 2)
        
        # Clear axes
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')

        # 1. Input Video
        self.axs[0, 0].set_title("Input Video")
        self.viz_artists['im1'] = self.axs[0, 0].imshow(self.viz_data['input'][self.current_frame])
        
        if full_viz:
            # 2. First frame with tracked points - use initial positions from pred_tracks
            self.axs[0, 1].set_title("Tracked Points (Frame 0)")
            self.axs[0, 1].imshow(self.viz_data['input'][0])
            # Get initial positions from first frame of predicted tracks
            initial_positions = self.viz_data['tracks'][0]  # (G, 2) - first frame positions
            self.axs[0, 1].scatter(initial_positions[:, 0], initial_positions[:, 1], c='r', s=5)
            
            # 3. Warped Result
            self.axs[1, 0].set_title("Warped Result")
            self.viz_artists['im3'] = self.axs[1, 0].imshow(self.viz_data['warped'][self.current_frame], cmap='gray')
            
            # 4. Tracking Trajectories
            self.axs[1, 1].set_title("Tracking Trajectories")
            self.viz_artists['im4'] = self.axs[1, 1].imshow(self.viz_data['input'][self.current_frame])
            current_tracks = self.viz_data['tracks'][self.current_frame]
            self.viz_artists['scat4'] = self.axs[1, 1].scatter(current_tracks[:, 0], current_tracks[:, 1], c='g', s=2)
        else:
            self.axs[0, 1].set_title("Tracked Points (Frame 0)")
            self.axs[1, 0].set_title("Warped Result")
            self.axs[1, 1].set_title("Tracking Trajectories")
        
        self.canvas.draw()

    def update_plot(self):
        if 'input' not in self.viz_data:
            return
            
        # Update Input
        self.viz_artists['im1'].set_data(self.viz_data['input'][self.current_frame])
        
        # Update others if full viz
        if 'warped' in self.viz_data and 'im3' in self.viz_artists:
            self.viz_artists['im3'].set_data(self.viz_data['warped'][self.current_frame])
            self.viz_artists['im4'].set_data(self.viz_data['input'][self.current_frame])
            
            current_tracks = self.viz_data['tracks'][self.current_frame]
            self.viz_artists['scat4'].set_offsets(current_tracks)
            
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = wx.App()
    gui = MotionCorrectionGUI()
    gui.Show()
    app.MainLoop()
