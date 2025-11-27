import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import numpy as np
import torch
import matplotlib
# Ensure matplotlib uses the correct backend
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mc_cotracker import cotrack_model, warping
from utils.utils import load_video, load_tiff, load_hdf5

class ProcessingThread(QThread):
    finished = pyqtSignal(object, object, object, object) # pred_tracks, pred_visibility, grid_pts, warped_video
    error = pyqtSignal(str)

    def __init__(self, video_data):
        super().__init__()
        self.video_data = video_data

    def run(self):
        try:
            # Run CoTracker model
            pred_tracks, pred_visibility, grid_pts = cotrack_model(self.video_data, grid_size=20)
            
            # Run Warping
            # warping expects predicted_tracks and frames
            warped_video = warping(pred_tracks.cpu(), self.video_data.cpu())
            
            self.finished.emit(pred_tracks, pred_visibility, grid_pts, warped_video)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class MotionCorrectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Correction GUI")
        self.resize(1000, 800)

        self.video_path = None
        self.video_data = None
        self.pred_tracks = None
        self.pred_visibility = None
        self.grid_pts = None
        self.warped_video = None
        self.ani = None

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Controls layout
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.select_file)
        controls_layout.addWidget(self.load_btn)

        self.process_btn = QPushButton("Run Processing")
        self.process_btn.clicked.connect(self.run_processing)
        self.process_btn.setEnabled(False)
        controls_layout.addWidget(self.process_btn)

        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Initialize subplots
        self.axs = self.fig.subplots(2, 2)
        self.axs[0, 0].set_title("Input Video")
        self.axs[0, 1].set_title("Tracked Points (Frame 0)")
        self.axs[1, 0].set_title("Warped Result")
        self.axs[1, 1].set_title("Tracking Trajectories")
        
        for ax in self.axs.flat:
            ax.axis('off')

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video files (*.avi *.mp4 *.tiff *.tif *.h5);;All files (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.status_label.setText(f"Loading: {os.path.basename(file_path)}")
            self.load_data()

    def load_data(self):
        if not self.video_path:
            return

        ext = os.path.splitext(self.video_path)[1].lower()
        print(f"Loading data from {self.video_path}...")
        self.status_label.setText("Loading data...")
        QApplication.processEvents()

        try:
            if ext in ['.avi', '.mp4']:
                self.video_data = load_video(self.video_path, resize=True)
            elif ext in ['.tiff', '.tif']:
                video_np = load_tiff(self.video_path, frames=100)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            elif ext in ['.h5']:
                video_np = load_hdf5(self.video_path, "/data", frames=100)
                self.video_data = torch.from_numpy(video_np).unsqueeze(1).repeat(1, 3, 1, 1).float()
            else:
                QMessageBox.warning(self, "Error", f"Unsupported file extension: {ext}")
                self.status_label.setText("Error loading data")
                return

            print(f"Data loaded. Shape: {self.video_data.shape}")
            self.status_label.setText(f"Loaded: {os.path.basename(self.video_path)}")
            self.process_btn.setEnabled(True)
            
            # Show first frame
            self.show_preview()

        except Exception as e:
            print(f"Error loading data: {e}")
            QMessageBox.critical(self, "Error", f"Error loading data: {e}")
            self.status_label.setText("Error loading data")
            self.video_data = None

    def show_preview(self):
        if self.video_data is not None:
            # Clear axes
            for ax in self.axs.flat:
                ax.clear()
                ax.axis('off')
            
            self.axs[0, 0].set_title("Input Video")
            self.axs[0, 1].set_title("Tracked Points (Frame 0)")
            self.axs[1, 0].set_title("Warped Result")
            self.axs[1, 1].set_title("Tracking Trajectories")

            # Show first frame
            frame = self.video_data[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            self.axs[0, 0].imshow(frame)
            self.canvas.draw()

    def run_processing(self):
        if self.video_data is None:
            return

        self.status_label.setText("Processing... (This may take a while)")
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        self.thread = ProcessingThread(self.video_data)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def on_processing_finished(self, pred_tracks, pred_visibility, grid_pts, warped_video):
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        self.grid_pts = grid_pts
        self.warped_video = warped_video
        
        self.status_label.setText("Processing complete")
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.visualize()

    def on_processing_error(self, error_msg):
        self.status_label.setText("Error during processing")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

    def visualize(self):
        if self.video_data is None or self.warped_video is None:
            return

        print("Preparing visualization...")
        
        # Stop existing animation
        if self.ani is not None:
            self.ani.event_source.stop()
        
        # Prepare data
        input_video_vis = self.video_data.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        warped_video_vis = self.warped_video
        
        grid_pts_np = self.grid_pts.cpu().numpy()
        grid_x = grid_pts_np[:, 1]
        grid_y = grid_pts_np[:, 2]
        
        tracks = self.pred_tracks[0].cpu().numpy() # (T, G, 2)
        T = input_video_vis.shape[0]
        
        # Clear axes
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')

        # 1. Input Video
        self.axs[0, 0].set_title("Input Video")
        im1 = self.axs[0, 0].imshow(input_video_vis[0])
        
        # 2. First frame with tracked points
        self.axs[0, 1].set_title("Tracked Points (Frame 0)")
        self.axs[0, 1].imshow(input_video_vis[0])
        self.axs[0, 1].scatter(grid_x, grid_y, c='r', s=5)
        
        # 3. Warped Result
        self.axs[1, 0].set_title("Warped Result")
        im3 = self.axs[1, 0].imshow(warped_video_vis[0], cmap='gray')
        
        # 4. Tracking Trajectories
        self.axs[1, 1].set_title("Tracking Trajectories")
        im4 = self.axs[1, 1].imshow(input_video_vis[0])
        scat4 = self.axs[1, 1].scatter(tracks[0, :, 0], tracks[0, :, 1], c='g', s=2)
        
        def update(frame):
            im1.set_data(input_video_vis[frame])
            im3.set_data(warped_video_vis[frame])
            im4.set_data(input_video_vis[frame])
            scat4.set_offsets(tracks[frame])
            return im1, im3, im4, scat4

        self.ani = animation.FuncAnimation(self.fig, update, frames=T, interval=50, blit=True)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MotionCorrectionGUI()
    gui.show()
    sys.exit(app.exec())
