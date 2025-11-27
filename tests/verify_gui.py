import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock PyQt6 and matplotlib before importing gui
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.animation'] = MagicMock()
sys.modules['matplotlib.backends.backend_qtagg'] = MagicMock()
sys.modules['matplotlib.figure'] = MagicMock()

# Mock torch if not available
try:
    import torch
except ImportError:
    print("Torch not found, mocking it.")
    sys.modules['torch'] = MagicMock()
    torch = sys.modules['torch']
    torch.from_numpy.return_value = MagicMock()
    torch.cuda.is_available.return_value = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock mc_cotracker and utils if needed, but let's try to use them if possible
# If torch is mocked, we might need to mock these too as they depend on torch
if isinstance(torch, MagicMock):
    sys.modules['mc_cotracker'] = MagicMock()
    sys.modules['utils.utils'] = MagicMock()
    from gui import MotionCorrectionGUI
else:
    # If torch is real, we can try to import gui which imports these
    # But we need to make sure we don't fail on other imports
    try:
        from gui import MotionCorrectionGUI
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.modules['mc_cotracker'] = MagicMock()
        sys.modules['utils.utils'] = MagicMock()
        from gui import MotionCorrectionGUI

class TestMotionCorrectionGUI(unittest.TestCase):
    def setUp(self):
        # Patch QApplication to avoid issues
        self.qapp_patch = patch('PyQt6.QtWidgets.QApplication')
        self.qapp_mock = self.qapp_patch.start()
        
    def tearDown(self):
        self.qapp_patch.stop()

    def test_initialization(self):
        gui = MotionCorrectionGUI()
        self.assertIsNotNone(gui)
        print("GUI initialized successfully")

    def test_load_data_logic(self):
        gui = MotionCorrectionGUI()
        gui.video_path = "test.avi"
        
        # Mock load_video
        with patch('gui.load_video') as mock_load:
            mock_load.return_value = MagicMock()
            gui.load_data()
            self.assertIsNotNone(gui.video_data)
            print("Data loading logic verified")

    def test_processing_logic(self):
        gui = MotionCorrectionGUI()
        gui.video_data = MagicMock()
        
        # Mock thread start to just call run logic or finished signal
        # Since we can't easily test QThread in this mock setup, we'll test the logic in run_processing
        # But run_processing starts a thread.
        # We can test that it starts the thread.
        
        with patch('gui.ProcessingThread') as MockThread:
            gui.run_processing()
            MockThread.assert_called()
            print("Processing trigger verified")

if __name__ == '__main__':
    unittest.main()
