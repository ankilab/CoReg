import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, Dict, List, Optional
import cv2


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1, img2: Grayscale images as numpy arrays
    
    Returns:
        SSIM value between -1 and 1 (higher is better)
    """
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Store an experience."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class FeatureExtractor(nn.Module):
    """Lightweight CNN for extracting visual features from frames."""
    
    def __init__(self, output_dim: int = 256):
        """
        Args:
            output_dim: Dimension of output feature vector
        """
        super().__init__()
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Will be set after first forward pass
        self.fc_input_dim = None
        self.fc = None
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frame (B, 1, H, W)
        Returns:
            Feature vector (B, output_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Initialize FC layer on first forward pass
        if self.fc is None:
            self.fc_input_dim = x.size(1)
            self.fc = nn.Linear(self.fc_input_dim, self.output_dim).to(x.device)
        
        x = F.relu(self.fc(x))
        return x


class QNetwork(nn.Module):
    """Q-Network for estimating action values."""
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 512):
        """
        Args:
            state_dim: Dimension of state representation
            num_actions: Number of possible actions (grid cells)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: State tensor (B, state_dim)
        Returns:
            Q-values for each action (B, num_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RegistrationEnvironment:
    """
    Gym-like environment for point selection in motion correction.
    
    The agent selects tracking points sequentially, and receives rewards
    based on registration quality improvements.
    """
    
    def __init__(self, 
                 grid_rows: int = 4, 
                 grid_cols: int = 4,
                 max_points: int = 16,
                 device: str = 'cpu'):
        """
        Args:
            grid_rows: Number of rows in regular grid
            grid_cols: Number of columns in regular grid
            max_points: Maximum number of points to select per episode
            device: 'cpu' or 'cuda'
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_grid_cells = grid_rows * grid_cols
        self.max_points = min(max_points, self.num_grid_cells)
        self.device = device
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(output_dim=256).to(device)
        
        # Current episode state
        self.video_data = None
        self.deformation_field = None
        self.selected_points = []
        self.selected_indices = set()
        self.current_mse = None
        self.current_ssim = None
        self.grid_points = None  # (N, 2) array of grid positions
        
    def reset(self, video_data: np.ndarray, deformation_field: Optional[np.ndarray] = None) -> Dict:
        """
        Reset environment for a new episode.
        
        Args:
            video_data: Video tensor (T, H, W) or (T, C, H, W)
            deformation_field: Optional deformation field from CoTracker (T, H, W, 2)
        
        Returns:
            Initial state dictionary
        """
        self.video_data = video_data
        self.deformation_field = deformation_field
        self.selected_points = []
        self.selected_indices = set()
        self.current_mse = None
        self.current_ssim = None
        
        # Generate regular grid points
        if len(video_data.shape) == 3:
            h, w = video_data.shape[1], video_data.shape[2]
        else:
            h, w = video_data.shape[2], video_data.shape[3]
        
        self.grid_points = self._generate_grid_points(h, w)
        
        return self._get_state()
    
    def _generate_grid_points(self, height: int, width: int) -> np.ndarray:
        """Generate regular grid points."""
        y_coords = np.linspace(height * 0.1, height * 0.9, self.grid_rows)
        x_coords = np.linspace(width * 0.1, width * 0.9, self.grid_cols)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        return np.array(points, dtype=np.float32)
    
    def _get_state(self) -> Dict:
        """Get current state representation."""
        # Extract frame features from first frame
        frame = self.video_data[0]
        if len(frame.shape) == 3:  # (C, H, W)
            if frame.shape[0] == 3:  # RGB
                frame = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            else:
                frame = frame[0]  # Take first channel
        
        # Normalize and convert to tensor
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(self.device)
        frame_tensor = frame_tensor / 255.0 if frame_tensor.max() > 1 else frame_tensor
        
        # Extract features
        with torch.no_grad():
            frame_features = self.feature_extractor(frame_tensor).cpu().numpy().flatten()
        
        # Create point mask
        point_mask = np.zeros(self.num_grid_cells, dtype=np.float32)
        for idx in self.selected_indices:
            point_mask[idx] = 1.0
        
        # Deformation field features (if available)
        if self.deformation_field is not None:
            # Compute mean magnitude of deformation per grid cell
            deform_features = self._compute_deformation_features()
        else:
            deform_features = np.zeros(self.num_grid_cells, dtype=np.float32)
        
        # Quality metrics
        quality = np.array([
            self.current_mse if self.current_mse is not None else 0.0,
            self.current_ssim if self.current_ssim is not None else 0.0
        ], dtype=np.float32)
        
        state = {
            'frame_features': frame_features,
            'deformation_features': deform_features,
            'point_mask': point_mask,
            'num_selected': len(self.selected_points),
            'quality': quality
        }
        
        return state
    
    def _compute_deformation_features(self) -> np.ndarray:
        """Compute deformation magnitude for each grid cell."""
        if self.deformation_field is None:
            return np.zeros(self.num_grid_cells, dtype=np.float32)
        
        # deformation_field shape: (T, H, W, 2)
        # Compute mean magnitude over time
        deform_mag = np.linalg.norm(self.deformation_field, axis=-1)  # (T, H, W)
        mean_deform = deform_mag.mean(axis=0)  # (H, W)
        
        # Sample at grid points
        features = []
        for point in self.grid_points:
            x, y = int(point[0]), int(point[1])
            x = np.clip(x, 0, mean_deform.shape[1] - 1)
            y = np.clip(y, 0, mean_deform.shape[0] - 1)
            features.append(mean_deform[y, x])
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Grid cell index to select
        
        Returns:
            next_state: Next state dictionary
            reward: Reward for this action
            done: Whether episode is complete
            info: Additional information
        """
        # Check if action is valid
        if action in self.selected_indices:
            # Invalid action - already selected
            return self._get_state(), -1.0, False, {'invalid_action': True}
        
        # Add point
        self.selected_indices.add(action)
        self.selected_points.append(self.grid_points[action])
        
        # Compute reward based on registration quality
        reward, new_mse, new_ssim = self._compute_reward()
        
        # Update quality metrics
        self.current_mse = new_mse
        self.current_ssim = new_ssim
        
        # Check if done
        done = len(self.selected_points) >= self.max_points
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'mse': new_mse,
            'ssim': new_ssim,
            'num_points': len(self.selected_points)
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self) -> Tuple[float, float, float]:
        """
        Compute reward based on registration quality.
        
        Returns:
            reward: Computed reward
            mse: Current MSE
            ssim_score: Current SSIM
        """
        # For now, return dummy values
        # In full implementation, this would:
        # 1. Run registration with current points
        # 2. Compute MSE and SSIM
        # 3. Calculate reward based on improvement
        
        # Placeholder implementation
        prev_mse = self.current_mse if self.current_mse is not None else 100.0
        prev_ssim = self.current_ssim if self.current_ssim is not None else 0.5
        
        # Simulate improvement (in real implementation, run actual registration)
        new_mse = prev_mse * 0.95  # Assume 5% improvement
        new_ssim = min(1.0, prev_ssim + 0.02)  # Assume small SSIM improvement
        
        # Compute reward
        delta_mse = prev_mse - new_mse
        delta_ssim = new_ssim - prev_ssim
        
        # Weighted reward
        alpha = 10.0  # SSIM weight
        beta = 1.0    # MSE weight
        gamma = 0.01  # Point penalty
        
        reward = alpha * delta_ssim - beta * delta_mse - gamma
        
        return reward, new_mse, new_ssim
    
    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions (not yet selected)."""
        mask = np.ones(self.num_grid_cells, dtype=bool)
        for idx in self.selected_indices:
            mask[idx] = False
        return mask


# Placeholder for future implementation
class PointSelectionAgent:
    """DQN agent for learning optimal point selection."""
    
    def __init__(self, state_dim: int, num_actions: int, device: str = 'cpu'):
        """
        Args:
            state_dim: Dimension of state representation
            num_actions: Number of possible actions
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.num_actions = num_actions
        
        # Q-networks
        self.q_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update_freq = 100
        self.steps = 0
    
    def select_action(self, state: Dict, valid_actions: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state dictionary
            valid_actions: Boolean mask of valid actions
            training: Whether in training mode
        
        Returns:
            Selected action index
        """
        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions)[0]
            return np.random.choice(valid_indices)
        
        # Greedy action
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        q_values[~valid_actions] = -np.inf
        
        return int(np.argmax(q_values))
    
    def _state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        # Concatenate all state components
        components = [
            state['frame_features'],
            state['deformation_features'],
            state['point_mask'],
            np.array([state['num_selected']], dtype=np.float32),
            state['quality']
        ]
        state_vector = np.concatenate(components)
        return torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
    
    def store_experience(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([self._state_to_tensor(e.state) for e in experiences]).squeeze(1)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self._state_to_tensor(e.next_state) for e in experiences]).squeeze(1)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(self.device)
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


if __name__ == "__main__":
    # Simple test
    print("RL module loaded successfully")
    print("Testing environment...")
    
    # Create dummy video
    dummy_video = np.random.rand(10, 256, 256).astype(np.float32)
    
    # Create environment
    env = RegistrationEnvironment(grid_rows=4, grid_cols=4)
    state = env.reset(dummy_video)
    
    print(f"State keys: {state.keys()}")
    print(f"Frame features shape: {state['frame_features'].shape}")
    print(f"Point mask shape: {state['point_mask'].shape}")
    print(f"Number of grid cells: {env.num_grid_cells}")
    
    # Test step
    next_state, reward, done, info = env.step(0)
    print(f"Reward: {reward:.3f}, Done: {done}, Info: {info}")
    
    print("\nEnvironment test passed!")
