"""
Training script for RL-based point selection agent.

This script trains the DQN agent to select optimal tracking points
for motion correction by interacting with the registration environment.

Usage:
    python train_rl.py --videos_dir <path> --epochs 100 --save_path model.pth
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl import RegistrationEnvironment, PointSelectionAgent


def load_video(video_path: str) -> np.ndarray:
    """
    Load a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Video as numpy array (T, H, W) or (T, C, H, W)
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames, dtype=np.float32)


def train_episode(env: RegistrationEnvironment, 
                  agent: PointSelectionAgent,
                  video: np.ndarray,
                  deformation_field: np.ndarray = None) -> dict:
    """
    Train agent for one episode.
    
    Args:
        env: Registration environment
        agent: RL agent
        video: Video data
        deformation_field: Optional deformation field
    
    Returns:
        Episode statistics
    """
    state = env.reset(video, deformation_field)
    done = False
    total_reward = 0
    num_steps = 0
    
    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Select action
        action = agent.select_action(state, valid_actions, training=True)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        # Train
        agent.train_step()
        
        # Update
        state = next_state
        total_reward += reward
        num_steps += 1
    
    return {
        'total_reward': total_reward,
        'num_steps': num_steps,
        'final_mse': info['mse'],
        'final_ssim': info['ssim'],
        'epsilon': agent.epsilon
    }


def train(args):
    """Main training loop."""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = RegistrationEnvironment(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        max_points=args.max_points,
        device=device
    )
    
    # Calculate state dimension
    # frame_features (256) + deformation_features (grid_cells) + point_mask (grid_cells) + num_selected (1) + quality (2)
    state_dim = 256 + env.num_grid_cells + env.num_grid_cells + 1 + 2
    
    # Create agent
    agent = PointSelectionAgent(
        state_dim=state_dim,
        num_actions=env.num_grid_cells,
        device=device
    )
    
    # Load videos
    print(f"Loading videos from {args.videos_dir}...")
    video_files = list(Path(args.videos_dir).glob('*.mp4')) + \
                  list(Path(args.videos_dir).glob('*.avi'))
    
    if len(video_files) == 0:
        print("No video files found! Please provide training videos.")
        return
    
    print(f"Found {len(video_files)} videos")
    
    # Load all videos (for small datasets)
    videos = []
    for vf in tqdm(video_files[:args.max_videos], desc="Loading videos"):
        try:
            video = load_video(str(vf))
            if video.shape[0] > 0:  # Valid video
                videos.append(video)
        except Exception as e:
            print(f"Error loading {vf}: {e}")
    
    print(f"Successfully loaded {len(videos)} videos")
    
    if len(videos) == 0:
        print("No videos loaded successfully!")
        return
    
    # Training metrics
    episode_rewards = []
    episode_mse = []
    episode_ssim = []
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_stats = []
        
        # Train on all videos
        for video in videos:
            stats = train_episode(env, agent, video)
            epoch_stats.append(stats)
        
        # Compute epoch metrics
        avg_reward = np.mean([s['total_reward'] for s in epoch_stats])
        avg_mse = np.mean([s['final_mse'] for s in epoch_stats])
        avg_ssim = np.mean([s['final_ssim'] for s in epoch_stats])
        epsilon = epoch_stats[0]['epsilon']
        
        episode_rewards.append(avg_reward)
        episode_mse.append(avg_mse)
        episode_ssim.append(avg_ssim)
        
        # Print progress
        if (epoch + 1) % args.print_freq == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"MSE: {avg_mse:.2f} | "
                  f"SSIM: {avg_ssim:.3f} | "
                  f"ε: {epsilon:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = f"{args.save_path}.epoch{epoch+1}"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    agent.save(args.save_path)
    print(f"\nTraining complete! Model saved to {args.save_path}")
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_mse': episode_mse,
        'episode_ssim': episode_ssim
    }
    
    metrics_path = args.save_path.replace('.pth', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Plot training curves
    if args.plot:
        plot_training_curves(episode_rewards, episode_mse, episode_ssim, args.save_path)


def plot_training_curves(rewards, mse, ssim, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(rewards)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(True)
    
    axes[1].plot(mse)
    axes[1].set_title('Registration MSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True)
    
    axes[2].plot(ssim)
    axes[2].set_title('Registration SSIM')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = save_path.replace('.pth', '_training.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agent for point selection')
    
    # Data
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing training videos')
    parser.add_argument('--max_videos', type=int, default=100,
                        help='Maximum number of videos to use')
    
    # Environment
    parser.add_argument('--grid_rows', type=int, default=4,
                        help='Number of grid rows')
    parser.add_argument('--grid_cols', type=int, default=4,
                        help='Number of grid columns')
    parser.add_argument('--max_points', type=int, default=16,
                        help='Maximum points per episode')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    # Saving
    parser.add_argument('--save_path', type=str, default='rl_agent.pth',
                        help='Path to save trained model')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_freq', type=int, default=1,
                        help='Print progress every N epochs')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training curves')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    
    # Train
    train(args)
