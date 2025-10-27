import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """Handles logging to console, files, and TensorBoard."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
        
        self.metrics_history = {
            'episode': [],
            'reward': [],
            'steps': [],
            'epsilon': [],
            'loss': []
        }
    
    def log_episode(self, episode: int, reward: float, steps: int, 
                   epsilon: float, loss: Optional[float] = None):
        """Log episode metrics."""
        self.metrics_history['episode'].append(episode)
        self.metrics_history['reward'].append(reward)
        self.metrics_history['steps'].append(steps)
        self.metrics_history['epsilon'].append(epsilon)
        self.metrics_history['loss'].append(loss if loss is not None else 0)
        
        # Console output
        print(f"Episode {episode:4d} | Reward: {reward:7.2f} | "
              f"Steps: {steps:4d} | ε: {epsilon:.3f}", end="")
        if loss is not None:
            print(f" | Loss: {loss:.4f}")
        else:
            print()
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Train/Reward', reward, episode)
            self.writer.add_scalar('Train/Steps', steps, episode)
            self.writer.add_scalar('Train/Epsilon', epsilon, episode)
            if loss is not None:
                self.writer.add_scalar('Train/Loss', loss, episode)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward
        axes[0, 0].plot(self.metrics_history['episode'], 
                       self.metrics_history['reward'])
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Steps
        axes[0, 1].plot(self.metrics_history['episode'], 
                       self.metrics_history['steps'])
        axes[0, 1].set_title('Episode Steps')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Epsilon
        axes[1, 0].plot(self.metrics_history['episode'], 
                       self.metrics_history['epsilon'])
        axes[1, 0].set_title('Exploration (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Loss
        if any(self.metrics_history['loss']):
            axes[1, 1].plot(self.metrics_history['episode'], 
                           self.metrics_history['loss'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'))
        
        plt.close()
    
    def close(self):
        """Close logger."""
        if self.writer:
            self.writer.close()