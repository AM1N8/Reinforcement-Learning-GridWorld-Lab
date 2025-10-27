import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import matplotlib.animation as animation


class EnvironmentRenderer:
    """Render environment episodes as animations."""
    
    def __init__(self, env):
        self.env = env
        self.frames = []
    
    def record_frame(self):
        """Record current environment state."""
        frame = self.env.render(mode='rgb_array')
        self.frames.append(frame)
    
    def save_animation(self, path: str, fps: int = 5):
        """Save recorded frames as GIF."""
        if not self.frames:
            print("No frames to save")
            return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        
        im = ax.imshow(self.frames[0])
        
        def update(frame):
            im.set_array(self.frames[frame])
            return [im]
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.frames),
            interval=1000//fps, blit=True
        )
        
        anim.save(path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved: {path}")
        
        self.frames = []  # Clear frames
    
    def clear_frames(self):
        """Clear recorded frames."""
        self.frames = []


def plot_comparison(results: dict, save_path: Optional[str] = None):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        results: Dict mapping algorithm names to metric lists
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for algo_name, metrics in results.items():
        episodes = range(1, len(metrics['rewards']) + 1)
        
        # Plot rewards
        axes[0].plot(episodes, metrics['rewards'], label=algo_name, alpha=0.7)
        
        # Plot steps
        axes[1].plot(episodes, metrics['steps'], label=algo_name, alpha=0.7)
    
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Episode Steps')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_q_values(q_network, env, save_path: Optional[str] = None):
    """
    Visualize Q-values across grid positions.
    
    Args:
        q_network: Trained Q-network
        env: GridWorld environment
        save_path: Path to save figure
    """
    import torch
    
    size = env.size
    q_map = np.zeros((size, size, 4))
    
    # Compute Q-values for each position
    for i in range(size):
        for j in range(size):
            state = np.zeros(size * size)
            state[i * size + j] = 1.0
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_network(state_tensor).cpu().numpy()[0]
                q_map[i, j] = q_values
    
    # Plot Q-values for each action
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    actions = ['Up', 'Right', 'Down', 'Left']
    
    for idx, (ax, action) in enumerate(zip(axes.flat, actions)):
        im = ax.imshow(q_map[:, :, idx], cmap='RdYlGn', interpolation='nearest')
        ax.set_title(f'Q-values: {action}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()