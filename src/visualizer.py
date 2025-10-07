import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class GridWorldVisualizer:
    """Visualizer for GridWorld environment and RL algorithms"""
    
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
    
    def plot_grid(self, ax=None):
        """Plot the basic grid with start and goal"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
        # Draw grid lines
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=1)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=1)
    
        # Mark goal
        goal = patches.Rectangle((self.cols - 1, self.rows - 1), 1, 1, 
                                  linewidth=2, edgecolor='green', 
                                  facecolor='lightgreen', alpha=0.5)
        ax.add_patch(goal)
        ax.text(self.cols - 0.5, self.rows - 0.5, 'G', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='darkgreen')
    
        # Mark start
        start = patches.Rectangle((0, 0), 1, 1, 
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.5)
        ax.add_patch(start)
        ax.text(0.5, 0.5, 'S', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='darkblue')
    
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('GridWorld Environment')
    
        return ax
    
    def plot_trajectory(self, trajectory, title="Agent Trajectory"):
        """Plot agent's trajectory through the grid"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
        
        trajectory = np.array(trajectory)
        x_coords = trajectory[:, 1] + 0.5  
        y_coords = trajectory[:, 0] + 0.5  
        
        # Plot path
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(x_coords, y_coords, 'ro', markersize=8, alpha=0.5)
        
        # Add step numbers
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.text(x, y, str(i), ha='center', va='center', 
                   fontsize=8, color='white', fontweight='bold')
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    def plot_state(self, state, title="Current State"):
        """Plot current agent state"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
        
        # Plot agent position
        x = state[1] + 0.5
        y = state[0] + 0.5
        ax.plot(x, y, 'ro', markersize=20, label='Agent')
        ax.text(x, y, 'A', ha='center', va='center', 
               fontsize=16, color='white', fontweight='bold')
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    def plot_value_function(self, V, title="Value Function"):
        """Visualize value function as heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(V, cmap='RdYlGn', origin='upper', aspect='auto')
        
        # Add value text to each cell
        for i in range(self.rows):
            for j in range(self.cols):
                text = ax.text(j, i, f'{V[i, j]:.2f}',
                              ha="center", va="center", color="black", 
                              fontsize=10, fontweight='bold')
        
        # Add grid lines
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1)
        
        # Mark start and goal
        ax.text(0, 0, 'S', ha='center', va='bottom', 
                fontsize=16, fontweight='bold', color='blue')
        ax.text(self.cols - 1, self.rows - 1, 'G', ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='green')
        
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='State Value')
        plt.tight_layout()
        
        return fig, ax
    
    def plot_policy(self, policy, title="Policy"):
        """Visualize policy with arrows"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self.plot_grid(ax)
        
        # Arrow symbols for each action
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        # Arrow directions for quiver plot
        arrow_dirs = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Skip goal state
                if i == self.rows - 1 and j == self.cols - 1:
                    continue
                
                action = int(policy[i, j])
                
                # Draw arrow using text
                ax.text(j + 0.5, i + 0.5, arrows[action],
                       ha='center', va='center', fontsize=24, 
                       color='darkblue', fontweight='bold')
                
                # Alternative: use quiver for arrows
                # dx, dy = arrow_dirs[action]
                # ax.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.15, 
                #          head_length=0.1, fc='blue', ec='blue')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_q_values(self, Q, model, title="Q-Values"):
        """Visualize Q-values for all state-action pairs"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Find min and max Q-values for consistent coloring
        all_q_values = []
        for i in range(self.rows):
            for j in range(self.cols):
                state_key = (i, j)
                if state_key in Q:
                    all_q_values.extend(Q[state_key])
        
        if all_q_values:
            vmin, vmax = min(all_q_values), max(all_q_values)
        else:
            vmin, vmax = -10, 10
        
        # Draw grid
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=2)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=2)
        
        # Plot Q-values for each action in each state
        action_positions = {
            0: (0.5, 0.2),   # UP - top
            1: (0.5, 0.8),   # DOWN - bottom
            2: (0.2, 0.5),   # LEFT - left
            3: (0.8, 0.5),   # RIGHT - right
        }
        
        for i in range(self.rows):
            for j in range(self.cols):
                state_key = (i, j)
                
                # Skip goal state
                if i == self.rows - 1 and j == self.cols - 1:
                    ax.text(j + 0.5, i + 0.5, 'GOAL', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='green')
                    continue
                
                if state_key in Q:
                    q_values = Q[state_key]
                    
                    for action in range(model.num_actions):
                        q_val = q_values[action]
                        rel_x, rel_y = action_positions[action]
                        
                        # Color based on Q-value
                        color_intensity = (q_val - vmin) / (vmax - vmin + 1e-8)
                        color = plt.cm.RdYlGn(color_intensity)
                        
                        # Draw Q-value
                        ax.text(j + rel_x, i + rel_y, f'{q_val:.1f}',
                               ha='center', va='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=color, alpha=0.7),
                               fontweight='bold')
        
        # Mark start
        ax.text(0.5, 0.1, 'START', ha='center', va='center',
               fontsize=10, fontweight='bold', color='blue')
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_learning_curve(self, rewards, title="Learning Curve", window=100):
        """Plot learning curve showing rewards over episodes"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        episodes = range(1, len(rewards) + 1)
        
        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Plot moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), moving_avg, 
                   color='red', linewidth=2, label=f'{window}-Episode Average')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_comparison(self, results, title="Algorithm Comparison"):
        """Compare trajectories of multiple algorithms"""
        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 6))
        
        if n_algorithms == 1:
            axes = [axes]
        
        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx]
            self.plot_grid(ax)
            
            trajectory = np.array(data['trajectory'])
            x_coords = trajectory[:, 1] + 0.5
            y_coords = trajectory[:, 0] + 0.5
            
            # Plot path
            ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7)
            ax.plot(x_coords, y_coords, 'ro', markersize=8, alpha=0.5)
            
            # Add step numbers
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.text(x, y, str(i), ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
            
            ax.set_title(f"{name}\nSteps: {data['steps']}, Reward: {data['reward']:.1f}")
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes


def visualize_episode(env, trajectory, title="Episode Visualization"):
    """Convenience function to visualize a single episode"""
    viz = GridWorldVisualizer(env.model.rows, env.model.cols)
    fig, ax = viz.plot_trajectory(trajectory, title)
    plt.show()
    return fig, ax