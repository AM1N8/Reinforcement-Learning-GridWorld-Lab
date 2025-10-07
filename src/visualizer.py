import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import os
import json

class GridWorldVisualizer:
    """Enhanced visualizer with advanced metrics and animation support"""
    
    def __init__(self, rows=4, cols=4, goal_states=None, start_states=None, obstacles=None,
                 alpha=None, gamma=None, epsilon=None, lambda_val=None):
        self.rows = rows
        self.cols = cols
        self.goal_states = goal_states if goal_states else [(rows-1, cols-1)]
        self.start_states = start_states if start_states else [(0, 0)]
        self.obstacles = obstacles if obstacles else set()
        
        # Store hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        
        # Animation control
        self.anim_paused = False
        self.current_frame = 0
    
    def plot_grid(self, ax=None):
        """Plot grid with all starts, goals, and obstacles"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Draw grid lines
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=1)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=1)
        
        # Draw obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle((obs[1], obs[0]), 1, 1,
                                     linewidth=2, edgecolor='black',
                                     facecolor='gray', alpha=0.8)
            ax.add_patch(rect)
            ax.text(obs[1] + 0.5, obs[0] + 0.5, 'X', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='white')
        
        # Draw goals
        for goal in self.goal_states:
            rect = patches.Rectangle((goal[1], goal[0]), 1, 1,
                                     linewidth=2, edgecolor='red',
                                     facecolor='lightcoral', alpha=0.5)
            ax.add_patch(rect)
            ax.text(goal[1] + 0.5, goal[0] + 0.5, 'G', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='darkred')
        
        # Draw starts
        for start in self.start_states:
            if start not in self.goal_states:
                rect = patches.Rectangle((start[1], start[0]), 1, 1,
                                         linewidth=2, edgecolor='green',
                                         facecolor='lightgreen', alpha=0.5)
                ax.add_patch(rect)
                ax.text(start[1] + 0.5, start[0] + 0.5, 'S', ha='center', va='center',
                       fontsize=20, fontweight='bold', color='darkgreen')
        
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
        """Plot agent's trajectory"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
        
        trajectory = np.array(trajectory)
        x_coords = trajectory[:, 1] + 0.5
        y_coords = trajectory[:, 0] + 0.5
        
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(x_coords, y_coords, 'bo', markersize=8, alpha=0.5)
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    def plot_value_function(self, V, title="Value Function"):
        """Visualize value function as heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        V_masked = V.copy()
        for obs in self.obstacles:
            V_masked[obs[0], obs[1]] = np.nan
        
        im = ax.imshow(V_masked, cmap='RdYlGn', origin='upper', aspect='auto')
        
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.obstacles:
                    ax.text(j, i, f'{V[i, j]:.2f}',
                           ha="center", va="center", color="black",
                           fontsize=10, fontweight='bold')
        
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1)
        
        for obs in self.obstacles:
            ax.text(obs[1], obs[0], 'X', ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
        
        for start in self.start_states:
            ax.text(start[1], start[0], 'S', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color='blue')
        for goal in self.goal_states:
            ax.text(goal[1], goal[0], 'G', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color='green')
        
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        hparam_str = self._get_hyperparam_str()
        if hparam_str:
            title += f"\n({hparam_str})"
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='State Value')
        plt.tight_layout()
        
        return fig, ax
    
    def plot_policy(self, policy, title="Policy"):
        """Visualize policy with arrows"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self.plot_grid(ax)
        
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.goal_states or (i, j) in self.obstacles:
                    continue
                
                action = int(policy[i, j])
                ax.text(j + 0.5, i + 0.5, arrows[action],
                       ha='center', va='center', fontsize=24,
                       color='darkblue', fontweight='bold')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_q_values(self, Q, model, title="Q-Values"):
        """Visualize Q-values for all state-action pairs"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        all_q_values = []
        for i in range(self.rows):
            for j in range(self.cols):
                state_key = (i, j)
                if state_key in Q and state_key not in self.obstacles:
                    all_q_values.extend(Q[state_key])
        
        if all_q_values:
            vmin, vmax = min(all_q_values), max(all_q_values)
        else:
            vmin, vmax = -10, 10
        
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=2)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=2)
        
        action_positions = {
            0: (0.5, 0.2),   # UP
            1: (0.5, 0.8),   # DOWN
            2: (0.2, 0.5),   # LEFT
            3: (0.8, 0.5),   # RIGHT
        }
        
        for i in range(self.rows):
            for j in range(self.cols):
                state_key = (i, j)
                
                if state_key in self.obstacles:
                    ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center',
                           fontsize=16, fontweight='bold', color='gray')
                    continue
                
                if state_key in self.goal_states:
                    ax.text(j + 0.5, i + 0.5, 'GOAL', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='green')
                    continue
                
                if state_key in Q:
                    q_values = Q[state_key]
                    
                    for action in range(model.num_actions):
                        q_val = q_values[action]
                        rel_x, rel_y = action_positions[action]
                        
                        color_intensity = (q_val - vmin) / (vmax - vmin + 1e-8)
                        color = plt.cm.RdYlGn(color_intensity)
                        
                        ax.text(j + rel_x, i + rel_y, f'{q_val:.1f}',
                               ha='center', va='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor=color, alpha=0.7),
                               fontweight='bold')
        
        for start in self.start_states:
            ax.text(start[1] + 0.5, start[0] + 0.1, 'START', ha='center', va='center',
                   fontsize=8, fontweight='bold', color='blue')
        
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
        
        return fig
    
    def plot_learning_curve(self, rewards, title="Learning Curve", window=100):
        """Plot learning curve with moving average"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
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
        
        return fig
    
    def plot_epsilon_decay(self, epsilon_history, title="Epsilon Decay"):
        """Plot epsilon decay curve"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        episodes = range(1, len(epsilon_history) + 1)
        ax.plot(episodes, epsilon_history, linewidth=2, color='purple')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (ε)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        
        return fig
    
    def plot_exploration_exploitation(self, exploration_counts, exploitation_counts, 
                                     title="Exploration vs Exploitation"):
        """Plot exploration vs exploitation ratio over time"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        total_counts = np.array(exploration_counts) + np.array(exploitation_counts)
        exploration_ratio = np.array(exploration_counts) / (total_counts + 1e-8)
        
        episodes = range(1, len(exploration_ratio) + 1)
        ax.plot(episodes, exploration_ratio, linewidth=2, color='green', 
               label='Exploration Ratio')
        ax.plot(episodes, 1 - exploration_ratio, linewidth=2, color='orange',
               label='Exploitation Ratio')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Ratio')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        
        return fig
    
    def plot_steps_vs_grid_size(self, grid_sizes, steps_data, title="Steps vs Grid Size"):
        """Plot steps per episode vs grid size"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.boxplot(steps_data, labels=grid_sizes)
        
        ax.set_xlabel('Grid Size (rows × cols)')
        ax.set_ylabel('Steps per Episode')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig
    
    def plot_return_vs_grid_size(self, grid_sizes, return_data, title="Average Return vs Grid Size"):
        """Plot average return vs grid size with violin plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        parts = ax.violinplot(return_data, positions=range(len(grid_sizes)), 
                             showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(grid_sizes)))
        ax.set_xticklabels(grid_sizes)
        ax.set_xlabel('Grid Size (rows × cols)')
        ax.set_ylabel('Total Return')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig
    
    def plot_training_summary(self, metrics_dict, title="Training Performance Summary"):
        """Create comprehensive training summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Learning curve
        ax1 = fig.add_subplot(gs[0, :2])
        rewards = metrics_dict.get('episode_rewards', [])
        episodes = range(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, alpha=0.3, color='blue')
        if len(rewards) >= 50:
            ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax1.plot(range(50, len(rewards) + 1), ma, 'r-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Learning Curve')
        ax1.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax2 = fig.add_subplot(gs[0, 2])
        epsilon_hist = metrics_dict.get('epsilon_history', [])
        if epsilon_hist:
            ax2.plot(range(1, len(epsilon_hist) + 1), epsilon_hist, 'purple', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Epsilon Decay')
            ax2.grid(True, alpha=0.3)
        
        # Steps per episode
        ax3 = fig.add_subplot(gs[1, :2])
        steps = metrics_dict.get('episode_steps', [])
        if steps:
            ax3.plot(range(1, len(steps) + 1), steps, 'g-', alpha=0.5)
            if len(steps) >= 50:
                ma_steps = np.convolve(steps, np.ones(50)/50, mode='valid')
                ax3.plot(range(50, len(steps) + 1), ma_steps, 'darkgreen', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Steps')
            ax3.set_title('Steps per Episode')
            ax3.grid(True, alpha=0.3)
        
        # Loss history
        ax4 = fig.add_subplot(gs[1, 2])
        loss_hist = metrics_dict.get('loss_history', [])
        if loss_hist:
            ax4.plot(loss_hist, 'orange', alpha=0.5)
            if len(loss_hist) >= 100:
                ma_loss = np.convolve(loss_hist, np.ones(100)/100, mode='valid')
                ax4.plot(range(100, len(loss_hist) + 1), ma_loss, 'red', linewidth=2)
            ax4.set_xlabel('Update Step')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss')
            ax4.grid(True, alpha=0.3)
        
        # Exploration vs Exploitation
        ax5 = fig.add_subplot(gs[2, :])
        explore_cnt = metrics_dict.get('exploration_count', 0)
        exploit_cnt = metrics_dict.get('exploitation_count', 0)
        total = explore_cnt + exploit_cnt
        if total > 0:
            labels = ['Exploration', 'Exploitation']
            sizes = [explore_cnt, exploit_cnt]
            colors = ['lightgreen', 'lightcoral']
            ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax5.set_title(f'Exploration vs Exploitation (Total: {total})')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
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
            
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
            ax.plot(x_coords, y_coords, 'bo', markersize=8, alpha=0.5)
            
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.text(x, y, str(i), ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
            
            ax.set_title(f"{name}\nSteps: {data['steps']}, Reward: {data['reward']:.1f}")
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def _get_hyperparam_str(self):
        """Generate hyperparameter string for display"""
        hparams = []
        if self.alpha is not None: 
            alpha_val = f"{self.alpha:.2f}" if isinstance(self.alpha, float) else "Decaying"
            hparams.append(f"$\\alpha={alpha_val}$")
        
        if self.gamma is not None: 
            hparams.append(f"$\\gamma={self.gamma:.2f}$")
        
        if self.epsilon is not None:
            epsilon_val = f"{self.epsilon:.2f}" if isinstance(self.epsilon, float) else "Decaying"
            hparams.append(f"$\\epsilon={epsilon_val}$")
        
        if self.lambda_val is not None: 
            hparams.append(f'$\\lambda$={self.lambda_val:.2f}')
        
        return ", ".join(hparams)

    def save_figure(self, fig, filename, savedir="../visualizations"):
        """Save figure to disk"""
        os.makedirs(savedir, exist_ok=True)
        filepath = os.path.join(savedir, filename)
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to: {filepath}")
        except Exception as e:
            print(f"Error saving figure to {filepath}: {e}")

    def animate_episode(self, trajectory, interval_ms=500, title="Episode Animation", 
                       save_path=None):
        """Animate agent's trajectory step by step with optional save"""
        if not trajectory:
            print("Trajectory is empty. Cannot animate.")
            return None, None
    
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
    
        line, = ax.plot([], [], 'r-', linewidth=3, alpha=0.7, label='Path')
        marker, = ax.plot([], [], 'bo', markersize=15, alpha=0.8, label='Agent')
    
        x_coords = [t[1] + 0.5 for t in trajectory]
        y_coords = [t[0] + 0.5 for t in trajectory]
    
        hparam_str = self._get_hyperparam_str()
        base_title = title
        if hparam_str:
            base_title += f"\n({hparam_str})"
        
        ax.set_title(base_title)

        def init():
            line.set_data([], [])
            marker.set_data([], [])
            return line, marker

        def update(frame):
            current_x = x_coords[:frame+1]
            current_y = y_coords[:frame+1]
            line.set_data(current_x, current_y)
            marker.set_data([current_x[-1]], [current_y[-1]])
            
            current_step_info = f"Step {frame}/{len(trajectory)-1} | Pos: ({trajectory[frame][0]},{trajectory[frame][1]})"
            ax.set_title(f"{base_title}\n{current_step_info}") 
            
            return line, marker

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                     init_func=init, blit=False,
                                     interval=interval_ms, repeat=False)
        
        if save_path:
            print(f"Saving animation to: {save_path}")
            try:
                ani.save(save_path, writer='pillow', fps=1000/interval_ms)
                print("Animation saved successfully!")
            except Exception as e:
                print(f"Error saving animation: {e}. Ensure 'Pillow' is installed.")

        print("Starting animation. Close the plot window to continue.")
        plt.show(block=True)
        return ani, fig
    
    def animate_learning_process(self, episode_trajectories, episode_rewards, 
                                 q_values_history=None, interval_ms=800,
                                 title="Learning Process Animation", save_path=None):
        """
        Animate the learning process episode by episode
        Shows trajectory and Q-values evolution
        """
        if not episode_trajectories:
            print("No trajectories to animate.")
            return None, None
        
        # Setup figure with subplots
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax_grid = fig.add_subplot(gs[0])
        ax_reward = fig.add_subplot(gs[1])
        
        # Prepare grid plot
        self.plot_grid(ax_grid)
        line, = ax_grid.plot([], [], 'r-', linewidth=2, alpha=0.7)
        marker, = ax_grid.plot([], [], 'bo', markersize=12, alpha=0.8)
        
        # Prepare reward plot
        ax_reward.set_xlim(0, len(episode_trajectories))
        ax_reward.set_ylim(min(episode_rewards) - 5, max(episode_rewards) + 5)
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Total Reward')
        ax_reward.set_title('Reward Over Time')
        ax_reward.grid(True, alpha=0.3)
        reward_line, = ax_reward.plot([], [], 'g-', linewidth=2)
        reward_scatter = ax_reward.scatter([], [], c='red', s=100, zorder=5)
        
        def init():
            line.set_data([], [])
            marker.set_data([], [])
            reward_line.set_data([], [])
            reward_scatter.set_offsets(np.empty((0, 2)))
            return line, marker, reward_line, reward_scatter
        
        def update(episode):
            # Update trajectory
            trajectory = episode_trajectories[episode]
            x_coords = [t[1] + 0.5 for t in trajectory]
            y_coords = [t[0] + 0.5 for t in trajectory]
            
            line.set_data(x_coords, y_coords)
            if x_coords:
                marker.set_data([x_coords[-1]], [y_coords[-1]])
            
            ax_grid.set_title(f"Episode {episode + 1}/{len(episode_trajectories)}\n"
                            f"Steps: {len(trajectory)}, Reward: {episode_rewards[episode]:.1f}")
            
            # Update reward plot
            episodes_so_far = list(range(1, episode + 2))
            rewards_so_far = episode_rewards[:episode + 1]
            reward_line.set_data(episodes_so_far, rewards_so_far)
            reward_scatter.set_offsets(np.c_[episode + 1, episode_rewards[episode]])
            
            return line, marker, reward_line, reward_scatter
        
        ani = animation.FuncAnimation(fig, update, frames=len(episode_trajectories),
                                     init_func=init, blit=False,
                                     interval=interval_ms, repeat=True)
        
        if save_path:
            print(f"Saving learning process animation to: {save_path}")
            try:
                ani.save(save_path, writer='pillow', fps=1000/interval_ms)
                print("Animation saved successfully!")
            except Exception as e:
                print(f"Error saving animation: {e}")
        
        plt.tight_layout()
        print("Starting learning animation. Close window to continue.")
        plt.show(block=True)
        return ani, fig
    
    def load_and_visualize_log(self, log_filepath, save_dir=None):
        """Load training log and generate all visualizations"""
        with open(log_filepath, 'r') as f:
            log_data = json.load(f)
        
        print(f"Loaded training log from {log_filepath}")
        print(f"Hyperparameters: {log_data.get('hyperparameters', {})}")
        
        figures = {}
        
        # Learning curve
        if 'episode_rewards' in log_data:
            fig = self.plot_learning_curve(log_data['episode_rewards'], 
                                          "Learning Curve from Log")
            figures['learning_curve'] = fig
            if save_dir:
                self.save_figure(fig, "learning_curve_from_log.png", save_dir)
        
        # Epsilon decay
        if 'epsilon_history' in log_data:
            fig = self.plot_epsilon_decay(log_data['epsilon_history'],
                                         "Epsilon Decay from Log")
            figures['epsilon_decay'] = fig
            if save_dir:
                self.save_figure(fig, "epsilon_decay_from_log.png", save_dir)
        
        # Training summary
        fig = self.plot_training_summary(log_data, "Training Summary from Log")
        figures['training_summary'] = fig
        if save_dir:
            self.save_figure(fig, "training_summary_from_log.png", save_dir)
        
        return figures


def visualize_episode(env, trajectory, title="Episode Visualization"):
    """Convenience function to visualize a single episode"""
    viz = GridWorldVisualizer(env.model.rows, env.model.cols,
                             goal_states=env.model.goal_states,
                             obstacles=env.model.obstacles)
    fig, ax = viz.plot_trajectory(trajectory, title)
    plt.show()
    return fig, ax