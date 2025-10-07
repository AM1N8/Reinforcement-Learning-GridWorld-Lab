import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class GridWorldVisualizer:
    
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
    
    def plot_grid(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=1)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=1)
    
        goal = patches.Rectangle((self.cols - 1, self.rows - 1), 1, 1, 
                                  linewidth=2, edgecolor='green', 
                                  facecolor='lightgreen', alpha=0.5)
        ax.add_patch(goal)
        ax.text(self.cols - 0.5, self.rows - 0.5, 'G', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='darkgreen')
    
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
        
        trajectory = np.array(trajectory)
        x_coords = trajectory[:, 1] + 0.5  
        y_coords = trajectory[:, 0] + 0.5  
        
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(x_coords, y_coords, 'ro', markersize=8, alpha=0.5)
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.text(x, y, str(i), ha='center', va='center', 
                   fontsize=8, color='white', fontweight='bold')
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    def plot_state(self, state, title="Current State"):

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot_grid(ax)
        

        x = state[1] + 0.5
        y = state[0] + 0.5
        ax.plot(x, y, 'ro', markersize=20, label='Agent')
        ax.text(x, y, 'A', ha='center', va='center', 
               fontsize=16, color='white', fontweight='bold')
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax



def visualize_episode(env, trajectory, title="Episode Visualization"):

    viz = GridWorldVisualizer(env.model.rows, env.model.cols)
    fig, ax = viz.plot_trajectory(trajectory, title)
    plt.show()
    return fig, ax