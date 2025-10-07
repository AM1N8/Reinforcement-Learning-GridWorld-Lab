import numpy as np
from typing import Tuple, Dict
from model import GridWorldModel


class GridWorldEnv:
    """
    Environment: Manages the current state and orchestrates interactions
    between Agent and Model
    """
    
    def __init__(self, rows=5, cols=5, render_mode=None):
        self.model = GridWorldModel(rows, cols)
        self.render_mode = render_mode
        
        # Environment state
        self.state = None
        self.steps = 0
        self.trajectory = []  # Store trajectory for visualization
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.array([0, 0])
        self.steps = 0
        self.trajectory = [self.state.copy()]
        
        if self.render_mode == "human":
            self.render()
        
        info = {"steps": self.steps}
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Use model to compute transition
        next_state = self.model.transition(self.state, action)
        
        # Get reward from model
        reward = self.model.get_reward(self.state, action, next_state)
        
        # Check if terminal
        terminated = self.model.is_terminal(next_state)
        
        # Update state
        self.state = next_state
        self.steps += 1
        self.trajectory.append(self.state.copy())
        
        truncated = False
        info = {"steps": self.steps, "action": action}
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            print("\n" + "=" * (self.model.cols * 3 + 2))
            for i in range(self.model.rows):
                row_str = "|"
                for j in range(self.model.cols):
                    if np.array_equal(self.state, [i, j]):
                        row_str += " A "
                    elif i == self.model.rows - 1 and j == self.model.cols - 1:
                        row_str += " G "
                    else:
                        row_str += " . "
                row_str += "|"
                print(row_str)
            print("=" * (self.model.cols * 3 + 2))
    
    def get_trajectory(self):
        """Return the trajectory taken"""
        return self.trajectory
    
    def close(self):
        """Clean up resources"""
        pass