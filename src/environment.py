import numpy as np
from typing import Tuple, Dict, Optional


class GridWorldEnv:
   
    def __init__(self, model=None, rows=5, cols=5, render_mode=None, **model_kwargs):
        
        if model is not None:
            self.model = model
        else:
            from model import GridWorldModel
            self.model = GridWorldModel(rows=rows, cols=cols, **model_kwargs)
        
        self.render_mode = render_mode
        
        self.state = None
        self.steps = 0
        self.trajectory = []
        self.max_steps = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to a random start state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.state = self.model.get_random_start_state()
        self.steps = 0
        self.trajectory = [self.state.copy()]
        
        if self.render_mode == "human":
            self.render()
        
        info = {"steps": self.steps}
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        next_state = self.model.transition(self.state, action)
        reward = self.model.get_reward(self.state, action, next_state)
        terminated = self.model.is_terminal(next_state)
        
        self.state = next_state
        self.steps += 1
        self.trajectory.append(self.state.copy())
        
        truncated = False
        if self.max_steps is not None and self.steps >= self.max_steps:
            truncated = True
        
        info = {"steps": self.steps, "action": action}
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment to console"""
        if self.render_mode == "human":
            print("\n" + "=" * (self.model.cols * 3 + 2))
            for i in range(self.model.rows):
                row_str = "|"
                for j in range(self.model.cols):
                    cell = (i, j)
                    if np.array_equal(self.state, [i, j]):
                        row_str += " A "  # Agent
                    elif cell in self.model.goal_states:
                        row_str += " G "  # Goal
                    elif cell in self.model.obstacles:
                        row_str += " X "  # Obstacle
                    elif cell in self.model.start_states:
                        row_str += " S "  # Start
                    else:
                        row_str += " . "  # Empty
                row_str += "|"
                print(row_str)
            print("=" * (self.model.cols * 3 + 2))
            print(f"Steps: {self.steps}, Position: ({self.state[0]}, {self.state[1]})")
    
    def get_trajectory(self):
        """Get the trajectory of states visited"""
        return self.trajectory
    
    def close(self):
        """Clean up resources"""
        pass