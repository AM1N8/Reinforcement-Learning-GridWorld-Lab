import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Import your custom GridWorld environment
# This relative import assumes this file is in src/utils/
from ..environment import GridWorld


class GridWorldGymWrapper(gym.Env):
    """
    Wraps the custom GridWorld environment to be compatible with 
    Gymnasium/Stable-Baselines3.
    """
    # Define metadata for rendering
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, env_config: Dict[str, Any], render_mode: Optional[str] = None):
        """
        Args:
            env_config: A dictionary with 'size', 'obstacles', 'seed'
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()
        
        # Initialize your custom environment
        self.grid_world = GridWorld(
            size=env_config.get('size', 8),
            num_obstacles=env_config.get('obstacles', 5),
            seed=env_config.get('seed', None)
        )
        
        self.render_mode = render_mode
        
        # --- Define SB3/Gymnasium-required spaces ---
        
        # 1. Action Space: Discrete set of actions
        # (0=Up, 1=Right, 2=Down, 3=Left)
        self.action_space = spaces.Discrete(self.grid_world.action_space_n)
        
        # 2. Observation Space: CHANGED to match new positional observation
        obs_shape = self.grid_world.observation_space_shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment.
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Set the seed in the custom environment if one is provided
        if seed is not None:
            self.grid_world.seed(seed)
            
        # Call your environment's reset method
        observation = self.grid_world.reset()
        
        # info dict is required by Gymnasium
        info = {} 
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment.
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Call your environment's step method
        observation, reward, done, info = self.grid_world.step(action)
        
        # --- Map 'done' to 'terminated' and 'truncated' ---
        # 'terminated' = The episode ended due to a terminal state (e.g., reached goal)
        # 'truncated'  = The episode ended due to an external limit (e.g., max_steps)
        
        terminated = False
        truncated = False
        
        if done:
            if self.grid_world.agent_pos == self.grid_world.goal_pos:
                # Reached the goal
                terminated = True
            elif info.get('steps', 0) >= self.grid_world.max_steps:
                # Hit max steps
                truncated = True
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment based on the render_mode."""
        # Use your environment's built-in renderer
        return self.grid_world.render(mode=self.render_mode)

    def close(self):
        """Closes any open resources (like matplotlib windows)."""
        if self.render_mode == 'human':
            plt.close()