import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from matplotlib.patches import Rectangle


class GridWorld:
    """
    A configurable GridWorld environment for reinforcement learning.
    
    The agent must navigate from a random start position to the goal
    while avoiding obstacles.
    """
    
    def __init__(self, size: int = 8, num_obstacles: int = 5, seed: Optional[int] = None):
        """
        Args:
            size: Grid dimensions (size x size)
            num_obstacles: Number of obstacle cells
            seed: Random seed for reproducibility
        """
        self.size = size
        self.num_obstacles = num_obstacles
        self.rng = np.random.RandomState(seed)
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space_n = 4
        # CHANGED: Now includes agent position AND goal position
        self.observation_space_shape = (4,)  # [agent_x, agent_y, goal_x, goal_y]
        
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = set()
        self.steps = 0
        self.max_steps = size * size * 2
        self.prev_pos = None  # For distance-based rewards
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.steps = 0
        
        # Place goal
        self.goal_pos = (
            self.rng.randint(0, self.size),
            self.rng.randint(0, self.size)
        )
        
        # Place obstacles (not on goal)
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            pos = (
                self.rng.randint(0, self.size),
                self.rng.randint(0, self.size)
            )
            if pos != self.goal_pos:
                self.obstacles.add(pos)
        
        # Place agent (not on goal or obstacles)
        while True:
            self.agent_pos = (
                self.rng.randint(0, self.size),
                self.rng.randint(0, self.size)
            )
            if self.agent_pos != self.goal_pos and self.agent_pos not in self.obstacles:
                break
        
        self.prev_pos = self.agent_pos
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return transition.
        
        Returns:
            observation, reward, done, info
        """
        self.steps += 1
        
        # Calculate new position
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        new_pos = (
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        )
        
        # Check boundaries
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos  # Stay in place
        
        # Check obstacles
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # Stay in place
        
        self.prev_pos = self.agent_pos
        self.agent_pos = new_pos
        
        # CHANGED: Improved reward structure
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Larger positive reward for success
            done = True
        elif self.steps >= self.max_steps:
            reward = -2.0  # Clear penalty for timeout
            done = True
        else:
            # Distance-based reward
            old_dist = abs(self.prev_pos[0] - self.goal_pos[0]) + abs(self.prev_pos[1] - self.goal_pos[1])
            new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            
            if new_dist < old_dist:
                reward = 0.3  # Positive reward for getting closer
            elif new_dist > old_dist:
                reward = -0.2  # Negative reward for moving away
            else:
                reward = -0.05  # Very small negative for no progress
            done = False
        
        info = {
            'steps': self.steps,
            'distance_to_goal': abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """CHANGED: Return positional observation with goal information."""
        obs = np.zeros(4, dtype=np.float32)  # [agent_x, agent_y, goal_x, goal_y]
        obs[0] = self.agent_pos[0] / self.size  # Normalized to [0, 1]
        obs[1] = self.agent_pos[1] / self.size
        obs[2] = self.goal_pos[0] / self.size
        obs[3] = self.goal_pos[1] / self.size
        return obs
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        grid = np.zeros((self.size, self.size, 3))
        
        # Draw obstacles (black)
        for obs_pos in self.obstacles:
            grid[obs_pos[0], obs_pos[1]] = [0, 0, 0]
        
        # Draw goal (green)
        grid[self.goal_pos[0], self.goal_pos[1]] = [0, 1, 0]
        
        # Draw agent (blue)
        grid[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 1]
        
        if mode == 'human':
            plt.clf()
            plt.imshow(grid, interpolation='nearest')
            plt.title(f'GridWorld - Steps: {self.steps}, Distance: {abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])}')
            plt.xticks([])
            plt.yticks([])
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            return (grid * 255).astype(np.uint8)
    
    def seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)