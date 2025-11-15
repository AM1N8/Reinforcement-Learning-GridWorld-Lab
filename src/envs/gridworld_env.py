"""
Custom GridWorld environment using Gymnasium API.
Features: agent, walls, rewards, terminal states, RGB rendering.
"""

from typing import Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger


class GridWorldEnv(gym.Env):
    """
    A simple GridWorld environment.
    
    The agent starts at (0, 0) and must reach the goal at (7, 7).
    Walls block movement and give negative rewards.
    
    Action Space: Discrete(4) - [UP, RIGHT, DOWN, LEFT]
    Observation Space: Box(0, 7, shape=(2,)) - [row, col]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 8):
        """
        Initialize GridWorld environment.
        
        Args:
            render_mode: One of ["human", "rgb_array", None]
            grid_size: Size of the square grid (default: 8x8)
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )
        
        # Define walls (list of (row, col) positions)
        self.walls = [
            (1, 1), (1, 2), (1, 3),
            (3, 3), (3, 4), (3, 5),
            (5, 1), (5, 2), (5, 3),
            (2, 6), (3, 6), (4, 6),
        ]
        
        # Goal position
        self.goal_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        
        # Agent position (initialized in reset)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        # Rendering
        self.window = None
        self.clock = None
        self.cell_size = 64  # pixels per cell
        
        logger.info(
            f"GridWorld-v0 initialized: {grid_size}x{grid_size} grid, "
            f"{len(self.walls)} walls, goal at {self.goal_pos}"
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial agent position [0, 0]
            info: Additional information (empty dict)
        """
        super().reset(seed=seed)
        
        # Reset agent to start position
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        observation = self.agent_pos.copy()
        info = {}
        
        logger.debug(f"Environment reset. Agent at {self.agent_pos}")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: One of [UP, RIGHT, DOWN, LEFT]
        
        Returns:
            observation: New agent position
            reward: Reward for this step
            terminated: Whether the episode ended (reached goal)
            truncated: Whether episode was truncated (always False here)
            info: Additional information
        """
        # Calculate new position based on action
        new_pos = self.agent_pos.copy()
        
        if action == self.UP:
            new_pos[0] -= 1
        elif action == self.RIGHT:
            new_pos[1] += 1
        elif action == self.DOWN:
            new_pos[0] += 1
        elif action == self.LEFT:
            new_pos[1] -= 1
        
        # Check boundaries
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        
        # Check if new position is a wall
        if tuple(new_pos) in self.walls:
            reward = -1.0  # Penalty for hitting wall
            # Stay in place
        else:
            # Valid move
            self.agent_pos = new_pos
            reward = -0.01  # Small penalty for each step (encourages shorter paths)
        
        # Check if goal reached
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        if terminated:
            reward = 10.0  # Large reward for reaching goal
            logger.info(f"Goal reached! Final position: {self.agent_pos}")
        
        truncated = False
        info = {
            "agent_pos": self.agent_pos.tolist(),
            "action_taken": action,
        }
        
        observation = self.agent_pos.copy()
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self) -> np.ndarray:
        """
        Create RGB array representation of the current state.
        
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Create RGB array
        img_size = self.grid_size * self.cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            img[pos:pos+2, :] = 200  # Horizontal lines
            img[:, pos:pos+2] = 200  # Vertical lines
        
        # Draw walls (dark gray)
        for wall in self.walls:
            row, col = wall
            self._fill_cell(img, row, col, color=(64, 64, 64))
        
        # Draw goal (green)
        goal_row, goal_col = self.goal_pos
        self._fill_cell(img, goal_row, goal_col, color=(0, 255, 0))
        
        # Draw agent (blue)
        agent_row, agent_col = self.agent_pos
        self._fill_cell(img, agent_row, agent_col, color=(0, 0, 255))
        
        if self.render_mode == "human":
            self._render_human(img)
        
        return img
    
    def _fill_cell(self, img: np.ndarray, row: int, col: int, color: Tuple[int, int, int]) -> None:
        """
        Fill a grid cell with the given color.
        
        Args:
            img: Image array to modify
            row: Grid row
            col: Grid column
            color: RGB color tuple
        """
        y_start = row * self.cell_size + 2
        y_end = (row + 1) * self.cell_size - 2
        x_start = col * self.cell_size + 2
        x_end = (col + 1) * self.cell_size - 2
        
        img[y_start:y_end, x_start:x_end] = color
    
    def _render_human(self, img: np.ndarray) -> None:
        """
        Render to human-viewable window using pygame.
        
        Args:
            img: RGB array to display
        """
        try:
            import pygame
        except ImportError:
            logger.warning("pygame not installed, falling back to rgb_array mode")
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(img.shape[:2][::-1])
            pygame.display.set_caption("GridWorld-v0")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Convert numpy array to pygame surface
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        
        # Maintain render FPS
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self) -> None:
        """Close rendering window if open."""
        if self.window is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass
            self.window = None
            self.clock = None
        
        logger.debug("Environment closed")


# Register the environment
gym.register(
    id="GridWorld-v0",
    entry_point="src.envs.gridworld_env:GridWorldEnv",
    max_episode_steps=200,
)

logger.info("GridWorld-v0 registered successfully")