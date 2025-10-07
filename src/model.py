import numpy as np
from typing import List, Tuple, Optional


class GridWorldModel:
    def __init__(
        self, 
        rows: int = 5, 
        cols: int = 5,
        start_states: Optional[List[Tuple[int, int]]] = None,
        goal_states: Optional[List[Tuple[int, int]]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goal_reward: float = 10.0,
        step_penalty: float = -1.0,
        obstacle_penalty: float = -5.0
        ):
        self.rows = rows
        self.cols = cols
        
        # Set default start states
        if start_states is None:
            self.start_states = [(0, 0)]
        else:
            self.start_states = [tuple(s) for s in start_states]
        
        # Set default goal states
        if goal_states is None:
            self.goal_states = [(rows - 1, cols - 1)]
        else:
            self.goal_states = [tuple(g) for g in goal_states]
        
        # Set obstacles
        if obstacles is None:
            self.obstacles = set()
        else:
            self.obstacles = set(tuple(o) for o in obstacles)
        
        # Validate that goals and starts are not obstacles
        for goal in self.goal_states:
            if goal in self.obstacles:
                raise ValueError(f"Goal state {goal} cannot be an obstacle")
        for start in self.start_states:
            if start in self.obstacles:
                raise ValueError(f"Start state {start} cannot be an obstacle")
        
        # Rewards
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        
        # Actions
        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
        }
        
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.num_actions = len(self.actions)
    
    def get_random_start_state(self) -> np.ndarray:
        """Get a random start state from available start states"""
        start = self.start_states[np.random.randint(len(self.start_states))]
        return np.array(start)
    
    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        direction = self.actions[action]
        new_state = state + np.array(direction)
        
        if self.is_valid_state(new_state) and not self.is_obstacle(new_state):
            return new_state
        else:
            return state  # Stay in place if invalid move
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        """Check if state is within grid bounds"""
        return (0 <= state[0] < self.rows) and (0 <= state[1] < self.cols)
    
    def is_obstacle(self, state: np.ndarray) -> bool:
        """Check if state is an obstacle"""
        return tuple(state) in self.obstacles
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if state is a goal state"""
        return tuple(state) in self.goal_states
    
    def get_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        if self.is_terminal(next_state):
            return self.goal_reward
        
        # Check if we tried to move into an obstacle or boundary
        direction = self.actions[action]
        attempted_state = state + np.array(direction)
        
        if (not self.is_valid_state(attempted_state) or 
            self.is_obstacle(attempted_state)):
            return self.obstacle_penalty
        
        return self.step_penalty
    
    def get_all_states(self) -> List[Tuple[int, int]]:
        """Get all valid (non-obstacle) states"""
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.obstacles:
                    states.append((i, j))
        return states
    
    def __repr__(self) -> str:
        """String representation of the grid"""
        lines = [f"GridWorld({self.rows}x{self.cols})"]
        lines.append(f"Start states: {self.start_states}")
        lines.append(f"Goal states: {self.goal_states}")
        lines.append(f"Obstacles: {len(self.obstacles)} obstacles")
        return "\n".join(lines)