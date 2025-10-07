import numpy as np


class GridWorldModel:
    
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        
        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
        }
        
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.num_actions = len(self.actions)
    
    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
    
        direction = self.actions[action]
        new_state = state + np.array(direction)
        
        if self.is_valid_state(new_state):
            return new_state
        else:
            return state  
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        return (0 <= state[0] < self.rows) and (0 <= state[1] < self.cols)
    
    def is_terminal(self, state: np.ndarray) -> bool:
        return np.array_equal(state, [self.rows - 1, self.cols - 1])
    
    def get_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        if self.is_terminal(next_state):
            return 10.0  
        return -1.0  