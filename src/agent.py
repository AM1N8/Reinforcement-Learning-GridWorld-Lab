"""
agent.py
Agent that makes decisions and takes actions
"""
import numpy as np


class Agent:
    """
    Agent: Makes decisions and takes actions
    Can have different policies (random, learned, etc.)
    """
    
    def __init__(self, num_actions: int, policy="random"):
        self.num_actions = num_actions
        self.policy = policy
        self.total_reward = 0
        self.steps = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action based on current policy"""
        if self.policy == "random":
            return np.random.randint(0, self.num_actions)
        elif self.policy == "right-down":
            # Simple heuristic: prefer right and down
            if state[1] < 4:  # Not at rightmost column
                return 3  # RIGHT
            else:
                return 1  # DOWN
        else:
            return np.random.randint(0, self.num_actions)
    
    def update(self, state, action, reward, next_state):
        """Update agent's knowledge (for learning agents)"""
        self.total_reward += reward
        self.steps += 1
    
    def reset(self):
        """Reset agent statistics"""
        self.total_reward = 0
        self.steps = 0