import numpy as np


class Agent:
    
    def __init__(self, num_actions: int, policy="random", algorithm=None):
        self.num_actions = num_actions
        self.policy = policy
        self.algorithm = algorithm  
        self.total_reward = 0
        self.steps = 0
    
    def select_action(self, state: np.ndarray, training=False) -> int:
        if self.policy == "learned" and self.algorithm is not None:
            return self.algorithm.get_action(state, training=training)
        elif self.policy == "random":
            return np.random.randint(0, self.num_actions)
        elif self.policy == "right-down":
            if state[1] < 4:
                return 3  
            else:
                return 1  
        else:
            return np.random.randint(0, self.num_actions)
    
    def update(self, state, action, reward, next_state):
        self.total_reward += reward
        self.steps += 1
    
    def reset(self):
        self.total_reward = 0
        self.steps = 0