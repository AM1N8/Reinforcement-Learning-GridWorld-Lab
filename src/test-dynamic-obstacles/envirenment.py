import numpy as np
from typing import Tuple, Dict, Optional , Set
from collections import deque

class GridWorldEnv:
   
    def __init__(self, model=None, rows=5, cols=5, render_mode=None,dynamic_mode:bool=False,num_dynamic_obstacles: int = 5, **model_kwargs):
        
        if model is not None:
            self.model = model
        else:
            from model import GridWorldModel
            self.model = GridWorldModel(rows=rows, cols=cols, **model_kwargs)
        
        self.render_mode = render_mode
        self.dynamic_mode = dynamic_mode
        self.num_dynamic_obstacles = num_dynamic_obstacles

        self.state = None
        self.steps = 0
        self.trajectory = []
        self.max_steps = None


    def _is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> bool:
        """Uses Breadth-First Search (BFS) to check if the goal is reachable from the start state 
           without passing through obstacles."""
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                return True
            
            # Check neighbors (UP, DOWN, LEFT, RIGHT)
            for action in self.model.actions:
                direction = self.model.actions[action]
                next_r, next_c = current[0] + direction[0], current[1] + direction[1]
                next_state = (next_r, next_c)
                
                # Check bounds, if not visited, and if not an obstacle
                if (0 <= next_r < self.model.rows and 
                    0 <= next_c < self.model.cols and
                    next_state not in visited and 
                    next_state not in obstacles):
                    visited.add(next_state)
                    queue.append(next_state)
                    
        return False

    def _randomize_features(self):
        """Randomly set goal and obstacles, ensuring reachability."""
        all_possible_states = [(r, c) for r in range(self.model.rows) for c in range(self.model.cols)]
        
        # 1. Choose a random start state (assuming only one start state for simplicity)
        start_state = self.model.start_states[0]
        
        # 2. Randomly choose a new goal state that is not the start state
        non_start_states = [s for s in all_possible_states if s != start_state]
        new_goal_state = non_start_states[np.random.choice(len(non_start_states))]
        
        # 3. Randomly choose obstacles, ensuring the start and goal are not blocked and are reachable
        valid_feature_states = [s for s in all_possible_states if s != start_state and s != new_goal_state]
        
        new_obstacles = set()
        # Keep trying until a reachable path is found
        while True:
            # Choose N random obstacles from the valid states
            if len(valid_feature_states) >= self.num_dynamic_obstacles:
                random_indices = np.random.choice(len(valid_feature_states), 
                                                size=self.num_dynamic_obstacles, 
                                                replace=False)
                new_obstacles = {valid_feature_states[i] for i in random_indices}
            else:
                # If N is too large, use all available states
                new_obstacles = set(valid_feature_states)

            # Check for reachability
            if self._is_reachable(start_state, new_goal_state, new_obstacles):
                break # Found a valid configuration
            
            # If not reachable, loop and try a different set of obstacles
            # (or you could try a new goal, depending on desired behavior)
        
        # Apply the new features to the model
        self.model._set_features(
            start_states=[start_state], 
            goal_states=[new_goal_state], 
            obstacles=list(new_obstacles)
        )

    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to a random start state"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.dynamic_mode:
            self._randomize_features()

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