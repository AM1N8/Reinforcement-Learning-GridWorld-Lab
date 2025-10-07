import numpy as np
from collections import defaultdict


class ValueIteration:
    """Value Iteration Algorithm"""
    
    def __init__(self, model, gamma=0.99, theta=1e-6):
        self.model = model
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((model.rows, model.cols))
        self.policy = np.zeros((model.rows, model.cols), dtype=int)
    
    def train(self, max_iterations=1000):
        """Run value iteration"""
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.zeros_like(self.V)
            
            for i in range(self.model.rows):
                for j in range(self.model.cols):
                    state = np.array([i, j])
                    
                    if self.model.is_terminal(state):
                        continue
                    
                    # Compute value for all actions
                    action_values = []
                    for action in range(self.model.num_actions):
                        next_state = self.model.transition(state, action)
                        reward = self.model.get_reward(state, action, next_state)
                        value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                        action_values.append(value)
                    
                    V_new[i, j] = max(action_values)
                    delta = max(delta, abs(V_new[i, j] - self.V[i, j]))
            
            self.V = V_new
            
            if delta < self.theta:
                print(f"Value Iteration converged in {iteration + 1} iterations")
                break
        
        # Extract policy
        self._extract_policy()
        return self.V, self.policy
    
    def _extract_policy(self):
        """Extract policy from value function"""
        for i in range(self.model.rows):
            for j in range(self.model.cols):
                state = np.array([i, j])
                
                if self.model.is_terminal(state):
                    continue
                
                action_values = []
                for action in range(self.model.num_actions):
                    next_state = self.model.transition(state, action)
                    reward = self.model.get_reward(state, action, next_state)
                    value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    action_values.append(value)
                
                self.policy[i, j] = np.argmax(action_values)
    
    def get_action(self, state, training=False):
        """Get action from learned policy"""
        return self.policy[state[0], state[1]]


class PolicyIteration:
    """Policy Iteration Algorithm"""
    
    def __init__(self, model, gamma=0.99, theta=1e-6):
        self.model = model
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((model.rows, model.cols))
        self.policy = np.random.randint(0, model.num_actions, (model.rows, model.cols))
    
    def _policy_evaluation(self):
        """Evaluate current policy"""
        while True:
            delta = 0
            V_new = np.zeros_like(self.V)
            
            for i in range(self.model.rows):
                for j in range(self.model.cols):
                    state = np.array([i, j])
                    
                    if self.model.is_terminal(state):
                        continue
                    
                    action = self.policy[i, j]
                    next_state = self.model.transition(state, action)
                    reward = self.model.get_reward(state, action, next_state)
                    V_new[i, j] = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    delta = max(delta, abs(V_new[i, j] - self.V[i, j]))
            
            self.V = V_new
            
            if delta < self.theta:
                break
    
    def _policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        
        for i in range(self.model.rows):
            for j in range(self.model.cols):
                state = np.array([i, j])
                
                if self.model.is_terminal(state):
                    continue
                
                old_action = self.policy[i, j]
                
                action_values = []
                for action in range(self.model.num_actions):
                    next_state = self.model.transition(state, action)
                    reward = self.model.get_reward(state, action, next_state)
                    value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    action_values.append(value)
                
                self.policy[i, j] = np.argmax(action_values)
                
                if old_action != self.policy[i, j]:
                    policy_stable = False
        
        return policy_stable
    
    def train(self, max_iterations=100):
        """Run policy iteration"""
        for iteration in range(max_iterations):
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            
            if policy_stable:
                print(f"Policy Iteration converged in {iteration + 1} iterations")
                break
        
        return self.V, self.policy
    
    def get_action(self, state, training=False):
        """Get action from learned policy"""
        return self.policy[state[0], state[1]]


class MonteCarlo:
    """Monte Carlo Control with Epsilon-Greedy"""
    
    def __init__(self, model, gamma=0.99, epsilon=0.1):
        self.model = model
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(model.num_actions))
        self.returns = defaultdict(list)
        self.policy = {}
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        state_key = tuple(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.model.num_actions)
        else:
            return np.argmax(self.Q[state_key])
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train using Monte Carlo control"""
        for episode in range(num_episodes):
            episode_data = []
            state, _ = env.reset()
            
            # Generate episode
            for step in range(max_steps):
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_data.append((state.copy(), action, reward))
                state = next_state
                
                if terminated:
                    break
            
            # Calculate returns and update Q-values
            G = 0
            visited = set()
            
            for t in reversed(range(len(episode_data))):
                state, action, reward = episode_data[t]
                state_key = tuple(state)
                G = reward + self.gamma * G
                
                # First-visit MC
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    self.returns[(state_key, action)].append(G)
                    self.Q[state_key][action] = np.mean(self.returns[(state_key, action)])
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")
        
        return self.Q


class QLearning:
    """Q-Learning Algorithm"""
    
    def __init__(self, model, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(model.num_actions))
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        state_key = tuple(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.model.num_actions)
        else:
            return np.argmax(self.Q[state_key])
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train using Q-Learning"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                state_key = tuple(state)
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state_key = tuple(next_state)
                
                # Q-Learning update
                best_next_action = np.argmax(self.Q[next_state_key])
                td_target = reward + self.gamma * self.Q[next_state_key][best_next_action]
                td_error = td_target - self.Q[state_key][action]
                self.Q[state_key][action] += self.alpha * td_error
                
                total_reward += reward
                state = next_state
                
                if terminated:
                    break
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return self.Q, episode_rewards
    

class SarsaLambda:
    """SARSA(lambda) Algorithm (On-Policy TD Control with Eligibility Traces)"""
    
    def __init__(self, model, alpha=0.1, gamma=0.99, epsilon=0.1, lambda_val=0.9):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_val = lambda_val  # The lambda parameter for eligibility traces
        
        # Initialize Q-table (action-value function)
        self.Q = defaultdict(lambda: np.zeros(model.num_actions))
        
        # Initialize E-table (eligibility traces)
        self.E = defaultdict(lambda: np.zeros(model.num_actions))
        
    def get_action(self, state, training=False):
        """Epsilon-greedy policy"""
        state_key = tuple(state)
        
        if training and np.random.rand() < self.epsilon:
            # Exploration: Choose random action
            return np.random.randint(self.model.num_actions)
        else:
            # Exploitation: Choose greedy action
            # Handle states not yet in Q-table (defaults to random choice)
            if state_key not in self.Q:
                return np.random.randint(self.model.num_actions)
            return np.argmax(self.Q[state_key])

    def reset_traces(self):
        """Reset eligibility traces at the start of an episode"""
        self.E = defaultdict(lambda: np.zeros(self.model.num_actions))

    def train(self, env, num_episodes=1000, max_steps=100):
        """Train using SARSA(lambda)"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            self.reset_traces() # Reset E-traces for new episode
            state, _ = env.reset()
            total_reward = 0
            
            # Select first action using current Q and Epsilon-greedy
            action = self.get_action(state, training=True)
            
            for step in range(max_steps):
                state_key = tuple(state)
                
                # Take step
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state_key = tuple(next_state)

                # Select next action (on-policy: SARSA)
                next_action = self.get_action(next_state, training=True)
                
                # Calculate TD Error
                if terminated:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * self.Q[next_state_key][next_action]
                
                td_error = td_target - self.Q[state_key][action]
                
                # Update Eligibility Trace for the current state-action pair
                self.E[state_key][action] += 1
                
                # Update Q-table and Eligibility Traces
                for s_key in self.Q:
                    for a in range(self.model.num_actions):
                        # SARSA(lambda) Update rule
                        self.Q[s_key][a] += self.alpha * td_error * self.E[s_key][a]
                        self.E[s_key][a] *= self.gamma * self.lambda_val # Decay E-trace
                
                total_reward += reward
                state = next_state
                action = next_action # SARSA uses the *next* action for the next step

                if terminated or truncated:
                    # Update E-traces for the final state if needed
                    # The general update loop above handles the last update
                    break
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

        return self.Q, episode_rewards