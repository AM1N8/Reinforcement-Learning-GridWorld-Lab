import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os
import json

# Experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQN:
    """Deep Q-Network Algorithm - Compatible with framework interface"""
    
    def __init__(self, model, state_dim=2, hidden_dims=[128, 128],
                 alpha=0.001, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, 
                 target_update_freq=10):
        
        self.model = model
        self.state_dim = state_dim
        self.action_dim = model.num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.policy_net = DQNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_net = DQNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics (for compatibility and logging)
        self.episode_count = 0
        self.epsilon_history = []
        self.loss_history = []
        self.exploration_count = 0
        self.exploitation_count = 0
        self.episode_rewards = []
        self.episode_steps = []
        
        # Q-table approximation for visualization (optional)
        self.Q = {}
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection - Interface compatible"""
        if training and random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randrange(self.action_dim)
        else:
            self.exploitation_count += 1
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def select_action(self, state, training=True):
        """Alias for get_action - Framework compatibility"""
        return self.get_action(state, training)
    
    def update(self):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_q_table_approximation(self, state):
        """Update Q-table approximation for visualization"""
        state_key = tuple(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            self.Q[state_key] = q_values.tolist()
    
    def train(self, env, num_episodes=500, max_steps=100, verbose_freq=50):
        """Train DQN agent - Framework compatible interface"""
        episode_rewards = []
        episode_steps = []
        episode_losses = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_loss = []
            
            for step in range(max_steps):
                # Select action
                action = self.get_action(state, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, float(done))
                
                # Update Q-table approximation periodically
                if step % 10 == 0:
                    self.update_q_table_approximation(state)
                
                # Update network
                loss = self.update()
                if loss is not None:
                    episode_loss.append(loss)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store metrics
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)
            self.epsilon_history.append(self.epsilon)
            self.episode_rewards = episode_rewards
            self.episode_steps = episode_steps
            
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_losses.append(avg_loss)
            self.loss_history.extend(episode_loss)
            
            self.episode_count += 1
            
            # Verbose output
            if (episode + 1) % verbose_freq == 0:
                avg_reward = np.mean(episode_rewards[-verbose_freq:])
                avg_steps = np.mean(episode_steps[-verbose_freq:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
        
        # Return both Q and episode_rewards for compatibility
        return self.Q, episode_rewards
    
    def get_exploration_ratio(self):
        """Get exploration vs exploitation ratio"""
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return 0.5
        return self.exploration_count / total
    
    def save_model(self, filepath):
        """Save model weights and training state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'Q': self.Q
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.Q = checkpoint.get('Q', {})
        print(f"Model loaded from {filepath}")
    
    def save_training_log(self, filepath):
        """Save training metrics for later analysis"""
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'epsilon_history': self.epsilon_history,
            'loss_history': self.loss_history,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'buffer_size': self.replay_buffer.buffer.maxlen,
                'target_update_freq': self.target_update_freq
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Training log saved to {filepath}")