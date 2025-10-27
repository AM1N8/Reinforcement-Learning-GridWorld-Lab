import os
import numpy as np
from typing import Dict, Any
from ..algorithms.base_algorithm import BaseAlgorithm
from ..environment import GridWorld
from ..utils.replay_buffer import ReplayBuffer
from ..utils.logger import Logger


class Trainer:
    """
    Algorithm-agnostic trainer for reinforcement learning.
    
    This trainer works with any algorithm that implements BaseAlgorithm.
    """
    
    def __init__(self, env: GridWorld, algorithm: BaseAlgorithm,
                 replay_buffer: ReplayBuffer, logger: Logger, config: Dict[str, Any]):
        """
        Args:
            env: Training environment
            algorithm: RL algorithm instance
            replay_buffer: Experience replay buffer
            logger: Logger instance
            config: Training configuration
        """
        self.env = env
        self.algorithm = algorithm
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.config = config
        
        # Training parameters
        self.episodes = config['train']['episodes']
        self.batch_size = config['train']['batch_size']
        self.eval_interval = config['train'].get('eval_interval', 50)
        self.save_interval = config['train'].get('save_interval', 100)
        self.save_dir = config['logging']['save_dir']
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.episodes} episodes...")
        print(f"Algorithm: {self.config['algorithm']}")
        print(f"Environment: GridWorld {self.config['env']['size']}x{self.config['env']['size']}")
        print(f"Observation space: {self.env.observation_space_shape}")
        print(f"Action space: {self.env.action_space_n}")
        print("-" * 70)
        
        best_eval_reward = -float('inf')
        
        for episode in range(1, self.episodes + 1):
            # Run episode
            episode_reward, episode_steps, episode_loss = self._run_episode(train=True)
            
            # Get algorithm metrics
            metrics = self.algorithm.get_metrics()
            
            # Log
            self.logger.log_episode(
                episode, episode_reward, episode_steps,
                metrics.get('epsilon', 0), episode_loss
            )
            
            # Evaluate
            if episode % self.eval_interval == 0:
                eval_reward, eval_steps = self._evaluate()
                print(f"  → Evaluation: Reward={eval_reward:.2f}, Steps={eval_steps}")
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_path = os.path.join(self.save_dir, 'best_model.pt')
                    self.algorithm.save(best_path)
                    print(f"  → New best model saved: {best_path} (reward: {eval_reward:.2f})")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                save_path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pt')
                self.algorithm.save(save_path)
                print(f"  → Checkpoint saved: {save_path}")
        
        # Final save and plots
        final_path = os.path.join(self.save_dir, 'final_model.pt')
        self.algorithm.save(final_path)
        self.logger.plot_metrics()
        self.logger.close()
        
        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"Final model saved: {final_path}")
        print(f"Best model saved: {os.path.join(self.save_dir, 'best_model.pt')}")
        print(f"Metrics plot saved: {self.logger.log_dir}/training_metrics.png")
    
    def _run_episode(self, train: bool = True) -> tuple:
        """Run single episode."""
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        done = False
        
        # Debug: Check if agent is stuck
        if train and episode_steps == 0 and np.random.random() < 0.01:  # 1% chance
            print(f"Debug: Start state - Agent: {self.env.agent_pos}, Goal: {self.env.goal_pos}")
        
        while not done:
            # Select action
            action = self.algorithm.act(state, train=train)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if train:
                # Store experience
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Learn from batch
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.algorithm.learn(batch)
                    episode_losses.append(metrics.get('loss', 0))
            
            state = next_state
            
            # Early termination if stuck for too long
            if episode_steps > self.env.max_steps * 2:
                print(f"Warning: Episode {episode_steps} steps - forcing termination")
                break
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        return episode_reward, episode_steps, avg_loss
    
    def _evaluate(self, num_episodes: int = 10) -> tuple:
        """Evaluate current policy."""
        rewards = []
        steps_list = []
        
        for _ in range(num_episodes):
            reward, steps, _ = self._run_episode(train=False)
            rewards.append(reward)
            steps_list.append(steps)
        
        return np.mean(rewards), np.mean(steps_list)