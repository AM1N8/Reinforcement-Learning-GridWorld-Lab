import argparse
import yaml
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.environment import GridWorld
from src.algorithms.dqn import DQN


def evaluate(config_path: str, checkpoint_path: str, num_episodes: int = 5,
            render: bool = True):
    """Evaluate trained agent."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = GridWorld(
        size=config['env']['size'],
        num_obstacles=config['env']['obstacles'],
        seed=config['env']['seed']
    )
    
    # Create and load algorithm
    algorithm = DQN(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n
    )
    algorithm.load(checkpoint_path)
    
    print(f"Evaluating for {num_episodes} episodes...")
    
    if render:
        plt.figure(figsize=(6, 6))
    
    rewards = []
    steps_list = []
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = algorithm.act(state, train=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        rewards.append(episode_reward)
        steps_list.append(episode_steps)
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    print(f"\nAverage Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Average Steps: {sum(steps_list)/len(steps_list):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.episodes, args.render)