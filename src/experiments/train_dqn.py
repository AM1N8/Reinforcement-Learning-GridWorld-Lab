import argparse
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.environment import GridWorld
from src.algorithms.dqn import DQN
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger
from src.trainer.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train DQN on GridWorld')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create environment
    env = GridWorld(
        size=config['env']['size'],
        num_obstacles=config['env']['obstacles'],
        seed=config['env']['seed']
    )
    
    # Create algorithm
    if config['algorithm'] == 'DQN':
        algorithm = DQN(
            state_dim=env.observation_space_shape[0],
            action_dim=env.action_space_n,
            lr=config['train']['lr'],
            gamma=config['train']['gamma'],
            epsilon_start=config['train']['epsilon_start'],
            epsilon_end=config['train']['epsilon_end'],
            epsilon_decay=config['train']['epsilon_decay'],
            target_update=config['train']['target_update']
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config['train']['replay_size'],
        state_dim=env.observation_space_shape[0]
    )
    
    # Create logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    # Create trainer
    trainer = Trainer(env, algorithm, replay_buffer, logger, config)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()