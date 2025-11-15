"""
Evaluate a trained RL model.
"""

import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.register_envs import CUSTOM_ENVS


# Map of algorithm names to classes
ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def detect_algorithm(model_path: Path) -> str:
    """
    Detect algorithm type from model filename.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Algorithm name (ppo, a2c, or dqn)
    """
    model_name = model_path.stem.lower()
    
    for algo in ALGO_MAP.keys():
        if algo in model_name:
            return algo
    
    logger.warning(f"Could not detect algorithm from filename: {model_path.name}")
    logger.info("Defaulting to PPO. Use --algo to specify manually.")
    return "ppo"


def evaluate_model(
    model_path: Path,
    env_id: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    algo: str = None,
) -> dict:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        env_id: Gymnasium environment ID
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        render: Render environment during evaluation
        algo: Algorithm name (ppo, a2c, dqn). Auto-detected if None.
    
    Returns:
        Dictionary with evaluation statistics
    """
    # Setup logging
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Environment: {env_id}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Deterministic: {deterministic}")
    
    # Detect or use specified algorithm
    if algo is None:
        algo = detect_algorithm(model_path)
    
    algo = algo.lower()
    if algo not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(ALGO_MAP.keys())}")
    
    logger.info(f"Algorithm: {algo.upper()}")
    
    # Load model
    logger.info("Loading model...")
    AlgoClass = ALGO_MAP[algo]
    model = AlgoClass.load(model_path)
    logger.success("Model loaded successfully!")
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    # Evaluate
    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        logger.info(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Reward={episode_reward:.2f}, Steps={steps}"
        )
    
    env.close()
    
    # Calculate statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    logger.info(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    logger.info(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model (.zip file)"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=list(ALGO_MAP.keys()),
        help="Algorithm type (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    evaluate_model(
        model_path=args.model_path,
        env_id=args.env,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()