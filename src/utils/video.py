"""
Utilities for recording videos of agent behavior.
"""

from pathlib import Path
from typing import Optional
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.base_class import BaseAlgorithm
from loguru import logger


def setup_video_recorder(
    env: gym.Env,
    video_folder: Path,
    video_length: int = 1000,
    name_prefix: str = "rl-video",
) -> gym.Env:
    """
    Wrap environment with video recording capability.
    
    Args:
        env: Gymnasium environment to wrap
        video_folder: Directory to save videos
        video_length: Maximum length of video in steps
        name_prefix: Prefix for video filenames
    
    Returns:
        Wrapped environment that records videos
    """
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)
    
    # Wrap with RecordVideo
    env = RecordVideo(
        env,
        video_folder=str(video_folder),
        episode_trigger=lambda x: True,  # Record all episodes
        video_length=video_length,
        name_prefix=name_prefix,
    )
    
    logger.info(f"Video recording enabled. Videos will be saved to: {video_folder}")
    
    return env


def record_agent_video(
    model: BaseAlgorithm,
    env_id: str,
    video_path: Path,
    n_episodes: int = 1,
    video_length: int = 1000,
    deterministic: bool = True,
) -> None:
    """
    Record video of trained agent playing in environment.
    
    Args:
        model: Trained SB3 model
        env_id: Gymnasium environment ID
        video_path: Path to save video
        n_episodes: Number of episodes to record
        video_length: Maximum steps per video
        deterministic: Use deterministic policy
    """
    video_path = Path(video_path)
    video_folder = video_path.parent
    video_name = video_path.stem
    
    logger.info(f"Recording {n_episodes} episode(s) of {env_id}")
    
    # Create environment with video recording
    env = gym.make(env_id, render_mode="rgb_array")
    env = setup_video_recorder(
        env,
        video_folder=video_folder,
        video_length=video_length,
        name_prefix=video_name,
    )
    
    # Record episodes
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < video_length:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        logger.info(
            f"Episode {episode + 1}/{n_episodes} - "
            f"Steps: {steps}, Reward: {episode_reward:.2f}"
        )
    
    env.close()
    logger.success(f"Video recording complete! Saved to: {video_folder}")