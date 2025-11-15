"""
Train A2C agent on CartPole-v1 environment.
Uses hyperparameters from hyperparams/a2c.yml.
"""

import argparse
from pathlib import Path
import yaml
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from loguru import logger

from src.utils.logger import setup_logger


def load_hyperparams(config_path: Path, env_id: str) -> dict:
    """Load hyperparameters from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if env_id not in config:
        logger.warning(f"No config for {env_id}, using defaults")
        return {}
    
    hyperparams = config[env_id].copy()
    
    # Handle policy_kwargs if it's a string (needs to be evaluated)
    if 'policy_kwargs' in hyperparams and isinstance(hyperparams['policy_kwargs'], str):
        try:
            # Safely evaluate the string as a Python dict
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])
            logger.debug(f"Parsed policy_kwargs: {hyperparams['policy_kwargs']}")
        except Exception as e:
            logger.warning(f"Could not parse policy_kwargs: {e}, removing it")
            del hyperparams['policy_kwargs']
    
    logger.info(f"Loaded hyperparameters for {env_id}: {hyperparams}")
    return hyperparams


def train_cartpole(
    total_timesteps: int = 100_000,
    checkpoint_freq: int = 10_000,
    log_dir: Path = Path("logs/cartpole"),
    tensorboard_dir: Path = Path("tensorboard/cartpole"),
    save_path: Path = Path("models/a2c_cartpole.zip"),
    hyperparams_path: Path = Path("hyperparams/a2c.yml"),
) -> None:
    """
    Train A2C agent on CartPole-v1.
    
    Args:
        total_timesteps: Total training timesteps
        checkpoint_freq: Save checkpoint every N timesteps
        log_dir: Directory for logs
        tensorboard_dir: Directory for TensorBoard logs
        save_path: Path to save final model
        hyperparams_path: Path to hyperparameters YAML file
    """
    # Setup logging
    setup_logger(log_dir=log_dir, log_level="INFO")
    logger.info("=" * 60)
    logger.info("Training A2C on CartPole-v1")
    logger.info("=" * 60)
    
    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load hyperparameters
    hyperparams = load_hyperparams(hyperparams_path, "CartPole-v1")
    
    # Extract training params
    n_timesteps = int(hyperparams.pop("n_timesteps", total_timesteps))
    policy = hyperparams.pop("policy", "MlpPolicy")
    
    # Create environment
    logger.info("Creating CartPole-v1 environment...")
    env = gym.make("CartPole-v1")
    env = Monitor(env, str(log_dir / "monitor"))
    
    # Create eval environment
    eval_env = gym.make("CartPole-v1")
    eval_env = Monitor(eval_env, str(log_dir / "eval_monitor"))
    
    # Create model
    logger.info(f"Initializing A2C model with policy: {policy}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    model = A2C(
        policy,
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        **hyperparams
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="a2c_cartpole",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train
    logger.info(f"Starting training for {n_timesteps:,} timesteps...")
    logger.info(f"Checkpoints every {checkpoint_freq:,} steps")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    try:
        model.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        logger.info(f"Saving final model to: {save_path}")
        model.save(save_path)
        
        logger.success("Training completed successfully!")
        logger.info(f"Final model saved to: {save_path}")
        logger.info(f"Best model saved to: {log_dir / 'best_model'}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(f"Saving interrupted model to: {save_path}")
        model.save(save_path)
    
    finally:
        env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train A2C on CartPole-v1")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/cartpole"),
        help="Directory for logs"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("tensorboard/cartpole"),
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("models/a2c_cartpole.zip"),
        help="Path to save final model"
    )
    parser.add_argument(
        "--hyperparams",
        type=Path,
        default=Path("hyperparams/a2c.yml"),
        help="Path to hyperparameters YAML file"
    )
    
    args = parser.parse_args()
    
    train_cartpole(
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        log_dir=args.log_dir,
        tensorboard_dir=args.tensorboard_dir,
        save_path=args.save_path,
        hyperparams_path=args.hyperparams,
    )


if __name__ == "__main__":
    main()