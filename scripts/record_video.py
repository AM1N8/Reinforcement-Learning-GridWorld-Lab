"""
Record video of trained agent playing in environment.
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.video import record_agent_video


# Map of algorithm names to classes
ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def detect_algorithm(model_path: Path) -> str:
    """Detect algorithm type from model filename."""
    model_name = model_path.stem.lower()
    
    for algo in ALGO_MAP.keys():
        if algo in model_name:
            return algo
    
    logger.warning(f"Could not detect algorithm from filename: {model_path.name}")
    logger.info("Defaulting to PPO. Use --algo to specify manually.")
    return "ppo"


def main():
    parser = argparse.ArgumentParser(description="Record video of trained agent")
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
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: videos/<env>_<algo>.mp4)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=1000,
        help="Maximum length of video in steps"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=list(ALGO_MAP.keys()),
        help="Algorithm type (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="INFO")
    
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Detect or use specified algorithm
    if args.algo is None:
        algo = detect_algorithm(args.model_path)
    else:
        algo = args.algo.lower()
    
    if algo not in ALGO_MAP:
        logger.error(f"Unknown algorithm: {algo}. Choose from {list(ALGO_MAP.keys())}")
        return
    
    logger.info(f"Using algorithm: {algo.upper()}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    AlgoClass = ALGO_MAP[algo]
    model = AlgoClass.load(args.model_path)
    logger.success("Model loaded successfully!")
    
    # Determine output path
    if args.output is None:
        env_name = args.env.replace("-", "_").lower()
        output = Path(f"videos/{env_name}_{algo}.mp4")
    else:
        output = args.output
    
    # Record video
    logger.info(f"Recording video to: {output}")
    record_agent_video(
        model=model,
        env_id=args.env,
        video_path=output,
        n_episodes=args.n_episodes,
        video_length=args.video_length,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()