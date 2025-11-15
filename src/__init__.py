# src/__init__.py
"""RL Pipeline package."""

__version__ = "0.1.0"

# src/envs/__init__.py
"""Custom Gymnasium environments."""

from src.envs.gridworld_env import GridWorldEnv

__all__ = ["GridWorldEnv"]

# src/utils/__init__.py
"""Utility modules."""

from src.utils.logger import setup_logger, get_logger
from src.utils.video import setup_video_recorder, record_agent_video
from src.utils.register_envs import CUSTOM_ENVS, verify_registration

__all__ = [
    "setup_logger",
    "get_logger",
    "setup_video_recorder",
    "record_agent_video",
    "CUSTOM_ENVS",
    "verify_registration",
]

# src/train/__init__.py
"""Training scripts."""

__all__ = []