"""
Centralized logging configuration using Loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure loguru logger with file and console output.

    Args:
        log_dir: Directory to save log files (default: logs/)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Custom log file name (default: training.log)
    """
    # Remove default logger
    logger.remove()

    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file logger if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = "training.log"

        log_path = log_dir / log_file

        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",  # Rotate after 10 MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip",  # Compress rotated logs
        )

        logger.info(f"Logging to file: {log_path}")

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        Loguru logger instance
    """
    return logger