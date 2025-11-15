"""
Automatic registration of custom environments.
Import this module to register all custom envs.
"""

from loguru import logger

# Import custom environments (this triggers registration)
try:
    from src.envs.gridworld_env import GridWorldEnv
    logger.info("Custom environments registered successfully")
except ImportError as e:
    logger.warning(f"Failed to register custom environments: {e}")

# List of all custom environment IDs
CUSTOM_ENVS = [
    "GridWorld-v0",
]


def verify_registration() -> bool:
    """
    Verify that all custom environments are registered with Gymnasium.
    
    Returns:
        True if all environments are registered, False otherwise
    """
    import gymnasium as gym
    
    all_registered = True
    for env_id in CUSTOM_ENVS:
        try:
            env = gym.make(env_id)
            env.close()
            logger.info(f"✓ {env_id} is registered and functional")
        except Exception as e:
            logger.error(f"✗ {env_id} registration failed: {e}")
            all_registered = False
    
    return all_registered


if __name__ == "__main__":
    # Test registration when run directly
    from src.utils.logger import setup_logger
    
    setup_logger(log_level="INFO")
    logger.info("Testing environment registration...")
    
    if verify_registration():
        logger.success("All custom environments are properly registered!")
    else:
        logger.error("Some environments failed registration checks")