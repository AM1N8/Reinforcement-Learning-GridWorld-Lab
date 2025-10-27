from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    
    All algorithms should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def act(self, state: np.ndarray, train: bool = True) -> int:
        """
        Select action given state.
        
        Args:
            state: Current state
            train: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def learn(self, batch: Tuple[np.ndarray, ...]) -> Dict[str, float]:
        """
        Update algorithm from experience batch.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            
        Returns:
            Dictionary of training metrics (e.g., {'loss': 0.5})
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save algorithm state to file."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load algorithm state from file."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get algorithm-specific metrics (e.g., epsilon for DQN)."""
        return {}