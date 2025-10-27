#!/usr/bin/env python
"""Verify RL Playground installation"""
import sys
import os

print("Testing RL Playground Installation...")
print("=" * 60)

# Test imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"✓ PyYAML")
except ImportError as e:
    print(f"✗ PyYAML import failed: {e}")
    sys.exit(1)

# Test project modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.environment import GridWorld
    print(f"✓ GridWorld environment")
except ImportError as e:
    print(f"✗ GridWorld import failed: {e}")
    sys.exit(1)

try:
    from src.algorithms.dqn import DQN
    print(f"✓ DQN algorithm")
except ImportError as e:
    print(f"✗ DQN import failed: {e}")
    sys.exit(1)

try:
    from src.utils.replay_buffer import ReplayBuffer
    print(f"✓ ReplayBuffer")
except ImportError as e:
    print(f"✗ ReplayBuffer import failed: {e}")
    sys.exit(1)

try:
    from src.utils.logger import Logger
    print(f"✓ Logger")
except ImportError as e:
    print(f"✗ Logger import failed: {e}")
    sys.exit(1)

# Quick functional test
print("\nRunning functional tests...")
env = GridWorld(size=5, num_obstacles=2, seed=42)
state = env.reset()
assert state.shape == (25,), f"State shape wrong: {state.shape}"
print("✓ Environment functional")

dqn = DQN(state_dim=25, action_dim=4)
action = dqn.act(state)
assert 0 <= action < 4, f"Invalid action: {action}"
print("✓ DQN functional")

buffer = ReplayBuffer(capacity=100, state_dim=25)
buffer.push(state, action, 0.0, state, False)
assert len(buffer) == 1, "Buffer not working"
print("✓ ReplayBuffer functional")

print("\n" + "=" * 60)
print("✅ All tests passed! Installation successful.")
print("You're ready to train agents!")