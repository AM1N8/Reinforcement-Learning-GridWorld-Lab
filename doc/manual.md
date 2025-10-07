# GridWorld Reinforcement Learning Framework - User Manual

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Framework Architecture](#framework-architecture)
- [Algorithms Overview](#algorithms-overview)
- [Complete Usage Guide](#complete-usage-guide)
- [Command-Line Arguments](#command-line-arguments)
- [Use Cases & Examples](#use-cases--examples)
- [Cheat Sheet](#cheat-sheet)
- [Understanding the Output](#understanding-the-output)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Introduction

The GridWorld Reinforcement Learning Framework is a comprehensive toolkit for learning, experimenting with, and visualizing reinforcement learning algorithms. It provides implementations of both classical and modern RL algorithms in a simple grid environment.

### Key Features
- 🎯 6 RL algorithms implemented (VI, PI, MC, Q-Learning, SARSA(λ), DQN)
- 📊 Rich visualization capabilities
- 🎬 Animated learning processes
- 💾 Model saving/loading
- 🔬 Built-in experiments
- 📈 Comprehensive metrics tracking

---

## Quick Start

### Basic Commands

```bash
# Run Q-Learning (default)
uv run src/main.py

# Run with animation
uv run src/main.py --animate

# Run Value Iteration on 10x10 grid
uv run src/main.py --algorithm vi --rows 10 --cols 10

# Compare all algorithms
uv run src/main.py --algorithm compare
```

---

## Framework Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│          GridWorld Framework            │
├─────────────────────────────────────────┤
│  model.py        │ Environment model    │
│  environment.py  │ Gym-style interface  │
│  agent.py        │ Agent wrapper        │
│  algorithms.py   │ Classical RL algos   │
│  dqn.py          │ Deep Q-Network       │
│  visualizer.py   │ Plotting & animation │
│  main.py         │ CLI interface        │
└─────────────────────────────────────────┘
```

### How It Works

1. **Model** (`model.py`): Defines the grid world structure, rewards, and transitions
2. **Environment** (`environment.py`): OpenAI Gym-compatible environment for training
3. **Algorithms**: Learn optimal policies through different strategies
4. **Agent** (`agent.py`): Executes learned policies
5. **Visualizer** (`visualizer.py`): Creates plots and animations

---

## Algorithms Overview

### 1. Value Iteration (VI)
**Type**: Dynamic Programming  
**Best for**: Small grids, exact solutions  
**How it works**: Iteratively updates state values until convergence

```bash
uv run src/main.py --algorithm vi --gamma 0.99
```

**Pros**: Guaranteed optimal policy, fast on small grids  
**Cons**: Requires model knowledge, scales poorly

---

### 2. Policy Iteration (PI)
**Type**: Dynamic Programming  
**Best for**: Small grids, few actions  
**How it works**: Alternates between policy evaluation and improvement

```bash
uv run src/main.py --algorithm pi --gamma 0.99
```

**Pros**: Often faster than VI, guaranteed optimal  
**Cons**: Requires model knowledge, scales poorly

---

### 3. Monte Carlo (MC)
**Type**: Model-free, Episode-based  
**Best for**: Episodic tasks, simple exploration  
**How it works**: Learns from complete episode returns

```bash
uv run src/main.py --algorithm mc --episodes 1000 --epsilon 0.1
```

**Pros**: Model-free, simple, unbiased estimates  
**Cons**: High variance, requires episode completion

---

### 4. Q-Learning (QL)
**Type**: Model-free, Off-policy TD  
**Best for**: General purpose learning  
**How it works**: Updates Q-values using max over next actions

```bash
uv run src/main.py --algorithm ql --alpha 0.1 --gamma 0.99 --epsilon 0.1
```

**Pros**: Off-policy, online learning, proven convergence  
**Cons**: Overestimation bias, tabular only

---

### 5. SARSA(λ) (SL)
**Type**: Model-free, On-policy TD(λ)  
**Best for**: Safe exploration, eligibility traces  
**How it works**: On-policy TD with eligibility traces

```bash
uv run src/main.py --algorithm sl --alpha 0.1 --lambda-val 0.9
```

**Pros**: On-policy (safer), multi-step credit assignment  
**Cons**: Slower convergence than Q-learning

---

### 6. Deep Q-Network (DQN)
**Type**: Deep RL, Off-policy  
**Best for**: Large state spaces, function approximation  
**How it works**: Neural network approximates Q-function

```bash
uv run src/main.py --algorithm dql --episodes 500 --dqn-hidden 128 128
```

**Pros**: Scales to large spaces, function approximation  
**Cons**: Sample inefficient, hyperparameter sensitive

---

## Complete Usage Guide

### Grid Configuration

#### Basic Grid Setup
```bash
# 5x5 grid (default)
uv run src/main.py --rows 5 --cols 5

# 10x10 grid
uv run src/main.py --rows 10 --cols 10
```

#### Setting Start and Goal States
```bash
# Start at (0,0), goal at (4,4)
uv run src/main.py --starts 0,0 --goals 4,4

# Multiple starts
uv run src/main.py --starts 0,0 0,4 --goals 4,4

# Multiple goals
uv run src/main.py --starts 0,0 --goals 4,4 4,0
```

#### Adding Obstacles
```bash
# Single obstacle
uv run src/main.py --obstacles 2,2

# Multiple obstacles (create a maze)
uv run src/main.py --obstacles 1,1 1,2 1,3 3,1 3,2 3,3
```

#### Reward Customization
```bash
# Custom rewards
uv run src/main.py --goal-reward 100 --step-penalty -0.1 --obstacle-penalty -10

# Sparse rewards (discourage wandering)
uv run src/main.py --goal-reward 10 --step-penalty -1

# Dense rewards (encourage exploration)
uv run src/main.py --goal-reward 10 --step-penalty -0.01
```

---

### Algorithm-Specific Parameters

#### Value/Policy Iteration
```bash
# Standard discount factor
uv run src/main.py --algorithm vi --gamma 0.99

# Higher discount (more forward-looking)
uv run src/main.py --algorithm pi --gamma 0.999
```

#### Q-Learning
```bash
# Standard hyperparameters
uv run src/main.py --algorithm ql \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.1 \
  --episodes 500

# Fast learning
uv run src/main.py --algorithm ql --alpha 0.3 --epsilon 0.2

# Conservative learning
uv run src/main.py --algorithm ql --alpha 0.01 --epsilon 0.05
```

#### SARSA(λ)
```bash
# High eligibility traces
uv run src/main.py --algorithm sl --lambda-val 0.9

# Low eligibility traces (closer to SARSA(0))
uv run src/main.py --algorithm sl --lambda-val 0.3
```

#### DQN
```bash
# Standard DQN
uv run src/main.py --algorithm dql \
  --dqn-hidden 128 128 \
  --dqn-buffer-size 10000 \
  --dqn-batch-size 64 \
  --dqn-target-update 10

# Deeper network
uv run src/main.py --algorithm dql --dqn-hidden 256 256 128

# Larger replay buffer
uv run src/main.py --algorithm dql --dqn-buffer-size 50000
```

---

### Visualization Options

#### Basic Visualization
```bash
# Default: shows all plots
uv run src/main.py --algorithm ql

# Disable visualization
uv run src/main.py --algorithm ql --no-viz
```

#### Animated Episodes
```bash
# Animate final episode
uv run src/main.py --algorithm ql --animate

# Save animation as GIF
uv run src/main.py --algorithm ql --animate --save-animation

# Animate entire learning process
uv run src/main.py --algorithm dql --animate-learning --save-animation
```

#### Saving Visualizations
```bash
# Save to default directory (../visualizations)
uv run src/main.py --algorithm ql

# Custom save directory
uv run src/main.py --algorithm ql --save-dir ./my_results
```

---

### Model Management

#### Saving Models
```bash
# Save DQN model
uv run src/main.py --algorithm dql --save-model

# Save with logs
uv run src/main.py --algorithm dql --save-model --save-logs
```

#### Loading Models
```bash
# Load pre-trained model
uv run src/main.py --algorithm dql --load-model ../visualizations/dqn_model_20241006_123456.pth

# Load and test
uv run src/main.py --algorithm dql --load-model path/to/model.pth --animate
```

#### Loading and Visualizing Logs
```bash
# Visualize training logs
uv run src/main.py --load-logs ../visualizations/dqn_log_20241006_123456.json
```

---

## Use Cases & Examples

### Use Case 1: Learning Basics
**Goal**: Understand how Q-Learning works

```bash
# Small grid, verbose output
uv run src/main.py --algorithm ql \
  --rows 3 --cols 3 \
  --episodes 100 \
  --verbose \
  --animate
```

**What to observe**:
- Q-values converging
- Policy becoming deterministic
- Rewards increasing over time

---

### Use Case 2: Maze Navigation
**Goal**: Solve a complex maze

```bash
# Create a maze with obstacles
uv run src/main.py --algorithm ql \
  --rows 7 --cols 7 \
  --obstacles 1,1 1,2 1,3 3,1 3,3 3,5 5,3 5,4 5,5 \
  --starts 0,0 \
  --goals 6,6 \
  --episodes 1000 \
  --animate
```

**Tips**:
- Increase episodes for complex mazes
- Lower step penalty to encourage exploration
- Use --animate to visualize the solution

---

### Use Case 3: Comparing Algorithms
**Goal**: Find the best algorithm for your problem

```bash
# Compare all algorithms
uv run src/main.py --algorithm compare \
  --rows 5 --cols 5 \
  --episodes 500
```

**Analysis**:
- VI/PI: Fastest on small grids
- Q-Learning: Best general-purpose
- DQN: Best for scaling up

---

### Use Case 4: Grid Size Experiment
**Goal**: Understand how algorithms scale

```bash
# Test multiple grid sizes
uv run src/main.py --grid-size-experiment \
  --grid-sizes 3x3 5x5 7x7 10x10 15x15 \
  --episodes 500 \
  --save-logs
```

**Insights**:
- Training time vs grid size
- Convergence speed
- Sample efficiency

---

### Use Case 5: Fine-tuning DQN
**Goal**: Optimize DQN performance

```bash
# Optimized DQN configuration
uv run src/main.py --algorithm dql \
  --rows 10 --cols 10 \
  --dqn-hidden 256 128 64 \
  --dqn-buffer-size 20000 \
  --dqn-batch-size 128 \
  --dqn-target-update 5 \
  --epsilon-decay 0.995 \
  --episodes 1000 \
  --save-model --save-logs \
  --animate-learning
```

---

### Use Case 6: Custom Reward Shaping
**Goal**: Design rewards for specific behavior

```bash
# Encourage fast solutions
uv run src/main.py --algorithm ql \
  --goal-reward 100 \
  --step-penalty -2 \
  --episodes 500

# Encourage exploration (minimize steps)
uv run src/main.py --algorithm ql \
  --goal-reward 10 \
  --step-penalty -0.1 \
  --episodes 500
```

---

## Cheat Sheet

### Quick Reference Commands

```bash
# Basic training
uv run src/main.py --algorithm [vi|pi|mc|ql|sl|dql]

# Common flags
--rows N --cols N              # Grid size
--starts r,c                   # Start position
--goals r,c                    # Goal position
--obstacles r,c r,c ...        # Obstacle positions
--episodes N                   # Training episodes
--animate                      # Show animation
--save-animation               # Save as GIF
--no-viz                       # No plots
--verbose                      # Detailed output
--save-model                   # Save trained model
--load-model path              # Load model
```

### Hyperparameter Recommendations

| Algorithm | Alpha | Gamma | Epsilon | Episodes |
|-----------|-------|-------|---------|----------|
| VI/PI     | N/A   | 0.99  | N/A     | N/A      |
| MC        | N/A   | 0.99  | 0.1     | 1000+    |
| Q-Learning| 0.1   | 0.99  | 0.1     | 500      |
| SARSA(λ)  | 0.1   | 0.99  | 0.1     | 500      |
| DQN       | 0.001 | 0.99  | 1.0→0.01| 500+     |

### File Outputs

```
visualizations/
├── *_q_values.png           # Q-value heatmaps
├── *_policy.png             # Policy arrows
├── *_trajectory.png         # Agent path
├── *_learning_curve.png     # Training progress
├── *_epsilon_decay.png      # Exploration schedule
├── *_training_summary.png   # Multi-panel summary
├── *.gif                    # Animations
└── *.json                   # Training logs
```

---

## Understanding the Output

### Console Output

```
============================================================
GridWorld: Reinforcement Learning Framework
============================================================

GridWorld(5x5)
  Start: [(0, 0)]
  Goals: [(4, 4)]
  Actions: 4 (↑, →, ↓, ←)

============================================================
Q-Learning
============================================================

Training Q-Learning...
Episode 100/500, Avg Reward: -12.4
Episode 200/500, Avg Reward: -8.1
Episode 500/500, Avg Reward: 5.2

Q-Learning Results:
Steps: 8, Total Reward: 2.0
```

**Interpreting**:
- **Avg Reward increasing**: Learning is progressing
- **Steps decreasing**: Finding shorter paths
- **Total Reward**: Final episode performance

---

### Visualizations

#### 1. Q-Values Plot
Shows learned value for each state-action pair
- **Brighter colors**: Higher Q-values (better actions)
- **Darker colors**: Lower Q-values (worse actions)

#### 2. Policy Plot
Shows the best action in each state
- **Arrows**: Indicate optimal action direction
- **Color**: State value (brighter = better)

#### 3. Trajectory Plot
Shows agent's path through the grid
- **Green square**: Start state
- **Red square**: Goal state
- **Yellow dots**: Agent's path
- **Gray squares**: Obstacles

#### 4. Learning Curve
Shows training progress
- **X-axis**: Episode number
- **Y-axis**: Total reward
- **Upward trend**: Successful learning
- **Plateau**: Convergence

---

## Advanced Features

### Custom Grid Configurations

Create complex environments by combining arguments:

```bash
# Four-room environment
uv run src/main.py \
  --rows 11 --cols 11 \
  --obstacles \
    5,0 5,1 5,2 5,3 5,4 5,6 5,7 5,8 5,9 5,10 \
    0,5 1,5 2,5 3,5 4,5 6,5 7,5 8,5 9,5 10,5 \
  --starts 1,1 \
  --goals 9,9 \
  --algorithm ql --episodes 1000
```

### Reproducibility

```bash
# Set random seed for reproducible results
uv run src/main.py --seed 42 --algorithm ql
```

### Batch Experiments

```bash
# Compare different learning rates
for alpha in 0.01 0.05 0.1 0.2; do
  uv run src/main.py --algorithm ql --alpha $alpha --save-logs
done
```

### Analyzing Saved Logs

```python
import json
import matplotlib.pyplot as plt

# Load training log
with open('visualizations/dqn_log_20241006_123456.json', 'r') as f:
    log = json.load(f)

# Plot episode rewards
plt.plot(log['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
```

---

## Troubleshooting

### Common Issues

#### 1. Agent doesn't reach goal
**Symptoms**: High negative rewards, no improvement

**Solutions**:
```bash
# Increase exploration
uv run src/main.py --epsilon 0.2

# More training episodes
uv run src/main.py --episodes 1000

# Higher learning rate
uv run src/main.py --alpha 0.2

# Lower step penalty
uv run src/main.py --step-penalty -0.1
```

#### 2. Slow convergence
**Symptoms**: Slow learning curve improvement

**Solutions**:
```bash
# Increase learning rate
uv run src/main.py --alpha 0.2

# Adjust discount factor
uv run src/main.py --gamma 0.95

# For DQN: larger batch size
uv run src/main.py --algorithm dql --dqn-batch-size 128
```

#### 3. DQN instability
**Symptoms**: Wild fluctuations in rewards

**Solutions**:
```bash
# Larger replay buffer
uv run src/main.py --algorithm dql --dqn-buffer-size 50000

# More frequent target updates
uv run src/main.py --algorithm dql --dqn-target-update 5

# Lower learning rate
uv run src/main.py --algorithm dql --alpha 0.0001
```

#### 4. Memory issues on large grids
**Symptoms**: Out of memory errors

**Solutions**:
```bash
# Use DQN for large grids
uv run src/main.py --algorithm dql --rows 20 --cols 20

# Reduce buffer size
uv run src/main.py --algorithm dql --dqn-buffer-size 5000

# Reduce network size
uv run src/main.py --algorithm dql --dqn-hidden 64 64
```

---

## Tips & Best Practices

### 1. Algorithm Selection
- **Small grids (<7x7)**: Use VI or PI for optimal solutions
- **Medium grids (7x10)**: Q-Learning or SARSA(λ)
- **Large grids (>10x10)**: DQN

### 2. Hyperparameter Tuning
- Start with defaults
- Adjust learning rate (alpha) first
- Then tune exploration (epsilon)
- Finally adjust discount (gamma)

### 3. Debugging
- Use `--verbose` to see step-by-step actions
- Use `--animate` to visualize behavior
- Check learning curves for convergence
- Compare with `--algorithm compare`

### 4. Performance Optimization
- Reduce `--max-steps` for faster training
- Use `--no-viz` for batch experiments
- Save models to avoid retraining

---

## Glossary

- **Alpha (α)**: Learning rate - how much to update Q-values
- **Gamma (γ)**: Discount factor - how much to value future rewards
- **Epsilon (ε)**: Exploration rate - probability of random action
- **Lambda (λ)**: Eligibility trace decay in SARSA(λ)
- **Episode**: One complete trajectory from start to goal
- **State**: Agent's position in the grid
- **Action**: Movement direction (up, right, down, left)
- **Reward**: Numerical feedback from environment
- **Policy**: Mapping from states to actions
- **Q-value**: Expected future reward for state-action pair
- **Value function**: Expected future reward for a state

---

## Further Reading

### Reinforcement Learning Resources
- Sutton & Barto: "Reinforcement Learning: An Introduction"
- David Silver's RL Course: https://www.davidsilver.uk/teaching/
- OpenAI Spinning Up: https://spinningup.openai.com/

### Framework Extensions
Consider implementing:
- Stochastic transitions
- Continuous action spaces
- Multi-agent scenarios
- Hierarchical RL
- Transfer learning

---

**Version**: 1.0  
**Last Updated**: 2024  
**License**: MIT