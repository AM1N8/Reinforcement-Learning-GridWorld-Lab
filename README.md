# GridWorld Reinforcement Learning Framework

A comprehensive Python framework for learning and experimenting with reinforcement learning algorithms in a grid world environment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Features

- **6 RL Algorithms**: Value Iteration, Policy Iteration, Monte Carlo, Q-Learning, SARSA(λ), Deep Q-Network (DQN)
- **Rich Visualizations**: Q-values, policies, trajectories, learning curves, and more
- **Animated Learning**: Watch agents learn in real-time with GIF export
- **Model Persistence**: Save and load trained models
- **Experiments**: Built-in grid size scaling experiments
- **Customizable Environments**: Configurable grids, obstacles, rewards
- **Comprehensive Logging**: Track and analyze training metrics

![DQL Learning Animation](visualizations/dql_learning_process.gif)


##  Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

##  Installation

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

### Setting Up the Project

```bash
# Clone the repository
git clone hhttps://github.com/AM1N8/Reinforcement-Learning-GridWorld-Lab.git
cd Reinforcement-Learning-GridWorld-Lab

# Create virtual environment and install dependencies
uv sync
```


##  Quick Start

### Basic Usage

```bash
# Run Q-Learning with default settings
uv run src/main.py

# Run with visualization and animation
uv run src/main.py --animate

# Try different algorithms
uv run src/main.py --algorithm vi    # Value Iteration
uv run src/main.py --algorithm pi    # Policy Iteration
uv run src/main.py --algorithm dql   # Deep Q-Learning
```

### Your First Custom Environment

```bash
# Create a 7x7 maze with obstacles
uv run src/main.py \
  --rows 7 --cols 7 \
  --starts 0,0 \
  --goals 6,6 \
  --obstacles 1,1 1,2 3,3 3,4 5,2 5,3 \
  --algorithm ql \
  --episodes 500 \
  --animate
```

##  Project Structure

```
gridworld-rl/
├── .venv/                      # Virtual environment (uv managed)
├── doc/
│   └── manual.md              # Comprehensive user manual
├── src/
│   ├── __pycache__/           # Python cache
│   ├── agent.py               # Agent implementation
│   ├── algorithms.py          # Classical RL algorithms
│   ├── dqn.py                 # Deep Q-Network
│   ├── environment.py         # Gym-compatible environment
│   ├── main.py               # CLI entry point
│   ├── model.py              # Grid world model
│   └── visualizer.py         # Plotting and animation
├── visualizations/            # Output directory for plots/GIFs
├── .gitignore
├── .python-version           # Python version for uv
├── main.py                   # Root-level script runner
├── pyproject.toml           # Project configuration (uv)
├── README.md                # This file
└── uv.lock                  # Dependency lock file (uv)
```

### Key Components

| File | Purpose |
|------|---------|
| `model.py` | Defines the GridWorld MDP (states, actions, rewards, transitions) |
| `environment.py` | OpenAI Gym-compatible environment wrapper |
| `agent.py` | Agent that executes policies |
| `algorithms.py` | VI, PI, MC, Q-Learning, SARSA(λ) implementations |
| `dqn.py` | Deep Q-Network with experience replay |
| `visualizer.py` | All plotting and animation functionality |
| `main.py` | Command-line interface |

##  Usage Examples

### Example 1: Compare All Algorithms

```bash
uv run src/main.py --algorithm compare --episodes 500
```

This will train and compare all 6 algorithms on the same environment.

### Example 2: Train DQN and Save Model

```bash
uv run src/main.py \
  --algorithm dql \
  --episodes 1000 \
  --dqn-hidden 256 128 \
  --save-model \
  --save-logs \
  --animate-learning
```

### Example 3: Load and Test Pre-trained Model

```bash
uv run src/main.py \
  --algorithm dql \
  --load-model visualizations/dqn_model_20241006_123456.pth \
  --animate
```

### Example 4: Grid Size Experiment

```bash
uv run src/main.py \
  --grid-size-experiment \
  --grid-sizes 3x3 5x5 7x7 10x10 15x15 \
  --episodes 500 \
  --save-logs
```

### Example 5: Custom Maze with Q-Learning

```bash
uv run src/main.py \
  --algorithm ql \
  --rows 10 --cols 10 \
  --starts 0,0 \
  --goals 9,9 \
  --obstacles 2,2 2,3 2,4 4,2 4,4 4,6 6,4 6,5 6,6 8,2 8,3 \
  --goal-reward 100 \
  --step-penalty -1 \
  --obstacle-penalty -10 \
  --episodes 1000 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.1 \
  --animate \
  --save-animation
```

##  Configuration

### Command-Line Arguments

#### Grid Configuration
```bash
--rows N              # Number of rows (default: 5)
--cols N              # Number of columns (default: 5)
--starts r,c [r,c]    # Start states (default: 0,0)
--goals r,c [r,c]     # Goal states (default: bottom-right)
--obstacles r,c [r,c] # Obstacle positions
```

#### Rewards
```bash
--goal-reward FLOAT       # Reward for reaching goal (default: 10.0)
--step-penalty FLOAT      # Penalty per step (default: -1.0)
--obstacle-penalty FLOAT  # Penalty for obstacles (default: -5.0)
```

#### Algorithm Selection
```bash
--algorithm {random|vi|pi|mc|ql|sl|dql|compare}
```

#### Hyperparameters
```bash
--gamma FLOAT          # Discount factor (default: 0.99)
--epsilon FLOAT        # Exploration rate (default: 0.1)
--alpha FLOAT          # Learning rate (default: 0.1)
--episodes N           # Training episodes (default: 500)
--max-steps N          # Max steps per episode (default: 100)
--lambda-val FLOAT     # λ for SARSA(λ) (default: 0.9)
```

#### DQN Specific
```bash
--dqn-hidden [N ...]         # Hidden layer sizes (default: 128 128)
--dqn-buffer-size N          # Replay buffer size (default: 10000)
--dqn-batch-size N           # Batch size (default: 64)
--dqn-target-update N        # Target update frequency (default: 10)
--epsilon-decay FLOAT        # Epsilon decay rate (default: 0.995)
```

#### Visualization & Output
```bash
--no-viz                # Disable visualization
--verbose               # Detailed output
--animate               # Animate final episode
--save-animation        # Save animations as GIF
--animate-learning      # Animate learning process
--save-dir PATH         # Output directory (default: ../visualizations)
--save-model            # Save trained model
--load-model PATH       # Load model from path
--save-logs             # Save training logs
--load-logs PATH        # Load and visualize logs
```

### Environment Variables

```bash
# Set random seed for reproducibility
export PYTHONHASHSEED=0

# Run with specific seed
uv run src/main.py --seed 42
```

##  Development

### Running from Root Directory

The project includes a convenience script at the root level:

```bash
# Option 1: Run from src directory
cd src
uv run main.py --algorithm ql

# Option 2: Run from root using the wrapper
uv run main.py --algorithm ql

# Option 3: Direct execution with Python
uv run python src/main.py --algorithm ql
```

### Development Workflow

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in editable mode
uv pip install -e .

# Run tests (if you add them)
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev pytest black mypy

# Update all dependencies
uv lock --upgrade

# Sync environment with lock file
uv sync
```

### Creating New Algorithms

To add a new algorithm:

1. Implement in `algorithms.py` or create new file
2. Add to algorithm registry in `main.py`
3. Update CLI choices in `parse_args()`
4. Create corresponding run function

Example:

```python
# In algorithms.py
class MyNewAlgorithm:
    def __init__(self, model, **kwargs):
        self.model = model
        # Initialize your algorithm
    
    def train(self, env, num_episodes, max_steps):
        # Training logic
        pass
    
    def get_action(self, state, training=False):
        # Action selection
        pass

# In main.py
def run_my_algorithm(args, model):
    """Run My New Algorithm"""
    env = GridWorldEnv(model=model)
    algo = MyNewAlgorithm(model, **params)
    Q = algo.train(env, args.episodes, args.max_steps)
    # Test and visualize
    env.close()

# Add to algorithms dictionary
algorithms = {
    'myalgo': run_my_algorithm,
    # ... other algorithms
}
```

##  Documentation

Comprehensive documentation is available in `doc/manual.md`:

- **Quick Start Guide**: Get running in minutes
- **Algorithm Explanations**: Detailed descriptions of each algorithm
- **Use Cases**: Real-world examples and scenarios
- **Cheat Sheet**: Quick reference for common commands
- **Troubleshooting**: Solutions to common issues
- **Best Practices**: Tips for optimal results

### Quick Documentation Access

```bash
# View manual (requires markdown viewer)
mdless doc/manual.md    # macOS
glow doc/manual.md      # Cross-platform
cat doc/manual.md       # Plain text
```

##  Educational Use

This framework is ideal for:

- **Learning RL Fundamentals**: Clear implementations of core algorithms
- **Teaching**: Visual feedback helps students understand concepts
- **Research**: Easy experimentation with different approaches
- **Prototyping**: Quick setup for testing new ideas

### Recommended Learning Path

1. **Start Simple**: Run random agent to understand environment
   ```bash
   uv run src/main.py --algorithm random --verbose
   ```

2. **Dynamic Programming**: Learn VI and PI on small grids
   ```bash
   uv run src/main.py --algorithm vi --rows 3 --cols 3 --animate
   ```

3. **Model-Free Learning**: Progress to Q-Learning
   ```bash
   uv run src/main.py --algorithm ql --episodes 500 --animate
   ```

4. **Deep RL**: Experiment with DQN
   ```bash
   uv run src/main.py --algorithm dql --episodes 1000 --animate-learning
   ```

5. **Compare**: Run all algorithms to see differences
   ```bash
   uv run src/main.py --algorithm compare
   ```

##  Output Files

The framework generates various outputs in the `visualizations/` directory:

### Plots
- `*_q_values.png` - Q-value heatmaps for each action
- `*_policy.png` - Optimal policy visualization with arrows
- `*_value_function.png` - State value heatmap
- `*_trajectory.png` - Agent's path through the grid
- `*_learning_curve.png` - Episode rewards over time
- `*_epsilon_decay.png` - Exploration rate decay
- `*_training_summary.png` - Multi-panel summary

### Animations
- `*_trajectory.gif` - Animated episode playback
- `*_learning_process.gif` - Episode-by-episode learning

### Data Files
- `dqn_model_*.pth` - Saved PyTorch models
- `*_log_*.json` - Training metrics and history
- `grid_experiment_*.json` - Experiment results

### Log File Format

```json
{
  "algorithm": "DQN",
  "hyperparameters": {
    "alpha": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0
  },
  "episode_rewards": [/* list of rewards */],
  "episode_steps": [/* list of steps */],
  "epsilon_history": [/* epsilon over time */],
  "loss_history": [/* training losses */],
  "training_time": 45.2
}
```

##  Testing

### Manual Testing

```bash
# Test each algorithm
for algo in vi pi mc ql sl dql; do
  echo "Testing $algo..."
  uv run src/main.py --algorithm $algo --episodes 100 --no-viz
done
```

### Validation Tests

```bash
# Test deterministic behavior
uv run src/main.py --algorithm vi --seed 42 --no-viz
uv run src/main.py --algorithm vi --seed 42 --no-viz
# Outputs should be identical

# Test saving and loading
uv run src/main.py --algorithm dql --episodes 100 --save-model --no-viz
uv run src/main.py --algorithm dql --load-model visualizations/dqn_model_*.pth --no-viz
```

##  Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Ensure virtual environment is activated
source .venv/bin/activate
uv sync
```

#### 2. Matplotlib Backend Issues
```bash
# Error: Cannot show plots
# Solution: Set backend
export MPLBACKEND=TkAgg  # or Qt5Agg, Agg for no display
```

#### 3. Memory Issues on Large Grids
```bash
# Error: Out of memory
# Solution: Use DQN with smaller buffer
uv run src/main.py --algorithm dql --rows 20 --cols 20 --dqn-buffer-size 5000
```

#### 4. Slow Training
```bash
# Solution: Reduce episodes or use faster algorithm
uv run src/main.py --algorithm vi  # Fast for small grids
uv run src/main.py --episodes 200  # Fewer episodes
```

### Getting Help

1. Check the [manual](doc/manual.md) for detailed usage
2. Run with `--verbose` for detailed output
3. Use `--no-viz` to isolate visualization issues
4. Test on small grids first (3x3 or 5x5)

##  Advanced Configuration

### Custom pyproject.toml

```toml
[project]
name = "gridworld-rl"
version = "1.0.0"
description = "GridWorld Reinforcement Learning Framework"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "torch>=2.0.0",
    "gymnasium>=0.29.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "isort>=5.12.0",
]

[project.scripts]
gridworld = "src.main:main"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

### Environment Setup Script

Create `setup.sh` for quick setup:

```bash
#!/bin/bash
# setup.sh - Quick setup script

echo "Setting up GridWorld RL Framework..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Create output directory
mkdir -p visualizations

echo "Setup complete! Run:"
echo "  source .venv/bin/activate"
echo "  uv run src/main.py --algorithm ql --animate"
```

##  Performance Tips

### 1. Speed Up Training
```bash
# Use vectorized operations (already optimized)
# Reduce max steps
uv run src/main.py --max-steps 50

# Use faster algorithms for small grids
uv run src/main.py --algorithm vi  # O(|S|²|A|) vs O(episodes)
```

### 2. Optimize DQN
```bash
# Larger batch size (if GPU available)
uv run src/main.py --algorithm dql --dqn-batch-size 256

# More efficient network
uv run src/main.py --algorithm dql --dqn-hidden 64 64
```

### 3. Parallel Experiments
```bash
# Run multiple experiments in background
for seed in 1 2 3 4 5; do
  uv run src/main.py --algorithm ql --seed $seed --no-viz --save-logs &
done
wait
```

##  Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/AM1N8/Reinforcement-Learning-GridWorld-Lab.git
cd Reinforcement-Learning-GridWorld-Lab


# Create feature branch
git checkout -b feature/my-new-feature

# Install dev dependencies
uv add --dev pytest black mypy isort

# Make changes and test
pytest tests/
black src/
mypy src/

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/my-new-feature
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described

### Areas for Contribution

- [ ] Add more RL algorithms (PPO, A3C, etc.)
- [ ] Implement continuous action spaces
- [ ] Add multi-agent support
- [ ] Create more complex environments
- [ ] Improve visualization options
- [ ] Add comprehensive test suite
- [ ] Enhance documentation
- [ ] Performance optimizations


##  Acknowledgments

- Inspired by Sutton & Barto's "Reinforcement Learning: An Introduction"
- Built with PyTorch and Gymnasium
- Uses uv for fast, reliable dependency management


##  Roadmap

### Version 1.1 (Planned)
- [ ] Add pytest test suite
- [ ] Implement continuous action spaces
- [ ] Add model zoo with pre-trained models
- [ ] Web-based visualization interface

### Version 1.2 (Future)
- [ ] Multi-agent GridWorld
- [ ] Hierarchical RL support
- [ ] Transfer learning capabilities
- [ ] Interactive Jupyter notebooks


---
