# RL Pipeline with stable-baselines3 & rl_zoo3

A complete, production-ready Reinforcement Learning pipeline using `stable-baselines3`, `rl_zoo3`, and custom environments. Built with `uv` for fast, reliable dependency management.

## 🚀 Features

- **Multiple Environments**: LunarLander-v3, CartPole-v1, and custom GridWorld-v0
- **Multiple Algorithms**: PPO, A2C, DQN with hyperparameter configs
- **rl_zoo3 Integration**: Training, evaluation, hyperparameter tuning, and TensorBoard logging
- **Custom GridWorld**: Fully functional grid environment with walls, rewards, and RGB rendering
- **Video Recording**: Automatic video saving of agent performance
- **Structured Logging**: Loguru-based logging across the entire pipeline
- **Type Hints**: Complete type annotations for better code quality

## 📋 Prerequisites

- Python 3.9-3.12
- [uv](https://docs.astral.sh/uv/) package manager

## 🛠️ Installation

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup Project

```bash
# Create project directory
mkdir rl-pipeline && cd rl-pipeline

# Copy all project files here

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv sync

# Verify installation
uv run python -c "import gymnasium; import stable_baselines3; from loguru import logger; logger.info('Setup complete!')"
```

## 🎮 Environments

### Classic Gymnasium Environments
- **LunarLander-v3**: Land a spacecraft safely
- **CartPole-v1**: Balance a pole on a cart

### Custom GridWorld-v0
A custom grid-based environment with:
- 8x8 grid with walls
- Agent starts at (0,0)
- Goal at (7,7) with +10 reward
- Walls return -1 reward
- RGB rendering with distinct colors
- Automatic video recording support

## 🏋️ Training

### Basic Training

```bash
# Train LunarLander with PPO (from hyperparams/ppo.yml)
uv run python src/train/train_lunarlander.py

# Train CartPole with A2C
uv run python src/train/train_cartpole.py

# Train GridWorld with DQN
uv run python src/train/train_gridworld.py
```

### Advanced Training Options

All training scripts support these command-line arguments:

```bash
uv run python src/train/train_lunarlander.py \
    --total-timesteps 500000 \
    --checkpoint-freq 10000 \
    --log-dir logs/lunarlander \
    --tensorboard-dir tensorboard/lunarlander \
    --save-path models/ppo_lunarlander_custom.zip
```

## 📊 Monitoring with TensorBoard

```bash
# Start TensorBoard
uv run tensorboard --logdir tensorboard/

# Open browser to http://localhost:6006
```

## 🎬 Evaluation & Video Recording

### Evaluate a Trained Model

```bash
uv run python scripts/eval_model.py \
    --model-path models/ppo_lunarlander.zip \
    --env LunarLander-v3 \
    --n-episodes 10
```

### Record Videos

```bash
uv run python scripts/record_video.py \
    --model-path models/ppo_lunarlander.zip \
    --env LunarLander-v3 \
    --video-length 1000 \
    --output videos/lunarlander_demo.mp4
```

## 🔧 Hyperparameter Configuration

Hyperparameters are stored in `hyperparams/*.yml` using rl_zoo3 format:

### Example: `hyperparams/ppo.yml`

```yaml
LunarLander-v3:
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 1024
  batch_size: 64
  gae_lambda: 0.98
  gamma: 0.999
  n_epochs: 4
  ent_coef: 0.01
```

## 📁 Project Structure

```
rl-pipeline/
├── pyproject.toml           # uv configuration & dependencies
├── README.md                # This file
├── src/
│   ├── envs/
│   │   └── gridworld_env.py    # Custom GridWorld environment
│   ├── utils/
│   │   ├── logger.py           # Loguru configuration
│   │   ├── video.py            # Video recording utilities
│   │   └── register_envs.py    # Environment registration
│   └── train/
│       ├── train_lunarlander.py
│       ├── train_cartpole.py
│       └── train_gridworld.py
├── scripts/
│   ├── eval_model.py           # Model evaluation
│   └── record_video.py         # Video recording
├── hyperparams/                # Algorithm configs
│   ├── ppo.yml
│   ├── a2c.yml
│   └── dqn.yml
├── logs/                       # Training logs
├── videos/                     # Recorded videos
├── models/                     # Saved models
└── tensorboard/                # TensorBoard logs
```

## 🔍 Using rl_zoo3 CLI

This project is fully compatible with rl_zoo3's command-line interface:

```bash
# Train using rl_zoo3 CLI
uv run python -m rl_zoo3.train \
    --algo ppo \
    --env LunarLander-v3 \
    --conf-file hyperparams/ppo.yml

# Hyperparameter optimization
uv run python -m rl_zoo3.train \
    --algo ppo \
    --env LunarLander-v3 \
    --optimize \
    --n-trials 100 \
    --n-jobs 4

# Enjoy trained agent
uv run python -m rl_zoo3.enjoy \
    --algo ppo \
    --env LunarLander-v3 \
    --folder models/
```

## 🐛 Troubleshooting

### Box2D Installation Issues

If you encounter Box2D errors (for LunarLander):

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install swig

# Reinstall gymnasium[box2d]
uv pip install gymnasium[box2d] --force-reinstall
```

### CUDA/GPU Support

For GPU acceleration:

```bash
# Install PyTorch with CUDA support
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Key Dependencies

- `stable-baselines3`: RL algorithms (PPO, A2C, DQN, etc.)
- `rl-zoo3`: Training scripts and hyperparameter optimization
- `gymnasium`: Environment interface (OpenAI Gym successor)
- `loguru`: Advanced logging
- `tensorboard`: Training visualization
- `torch`: Neural network backend

## 🤝 Contributing

1. Follow PEP 8 style guide
2. Add type hints to all functions
3. Use loguru for logging
4. Test with `uv run pytest`

## 📝 License

MIT License - feel free to use this project as a template for your RL experiments!

## 🔗 Resources

- [stable-baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [rl_zoo3 GitHub](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [uv Documentation](https://docs.astral.sh/uv/)