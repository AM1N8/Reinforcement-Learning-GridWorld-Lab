.PHONY: help setup init install clean test format lint run-lunarlander run-cartpole run-gridworld tensorboard

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)RL Pipeline Makefile Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Create complete project structure with empty files
	@echo "$(BLUE)Creating project structure...$(NC)"
	@mkdir -p src/envs
	@mkdir -p src/utils
	@mkdir -p src/train
	@mkdir -p scripts
	@mkdir -p hyperparams
	@mkdir -p logs
	@mkdir -p videos
	@mkdir -p models
	@mkdir -p tensorboard
	@touch src/__init__.py
	@touch src/envs/__init__.py
	@touch src/envs/gridworld_env.py
	@touch src/utils/__init__.py
	@touch src/utils/logger.py
	@touch src/utils/video.py
	@touch src/utils/register_envs.py
	@touch src/train/__init__.py
	@touch src/train/train_lunarlander.py
	@touch src/train/train_cartpole.py
	@touch src/train/train_gridworld.py
	@touch scripts/eval_model.py
	@touch scripts/record_video.py
	@touch hyperparams/ppo.yml
	@touch hyperparams/a2c.yml
	@touch hyperparams/dqn.yml
	@touch pyproject.toml
	@touch README.md
	@touch .gitignore
	@touch .env
	@touch logs/.gitkeep
	@touch videos/.gitkeep
	@touch models/.gitkeep
	@touch tensorboard/.gitkeep
	@echo "$(GREEN)✓ Project structure created successfully!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy the provided file contents into each file"
	@echo "  2. Run: make init"
	@echo "  3. Run: make install"
	@echo ""

init: ## Initialize uv virtual environment
	@echo "$(BLUE)Initializing virtual environment with uv...$(NC)"
	@if ! command -v uv &> /dev/null; then \
		echo "$(RED)Error: uv is not installed$(NC)"; \
		echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@uv venv
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo ""
	@echo "$(YELLOW)Activate with:$(NC)"
	@echo "  source .venv/bin/activate  # Linux/macOS"
	@echo "  .venv\\Scripts\\activate     # Windows"
	@echo ""

install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@uv sync
	@echo "$(GREEN)✓ Dependencies installed successfully!$(NC)"

verify: ## Verify installation and environment registration
	@echo "$(BLUE)Verifying installation...$(NC)"
	@uv run python -c "import gymnasium; import stable_baselines3; from loguru import logger; logger.info('✓ All imports successful!')"
	@echo "$(BLUE)Testing environment registration...$(NC)"
	@uv run python src/utils/register_envs.py
	@echo "$(GREEN)✓ Verification complete!$(NC)"

clean: ## Clean generated files and caches
	@echo "$(BLUE)Cleaning project...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .mypy_cache 2>/dev/null || true
	@rm -rf dist 2>/dev/null || true
	@rm -rf build 2>/dev/null || true
	@echo "$(GREEN)✓ Project cleaned$(NC)"

clean-all: clean ## Clean everything including logs, videos, and models
	@echo "$(BLUE)Deep cleaning project...$(NC)"
	@rm -rf logs/* videos/* models/* tensorboard/* 2>/dev/null || true
	@touch logs/.gitkeep videos/.gitkeep models/.gitkeep tensorboard/.gitkeep
	@rm -rf .venv 2>/dev/null || true
	@rm -f uv.lock 2>/dev/null || true
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@uv run pytest tests/ -v

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	@uv run black src/ scripts/ --line-length 100
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code...$(NC)"
	@uv run ruff check src/ scripts/
	@echo "$(GREEN)✓ Linting complete$(NC)"

type-check: ## Type check with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	@uv run mypy src/ scripts/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Training targets
run-lunarlander: ## Train PPO on LunarLander-v3
	@echo "$(BLUE)Training LunarLander with PPO...$(NC)"
	@uv run python src/train/train_lunarlander.py

run-cartpole: ## Train A2C on CartPole-v1
	@echo "$(BLUE)Training CartPole with A2C...$(NC)"
	@uv run python src/train/train_cartpole.py

run-gridworld: ## Train DQN on GridWorld-v0
	@echo "$(BLUE)Training GridWorld with DQN...$(NC)"
	@uv run python src/train/train_gridworld.py

# Evaluation targets
eval: ## Evaluate a model (usage: make eval MODEL=models/ppo_lunarlander.zip ENV=LunarLander-v3)
	@if [ -z "$(MODEL)" ] || [ -z "$(ENV)" ]; then \
		echo "$(RED)Error: MODEL and ENV parameters required$(NC)"; \
		echo "Usage: make eval MODEL=models/ppo_lunarlander.zip ENV=LunarLander-v3"; \
		exit 1; \
	fi
	@echo "$(BLUE)Evaluating model...$(NC)"
	@uv run python scripts/eval_model.py --model-path $(MODEL) --env $(ENV) --n-episodes 10

record: ## Record video (usage: make record MODEL=models/ppo_lunarlander.zip ENV=LunarLander-v3)
	@if [ -z "$(MODEL)" ] || [ -z "$(ENV)" ]; then \
		echo "$(RED)Error: MODEL and ENV parameters required$(NC)"; \
		echo "Usage: make record MODEL=models/ppo_lunarlander.zip ENV=LunarLander-v3"; \
		exit 1; \
	fi
	@echo "$(BLUE)Recording video...$(NC)"
	@uv run python scripts/record_video.py --model-path $(MODEL) --env $(ENV)

# Monitoring
tensorboard: ## Start TensorBoard server
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "$(YELLOW)Open your browser to: http://localhost:6006$(NC)"
	@uv run tensorboard --logdir tensorboard/

# rl_zoo3 integration
zoo-train: ## Train with rl_zoo3 (usage: make zoo-train ALGO=ppo ENV=LunarLander-v3)
	@if [ -z "$(ALGO)" ] || [ -z "$(ENV)" ]; then \
		echo "$(RED)Error: ALGO and ENV parameters required$(NC)"; \
		echo "Usage: make zoo-train ALGO=ppo ENV=LunarLander-v3"; \
		exit 1; \
	fi
	@echo "$(BLUE)Training with rl_zoo3...$(NC)"
	@uv run python -m rl_zoo3.train \
		--algo $(ALGO) \
		--env $(ENV) \
		--conf-file hyperparams/$(ALGO).yml \
		--tensorboard-log tensorboard/

zoo-optimize: ## Hyperparameter optimization (usage: make zoo-optimize ALGO=ppo ENV=CartPole-v1)
	@if [ -z "$(ALGO)" ] || [ -z "$(ENV)" ]; then \
		echo "$(RED)Error: ALGO and ENV parameters required$(NC)"; \
		echo "Usage: make zoo-optimize ALGO=ppo ENV=CartPole-v1"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running hyperparameter optimization...$(NC)"
	@uv run python -m rl_zoo3.train \
		--algo $(ALGO) \
		--env $(ENV) \
		--optimize \
		--n-trials 100 \
		--n-jobs 4

zoo-enjoy: ## Watch trained agent (usage: make zoo-enjoy ALGO=ppo ENV=LunarLander-v3)
	@if [ -z "$(ALGO)" ] || [ -z "$(ENV)" ]; then \
		echo "$(RED)Error: ALGO and ENV parameters required$(NC)"; \
		echo "Usage: make zoo-enjoy ALGO=ppo ENV=LunarLander-v3"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running trained agent...$(NC)"
	@uv run python -m rl_zoo3.enjoy \
		--algo $(ALGO) \
		--env $(ENV) \
		--folder models/

# Quick start
quickstart: setup ## Complete setup from scratch
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Project structure created!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy all file contents from the artifacts into the created files"
	@echo "  2. Run: $(GREEN)make init$(NC) to create virtual environment"
	@echo "  3. Run: $(GREEN)source .venv/bin/activate$(NC) to activate it"
	@echo "  4. Run: $(GREEN)make install$(NC) to install dependencies"
	@echo "  5. Run: $(GREEN)make verify$(NC) to verify installation"
	@echo "  6. Run: $(GREEN)make run-lunarlander$(NC) to start training!"
	@echo ""

# Development workflow
dev-setup: init install verify ## Complete development setup
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@echo "  $(GREEN)make run-lunarlander$(NC)  - Train LunarLander"
	@echo "  $(GREEN)make run-cartpole$(NC)     - Train CartPole"
	@echo "  $(GREEN)make run-gridworld$(NC)    - Train GridWorld"
	@echo "  $(GREEN)make tensorboard$(NC)       - Start TensorBoard"
	@echo "  $(GREEN)make help$(NC)              - Show all commands"
	@echo ""

# Information targets
info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "===================="
	@echo "Project: RL Pipeline"
	@echo "Python: $(shell python --version 2>/dev/null || echo 'Not activated')"
	@echo "UV: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "$(BLUE)Directory Status:$(NC)"
	@ls -la src/ 2>/dev/null || echo "src/ not found"
	@echo ""
	@echo "$(BLUE)Models:$(NC)"
	@ls -lh models/*.zip 2>/dev/null || echo "No models found"
	@echo ""
	@echo "$(BLUE)Videos:$(NC)"
	@ls -lh videos/*.mp4 2>/dev/null || echo "No videos found"

tree: ## Show project tree structure
	@echo "$(BLUE)Project Structure:$(NC)"
	@tree -L 3 -I '__pycache__|*.pyc|.venv|*.egg-info' || \
		(echo "$(YELLOW)tree command not found, using ls:$(NC)" && \
		find . -not -path '*/\.*' -not -path '*/__pycache__/*' -not -path '*/\.venv/*' | head -50)

# CI/CD targets
ci: format lint type-check test ## Run all CI checks
	@echo "$(GREEN)✓ All CI checks passed!$(NC)"

# Documentation
docs: ## Show documentation links
	@echo "$(BLUE)Documentation Links:$(NC)"
	@echo "===================="
	@echo "uv:              https://docs.astral.sh/uv/"
	@echo "stable-baselines3: https://stable-baselines3.readthedocs.io/"
	@echo "rl_zoo3:         https://github.com/DLR-RM/rl-baselines3-zoo"
	@echo "Gymnasium:       https://gymnasium.farama.org/"
	@echo "Loguru:          https://loguru.readthedocs.io/"
	@echo ""