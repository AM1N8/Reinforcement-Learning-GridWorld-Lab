```
rl-playground/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_algorithm.py
│   │   └── dqn.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── logger.py
│   │   └── plotter.py
│   ├── experiments/
│   │   ├── train_dqn.py
|   |   ├── train_sb3.py
│   │   └── evaluate.py
│   ├── __init__.py
│   └── environment.py
├── configs/
│   └── default.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

## run files : 
```bash
uv sync
.venv\Scripts\activate
```
```
uv run src/experiments/evaluate.py --checkpoint checkpoints/final_model.pt --episodes 10

uv run src/experiments/evaluate.py --checkpoint checkpoints/final_model.pt --episodes 5 --render
```


