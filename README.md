Feature Comparison

| Feature Type       | SimpleExtractor | AdvancedExtractor | ExpertExtractor | DeepExtractor |
|-------------------|----------------|-----------------|----------------|---------------|
| Food Distance      | ✓              | ✓               | ✓              | ✓             |
| Food Density       | ✗              | ✓               | ✓              | ✓             |
| Ghost Danger       | Basic          | Advanced        | Predictive     | Interactive   |
| Scared Ghosts      | ✗              | ✓               | Strategic      | ✓             |
| Capsule Strategy   | ✗              | ✓               | ✓              | ✓             |
| Path Safety        | ✗              | ✓               | ✓              | ✓             |
| Territory Control  | ✗              | ✗               | ✓              | ✓             |
| Escape Routes      | ✗              | ✗               | ✓              | ✓             |
| Interaction Terms  | ✗              | ✗               | ✗              | ✓             |


These extractors should significantly improve Pacman's performance on challenging

```
# Test AdvancedExtractor
python pacman.py -p ApproximateQAgent -a extractor=AdvancedExtractor -x 50 -n 60 -l mediumGrid

# Test ExpertExtractor  
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l mediumClassic

# Test DeepExtractor (may need more training)
python pacman.py -p ApproximateQAgent -a extractor=DeepExtractor -x 200 -n 210 -l mediumClassic

# Large challenging layout
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l openClassic
```

1. AdvancedExtractor - Comprehensive Game Understanding
Key Features:

Food Features: Density analysis, local clustering
Ghost Intelligence: Separates active vs scared ghosts, danger zones, exponential danger scoring
Capsule Strategy: Strategic capsule pursuit when under pressure
Tactical Awareness: Dead-end detection, mobility scoring, path safety lookahead
Movement Optimization: Penalizes stopping, rewards momentum

Use Case: General improvement for medium-sized layouts
2. ExpertExtractor - Advanced Game Theory
Key Features:

Game Phase Detection: Adapts strategy for early/mid/endgame
Territory Control: Analyzes which food is actually reachable before ghosts
Ghost Prediction: Predicts where ghosts will move (assumes they chase Pacman)
Scared Ghost Hunting: Calculates value of pursuing scared ghosts with time constraints
Food Clustering: Prefers dense food areas for efficiency
Escape Route Analysis: Counts viable escape paths, detects traps
Competition Metrics: Compares Pacman's distance to food vs ghost distances

Use Case: Large layouts and competitive scenarios
3. DeepExtractor - Feature Engineering with Interactions
Key Features:

Interaction Terms: Combines features (e.g., food-ghost-interaction)
Risk-Reward Ratios: Balances danger against opportunity
Polynomial Features: Non-linear relationships (squared distances)
Composite Metrics: Complex derived features

Use Case: Maximum learning capacity with sufficient training data
