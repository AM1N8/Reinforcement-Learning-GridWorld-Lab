# Pacman Feature Extractors

## Feature Comparison

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

These extractors significantly improve Pacman's performance on challenging layouts.

---

## Running the Agents

```bash
# Test AdvancedExtractor
python pacman.py -p ApproximateQAgent -a extractor=AdvancedExtractor -x 50 -n 60 -l mediumGrid

# Test ExpertExtractor  
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l mediumClassic

# Test DeepExtractor (may need more training)
python pacman.py -p ApproximateQAgent -a extractor=DeepExtractor -x 200 -n 210 -l mediumClassic

# Large challenging layout
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l openClassic
```
```
┌──(amine㉿DESKTOP)-[ DeepRL]-[~/Projects/INBOX/DeepRL/projet3]-[pacman]                                  
└─$ python pacman.py -p ApproximateQAgent -a extractor=GhostBustersExtractor -x 500 -n 510 -l mediumClassic
Beginning 500 episodes of Training
Reinforcement Learning Status:
        Completed 100 out of 500 training episodes
        Average Rewards over all training: 954.31
        Average Rewards for last 100 episodes: 954.31
        Episode took 19.50 seconds
Reinforcement Learning Status:
        Completed 200 out of 500 training episodes
        Average Rewards over all training: 1236.60
        Average Rewards for last 100 episodes: 1518.88
        Episode took 18.45 seconds
Reinforcement Learning Status:
        Completed 300 out of 500 training episodes
        Average Rewards over all training: 1320.09
        Average Rewards for last 100 episodes: 1487.08
        Episode took 18.66 seconds
Reinforcement Learning Status:
        Completed 400 out of 500 training episodes
        Average Rewards over all training: 1356.05
        Average Rewards for last 100 episodes: 1463.92
        Episode took 19.29 seconds
Reinforcement Learning Status:
        Completed 500 out of 500 training episodes
        Average Rewards over all training: 1361.36
        Average Rewards for last 100 episodes: 1382.61
        Episode took 17.29 seconds
Training Done (turning off epsilon and alpha)
---------------------------------------------
Pacman died! Score: -184
Pacman emerges victorious! Score: 2064
Pacman emerges victorious! Score: 1637
Pacman emerges victorious! Score: 1860
Pacman emerges victorious! Score: 1806
Pacman emerges victorious! Score: 1928
```


## 1. AdvancedExtractor – Comprehensive Game Understanding

**Key Features:**

- **Food Features:** Density analysis and local clustering.
- **Ghost Intelligence:** Differentiates active vs scared ghosts, identifies danger zones, applies exponential danger scoring.
- **Capsule Strategy:** Pursues capsules strategically when under pressure.
- **Tactical Awareness:** Detects dead-ends, evaluates mobility, and looks ahead for path safety.
- **Movement Optimization:** Penalizes stopping and rewards smooth momentum.

**Use Case:** Ideal for medium-sized layouts to improve general gameplay performance.

---

## 2. ExpertExtractor – Advanced Game Theory

**Key Features:**

- **Game Phase Detection:** Adjusts strategy for early, mid, and endgame phases.
- **Territory Control:** Analyzes which food is reachable given ghost positions.
- **Ghost Prediction:** Predicts ghost movements assuming they chase Pacman.
- **Scared Ghost Hunting:** Evaluates opportunities for pursuing scared ghosts efficiently.
- **Food Clustering:** Prefers dense food areas for optimal collection.
- **Escape Route Analysis:** Counts viable escape paths and detects potential traps.
- **Competition Metrics:** Compares Pacman’s distance to food relative to ghosts.

**Use Case:** Best suited for large layouts or competitive scenarios requiring advanced planning.

---

## 3. DeepExtractor – Feature Engineering with Interactions

**Key Features:**

- **Interaction Terms:** Combines multiple features (e.g., food-ghost interactions) for richer representations.
- **Risk-Reward Ratios:** Balances danger against potential rewards.
- **Polynomial Features:** Captures non-linear relationships such as squared distances.
- **Composite Metrics:** Derives complex features from multiple sources for improved learning.

**Use Case:** Maximizes learning capacity on complex layouts when sufficient training data is available.
