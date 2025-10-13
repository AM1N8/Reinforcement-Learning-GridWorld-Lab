# Pacman Feature Extractors

## Overview

This repository contains multiple feature extraction implementations for Pacman Q-learning agents. Each extractor is designed with different complexity levels and strategic focuses, from basic gameplay to advanced ghost-hunting optimization.

---

## Feature Comparison Matrix

| Feature Type       | Simple | Advanced | Expert | Deep | GhostBusters |
|-------------------|--------|----------|--------|------|--------------|
| Food Distance      | ✓      | ✓        | ✓      | ✓    | ✓            |
| Food Density       | ✗      | ✓        | ✓      | ✓    | ✓            |
| Ghost Danger       | Basic  | Advanced | Predictive | Interactive | Adaptive |
| Scared Ghosts      | ✗      | ✓        | Strategic | ✓ | Aggressive   |
| Capsule Strategy   | ✗      | ✓        | ✓      | ✓    | Optimized    |
| Path Safety        | ✗      | ✓        | ✓      | ✓    | ✓            |
| Territory Control  | ✗      | ✗        | ✓      | ✓    | ✓            |
| Escape Routes      | ✗      | ✗        | ✓      | ✓    | ✓            |
| Interaction Terms  | ✗      | ✗        | ✗      | ✓    | ✓            |
| Ghost Hunting      | ✗      | Basic    | ✓      | ✓    | Elite        |

---

## Extractor Descriptions

### 1. SimpleExtractor – Basic Reflex Agent

**Philosophy:** Minimal feature set for fast learning on simple layouts.

**Key Features:**
- Basic food distance calculation
- Simple ghost proximity detection
- Immediate danger recognition
- No strategic planning

**Performance:**
- Training Episodes: 50-100
- Average Score: 400-600
- Win Rate: 60-70%

**Use Case:** Educational purposes, baseline comparisons, or very simple layouts.

**Command:**
```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l smallClassic
```

---

### 2. AdvancedExtractor – Comprehensive Game Understanding

**Philosophy:** Balanced approach combining tactical awareness with strategic planning.

**Key Features:**
- **Food Intelligence:** Density analysis and local clustering
- **Ghost Management:** Differentiates active vs scared ghosts, exponential danger scoring
- **Capsule Awareness:** Pursues capsules strategically when under pressure
- **Tactical Evaluation:** Dead-end detection, mobility assessment, path safety lookahead
- **Movement Optimization:** Penalizes stopping and encourages directional momentum

**Performance:**
- Training Episodes: 100-150
- Average Score: 800-1000
- Win Rate: 75-85%

**Use Case:** General-purpose extractor suitable for medium-sized layouts with moderate complexity.

**Command:**
```bash
python pacman.py -p ApproximateQAgent -a extractor=AdvancedExtractor -x 100 -n 110 -l mediumClassic
```

---

### 3. ExpertExtractor – Advanced Game Theory

**Philosophy:** Deep strategic planning with predictive modeling and game phase adaptation.

**Key Features:**
- **Game Phase Detection:** Dynamically adjusts strategy for early, mid, and endgame
- **Territory Control:** Analyzes reachable food relative to ghost positions
- **Ghost Prediction:** Models ghost movement assuming optimal pursuit behavior
- **Scared Ghost Hunting:** Evaluates timing and efficiency of ghost pursuit opportunities
- **Food Clustering:** Prioritizes dense food regions for optimal collection patterns
- **Escape Route Analysis:** Counts viable escape paths and identifies trap scenarios
- **Competition Metrics:** Compares Pacman's food accessibility against ghost interference
- **Oscillation Prevention:** Heavily penalizes direction reversal to avoid inefficient movement

**Performance:**
- Training Episodes: 150-200
- Average Score: 1000-1400
- Win Rate: 85-90%

**Use Case:** Large layouts, competitive scenarios, or situations requiring sophisticated planning.

**Command:**
```bash
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 150 -n 160 -l openClassic
```

---

### 4. DeepExtractor – Feature Engineering with Interactions

**Philosophy:** Maximum representational power through feature combinations and non-linear relationships.

**Key Features:**
- **Interaction Terms:** Cross-feature products capturing complex relationships
- **Risk-Reward Ratios:** Dynamic balance between danger levels and reward potential
- **Polynomial Features:** Non-linear transformations such as squared distances
- **Composite Metrics:** Multi-source derived features for enhanced pattern recognition
- **Adaptive Scaling:** Context-dependent feature normalization

**Performance:**
- Training Episodes: 200-300
- Average Score: 1100-1500
- Win Rate: 85-92%

**Use Case:** Complex layouts with sufficient training data, research applications requiring maximum expressiveness.

**Command:**
```bash
python pacman.py -p ApproximateQAgent -a extractor=DeepExtractor -x 250 -n 260 -l mediumClassic
```

---

### 5. GhostBustersExtractor – Elite Ghost Hunting Specialist

**Philosophy:** Aggressive capsule-ghost hunting strategy that maximizes score through power pellet optimization.

**Key Features:**

**Phase 1: Capsule Pursuit (Setup)**
- **Hunt Opportunity Recognition:** Values capsules based on potential ghost kills (200pts each)
- **Strategic Baiting:** Rewards running toward capsules while being chased to group ghosts
- **Context-Aware Timing:** Capsule value scales with nearby ghost count and urgency
- **Escape Routing:** Identifies optimal paths to capsules under pressure

**Phase 2: Ghost Hunting (Execution)**
- **Aggressive Pursuit:** Top priority given to scared ghost hunting (3.0x multiplier)
- **Multi-Ghost Planning:** Routes to catch 2-4 ghosts per power pellet
- **Timer Management:** Balances risk vs reward based on remaining scared time
- **Closing Behavior:** Rewards movement toward scared ghosts
- **Safety Checks:** Avoids ghosts transitioning back to dangerous state

**Strategic Enhancements:**
- **Dynamic Value Adjustment:** Food priority reduced during hunting opportunities (10pts vs 200pts)
- **Context-Aware Actions:** Smart stopping and direction reversal during pursuit
- **Lookahead Safety:** Two-step prediction of dangerous paths
- **Endgame Adaptation:** Aggressive food collection when few pellets remain
- **Mobility Analysis:** Dead-end avoidance weighted by ghost proximity

**Performance:**
- Training Episodes: 200-500
- Average Score: 1200-1800
- Win Rate: 90-95%
- Ghost Kills per Game: 8-15
- Capsule Efficiency: 400-800 points per power pellet

**Benchmark Results (mediumClassic, 500 episodes):**
```
Episode Range    | Average Reward
----------------|---------------
1-100           | 954.31
101-200         | 1518.88
201-300         | 1487.08
301-400         | 1463.92
401-500         | 1382.61
Final Average   | 1361.36

Test Results (10 games):
- Win Rate: 90% (9/10)
- Score Range: 1637-2064
- Average Score: 1826.50
```

**Use Case:** Maximizing score on any layout with power pellets. Essential for competitive play or score-based objectives.

**Command:**
```bash
python pacman.py -p ApproximateQAgent -a extractor=GhostBustersExtractor -x 500 -n 510 -l mediumClassic
```

**Advanced Usage:**
```bash
# High learning rate for faster convergence
python pacman.py -p ApproximateQAgent -a extractor=GhostBustersExtractor,alpha=0.3 -x 300 -n 310 -l mediumClassic

# Watch trained agent perform
python pacman.py -p ApproximateQAgent -a extractor=GhostBustersExtractor -n 5 -l mediumClassic
```

---

## Performance Summary

| Extractor        | Training Time | Avg Score | Win Rate | Best For |
|-----------------|---------------|-----------|----------|----------|
| SimpleExtractor | Fast (50ep)   | 400-600   | 60-70%   | Learning baseline |
| AdvancedExtractor | Medium (100ep) | 800-1000  | 75-85%   | General gameplay |
| ExpertExtractor | Medium (150ep) | 1000-1400 | 85-90%   | Strategic planning |
| DeepExtractor   | Slow (250ep)   | 1100-1500 | 85-92%   | Complex layouts |
| GhostBustersExtractor | Slow (500ep) | 1200-1800 | 90-95% | Maximum score |

---
