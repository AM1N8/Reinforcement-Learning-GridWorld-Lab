# Reinforcement Learning Framework: Algorithm Documentation

## Table of Contents
1. [Overview](#overview)
2. [Dynamic Programming Methods](#dynamic-programming-methods)
3. [Temporal-Difference Learning](#temporal-difference-learning)
4. [Deep Reinforcement Learning](#deep-reinforcement-learning)
5. [Optimization Techniques](#optimization-techniques)
6. [Architectural Patterns](#architectural-patterns)

---

## Overview

This framework implements a comprehensive suite of reinforcement learning algorithms for grid-world environments, ranging from classical dynamic programming to modern deep learning approaches. The algorithms are designed with a unified interface while maintaining their theoretical foundations.

### Core RL Components

**State Space**: Discrete 2D grid positions represented as `(row, col)` tuples or numpy arrays.

**Action Space**: Four cardinal directions (UP, DOWN, LEFT, RIGHT) with deterministic transitions.

**Reward Structure**:
- Goal reward: Positive reinforcement for reaching terminal states
- Step penalty: Negative cost per action (encourages efficiency)
- Obstacle penalty: Negative reinforcement for invalid moves

---

## Dynamic Programming Methods

Dynamic programming algorithms leverage complete knowledge of the environment model (transition dynamics and rewards) to compute optimal policies through systematic value function updates.

### Value Iteration

**Algorithm Class**: Model-based, off-policy planning

**Core Principle**: Iteratively applies the Bellman optimality operator until convergence.

**Mathematical Foundation**:
```
V(s) ← max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
```

In deterministic environments (our case):
```
V(s) ← max_a [R(s,a) + γ V(s')]
```

**Implementation Details**:

1. **Initialization**: Zero-valued function for all non-terminal states
2. **Sweep Strategy**: Synchronous updates across entire state space
3. **Convergence Criterion**: Maximum value change below threshold θ (default: 1e-6)
4. **Policy Extraction**: Greedy policy derived from converged value function

**Computational Complexity**: O(|S|² |A| K) where K is iterations to convergence

**Advantages**:
- Guaranteed convergence to optimal policy
- No exploration required
- Fast convergence in small state spaces

**Limitations**:
- Requires complete model knowledge
- Memory intensive for large state spaces

### Policy Iteration

**Algorithm Class**: Model-based, on-policy planning

**Two-Phase Approach**:

1. **Policy Evaluation**: Iteratively compute V^π for current policy
   ```
   V^π(s) ← R(s,π(s)) + γ V^π(s')
   ```

2. **Policy Improvement**: Greedily update policy
   ```
   π'(s) ← argmax_a [R(s,a) + γ V^π(s')]
   ```

**Implementation Details**:

- **Evaluation Termination**: Converges when ΔV < θ
- **Improvement Check**: Monitors policy stability
- **Early Stopping**: Terminates when policy unchanged

**Convergence Properties**:
- Typically fewer iterations than value iteration
- Each iteration more computationally expensive
- Guaranteed monotonic improvement

**Optimization**:
- Random initial policy spreads exploration
- Synchronous evaluation updates
- Policy stability check prevents unnecessary evaluations

---

## Temporal-Difference Learning

TD methods learn directly from experience without requiring a model, using bootstrapping to update value estimates based on other estimates.

### Monte Carlo Control

**Algorithm Class**: Model-free, on-policy learning

**Learning Strategy**: Episode-based updates using complete returns.

**Core Update Rule**:
```
Q(s,a) ← Q(s,a) + α[G_t - Q(s,a)]
```

Where G_t is the actual return from time t to episode end.

**Implementation Characteristics**:

1. **First-Visit MC**: Updates only first occurrence of (s,a) pair
2. **Return Calculation**: Backward accumulation with discount
   ```
   G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
   ```
3. **Value Estimation**: Running average of observed returns

**Exploration Strategy**: ε-greedy policy
```
π(a|s) = {
  1 - ε + ε/|A|  if a = argmax Q(s,a)
  ε/|A|          otherwise
}
```

**Advantages**:
- Unbiased estimates (uses actual returns)
- Simple to understand and implement
- No bootstrapping bias

**Limitations**:
- High variance in estimates
- Requires complete episodes
- Slow learning in sparse reward environments

### Q-Learning

**Algorithm Class**: Model-free, off-policy TD control

**Fundamental Innovation**: Learns optimal Q* regardless of behavior policy.

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**TD Target**: `r + γ max_a' Q(s',a')`
**TD Error**: `δ = TD_target - Q(s,a)`

**Key Properties**:

1. **Off-Policy Learning**: Behavior policy (ε-greedy) differs from target policy (greedy)
2. **Bootstrapping**: Updates based on estimated future values
3. **One-Step Updates**: Learns after each transition
4. **Optimistic Initialization**: Q-table initialized to zeros

**Convergence Requirements**:
- All state-action pairs visited infinitely often
- Learning rate satisfies Robbins-Monro conditions: Σα = ∞, Σα² < ∞
- Step-size schedule: α_t(s,a) = 1/(1 + visits(s,a))^ω where 0.5 < ω ≤ 1

**Implementation Optimizations**:

- **defaultdict with lambda**: Automatic Q-value initialization
- **Tuple state keys**: Hashable state representation
- **Constant step-size**: Trades guaranteed convergence for faster adaptation

**Exploration Techniques**:
- ε-greedy with constant ε
- Decaying ε strategies possible
- Optimistic initial values encourage exploration

### SARSA(λ)

**Algorithm Class**: Model-free, on-policy TD control with eligibility traces

**Extended Credit Assignment**: Bridges one-step and Monte Carlo methods through eligibility traces.

**Update Mechanism**:

1. **TD Error Calculation**:
   ```
   δ_t = r_{t+1} + γQ(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
   ```

2. **Eligibility Trace Update**:
   ```
   E(s,a) ← E(s,a) + 1  (accumulating traces)
   ```

3. **Q-Value Updates** (for all state-action pairs):
   ```
   Q(s,a) ← Q(s,a) + α δ_t E(s,a)
   E(s,a) ← γλ E(s,a)
   ```

**Trace Decay Parameter (λ)**:
- λ = 0: Equivalent to one-step SARSA
- λ = 1: Equivalent to Monte Carlo (in episodic tasks)
- 0 < λ < 1: Smooth interpolation

**Eligibility Trace Properties**:

- **Short-term memory**: Recent state-actions have higher eligibility
- **Temporal credit assignment**: Distributes updates across visited states
- **Faster learning**: Propagates rewards backward more efficiently

**Implementation Details**:

1. **Trace Reset**: E-table cleared at episode start
2. **Accumulating Traces**: Increments on revisits (alternative: replacing traces)
3. **Full Sweep Updates**: All (s,a) pairs updated per step (sparse in practice)

**On-Policy Nature**:
- Next action selected from current policy (ε-greedy)
- Updates reflect actual behavior
- More stable than Q-learning in some domains

**Advantages over Standard SARSA**:
- Faster convergence through multi-step credit assignment
- Better in domains with delayed rewards
- Tunable bias-variance tradeoff via λ

---

## Deep Reinforcement Learning

### Deep Q-Network (DQN)

**Algorithm Class**: Model-free, off-policy, value-based deep RL

**Motivation**: Function approximation for large/continuous state spaces where tabular methods fail.

**Neural Network Architecture**:

```
Input Layer: State representation (2D position)
    ↓
Hidden Layer 1: Fully connected (128 units) + ReLU
    ↓
Hidden Layer 2: Fully connected (128 units) + ReLU
    ↓
Output Layer: Q-values for all actions (4 units)
```

**Key Innovations**:

#### 1. Experience Replay

**Replay Buffer**: Fixed-size deque storing transitions (s, a, r, s', done)

**Sampling Strategy**: Uniform random sampling of minibatches

**Benefits**:
- **Breaks temporal correlation**: Samples i.i.d. from memory
- **Improves data efficiency**: Each experience used multiple times
- **Stabilizes training**: Smooths over recent experiences

**Implementation**:
```python
buffer = deque(maxlen=10000)
batch = random.sample(buffer, batch_size=64)
```

#### 2. Target Network

**Dual Network Architecture**:
- **Policy Network** (θ): Updated every step
- **Target Network** (θ⁻): Updated every N steps

**Update Rule**:
```
y_i = r_i + γ max_a' Q(s', a'; θ⁻)
Loss = MSE(Q(s, a; θ), y_i)
```

**Stabilization Mechanism**:
- Fixes TD target for N updates
- Reduces moving target problem
- Prevents divergence from correlation between Q and target

**Update Frequency**: θ⁻ ← θ every 10 episodes (configurable)

#### 3. Deep Q-Learning Update

**Loss Function**: Mean Squared Error (Huber loss alternative available)
```
L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
```

**Optimization**:
- **Optimizer**: Adam (adaptive learning rates)
- **Learning Rate**: 0.001 (default)
- **Gradient Clipping**: Prevents exploding gradients (optional)

**Training Loop**:
1. Select action via ε-greedy
2. Execute action, observe transition
3. Store transition in replay buffer
4. Sample minibatch and compute TD targets
5. Perform gradient descent step
6. Periodically update target network

#### 4. Exploration Strategy

**Epsilon-Greedy with Decay**:
```
ε_t = max(ε_end, ε_start × decay^t)
```

**Parameters**:
- ε_start = 1.0 (full exploration)
- ε_end = 0.01 (minimum exploration)
- decay = 0.995 (exponential decay)

**Adaptive Exploration**: High initial exploration transitions to exploitation

#### 5. Implementation Optimizations

**GPU Acceleration**:
- Automatic device detection (CUDA/CPU)
- Batch tensor operations
- Efficient memory transfers

**State Preprocessing**:
- Normalization to [0,1] range (optional)
- FloatTensor conversion
- Batch dimension handling

**Q-Table Approximation**:
- Periodic state sweeps for visualization
- Sparse updates during training
- Compatible with visualization tools

**Model Persistence**:
- Save/load functionality for checkpoints
- Optimizer state preservation
- Training metrics logging (JSON)

**Convergence Monitoring**:
- Episode rewards tracking
- Loss history
- Epsilon decay logging
- Exploration/exploitation ratio

---

## Optimization Techniques

### Hyperparameter Tuning

**Learning Rate (α)**:
- **Range**: [0.001, 0.5]
- **Impact**: Controls step size in Q-value updates
- **Trade-off**: High α → fast learning but instability; Low α → slow but stable

**Discount Factor (γ)**:
- **Range**: [0.9, 0.99]
- **Impact**: Balances immediate vs. future rewards
- **Selection**: High γ for long-term planning; Low γ for myopic tasks

**Exploration Rate (ε)**:
- **Range**: [0.01, 0.3]
- **Impact**: Exploration-exploitation balance
- **Strategy**: Decay from high to low over training

**Eligibility Trace Decay (λ)**:
- **Range**: [0.8, 0.95]
- **Impact**: Multi-step credit assignment strength
- **Selection**: High λ for sparse rewards; Low λ for noisy environments

### Convergence Acceleration

**Value Function Initialization**:
- Optimistic initialization encourages exploration
- Zero initialization for unbiased estimates
- Heuristic initialization for domain knowledge

**Policy Initialization**:
- Random policy for Policy Iteration
- Warm-start from simpler algorithms

**Update Scheduling**:
- Synchronous updates (dynamic programming)
- Asynchronous updates (TD learning)
- Prioritized sweeping (focus on important states)

### Numerical Stability

**Convergence Thresholds**:
- θ = 1e-6 for value iteration
- Prevents unnecessary iterations
- Balances accuracy and computation

**Floating-Point Precision**:
- Float32 for neural networks (speed)
- Float64 for tabular methods (accuracy)

**Gradient Clipping** (DQN):
- Prevents exploding gradients
- Norm-based or value-based clipping

---

## Architectural Patterns

### Modular Design

**Component Separation**:

1. **Model**: Environment dynamics (model.py)
2. **Environment**: Gym-compatible interface (environment.py)
3. **Algorithms**: Learning methods (algorithms.py, dqn.py)
4. **Agent**: Policy execution wrapper (agent.py)
5. **Visualizer**: Rendering and analysis (visualizer.py)

**Benefits**:
- Easy algorithm swapping
- Independent testing
- Clear interfaces

### Unified Interface

**Algorithm Interface Contract**:
```python
def get_action(state, training=False) -> int
def train(env, num_episodes, max_steps) -> (Q, rewards)
```

**Environment Interface** (Gymnasium-compatible):
```python
def reset() -> (state, info)
def step(action) -> (next_state, reward, terminated, truncated, info)
```

### Design Patterns

**Strategy Pattern**: Interchangeable algorithms
**Factory Pattern**: Agent creation with algorithm selection
**Observer Pattern**: Training metrics collection
**Template Method**: Common training loop structure

### State Representation

**Tabular Methods**:
- Tuple keys: `(row, col)` for dictionary-based Q-tables
- Hashable and efficient
- Direct state-action mapping

**Function Approximation**:
- Numpy arrays: `[row, col]` for neural network input
- Float tensors for PyTorch
- Batch-friendly representation

### Memory Management

**Replay Buffer**:
- Fixed-size circular buffer (deque)
- Automatic old transition removal
- Memory-efficient for large buffers

**State Space Storage**:
- Sparse representation (defaultdict)
- Only visited states stored
- Scales to larger environments

### Exploration Strategies

**Implemented**:
1. **ε-greedy**: Balanced random/greedy selection
2. **Decaying ε**: Annealing exploration over time
3. **Optimistic Initialization**: Encourages state-action coverage

**Potential Extensions**:
- Boltzmann exploration (softmax)
- UCB (Upper Confidence Bound)
- Thompson sampling

### Visualization Integration

**Training Monitoring**:
- Real-time reward curves
- Value function heatmaps
- Policy arrow plots
- Q-value visualizations

**Episode Analysis**:
- Trajectory plotting
- Step-by-step animation
- Learning process videos

**Performance Metrics**:
- Convergence analysis
- Exploration statistics
- Loss evolution (DQN)

---

## Algorithm Comparison

| Algorithm | Model-Free | Sample Efficiency | Convergence | Scalability |
|-----------|------------|-------------------|-------------|-------------|
| Value Iteration | No | N/A | Guaranteed | Small spaces |
| Policy Iteration | No | N/A | Guaranteed | Small spaces |
| Monte Carlo | Yes | Low | Slow | Medium |
| Q-Learning | Yes | Medium | Asymptotic | Medium |
| SARSA(λ) | Yes | High | Asymptotic | Medium |
| DQN | Yes | High | Empirical | Large spaces |

**Selection Guidelines**:
- **Known model + small space**: Value/Policy Iteration
- **Unknown model + tabular**: Q-Learning or SARSA(λ)
- **Large/continuous spaces**: DQN
- **Sample-limited scenarios**: SARSA(λ) or DQN with replay

---

## Theoretical Foundations

### Bellman Equations

**Bellman Expectation** (Policy Evaluation):
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

**Bellman Optimality** (Optimal Value):
```
V*(s) = max_a E[R_{t+1} + γV*(S_{t+1}) | S_t = s, A_t = a]
```

### Contraction Mapping

**Value Iteration Convergence**: Bellman operator is a γ-contraction
```
||T V - T U|| ≤ γ ||V - U||
```

**Implication**: Exponential convergence to unique fixed point

### TD Error

**General Form**:
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

**Expected TD Error**: Zero for correct value function under policy

### Function Approximation Challenges

**Deadly Triad** (Sutton & Barto):
1. Function approximation
2. Bootstrapping
3. Off-policy learning

**DQN Mitigations**:
- Experience replay (reduces correlation)
- Target networks (stabilizes bootstrapping)
- On-policy correction methods (optional)

---

## Future Extensions

**Potential Enhancements**:
- Double DQN (overestimation bias reduction)
- Dueling DQN (value/advantage decomposition)
- Prioritized Experience Replay (important sample weighting)
- Rainbow DQN (combination of improvements)
- Policy gradient methods (REINFORCE, A3C)
- Actor-Critic architectures
- Model-based RL (Dyna-Q)

**Scalability Improvements**:
- Distributed training
- Parallel environment simulation
- Sparse Q-table representations
- Function approximation for tabular methods

---

## References

**Foundational Papers**:
- Bellman, R. (1957). Dynamic Programming
- Watkins, C. (1989). Q-Learning
- Rummery & Niranjan (1994). SARSA
- Mnih et al. (2015). Human-level control through deep RL

**Textbook**:
- Sutton & Barto (2018). Reinforcement Learning: An Introduction

---

*This documentation covers the theoretical foundations, implementation details, and optimization techniques of the RL framework. For usage instructions, refer to the user guide.*
