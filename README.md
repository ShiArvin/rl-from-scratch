If you find this repository useful, consider giving it a ⭐.
# Reinforcement Learning Implementations and Experiments

This repository contains my reinforcement learning implementations and experiments completed during Summer 2026.

The project covers four representative deep reinforcement learning algorithms:

- **DQN** — value-based off-policy reinforcement learning
- **TRPO** — trust-region policy optimization
- **PPO** — clipped policy-gradient optimization
- **SAC** — maximum-entropy off-policy actor-critic

The main goal of this project was to understand and implement the core mechanisms of these algorithms within a common reinforcement learning codebase, and to study their empirical behavior through experiments.

---

## 1. Deep Q-Network (DQN)

DQN was implemented and evaluated on **Atari Pong**.

### Main Components

- convolutional Q-network
- experience replay
- target network
- epsilon-greedy exploration
- TD learning
- target-network updates
- configurable exploration schedules

### Experiments

The experiments study:

- baseline performance on Pong
- the effect of learning rate
- the effect of epsilon-decay schedule
- training dynamics of reward, Q-value and TD loss

### Code

```text
DQN-task/rlfromscratch/
```

### Detailed Experimental Report

[View DQN Report](DQN-task/DQN_Report.pdf)

---

## 2. Trust Region Policy Optimization (TRPO)

TRPO was implemented for **MuJoCo continuous-control environments**.

### Main Components

- stochastic Gaussian policy
- actor-critic architecture
- generalized advantage estimation (GAE)
- surrogate policy objective
- KL-divergence constraint
- conjugate-gradient method
- Hessian-vector product
- backtracking line search

Both **rollout-based** and **trajectory-based** sampling schemes were studied.

### Code

```text
TRPO-task/RLfromscratch/
```

### Main Entry

```text
TRPO-task/RLfromscratch/run_trpo.py
```

### Detailed Experimental Report

[View TRPO Report](TRPO-task/TRPO_Report.pdf)

---

## 3. Proximal Policy Optimization (PPO)

PPO was implemented for **MuJoCo continuous-control tasks**.

### Main Components

- clipped surrogate objective
- actor-critic architecture
- generalized advantage estimation (GAE)
- advantage normalization
- multiple optimization epochs
- shuffled minibatch updates
- gradient clipping
- value-function clipping
- entropy regularization
- learning-rate decay

The implementation focuses on how PPO approximates trust-region policy optimization using a simpler clipped objective.

### Code

```text
PPO_SAC-task/rlfromscratch/
```

### Main Entry

```text
PPO_SAC-task/rlfromscratch/run_ppo.py
```

### Detailed Experimental Report

[View PPO & SAC Report](PPO_SAC-task/PPO_SAC_Report.pdf)

---

## 4. Soft Actor-Critic (SAC)

SAC was implemented for **MuJoCo continuous-control tasks** as an off-policy maximum-entropy actor-critic algorithm.

### Main Components

- stochastic Gaussian actor
- reparameterization trick
- twin Q-networks
- target Q-networks
- experience replay
- entropy-regularized policy objective
- automatic temperature tuning
- soft target-network updates

Experiments were conducted across multiple continuous-control environments, including comparisons involving entropy-temperature settings.

### Code

```text
PPO_SAC-task/rlfromscratch/
```

### Main Entry

```text
PPO_SAC-task/rlfromscratch/run_sac.py
```

### Detailed Experimental Report

[View PPO & SAC Report](PPO_SAC-task/PPO_SAC_Report.pdf)

---

## Project Structure

```text
rl-from-scratch/
│
├── DQN-task/
│   ├── rlfromscratch/
│   ├── DQN_pong_experiments/
│   └── DQN_Report.pdf
│
├── TRPO-task/
│   ├── RLfromscratch/
│   └── TRPO_Report.pdf
│
├── PPO_SAC-task/
│   ├── rlfromscratch/
│   └── PPO_SAC_Report.pdf
│
├── .gitignore
└── README.md
```

The reinforcement learning framework is organized into several reusable modules:

```text
agent/       policy and value-function agents
algorithm/   reinforcement learning algorithms
config/      experiment configurations
memory/      replay buffers and trajectory storage
logger/      training and evaluation logging
utils/       neural networks and utility functions
```

---

## Experimental Environments

### Atari

- Pong

### MuJoCo

- HalfCheetah
- Hopper
- Walker2d

Detailed hyperparameters, learning curves, ablation studies and experimental analysis are provided in the corresponding PDF reports.

---

## Tech Stack

- Python
- PyTorch
- Gymnasium
- MuJoCo
- NumPy
- Hydra
- SwanLab / TensorBoard

---

## Learning Progression

The project roughly follows the progression:

```text
DQN
 ↓
TRPO
 ↓
PPO
 ↓
SAC
```

Through these implementations and experiments, I focused on several central topics in deep reinforcement learning:

- value-based vs. policy-based methods
- on-policy vs. off-policy learning
- exploration vs. exploitation
- trust-region optimization
- policy clipping
- advantage estimation
- entropy regularization
- sample efficiency
- training stability

For detailed experimental settings, results and analysis, please refer to the corresponding reports.
