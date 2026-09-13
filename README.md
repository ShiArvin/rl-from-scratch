# Reinforcement Learning Implementations and Experiments

A collection of reinforcement learning implementations and empirical studies completed during Summer 2026.

The project covers four representative deep reinforcement learning algorithms:

- **DQN** — value-based, off-policy reinforcement learning
- **TRPO** — trust-region policy optimization
- **PPO** — clipped policy-gradient optimization
- **SAC** — maximum-entropy, off-policy actor-critic

The implementations are based on PyTorch and Gymnasium, with experiments on Atari and MuJoCo environments.

---

## Algorithms

### DQN

Deep Q-Network implementation for Atari Pong.

Key components include:

- convolutional Q-network
- experience replay
- target network
- epsilon-greedy exploration
- hard target-network updates
- configurable learning rate and epsilon schedule

Experiments include:

- baseline training on Pong
- learning-rate ablation
- epsilon-decay ablation

The baseline agent reaches a test reward close to **21** on Pong after sufficient environment interaction.

Code:

```text
DQN-task/rlfromscratch/