Proximal Policy Optimization (PPO) Implementation

This module implements the PPO algorithm for reinforcement learning, supporting
Atari environments.

Key components:
    - PPOAgent: Handles environment interaction and data collection
    - ReplayMemory: Stores rollout data for training
    - PPOTrainer: Orchestrates the training loop (rollout + learning phases)
    - Network architectures: Actor (policy) and Critic (value) networks
    - Loss functions: Clipped surrogate objective, value loss, entropy bonus

I used this PPO implementation to train an agent to play Atari Breakout for 2,000,000 timesteps.

### Demo
