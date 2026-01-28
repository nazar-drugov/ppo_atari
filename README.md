# Proximal Policy Optimization (PPO) Implementation


This module implements the PPO algorithm for reinforcement learning, supporting
Atari environments. I used this PPO implementation to train an agent to play Atari Breakout for 2,000,000 timesteps.

Key components:
- PPOAgent: Handles environment interaction and data collection
- ReplayMemory: Stores rollout data for training
- PPOTrainer: Orchestrates the training loop (rollout + learning phases)
- Network architectures: Actor (policy) and Critic (value) networks
- Loss functions: Clipped surrogate objective, value loss, entropy bonus

### Demo
https://github.com/user-attachments/assets/d2c57a73-65a8-41e4-bf95-a373865cfc09
