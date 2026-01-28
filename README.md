# PPO for Atari Games

A PyTorch implementation of Proximal Policy Optimization (PPO) for training agents on Atari games using Gymnasium.

## Overview

This repository contains a clean, well-documented implementation of the PPO algorithm (Schulman et al., 2017) designed for Atari environments. The implementation follows best practices from OpenAI baselines and includes proper environment wrappers for Atari preprocessing.

## Features

- PPO algorithm with clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Atari environment wrappers (frame stacking, preprocessing, etc.)
- Weights & Biases integration for experiment tracking
- GPU support
- Visualization utilities with Plotly

## Project Structure

```
.
├── ppo.py              # Main PPO implementation
├── atari_wrappers.py   # Atari environment wrappers
├── rl_utils.py         # RL utility functions
├── gpu_env.py          # GPU environment setup
├── plotly_utils.py     # Visualization utilities
├── utils.py            # General utilities
└── play_breakout.py    # Interactive demo script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppo_atari.git
cd ppo_atari
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install ROMs for Atari games:
```bash
pip install gymnasium[accept-rom-license]
```

## Usage

### Training an Agent

Run the main PPO script to train an agent:

```bash
python ppo.py
```

The script will automatically use GPU if available and log training metrics to Weights & Biases.

### Playing Breakout Manually

Try the interactive demo to play Breakout yourself:

```bash
python play_breakout.py
```

Controls:
- Left/Right arrows: Move paddle
- F: Fire ball
- R: Reset
- Q: Quit

## Key Components

### PPO Algorithm

The implementation includes:
- **PPOAgent**: Handles environment interaction and rollout collection
- **PPOTrainer**: Orchestrates the training loop
- **ReplayMemory**: Stores and samples trajectory data
- Actor-Critic architecture with CNN for Atari
- Clipped surrogate objective loss
- Value function loss with optional clipping
- Entropy bonus for exploration

### Atari Wrappers

Standard preprocessing wrappers:
- Frame resizing and grayscaling
- Frame stacking
- Reward clipping
- Episode life management
- No-op randomization

## Configuration

Key hyperparameters can be modified in `ppo.py`:
- Learning rate
- Batch size and minibatch size
- PPO clip range
- GAE lambda
- Entropy coefficient
- Number of training epochs per rollout

## Dependencies

- PyTorch
- Gymnasium
- ALE (Arcade Learning Environment)
- OpenCV
- Matplotlib
- Plotly
- Weights & Biases
- einops
- jaxtyping

See `requirements.txt` for full details.

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [OpenAI Baselines](https://github.com/openai/baselines)

## License

MIT License
