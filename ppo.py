"""
Proximal Policy Optimization (PPO) Implementation

This module implements the PPO algorithm for reinforcement learning, supporting
classic control, Atari, and MuJoCo environments. The implementation follows
the PPO paper (Schulman et al., 2017) with best practices from the OpenAI
baselines implementation.

Key components:
    - PPOAgent: Handles environment interaction and data collection
    - ReplayMemory: Stores rollout data for training
    - PPOTrainer: Orchestrates the training loop (rollout + learning phases)
    - Network architectures: Actor (policy) and Critic (value) networks
    - Loss functions: Clipped surrogate objective, value loss, entropy bonus

The code is organized as follows:
    1. Imports and setup
    2. Utility functions (seeding, episode data extraction)
    3. Network architecture functions
    4. Loss computation functions
    5. Data structures (ReplayMemory, ReplayMinibatch)
    6. Agent and Trainer classes
    7. Main execution
"""

import itertools
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Import ale_py FIRST to register ALE namespace with gymnasium
# This must happen before importing gymnasium
import ale_py
import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_utils import make_env, prepare_atari_env
from plotly_utils import plot_cartpole_obs_and_dones


def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Purpose:
        Extract episode statistics (length, reward, duration) from the first terminated
        environment in the provided info dictionary. This is used for logging training
        progress to wandb.

    Parameters:
     * infos (dict) : dictionary containing environment step information, including
                     "final_info" list with episode data for terminated environments

    Returns:
        A dictionary containing episode_length, episode_reward, and episode_duration
        if at least one environment terminated, otherwise None.
    """
    """
    Extract episode statistics (length, reward, duration) from the first terminated
    environment in the provided info dictionary. This is used for logging training
    progress.

    Parameters:
        infos (dict) : dictionary containing environment step information, including
                       "final_info" list with episode data for terminated environments

    Returns:
        A dictionary containing episode_length, episode_reward, and episode_duration
        if at least one environment terminated, otherwise None.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }
    return None

def set_global_seeds(seed: int) -> None:
    """
    Purpose:
        Set seeds for Python's random module, NumPy, and PyTorch to ensure
        experiments are reproducible across runs.

    Parameters:
     * seed (int) : the seed value to use for all random number generators

    Returns:
        None
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    if t.backends.mps.is_available():
        t.mps.manual_seed(seed)

# Import probes from the probes module (parent directory)
from probes import Probe1, Probe2, Probe3, Probe4, Probe5

def arg_help(args: "PPOArgs", print_df: bool = False) -> None:
    """
    Purpose:
        Display a formatted list of all PPO hyperparameters and their values.
        Useful for debugging and ensuring correct configuration.

    Parameters:
     * args (PPOArgs) : the configuration dataclass to display
     * print_df (bool) : if True, print as pandas DataFrame; otherwise print as formatted text

    Returns:
        None (prints to stdout)
    """
    if print_df:
        import pandas as pd
        df = pd.DataFrame([
            (repr(getattr(args, name)), name) 
            for name in args.__dataclass_fields__
        ])
        df.columns = ["default value", "arg"]
        print(df)
    else:
        print("PPO Arguments:")
        for name in args.__dataclass_fields__:
            print(f"  {name}: {repr(getattr(args, name))}")

# ============================================================================
# Global Configuration
# ============================================================================

# Register probe environments for testing
probes = [Probe1, Probe2, Probe3, Probe4, Probe5]
for idx, probe in enumerate(probes):
    gym.envs.registration.register(id=f"Probe{idx + 1}-v0", entry_point=probe)

# Type alias for numpy arrays
Arr = np.ndarray

# Device selection: prefer MPS (Apple Silicon) > CUDA > CPU
# MPS provides GPU acceleration on Mac, CUDA on Linux/Windows with NVIDIA GPUs
device = t.device(
    "mps" if t.backends.mps.is_available() 
    else "cuda" if t.cuda.is_available() 
    else "cpu"
)


@dataclass
class PPOArgs:
    """
    Purpose:
        Configuration dataclass containing all hyperparameters for PPO training.
        Automatically computes derived parameters like batch_size and total_phases
        based on the provided settings.
    """
    
    # === Basic Configuration ===
    seed: int = 1  # Random seed for reproducibility
    env_id: str = "CartPole-v1"  # Gymnasium environment ID
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control"  # Environment type

    # === Logging Configuration ===
    use_wandb: bool = False  # Whether to log to Weights & Biases
    video_log_freq: int | None | Literal["last"] = None  # Frequency of video logging
                                                         # None = disabled, int = every N episodes,
                                                         # "last" = only final episode after training
    num_training_videos: int = 10  # Number of videos to record during training (at step milestones)
    num_final_videos: int = 10  # Number of videos to record at the end of training
    checkpoint_freq: int = 250_000  # Save checkpoint every N steps (0 = only at end)
    wandb_project_name: str = "PPOCartPole"  # W&B project name
    wandb_entity: str = None  # W&B entity/username (None = default)

    # === Rollout Configuration ===
    total_timesteps: int = 1_000_000  # Total environment steps to collect
    num_envs: int = 4  # Number of parallel environments
    num_steps_per_rollout: int = 128  # Steps to collect before each learning phase
    num_minibatches: int = 4  # Number of minibatches per learning phase
    batches_per_learning_phase: int = 4  # Number of times to reuse rollout data

    # === Optimization Hyperparameters ===
    lr: float = 2.5e-4  # Learning rate for Adam optimizer
    max_grad_norm: float = 0.5  # Maximum gradient norm for clipping

    # === RL Hyperparameters ===
    gamma: float = 0.99  # Discount factor for future rewards

    # === PPO-Specific Hyperparameters ===
    gae_lambda: float = 0.95  # GAE parameter (bias-variance trade-off)
    clip_coef: float = 0.2  # PPO clipping coefficient (epsilon in paper)
    ent_coef: float = 0.01  # Entropy bonus coefficient (encourages exploration)
    vf_coef: float = 0.25  # Value function loss coefficient

    def __post_init__(self):
        """
        Purpose:
            Compute derived parameters that depend on the configuration.
            Validates that batch_size is compatible with minibatch settings.
        """
        # Total number of transitions collected per rollout
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        # Ensure batch can be evenly divided into minibatches
        assert self.batch_size % self.num_minibatches == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"num_minibatches ({self.num_minibatches})"
        )
        
        # Size of each training minibatch
        self.minibatch_size = self.batch_size // self.num_minibatches
        
        # Total number of rollout phases (each phase collects one batch)
        self.total_phases = self.total_timesteps // self.batch_size
        
        # Total number of gradient updates (accounts for multiple passes over data)
        self.total_training_steps = (
            self.total_phases * self.batches_per_learning_phase * self.num_minibatches
        )

        # Set video save path relative to the script directory
        script_dir = Path(__file__).parent
        self.video_save_path = script_dir / "videos"

        # Calculate video milestone steps (record videos at 1/10 intervals during training)
        if self.num_training_videos > 0:
            self.video_milestone_steps = [
                int(self.total_timesteps * (i + 1) / (self.num_training_videos + 1))
                for i in range(self.num_training_videos)
            ]
        else:
            self.video_milestone_steps = []


# Example configuration (commented out - uncomment to use)
# args = PPOArgs(
#     num_minibatches=2
# )
# arg_help(args)


def layer_init(layer: nn.Linear | nn.Conv2d, std: float = np.sqrt(2), bias_const: float = 0.0):
    """
    Purpose:
        Initialize a neural network layer with orthogonal weight initialization and
        constant bias. Orthogonal initialization helps with training stability by
        preserving gradient magnitudes through the network.

    Parameters:
     * layer (nn.Linear | nn.Conv2d) : the layer to initialize
     * std (float) : standard deviation for orthogonal weight initialization (default: sqrt(2))
     * bias_const (float) : constant value to initialize bias to (default: 0.0)

    Returns:
        The initialized layer (same object, modified in-place)
    """
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(
    envs: gym.vector.SyncVectorEnv,
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control",
) -> tuple[nn.Module, nn.Module]:
    """
    Purpose:
        Create and return the actor (policy) and critic (value) networks for PPO.
        The architecture depends on the environment mode: classic-control uses MLPs,
        atari uses CNNs, and mujoco uses different architectures.

    Parameters:
     * envs (gym.vector.SyncVectorEnv) : the vectorized environment to extract observation
                                        and action space dimensions from
     * mode (Literal) : the environment mode, determines network architecture
                       ("classic-control", "atari", or "mujoco")

    Returns:
        A tuple of (actor, critic) networks, both moved to the appropriate device
    """
    assert mode in ["classic-control", "atari", "mujoco"]

    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else np.array(envs.single_action_space.shape).prod()
    )

    if mode == "classic-control":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)
    if mode == "atari":
        actor, critic = get_actor_and_critic_atari(
            obs_shape, num_actions
        )  
    if mode == "mujoco":
        actor, critic = get_actor_and_critic_mujoco(
            num_obs, num_actions
        )  

    return actor.to(device), critic.to(device)


def get_actor_and_critic_classic(num_obs: int, num_actions: int) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Purpose:
        Create actor and critic networks for classic control environments (e.g., CartPole).
        Both networks use a simple MLP architecture with two hidden layers and Tanh activations.
        The actor outputs logits for action selection, while the critic outputs a single value.

    Parameters:
     * num_obs (int) : dimensionality of the observation space
     * num_actions (int) : number of discrete actions available

    Returns:
        A tuple of (actor, critic) networks as nn.Sequential modules
    """
    # Actor network: maps observations to action logits
    # Small std (0.01) on final layer encourages exploration early in training
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01)  # Small std for stable initial policy
    )

    # Critic network: maps observations to state values
    # Standard std (1.0) on final layer for value estimation
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0)  # Standard initialization for value function
    )
    
    return actor, critic


@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs"],
    next_terminated: Bool[Tensor, "num_envs"],
    rewards: Float[Tensor, "buffer_size num_envs"],
    values: Float[Tensor, "buffer_size num_envs"],
    terminated: Bool[Tensor, "buffer_size num_envs"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs"]:
    """
    Purpose:
        Compute advantages using Generalized Advantage Estimation (GAE). GAE provides
        a bias-variance trade-off for advantage estimation by using a weighted average
        of n-step returns. This is more stable than Monte Carlo returns and less biased
        than TD(0) returns.

    Parameters:
     * next_value (Float[Tensor, "num_envs"]) : value estimates for the next state after
                                                the rollout buffer
     * next_terminated (Bool[Tensor, "num_envs"]) : whether each environment terminated
                                                    after the rollout
     * rewards (Float[Tensor, "buffer_size num_envs"]) : rewards received at each timestep
     * values (Float[Tensor, "buffer_size num_envs"]) : value estimates for each state
                                                        in the buffer
     * terminated (Bool[Tensor, "buffer_size num_envs"]) : whether each environment
                                                           terminated at each timestep
     * gamma (float) : discount factor for future rewards
     * gae_lambda (float) : GAE parameter controlling bias-variance trade-off
                           (0 = high bias/low variance, 1 = low bias/high variance)

    Returns:
        Advantages for each timestep in the buffer, computed using GAE
    """
    buffer_size, num_envs = rewards.shape
    advantages = t.empty((buffer_size, num_envs), dtype=t.float32, device=rewards.device)

    # Compute the last advantage A_{T-1} separately
    # This uses the next_value as the bootstrap, accounting for termination
    advantages[-1] = (
        rewards[-1] 
        + (1 - next_terminated.float()) * gamma * next_value 
        - values[-1]
    )

    # Compute advantages backwards through time using GAE
    # GAE recursively combines TD errors with exponential decay controlled by gae_lambda
    for t_step in range(buffer_size - 2, -1, -1):
        # TD error: difference between actual reward + next value and current value estimate
        delta_t = (
            rewards[t_step] 
            + (1 - terminated[t_step + 1].float()) * gamma * values[t_step + 1] 
            - values[t_step]
        )
        # GAE advantage: TD error + discounted next advantage (if not terminated)
        advantages[t_step] = (
            delta_t 
            + (1 - terminated[t_step + 1].float()) * gamma * gae_lambda * advantages[t_step + 1]
        )

    return advantages


def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """
    Purpose:
        Generate random minibatch indices for sampling from the rollout buffer.
        Each minibatch contains a random subset of indices, and all minibatches together
        cover the entire batch without overlap. This shuffling helps decorrelate samples
        and improves training stability.

    Parameters:
     * rng (Generator) : NumPy random number generator for reproducibility
     * batch_size (int) : total number of samples in the batch
     * minibatch_size (int) : size of each minibatch (must divide batch_size evenly)

    Returns:
        A list of arrays, where each array contains minibatch_size random indices
        and together they partition [0, 1, ..., batch_size - 1]
    """
    assert batch_size % minibatch_size == 0, "batch_size must be divisible by minibatch_size"
    
    num_minibatches = batch_size // minibatch_size
    # Shuffle indices and split into minibatches
    shuffled_indices = rng.permutation(batch_size)
    minibatch_indices = shuffled_indices.reshape((num_minibatches, minibatch_size))
    return list(minibatch_indices)


@dataclass
class ReplayMinibatch:
    """
    Purpose:
        Container for a minibatch of transitions ready for neural network training.
        All data is stored as PyTorch tensors on the appropriate device. This represents
        a random subset of the rollout buffer, used for one gradient update.

    Data fields represent:
        - obs: states s_t
        - actions: actions a_t taken in those states
        - logprobs: log probabilities log π_old(a_t|s_t) under the old policy
        - advantages: advantage estimates A_t computed via GAE
        - returns: target returns A_t + V_old(s_t) for value function learning
        - terminated: termination flags d_{t+1} indicating if next state is terminal
    """

    obs: Float[Tensor, " minibatch_size *obs_shape"]
    actions: Int[Tensor, " minibatch_size *action_shape"]
    logprobs: Float[Tensor, " minibatch_size"]
    advantages: Float[Tensor, " minibatch_size"]
    returns: Float[Tensor, " minibatch_size"]
    terminated: Bool[Tensor, " minibatch_size"]


class ReplayMemory:
    """
    Purpose:
        Stores rollout data collected during the rollout phase. The buffer grows as
        transitions are added, then is consumed during the learning phase to create
        training minibatches. After extraction, the buffer is reset for the next rollout.

    The buffer stores data in numpy arrays (for efficiency) and converts to PyTorch
    tensors when creating minibatches. Data is organized as (timesteps, num_envs, ...).
    """

    rng: Generator
    obs: Float[Arr, " buffer_size num_envs *obs_shape"]
    actions: Int[Arr, " buffer_size num_envs *action_shape"]
    logprobs: Float[Arr, " buffer_size num_envs"]
    values: Float[Arr, " buffer_size num_envs"]
    rewards: Float[Arr, " buffer_size num_envs"]
    terminated: Bool[Arr, " buffer_size num_envs"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        """
        Purpose:
            Initialize the replay memory buffer with empty arrays. The buffer will
            grow dynamically as transitions are added during rollout collection.

        Parameters:
         * num_envs (int) : number of parallel environments
         * obs_shape (tuple) : shape of a single observation
         * action_shape (tuple) : shape of a single action
         * batch_size (int) : total number of transitions in a full rollout
         * minibatch_size (int) : size of each training minibatch
         * batches_per_learning_phase (int) : number of times to reuse the rollout data
         * seed (int) : random seed for minibatch shuffling
        """
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """
        Purpose:
            Clear all stored transitions from the buffer. Called after extracting
            minibatches to prepare for the next rollout phase.

        Parameters:
            None (uses self attributes)

        Returns:
            None (modifies self in place)
        """
        # Initialize empty arrays with correct shapes and dtypes
        # Shape: (0, num_envs, ...) will grow to (T, num_envs, ...) as we add timesteps
        self.obs = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.terminated = np.empty((0, self.num_envs), dtype=bool)

    def add(
        self,
        obs: Float[Arr, " num_envs *obs_shape"],
        actions: Int[Arr, " num_envs *action_shape"],
        logprobs: Float[Arr, " num_envs"],
        values: Float[Arr, " num_envs"],
        rewards: Float[Arr, " num_envs"],
        terminated: Bool[Arr, " num_envs"],
    ) -> None:
        """
        Purpose:
            Add a batch of transitions (one step from all environments) to the replay buffer.
            The buffer grows as we collect more experience during the rollout phase.

        Parameters:
         * obs (Float[Arr, " num_envs *obs_shape"]) : observations for all environments
         * actions (Int[Arr, " num_envs *action_shape"]) : actions taken in all environments
         * logprobs (Float[Arr, " num_envs"]) : log probabilities of actions under old policy
         * values (Float[Arr, " num_envs"]) : value estimates from old critic network
         * rewards (Float[Arr, " num_envs"]) : rewards received from environment
         * terminated (Bool[Arr, " num_envs"]) : whether each environment terminated

        Returns:
            None (modifies self in place)
        """
        # Validate shapes and types before adding to buffer
        for data, expected_shape in zip(
            [obs, actions, logprobs, values, rewards, terminated],
            [self.obs_shape, self.action_shape, (), (), (), ()],
        ):
            assert isinstance(data, np.ndarray), f"Expected numpy array, got {type(data)}"
            assert data.shape == (self.num_envs, *expected_shape), \
                f"Shape mismatch: got {data.shape}, expected {(self.num_envs, *expected_shape)}"

        # Append new timestep to buffer (add new dimension for time)
        # This grows the buffer: (T, num_envs, ...) -> (T+1, num_envs, ...)
        self.obs = np.concatenate((self.obs, obs[None, :]))
        self.actions = np.concatenate((self.actions, actions[None, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :]))
        self.values = np.concatenate((self.values, values[None, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :]))

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """
        Purpose:
            Convert stored rollout data into training minibatches. Computes advantages
            using GAE, then creates multiple random minibatches for multiple passes over
            the data. This improves sample efficiency by reusing the same rollout data
            multiple times with different randomizations.

        Parameters:
         * next_value (Tensor) : value estimates for states after the rollout (for GAE bootstrap)
         * next_terminated (Tensor) : termination flags for states after the rollout
         * gamma (float) : discount factor
         * gae_lambda (float) : GAE parameter

        Returns:
            A list of ReplayMinibatch objects. The total number is:
            batches_per_learning_phase * num_minibatches
            This allows multiple passes over the same data with different randomizations.
        """
        # Convert numpy arrays to PyTorch tensors on the correct device
        # Explicit dtypes are important for MPS compatibility (MPS doesn't support float64)
        obs = t.tensor(self.obs, device=device, dtype=t.float32)
        actions = t.tensor(self.actions, device=device, dtype=t.int64)
        logprobs = t.tensor(self.logprobs, device=device, dtype=t.float32)
        values = t.tensor(self.values, device=device, dtype=t.float32)
        rewards = t.tensor(self.rewards, device=device, dtype=t.float32)
        terminated = t.tensor(self.terminated, device=device, dtype=t.bool)

        # Compute advantages using Generalized Advantage Estimation
        # This provides better bias-variance trade-off than simple TD or Monte Carlo
        advantages = compute_advantages(
            next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda
        )
        
        # Returns are advantages + old values (used as targets for value function learning)
        returns = advantages + values

        # Create minibatches: multiple passes over the data with random shuffling
        # This improves sample efficiency by reusing the same rollout multiple times
        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            # Get random minibatch indices for this pass
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                # Flatten time and env dimensions: (T, num_envs, ...) -> (T*num_envs, ...)
                # Then index into the flattened array to get the minibatch
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 1)[indices]
                            for data in [obs, actions, logprobs, advantages, returns, terminated]
                        ]
                    )
                )

        # Clear memory after extracting minibatches (we only use each rollout once)
        self.reset()

        return minibatches
    

class PPOAgent:
    """
    Purpose:
        Handles the interaction between the PPO agent and the environment. Manages
        action selection, environment stepping, and data collection. Tracks the
        current state of all parallel environments.
    """

    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        actor: nn.Module,
        critic: nn.Module,
        memory: ReplayMemory,
    ):
        """
        Purpose:
            Initialize the PPO agent with networks, environments, and memory buffer.
            Resets all environments and initializes tracking variables.

        Parameters:
         * envs (gym.vector.SyncVectorEnv) : vectorized environments for parallel data collection
         * actor (nn.Module) : policy network for action selection
         * critic (nn.Module) : value network for state value estimation
         * memory (ReplayMemory) : buffer for storing collected transitions

        Returns:
            None (initializes the PPOAgent instance)
        """
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        # Initialize tracking variables
        self.step = 0  # Total number of environment steps taken (across all environments)
        
        # Reset environments and get initial observations
        # Convert to tensor and move to device for neural network processing
        obs_np, _ = envs.reset()
        self.next_obs = t.from_numpy(obs_np).to(device, dtype=t.float32)
        
        # Initialize termination flags (all environments start as non-terminated)
        self.next_terminated = t.zeros(envs.num_envs, device=device, dtype=t.bool)

    def play_step(self) -> list[dict]:
        """
        Purpose:
            Execute a single interaction step between the agent and all environments.
            The agent uses its current policy to sample actions, steps the environments,
            and stores the resulting transitions in replay memory for later learning.

        Parameters:
            None (uses self attributes)

        Returns:
            A list of info dictionaries from the environment step, containing episode
            statistics if any episodes terminated
        """
        # Get current observations and termination status
        obs = self.next_obs
        terminated = self.next_terminated

        # Sample actions from the current policy (no gradients needed here)
        with t.inference_mode():
            # Actor network outputs logits for each action
            logits = self.actor(obs)
            dist = t.distributions.Categorical(logits=logits)
            
            # Sample actions stochastically (exploration)
            actions = dist.sample()
            
            # Store log probabilities for importance sampling ratio computation later
            logprobs = dist.log_prob(actions)

            # Critic network estimates state values (used for advantage computation)
            values = self.critic(obs).flatten().cpu().numpy()

        # Step all environments in parallel
        # Note: environments expect numpy arrays, so we convert from tensors
        next_obs, rewards, next_terminated, next_truncated, infos = self.envs.step(
            actions.cpu().numpy()
        )

        # Store transition in replay memory for later learning
        # All data is converted to numpy for storage (memory is numpy-based)
        self.memory.add(
            obs.cpu().numpy(),
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values,
            rewards,
            terminated.cpu().numpy()
        )
        
        # Update tracking variables for next step
        self.step += self.envs.num_envs
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float32)
        self.next_terminated = t.from_numpy(next_terminated).to(device, dtype=t.bool)

        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Purpose:
            Extract minibatches from replay memory for learning. Computes advantages
            using GAE and returns shuffled minibatches ready for training. The memory
            is reset after extraction since we only use each rollout once.

        Parameters:
         * gamma (float) : discount factor for future rewards
         * gae_lambda (float) : GAE parameter for advantage estimation

        Returns:
            A list of ReplayMinibatch objects, each containing a random subset of
            the rollout data with computed advantages and returns
        """
        # Compute value estimate for the next state (needed for GAE bootstrap)
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        
        # Get minibatches with advantages computed via GAE
        minibatches = self.memory.get_minibatches(
            next_value, self.next_terminated, gamma, gae_lambda
        )
        
        # Memory is automatically reset inside get_minibatches
        return minibatches


def calc_clipped_surrogate_objective(
    dist: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """
    Purpose:
        Compute the clipped surrogate objective for PPO. This is the core of PPO's policy
        update mechanism. It prevents large policy updates by clipping the probability
        ratio, ensuring the new policy doesn't deviate too far from the old policy.
        This clipping mechanism is what makes PPO stable and sample-efficient.

    Parameters:
     * dist (Categorical) : action distribution from the current (updated) policy network
     * mb_action (Int[Tensor, "minibatch_size"]) : actions that were taken in the minibatch
     * mb_advantages (Float[Tensor, "minibatch_size"]) : advantage estimates for each sample
     * mb_logprobs (Float[Tensor, "minibatch_size"]) : log probabilities of actions under
                                                      the old policy (used for importance sampling)
     * clip_coef (float) : clipping coefficient (epsilon in the PPO paper), typically 0.1-0.3
     * eps (float) : small constant to prevent division by zero when normalizing advantages

    Returns:
        A tuple of (loss, clipfrac, approxkl) where:
        - loss: the clipped surrogate objective to maximize
        - clipfrac: fraction of samples where clipping was active (useful for debugging)
        - approxkl: approximate KL divergence between old and new policies
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    
    # Get log probabilities under the current (updated) policy
    theta_logprobs = dist.log_prob(mb_action)

    # Compute importance sampling ratio: π_new(a|s) / π_old(a|s)
    # This tells us how much more/less likely the action is under the new policy
    logratio = theta_logprobs - mb_logprobs
    ratios = t.exp(logratio)
    
    # Approximate KL divergence: (-logratio).mean()
    # This measures how different the new policy is from the old one
    approxkl = (-logratio).mean()

    # Normalize advantages to reduce variance and improve training stability
    # This doesn't change the policy gradient direction, just scales it
    mb_advantages_normalized = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    # Compute both clipped and unclipped objectives
    # We'll take the minimum to implement the pessimistic clipping
    unclipped_objective = ratios * mb_advantages_normalized
    clipped_objective = t.clamp(ratios, 1 - clip_coef, 1 + clip_coef) * mb_advantages_normalized
    
    # Take minimum to implement pessimistic clipping (prevents large updates)
    # This ensures we don't over-optimize when the ratio is very large
    objective = t.minimum(unclipped_objective, clipped_objective)
    loss = objective.mean()
    
    # Track how often clipping is active (useful diagnostic)
    clipfrac = (t.abs(ratios - 1.0) > clip_coef).float().mean()
    
    return loss, clipfrac, approxkl


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float,
) -> Float[Tensor, ""]:
    """
    Purpose:
        Compute the value function loss component of the PPO objective. The critic network
        learns to predict the expected return (value) for each state. This loss encourages
        the critic to accurately estimate state values, which are used for advantage computation.

    Parameters:
     * values (Float[Tensor, "minibatch_size"]) : value predictions from the updated critic network
     * mb_returns (Float[Tensor, "minibatch_size"]) : target returns computed as advantages + old values
                                                    (these are the "labels" for value function learning)
     * vf_coef (float) : coefficient weighting the value loss in the total objective (c_1 in paper)

    Returns:
        The value function loss, scaled by vf_coef
    """
    assert values.shape == mb_returns.shape, "Values and returns must have the same shape"

    # Mean squared error between predicted values and target returns
    # The critic learns to minimize this, improving its value estimates
    value_loss = nn.MSELoss()(values, mb_returns)
    return vf_coef * value_loss


def calc_entropy_bonus(dist: Categorical, ent_coef: float) -> Float[Tensor, ""]:
    """
    Purpose:
        Compute the entropy bonus term to encourage exploration. Higher entropy means
        the policy is more uniform (explores more), while lower entropy means it's more
        deterministic (exploits more). This bonus prevents the policy from becoming too
        greedy too quickly, maintaining exploration throughout training.

    Parameters:
     * dist (Categorical) : action distribution from the current policy
     * ent_coef (float) : coefficient weighting the entropy bonus (c_2 in paper)
                         Higher values encourage more exploration

    Returns:
        The entropy bonus, scaled by ent_coef (to be added to the objective)
    """
    return ent_coef * dist.entropy().mean()


class PPOScheduler:
    """
    Purpose:
        Linear learning rate scheduler for PPO. Gradually reduces learning rate from
        initial_lr to end_lr over the course of training. This helps with convergence
        by allowing larger updates early and smaller, more stable updates later.
    """
    
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_phases: int):
        """
        Purpose:
            Initialize the scheduler with decay parameters.

        Parameters:
         * optimizer (Optimizer) : the optimizer whose learning rate will be scheduled
         * initial_lr (float) : starting learning rate
         * end_lr (float) : final learning rate after decay
         * total_phases (int) : number of training phases over which to decay
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0
        
        # Precompute linear decay parameters: lr = slope * step + intercept
        self.slope = (self.end_lr - self.initial_lr) / self.total_phases
        self.intercept = self.initial_lr

    def step(self) -> None:
        """
        Purpose:
            Update the learning rate according to linear decay schedule. Called once
            per training phase to gradually reduce the learning rate.

        Parameters:
            None (uses self attributes)

        Returns:
            None (modifies optimizer in place)
        """
        self.n_step_calls += 1

        # Linear decay: lr = initial_lr + (end_lr - initial_lr) * (step / total_phases)
        if self.n_step_calls < self.total_phases:
            lr = self.slope * self.n_step_calls + self.intercept
        else:
            # After total_phases, keep learning rate at end_lr
            lr = self.end_lr

        # Update learning rate for all parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def make_optimizer(
    actor: nn.Module, critic: nn.Module, total_phases: int, initial_lr: float, end_lr: float = 0.0
) -> tuple[optim.AdamW, PPOScheduler]:
    """
    Purpose:
        Create an optimizer and learning rate scheduler for PPO training. The optimizer
        is configured to maximize the objective (since PPO maximizes rather than minimizes).
        The scheduler linearly decays the learning rate over training.

    Parameters:
     * actor (nn.Module) : the policy network
     * critic (nn.Module) : the value network
     * total_phases (int) : total number of training phases (for scheduler)
     * initial_lr (float) : starting learning rate
     * end_lr (float) : final learning rate after decay (default: 0.0)

    Returns:
        A tuple of (optimizer, scheduler) ready for training
    """
    # AdamW optimizer with maximize=True (PPO maximizes the objective)
    # Both actor and critic parameters are optimized together
    optimizer = optim.AdamW(
        itertools.chain(actor.parameters(), critic.parameters()),
        lr=initial_lr,
        eps=1e-5,  # Small epsilon for numerical stability
        maximize=True,  # PPO maximizes the objective function
    )
    
    # Learning rate scheduler: linear decay from initial_lr to end_lr
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    
    return optimizer, scheduler


class PPOTrainer:
    def __init__(self, args: PPOArgs):
        """
        Purpose:
            Initialize the PPO trainer with environments, networks, optimizer, and agent.
            Sets up all components needed for training a PPO agent.

        Parameters:
         * args (PPOArgs) : configuration dataclass containing all hyperparameters

        Returns:
            None (initializes the PPOTrainer instance)
        """
        # Set random seeds for reproducibility
        set_global_seeds(args.seed)
        self.args = args
        
        # Create unique run name for logging and checkpointing
        self.run_name = (
            f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__"
            f"{time.strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Create vectorized environments for parallel data collection
        # Each environment runs independently, allowing faster rollouts
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(idx=idx, run_name=self.run_name, **args.__dict__)
                for idx in range(args.num_envs)
            ]
        )
 
        # Extract environment properties needed for network and memory setup
        self.num_envs = self.envs.num_envs
        self.action_shape = self.envs.single_action_space.shape
        self.obs_shape = self.envs.single_observation_space.shape

        # Initialize replay memory for storing rollout data
        # Memory will grow during rollout phase and be consumed during learning phase
        self.memory = ReplayMemory(
            self.num_envs,
            self.obs_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        # Create actor (policy) and critic (value) networks
        # Architecture depends on environment mode (classic-control, atari, mujoco)
        self.actor, self.critic = get_actor_and_critic(self.envs, mode=args.mode)
        
        # Create optimizer with learning rate scheduler
        # Optimizer is configured to maximize (since PPO maximizes the objective)
        self.optimizer, self.scheduler = make_optimizer(
            self.actor, self.critic, args.total_training_steps, args.lr
        )

        # Create agent that handles environment interaction and data collection
        self.agent = PPOAgent(self.envs, self.actor, self.critic, self.memory)

        # Store checkpoint directory path for model saving
        self.checkpoint_dir = Path(__file__).parent / "checkpoints" / self.run_name

        # Initialize step tracking for video milestones
        self.total_steps_completed = 0
        self.next_video_milestone_idx = 0
        self.training_videos_recorded = []
        self.last_checkpoint_step = 0

    def rollout_phase(self) -> dict | None:
        """
        Purpose:
            Collect a rollout of experiences by interacting with the environment.
            This phase fills the replay memory with new transitions that will be
            used for learning in the next phase. Also logs episode statistics to
            wandb when episodes complete.

        Parameters:
            None (uses self attributes)

        Returns:
            The most recent info dictionary from environment steps, or None if
            no episodes completed during this rollout
        """
        infos = None

        # Collect num_steps_per_rollout steps of experience
        for step in range(self.args.num_steps_per_rollout):
            new_infos = self.agent.play_step()

            # Keep track of the most recent infos (for progress bar display)
            if new_infos is not None:
                infos = new_infos

                # Log episode completion statistics to wandb
                if self.args.use_wandb:
                    episode_data = get_episode_data_from_infos(infos)
                    if episode_data is not None:
                        wandb.log(episode_data)

        # Update total steps completed
        steps_in_this_rollout = self.args.num_steps_per_rollout * self.args.num_envs
        self.total_steps_completed += steps_in_this_rollout

        # Check if we've crossed a video milestone
        if (self.next_video_milestone_idx < len(self.args.video_milestone_steps) and
            self.total_steps_completed >= self.args.video_milestone_steps[self.next_video_milestone_idx]):
            milestone_step = self.args.video_milestone_steps[self.next_video_milestone_idx]
            print(f"\n📹 Recording milestone video at step {milestone_step} ({self.next_video_milestone_idx + 1}/{self.args.num_training_videos})")
            self._record_milestone_video(self.next_video_milestone_idx, milestone_step)
            self.next_video_milestone_idx += 1

        # Check if we should save a periodic checkpoint
        if (self.args.checkpoint_freq > 0 and
            self.total_steps_completed - self.last_checkpoint_step >= self.args.checkpoint_freq):
            print(f"\n💾 Saving checkpoint at step {self.total_steps_completed}...")
            self._save_model(suffix=f"_step_{self.total_steps_completed}")
            self.last_checkpoint_step = self.total_steps_completed

        return infos

    def learning_phase(self) -> None:
        """
        Purpose:
            Perform the learning phase of PPO. This updates both the actor (policy)
            and critic (value) networks using the collected rollout data. The phase
            involves multiple passes over the data (batches_per_learning_phase) with
            random minibatch sampling to improve sample efficiency.

        Parameters:
            None (uses self attributes)

        Returns:
            None
        """
        # Get minibatches from replay memory with computed advantages
        minibatches_list = self.agent.get_minibatches(self.args.gamma, self.args.gae_lambda)

        # Track debug metrics across all minibatches for logging
        debug_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "clipfrac": [],
            "approxkl": [],
        }
        
        # Train on each minibatch
        for minibatch in minibatches_list:
            # Compute PPO objective and get debug metrics
            objective, debug_dict = self.compute_ppo_objective(minibatch)
            
            # Backpropagate gradients through the objective
            objective.backward()

            # Clip gradients to prevent exploding gradients
            # This is important for training stability, especially with RNNs or deep networks
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.args.max_grad_norm
            )

            # Update network parameters (maximize=True was set during optimizer creation)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Accumulate metrics for logging
            for key in debug_metrics:
                debug_metrics[key].append(debug_dict[key])

        # Log averaged metrics to wandb for monitoring training progress
        if self.args.use_wandb:
            wandb_log = {
                key: np.mean(values) for key, values in debug_metrics.items()
            }
            wandb.log(wandb_log)

        # Update learning rate according to schedule
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> tuple[Float[Tensor, ""], dict]:
        """
        Purpose:
            Compute the complete PPO objective function for a single minibatch.
            The objective combines three components: policy improvement (clipped surrogate),
            value function accuracy, and exploration bonus (entropy). Also computes
            debug metrics useful for monitoring training.

        Parameters:
         * minibatch (ReplayMinibatch) : a batch of transitions with computed advantages
                                        and returns

        Returns:
            A tuple of (ppo_objective, debug_dict) where:
            - ppo_objective: the total objective to maximize (scalar tensor)
            - debug_dict: dictionary with component losses and diagnostics for logging
        """
        # Unpack minibatch data
        obs = minibatch.obs
        mb_action = minibatch.actions
        mb_logprobs = minibatch.logprobs  # Old policy logprobs (for importance sampling)
        mb_advantages = minibatch.advantages  # GAE-computed advantages
        mb_returns = minibatch.returns  # Target returns for value function learning

        # Forward pass through actor network to get current policy
        logits = self.actor(obs)
        dist = t.distributions.Categorical(logits=logits)

        # Compute clipped surrogate objective (policy improvement term)
        # This prevents large policy updates and includes clipping diagnostics
        clipped_surrogate_obj, clipfrac, approxkl = calc_clipped_surrogate_objective(
            dist, mb_action, mb_advantages, mb_logprobs, self.args.clip_coef
        )

        # Compute value function loss (critic learning term)
        # The critic learns to predict returns accurately
        values = self.critic(obs).flatten()
        value_loss = calc_value_function_loss(values, mb_returns, self.args.vf_coef)

        # Compute entropy bonus (exploration term)
        # Encourages policy to maintain exploration
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)

        # Total PPO objective: maximize policy improvement, minimize value error, maximize entropy
        # Note: value_loss is subtracted because we want to minimize it (but optimizer maximizes)
        ppo_objective = clipped_surrogate_obj - value_loss + entropy_bonus
        
        # Prepare debug metrics for logging and monitoring
        debug_dict = {
            "policy_loss": clipped_surrogate_obj.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_bonus.item(),
            "clipfrac": clipfrac.item(),  # How often clipping was active
            "approxkl": approxkl.item(),  # Policy change magnitude
        }
        
        return ppo_objective, debug_dict


    def train(self) -> None:
        """
        Purpose:
            Run the main training loop for PPO. Alternates between collecting rollouts
            (rollout_phase) and updating the policy/value networks (learning_phase).
            Logs progress to wandb and displays a progress bar.

        Parameters:
            None (uses self attributes)

        Returns:
            None
        """
        # Initialize wandb for experiment tracking
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            # Track gradients and parameters for debugging
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        # Main training loop: alternate between rollout and learning phases
        pbar = tqdm(range(self.args.total_phases))
        last_logged_time = time.time()  # Throttle progress bar updates for performance

        for phase in pbar:
            # Phase 1: Collect rollout data by interacting with environments
            data = self.rollout_phase()
            
            # Update progress bar with episode statistics (throttled to avoid lag)
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(phase=phase, **data)

            # Phase 2: Update networks using collected rollout data
            self.learning_phase()

        # Cleanup
        self.envs.close()

        # Save the trained model
        self._save_model()

        # Record final videos (always, if num_final_videos > 0)
        if self.args.num_final_videos > 0:
            self._save_final_videos(num_videos=self.args.num_final_videos)

        if self.args.use_wandb:
            wandb.finish()
    
    def _convert_to_native_types(self, obj):
        """
        Recursively convert NumPy types to native Python types for JSON serialization.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        else:
            return obj
    
    def _save_model(self, suffix: str = "") -> None:
        """
        Purpose:
            Save the trained actor and critic models to disk so they can be loaded
            later for evaluation or re-recording videos.

        Parameters:
         * suffix (str) : optional suffix to add to checkpoint filenames (e.g., "_step_250000")

        Returns:
            None
        """
        # Create checkpoint directory
        checkpoint_dir = self.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dicts (with optional suffix for periodic checkpoints)
        actor_path = checkpoint_dir / f"actor{suffix}.pt"
        critic_path = checkpoint_dir / f"critic{suffix}.pt"

        t.save(self.actor.state_dict(), actor_path)
        t.save(self.critic.state_dict(), critic_path)

        # Save training configuration for reproducibility
        import json
        config_path = checkpoint_dir / f"config{suffix}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        obs_shape_list = [int(x) for x in self.obs_shape]
        action_shape_list = [int(x) for x in self.action_shape] if hasattr(self, 'action_shape') else None
        num_actions = int(self.envs.single_action_space.n) if hasattr(self.envs.single_action_space, 'n') else None
        
        config_dict = {
            "env_id": self.args.env_id,
            "mode": self.args.mode,
            "seed": int(self.args.seed),
            "obs_shape": obs_shape_list,
            "action_shape": action_shape_list,
            "num_actions": num_actions,
        }
        
        # Recursively convert all NumPy types to native Python types
        config_dict = self._convert_to_native_types(config_dict)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        if suffix:
            print(f"  ✓ Checkpoint saved: {actor_path.name}, {critic_path.name}")
        else:
            print(f"✓ Final model saved to: {checkpoint_dir}")

        # Also log model to wandb if enabled
        if self.args.use_wandb:
            wandb.save(str(actor_path))
            wandb.save(str(critic_path))
            wandb.save(str(config_path))

    def _record_milestone_video(self, milestone_idx: int, step_count: int) -> None:
        """
        Purpose:
            Record a single video at a training milestone (e.g., every 1/10 of total steps).
            This provides visibility into the agent's learning progress throughout training.

        Parameters:
         * milestone_idx (int) : index of this milestone (0-based)
         * step_count (int) : total steps completed when this milestone was reached

        Returns:
            None
        """
        try:
            # Check if moviepy is available
            import moviepy  # noqa: F401
        except ImportError:
            print("⚠ Warning: moviepy is not installed. Skipping video recording.")
            return

        video_save_path = self.args.video_save_path / self.run_name / "training_milestones"
        video_save_path.mkdir(parents=True, exist_ok=True)

        try:
            # Create a fresh environment for this video
            if self.args.mode == "atari":
                base_env = gym.make(self.args.env_id, render_mode="rgb_array")
                base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
                video_env = prepare_atari_env(base_env)

                # Wrap with RecordVideo
                video_env = gym.wrappers.RecordVideo(
                    video_env,
                    video_folder=str(video_save_path),
                    episode_trigger=lambda episode_id: episode_id == 0,
                    disable_logger=True,
                    name_prefix=f"milestone_{milestone_idx:02d}_step_{step_count}",
                )
            else:
                # For non-Atari environments
                base_env = gym.make(self.args.env_id, render_mode="rgb_array")
                base_env = gym.wrappers.RecordEpisodeStatistics(base_env)

                video_env = gym.wrappers.RecordVideo(
                    base_env,
                    video_folder=str(video_save_path),
                    episode_trigger=lambda episode_id: episode_id == 0,
                    disable_logger=True,
                    name_prefix=f"milestone_{milestone_idx:02d}_step_{step_count}",
                )

            # Reset environment
            obs, info = video_env.reset(seed=self.args.seed + milestone_idx)
            video_env.action_space.seed(self.args.seed + milestone_idx)
            video_env.observation_space.seed(self.args.seed + milestone_idx)

            done = False
            step = 0
            total_reward = 0.0

            # Run one episode
            while not done and step < 10000:  # Safety limit
                # Convert observation to tensor
                obs_array = np.array(obs, dtype=np.float32)

                # Reshape based on environment mode
                if self.args.mode == "atari":
                    if obs_array.ndim == 3:
                        h, w, c = obs_array.shape
                        if c == 4:  # (H, W, 4) - channels last
                            obs_array = np.transpose(obs_array, (2, 0, 1))  # -> (4, H, W)
                        elif h == 4:  # (4, H, W) - already channels first
                            pass
                        elif w == 4:  # (H, 4, W) - wrong format
                            obs_array = np.transpose(obs_array, (1, 0, 2))  # -> (4, H, W)
                else:
                    obs_array = obs_array.flatten()

                obs_tensor = t.from_numpy(obs_array).unsqueeze(0).to(device)

                with t.inference_mode():
                    logits = self.actor(obs_tensor)
                    dist = t.distributions.Categorical(logits=logits)
                    action = dist.mode.item()  # Use mode for deterministic behavior

                obs, reward, terminated, truncated, info = video_env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

            video_env.close()

            # Wait for video file to be written
            import time
            time.sleep(0.5)

            # Find the recorded video file
            video_files = sorted(
                video_save_path.glob(f"milestone_{milestone_idx:02d}_step_{step_count}-episode-*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if video_files:
                video_path = video_files[0]
                self.training_videos_recorded.append(video_path)
                print(f"  ✓ Milestone video saved: {video_path.name} (reward: {total_reward:.1f}, steps: {step})")

                # Log to wandb if enabled
                if self.args.use_wandb:
                    wandb.log({
                        "milestone_video": wandb.Video(str(video_path), fps=30, format="mp4"),
                        "milestone_step": step_count,
                        "milestone_reward": total_reward,
                    })
            else:
                print(f"  ⚠ Could not find milestone video file")

        except Exception as e:
            print(f"  ⚠ Error recording milestone video: {e}")
            import traceback
            traceback.print_exc()

    def _save_final_videos(self, num_videos: int = 10) -> None:
        """
        Purpose:
            Record the last N episodes with the trained agent and save them as videos.
            Videos are logged to wandb and saved locally. This allows you to see the
            agent's final performance and re-record if needed.

        Parameters:
         * num_videos (int) : number of episodes to record (default: 10)

        Returns:
            None
        """
        print(f"\nRecording {num_videos} final videos...")
        
        try:
            # Check if moviepy is available
            import moviepy  # noqa: F401
        except ImportError:
            print("⚠ Warning: moviepy is not installed. Skipping video recording.")
            print("  To enable video recording, install: pip install moviepy")
            return
        
        video_save_path = self.args.video_save_path / self.run_name
        video_save_path.mkdir(parents=True, exist_ok=True)
        
        recorded_videos = []
        
        for episode_num in range(num_videos):
            video_env = None
            try:
                # Create a fresh environment for each episode
                # Correct wrapper order: base_env -> RecordEpisodeStatistics -> preprocessing -> RecordVideo
                # RecordVideo needs RecordEpisodeStatistics inside it to track episode_id
                # RecordVideo can unwrap to access base_env.render()
                if self.args.mode == "atari":
                    # For Atari: create base env, add statistics, add preprocessing, then RecordVideo
                    base_env = gym.make(self.args.env_id, render_mode="rgb_array")
                    base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
                    video_env = prepare_atari_env(base_env)
                    
                    # Wrap with RecordVideo LAST (it can unwrap to access base_env.render())
                    video_env = gym.wrappers.RecordVideo(
                        video_env,
                        video_folder=str(video_save_path),
                        episode_trigger=lambda episode_id: episode_id == 0,
                        disable_logger=True,
                        name_prefix=f"episode_{episode_num}",
                    )
                else:
                    # For non-Atari: simpler setup
                    base_env = gym.make(self.args.env_id, render_mode="rgb_array")
                    base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
                    
                    # Wrap with RecordVideo
                    video_env = gym.wrappers.RecordVideo(
                        base_env,
                        video_folder=str(video_save_path),
                        episode_trigger=lambda episode_id: episode_id == 0,
                        disable_logger=True,
                        name_prefix=f"episode_{episode_num}",
                    )
                
                # NOW reset - this starts episode 0 which will be recorded
                obs, info = video_env.reset(seed=self.args.seed + episode_num)
                video_env.action_space.seed(self.args.seed + episode_num)
                video_env.observation_space.seed(self.args.seed + episode_num)
                
                done = False
                step_count = 0
                total_reward = 0.0
                
                # Run the episode
                while not done:
                    # Convert observation to tensor (handle LazyFrames from FrameStack)
                    # Match the format used during training: observations from vectorized envs
                    # are converted directly with from_numpy, so we do the same here
                    obs_array = np.array(obs, dtype=np.float32)
                    
                    # Reshape based on environment mode to match network input
                    if self.args.mode == "atari":
                        # Atari: FrameStack concatenates frames along last axis: (84, 84, 4)
                        # Network expects channels first: (4, 84, 84)
                        # Handle different possible observation shapes
                        if obs_array.ndim == 3:
                            # Check the shape and fix accordingly
                            h, w, c = obs_array.shape
                            if c == 4:  # (H, W, 4) - channels last, correct format
                                obs_array = np.transpose(obs_array, (2, 0, 1))  # -> (4, H, W)
                            elif h == 4:  # (4, H, W) - already channels first
                                pass  # Already correct
                            elif w == 4:  # (H, 4, W) - wrong format, need to fix
                                # This is the problematic case: (84, 4, 84)
                                # Reshape to (84, 84, 4) assuming frames are stacked incorrectly
                                # Actually, this might be frames stacked along wrong axis
                                # Try to reshape: (84, 4, 84) -> treat as (84*4, 84) -> (84, 84, 4)?
                                # Better: transpose to get channels in right place
                                # If shape is (84, 4, 84), we want (4, 84, 84)
                                obs_array = np.transpose(obs_array, (1, 0, 2))  # (84, 4, 84) -> (4, 84, 84)
                            else:
                                raise ValueError(f"Unexpected Atari observation shape: {obs_array.shape}. Expected (84, 84, 4), (4, 84, 84), or (84, 4, 84)")
                        else:
                            raise ValueError(f"Unexpected Atari observation ndim: {obs_array.ndim}, shape: {obs_array.shape}")
                    else:
                        # Classic control: flatten to 1D
                        obs_array = obs_array.flatten()
                    
                    obs_tensor = t.from_numpy(obs_array).unsqueeze(0).to(device)
                    
                    with t.inference_mode():
                        logits = self.actor(obs_tensor)
                        dist = t.distributions.Categorical(logits=logits)
                        action = dist.mode.item()  # Use mode for deterministic behavior
                    
                    obs, reward, terminated, truncated, info = video_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step_count += 1
                    
                    # Safety: prevent infinite loops
                    if step_count > 10000:
                        print(f"  Episode {episode_num}: Stopped after 10000 steps (safety limit)")
                        break
                    
                    # Debug: warn if episode is very short (might indicate a problem)
                    if done and step_count < 5:
                        print(f"  ⚠ Episode {episode_num}: Very short episode ({step_count} steps) - agent may be poorly trained")
                
                # Close environment - this finalizes the video
                # RecordVideo automatically saves the video when the episode completes
                video_env.close()
                
                # Wait for video file to be written
                import time
                time.sleep(0.5)
                
                # Find the recorded video file
                # RecordVideo creates files with pattern: {name_prefix}-episode-{episode_id}.mp4
                video_files = sorted(
                    video_save_path.glob(f"episode_{episode_num}-episode-*.mp4"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                
                # Also check for rl-video pattern as fallback
                if not video_files:
                    video_files = sorted(
                        video_save_path.glob("rl-video-episode-*.mp4"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                
                if video_files:
                    video_path = video_files[0]
                    file_size = video_path.stat().st_size
                    if file_size > 0:
                        recorded_videos.append(str(video_path))
                        print(f"  Episode {episode_num}: {step_count} steps, reward={total_reward:.1f}, video={video_path.name} ({file_size / 1024:.1f} KB)")
                    else:
                        print(f"  Episode {episode_num}: Video file is empty (0 bytes)")
                else:
                    print(f"  Episode {episode_num}: Video file not found (episode had {step_count} steps)")
                
            except Exception as e:
                print(f"  Episode {episode_num}: Failed to record - {e}")
                import traceback
                traceback.print_exc()
                if video_env is not None:
                    try:
                        video_env.close()
                    except:
                        pass
                continue
        
        # Log videos to wandb
        if self.args.use_wandb and recorded_videos:
            print(f"\nUploading {len(recorded_videos)} videos to wandb...")
            for idx, video_path in enumerate(recorded_videos):
                try:
                    # Use higher FPS for better quality videos (Atari typically runs at 60 FPS, but we'll use 30)
                    fps = 30 if self.args.mode == "atari" else 4
                    wandb.log({f"final_video_{idx}": wandb.Video(video_path, fps=fps, format="mp4")})
                except Exception as e:
                    print(f"  Warning: Failed to upload video {idx}: {e}")
            print("✓ Videos uploaded to wandb")
        
        if recorded_videos:
            print(f"\n✓ {len(recorded_videos)} videos saved to: {video_save_path}")
        else:
            print("\n⚠ No videos were recorded (episodes may have been too short or recording failed)")
    
    @classmethod
    def load_model(cls, checkpoint_dir: Path | str, args: "PPOArgs") -> "PPOTrainer":
        """
        Purpose:
            Load a saved model from a checkpoint directory. This allows you to reload
            a trained model to evaluate it or re-record videos.

        Parameters:
         * checkpoint_dir (Path | str) : path to the checkpoint directory
         * args (PPOArgs) : configuration (must match the saved model)

        Returns:
            A PPOTrainer instance with loaded models (environments not initialized)
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load configuration
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loading model from: {checkpoint_dir}")
            print(f"  Environment: {config['env_id']}")
            print(f"  Mode: {config['mode']}")
        
        # Create a minimal trainer instance (we'll only use the networks)
        trainer = cls.__new__(cls)
        trainer.args = args
        
        # Create dummy environments just to get network architecture
        # (we won't actually use these environments)
        dummy_envs = gym.vector.SyncVectorEnv([
            make_env(idx=0, run_name="dummy", **args.__dict__)
            for _ in range(1)
        ])
        
        # Create networks with correct architecture
        trainer.actor, trainer.critic = get_actor_and_critic(dummy_envs, mode=args.mode)
        
        # Load saved weights
        actor_path = checkpoint_dir / "actor.pt"
        critic_path = checkpoint_dir / "critic.pt"
        
        if actor_path.exists():
            trainer.actor.load_state_dict(t.load(actor_path, map_location=device))
            print(f"✓ Loaded actor from: {actor_path}")
        else:
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
        
        if critic_path.exists():
            trainer.critic.load_state_dict(t.load(critic_path, map_location=device))
            print(f"✓ Loaded critic from: {critic_path}")
        else:
            raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
        
        dummy_envs.close()
        
        return trainer


def reload_and_record_videos(checkpoint_dir: Path | str, num_videos: int = 10) -> None:
    """
    Purpose:
        Helper function to reload a saved model and re-record videos. Useful if
        the initial videos didn't turn out well or you want to record more videos
        from a previously trained model.

    Parameters:
     * checkpoint_dir (Path | str) : path to the checkpoint directory containing saved model
     * num_videos (int) : number of episodes to record (default: 10)

    Returns:
        None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load configuration
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create args matching the saved model
    args = PPOArgs(
        env_id=config['env_id'],
        mode=config['mode'],
        seed=config['seed'],
        use_wandb=False,  # Don't log to wandb when reloading
        video_log_freq="last",
    )
    
    # Load the model
    trainer = PPOTrainer.load_model(checkpoint_dir, args)
    
    # Record videos
    trainer._save_final_videos(num_videos=num_videos)


def load_weights_from_files(
    actor_path: Path | str,
    critic_path: Path | str,
    env_id: str = "ALE/SpaceInvaders-v5",
    mode: str = "atari",
    seed: int = 1,
) -> PPOTrainer:
    """
    Purpose:
        Load actor and critic weights directly from .pt files (e.g., downloaded from wandb)
        and create a trainer instance for video recording or evaluation.
        
    Parameters:
     * actor_path (Path | str) : path to actor.pt file
     * critic_path (Path | str) : path to critic.pt file
     * env_id (str) : environment ID (default: "ALE/SpaceInvaders-v5")
     * mode (str) : environment mode (default: "atari")
     * seed (int) : random seed (default: 1)
    
    Returns:
        PPOTrainer instance with loaded weights (ready for video recording)
    """
    actor_path = Path(actor_path)
    critic_path = Path(critic_path)
    
    print(f"Loading weights from files:")
    print(f"  Actor: {actor_path}")
    print(f"  Critic: {critic_path}")
    
    # Create args matching the model
    args = PPOArgs(
        env_id=env_id,
        mode=mode,
        seed=seed,
        use_wandb=False,
        video_log_freq="last",
    )
    
    # Create a minimal trainer instance
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.args = args
    
    # Set run_name (needed for video saving)
    # Extract run name from actor file name or create a default one
    import time
    if "SpaceInvaders" in str(actor_path):
        # Try to extract run name from filename
        name_parts = actor_path.stem.replace("_actor", "").split("__")
        if len(name_parts) >= 2:
            trainer.run_name = "__".join(name_parts[:2]) + f"__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        else:
            trainer.run_name = f"{args.env_id}__PPOSpaceInvaders__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        trainer.run_name = f"{args.env_id}__PPO__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Create dummy environment to get network architecture
    dummy_envs = gym.vector.SyncVectorEnv([
        make_env(idx=0, run_name="dummy", **args.__dict__)
        for _ in range(1)
    ])
    
    # Create networks with correct architecture
    trainer.actor, trainer.critic = get_actor_and_critic(dummy_envs, mode=args.mode)
    
    # Load weights from files
    if actor_path.exists():
        trainer.actor.load_state_dict(t.load(actor_path, map_location=device))
        print(f"✓ Loaded actor weights")
    else:
        raise FileNotFoundError(f"Actor file not found: {actor_path}")
    
    if critic_path.exists():
        trainer.critic.load_state_dict(t.load(critic_path, map_location=device))
        print(f"✓ Loaded critic weights")
    else:
        raise FileNotFoundError(f"Critic file not found: {critic_path}")
    
    dummy_envs.close()
    
    return trainer


def find_latest_checkpoint(env_id: str = None) -> Path | None:
    """
    Purpose:
        Find the most recently created checkpoint directory.
        
    Parameters:
     * env_id (str, optional) : filter by environment ID (e.g., "ALE/SpaceInvaders-v5")
    
    Returns:
        Path to the latest checkpoint directory, or None if none found
    """
    checkpoints_dir = Path(__file__).parent / "checkpoints"
    
    if not checkpoints_dir.exists():
        return None
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    
    # Filter by env_id if provided
    if env_id:
        checkpoint_dirs = [d for d in checkpoint_dirs if env_id in d.name]
    
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time (most recent first)
    latest = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    return latest


# ============================================================================
# Atari Environment Setup
# ============================================================================
def play_atari() -> None:
    """
    Purpose:
        Demonstration function for visualizing Atari environments before and after
        preprocessing. Shows raw frames and preprocessed frames side-by-side to
        understand the effect of wrappers.

    Parameters:
        None

    Returns:
        None (displays visualizations)
    """
    # Create raw Atari environment
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    print(f"Action space: {env.action_space}")  # Discrete(4): 4 actions available
    print(f"Observation space: {env.observation_space}")  # Box(0, 255, (210, 160, 3), uint8)
    print(f"Action meanings: {env.get_action_meanings()}")

    def display_frames(frames: np.ndarray, figsize: tuple = (12, 3)) -> None:
        """Helper to display a sequence of frames as an animation."""
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(frames[0])
        plt.close()

        def update(frame):
            im.set_array(frame)
            return [im]

        ani = FuncAnimation(fig, update, frames=frames, interval=100)
        display(HTML(ani.to_jshtml()))

    # Collect frames from raw environment
    nsteps = 150
    frames = []
    obs, info = env.reset()
    for _ in tqdm(range(nsteps), desc="Collecting raw frames"):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)

    display_frames(np.stack(frames))

    # Apply Atari preprocessing wrappers (frame stacking, grayscale, etc.)
    env_wrapped = prepare_atari_env(env)

    # Collect frames from preprocessed environment
    frames = []
    obs, info = env_wrapped.reset()
    for _ in tqdm(range(nsteps), desc="Collecting preprocessed frames"):
        action = env_wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = env_wrapped.step(action)
        # Stack frames horizontally for visualization
        obs = einops.repeat(
            np.array(obs), "frames h w -> h (frames w) 3"
        )
        frames.append(obs)

    display_frames(np.stack(frames), figsize=(12, 3))


def get_actor_and_critic_atari(
    obs_shape: tuple[int, ...], num_actions: int
) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Purpose:
        Create actor and critic networks for Atari environments. Uses a CNN architecture
        to process image observations (stacked frames). The networks share convolutional
        layers for feature extraction, then branch into separate heads for policy and value.

    Parameters:
     * obs_shape (tuple[int, ...]) : shape of observation (should be (4, 84, 84) for
                                    4 stacked 84x84 frames after preprocessing)
     * num_actions (int) : number of discrete actions available in the environment

    Returns:
        A tuple of (actor, critic) networks as nn.Sequential modules
    """
    # Verify observation shape is compatible with our architecture
    # Expected: (4, 84, 84) -> after 3 conv layers with specified strides
    assert obs_shape[-1] % 8 == 4, f"Observation shape {obs_shape} incompatible with architecture"

    # Calculate spatial dimensions after convolutions
    # Each conv layer reduces spatial size: 84 -> 20 -> 9 -> 7
    L_after_convolutions = (obs_shape[-1] // 8) - 3
    in_features = 64 * L_after_convolutions * L_after_convolutions

    # Shared feature extractor: 3 conv layers + 1 linear layer
    # This processes the image frames into a feature vector
    hidden = nn.Sequential(
        # First conv: 4 channels (stacked frames) -> 32 channels, large stride for downsampling
        layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)),
        nn.ReLU(),
        # Second conv: further downsampling and channel expansion
        layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
        nn.ReLU(),
        # Third conv: fine-grained features, no downsampling
        layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
        nn.ReLU(),
        # Flatten spatial dimensions
        nn.Flatten(),
        # Project to hidden dimension
        layer_init(nn.Linear(in_features, 512)),
        nn.ReLU(),
    )

    # Actor head: outputs logits for action selection
    actor = nn.Sequential(
        hidden,
        layer_init(nn.Linear(512, num_actions), std=0.01)  # Small std for stable initial policy
    )
    
    # Critic head: outputs single value estimate
    critic = nn.Sequential(
        hidden,
        layer_init(nn.Linear(512, 1), std=1.0)  # Standard initialization for value function
    )

    return actor, critic


# ============================================================================
# Main Execution
# ============================================================================

def estimate_training_time(args: "PPOArgs") -> None:
    """
    Purpose:
        Estimate training time based on configuration and device capabilities.
        Provides rough estimates to help plan training runs.

    Parameters:
     * args (PPOArgs) : configuration to estimate time for

    Returns:
        None (prints time estimates)
    """
    print("=" * 60)
    print("Training Time Estimate")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total phases: {args.total_phases:,}")
    print()
    
    # Rough estimates per phase (in seconds)
    # These are conservative estimates based on typical performance
    if args.mode == "atari":
        # Atari: CNN processing is GPU-intensive
        rollout_time_per_phase = 2.0  # seconds (8 parallel envs, 128 steps)
        learning_time_per_phase = 1.5  # seconds (CNN forward/backward passes)
        if device.type == "mps":
            # MPS is typically 2-3x faster than CPU for CNNs
            learning_time_per_phase = 0.8
        elif device.type == "cuda":
            # CUDA is typically 5-10x faster than CPU
            learning_time_per_phase = 0.3
    else:
        # Classic control: simpler MLPs
        rollout_time_per_phase = 0.5
        learning_time_per_phase = 0.2
        if device.type == "mps":
            learning_time_per_phase = 0.15
        elif device.type == "cuda":
            learning_time_per_phase = 0.1
    
    time_per_phase = rollout_time_per_phase + learning_time_per_phase
    total_seconds = args.total_phases * time_per_phase
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    print("Estimated time per phase:")
    print(f"  Rollout: ~{rollout_time_per_phase:.2f}s")
    print(f"  Learning: ~{learning_time_per_phase:.2f}s")
    print(f"  Total per phase: ~{time_per_phase:.2f}s")
    print()
    print("Estimated total training time:")
    print(f"  {total_hours:.1f} hours ({total_minutes:.0f} minutes)")
    print()
    print("Note: Actual time may vary based on:")
    print("  - Environment step speed")
    print("  - GPU utilization")
    print("  - System load")
    print("=" * 60)
    print()


def test_video_recording(env_id: str = "CartPole-v1", mode: str = "classic-control", num_steps: int = 50) -> None:
    """
    Purpose:
        Test that video recording works correctly. Creates a simple environment,
        wraps it with RecordVideo, runs a few steps with random actions, and
        verifies that a video file was created with non-zero duration.
        
    Parameters:
     * env_id (str) : environment ID to test (default: "CartPole-v1")
     * mode (str) : environment mode (default: "classic-control")
     * num_steps (int) : number of steps to run (default: 50)
    
    Returns:
        None (raises AssertionError if test fails)
    """
    print("=" * 60)
    print("Video Recording Test")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Mode: {mode}")
    print(f"Steps: {num_steps}")
    print()
    
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print("⚠ SKIPPED: moviepy is not installed")
        print("  Install with: pip install moviepy")
        return
    
    # Create temporary directory for test videos
    test_video_dir = Path(__file__).parent / "videos" / "test_recording"
    test_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing test videos
    for old_video in test_video_dir.glob("*.mp4"):
        old_video.unlink()
    
    try:
        # Create environment with video recording
        if mode == "atari":
            base_env = gym.make(env_id, render_mode="rgb_array")
            base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
            env = prepare_atari_env(base_env)
        else:
            base_env = gym.make(env_id, render_mode="rgb_array")
            base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
            env = base_env
        
        # Wrap with RecordVideo
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(test_video_dir),
            episode_trigger=lambda episode_id: episode_id == 0,
            disable_logger=True,
            name_prefix="test_video",
        )
        
        # Reset and run episode
        obs, info = env.reset(seed=42)
        step_count = 0
        
        print("Running episode with random actions...")
        while step_count < num_steps:
            # Use random actions for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                print(f"  Episode ended early at step {step_count}")
                break
        
        # Close environment to finalize video
        env.close()
        
        # Wait for video file to be written
        import time
        time.sleep(1.0)
        
        # Check that video file was created
        video_files = list(test_video_dir.glob("test_video-episode-*.mp4"))
        if not video_files:
            # Try alternative naming pattern
            video_files = list(test_video_dir.glob("rl-video-episode-*.mp4"))
        
        assert len(video_files) > 0, "No video file was created!"
        
        video_path = video_files[0]
        file_size = video_path.stat().st_size
        
        assert file_size > 0, f"Video file is empty (0 bytes): {video_path}"
        
        # Try to get video duration using moviepy
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()
            
            assert duration > 0, f"Video has 0 duration: {video_path}"
            print(f"✓ Video created successfully!")
            print(f"  File: {video_path.name}")
            print(f"  Size: {file_size / 1024:.1f} KB")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Steps recorded: {step_count}")
            print()
            print("=" * 60)
            print("✓ TEST PASSED: Video recording works correctly!")
            print("=" * 60)
        except Exception as e:
            # If we can't read duration, at least verify file exists and has content
            print(f"✓ Video file created (could not verify duration): {e}")
            print(f"  File: {video_path.name}")
            print(f"  Size: {file_size / 1024:.1f} KB")
            print("=" * 60)
            print("✓ TEST PASSED: Video file created successfully!")
            print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up test videos (optional - comment out to keep them for inspection)
        # for video_file in test_video_dir.glob("*.mp4"):
        #     video_file.unlink()
        pass


def test_ppo_quick() -> None:
    """
    Purpose:
        Run a quick test of the PPO implementation to verify everything works.
        Uses CartPole (simple environment) with minimal training steps for fast verification.

    Parameters:
        None

    Returns:
        None (prints test results)
    """
    print("=" * 60)
    print("PPO Quick Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"MPS available: {t.backends.mps.is_available()}")
    print(f"CUDA available: {t.cuda.is_available()}")
    print()
    
    # Quick test configuration: CartPole with minimal steps
    args = PPOArgs(
        env_id="CartPole-v1",
        wandb_project_name="PPOTest",
        use_wandb=False,  # Disable wandb for quick test
        mode="classic-control",
        num_envs=4,
        num_steps_per_rollout=32,  # Small rollout for quick test
        total_timesteps=1000,  # Just 1000 steps total
        num_minibatches=2,
        batches_per_learning_phase=2,
        video_log_freq=None,
    )
    
    print("Configuration:")
    print(f"  Environment: {args.env_id}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total phases: {args.total_phases}")
    print()
    
    try:
        print("Initializing trainer...")
        trainer = PPOTrainer(args)
        print("✓ Trainer initialized successfully")
        print(f"  Actor parameters: {sum(p.numel() for p in trainer.actor.parameters()):,}")
        print(f"  Critic parameters: {sum(p.numel() for p in trainer.critic.parameters()):,}")
        print()
        
        print("Running training test...")
        trainer.train()
        print()
        print("=" * 60)
        print("✓ TEST PASSED: PPO training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Option 1: Test video recording (quick test - run this first!)
    # test_video_recording(env_id="CartPole-v1", mode="classic-control", num_steps=100)
    # test_video_recording(env_id="ALE/SpaceInvaders-v5", mode="atari", num_steps=200)
    
    # # Option 2: Run quick PPO test to verify everything works
    # test_ppo_quick()
    
    # Option 3: Reload a trained model and record videos (no re-training needed!)
    # Uncomment and modify the path to your checkpoint:
    # latest_checkpoint = find_latest_checkpoint(env_id="ALE/SpaceInvaders-v5")
    # if latest_checkpoint:
    #     print(f"Found checkpoint: {latest_checkpoint}")
    #     reload_and_record_videos(latest_checkpoint, num_videos=10)
    # else:
    #     print("No checkpoint found!")
    
    # Option 4: Train on Breakout with automatic video recording
    args = PPOArgs(
        env_id="ALE/Breakout-v5",
        wandb_project_name="PPOBreakout",
        use_wandb=True,
        mode="atari",
        clip_coef=0.1,
        num_envs=8,
        num_training_videos=10,  # Record 10 videos during training (at 1/10 intervals)
        num_final_videos=10,  # Record 10 videos at the end of training
        total_timesteps=2_000_000,  # 2M timesteps for decent Breakout performance
    )
    
    # Estimate training time before starting
    estimate_training_time(args)
    
    trainer = PPOTrainer(args)
    trainer.train()