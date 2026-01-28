# Import ale_py FIRST to register ALE namespace with gymnasium
# This must happen before importing gymnasium
import ale_py
import gymnasium as gym
from gpu_env import CartPole
import torch as t
from tqdm import tqdm
import numpy as np
import time
from IPython.display import HTML

from gymnasium.wrappers import (
    ClipAction,
    GrayscaleObservation,
    NormalizeObservation,
    NormalizeReward,
    ResizeObservation,
    TransformObservation,
    TransformReward,
)

# Import Atari-specific wrappers
# Import from current directory (files are in same directory)
from atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
    FrameStack,
)

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    run_name: str,
    mode: str = "classic-control",
    video_log_freq: int | None | str = None,
    video_save_path: str = None,
    **kwargs,
):
    """
    Return a function that returns an environment after setting up boilerplate.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        # RecordEpisodeStatistics is essential - it provides episode_id for video recording
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Only record video for the first environment (idx == 0)
        # If video_log_freq is "last", skip video recording here (will be done after training)
        if idx == 0 and video_log_freq and video_log_freq != "last":
            # Periodic video logging during training
            try:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"{video_save_path}/{run_name}",
                    episode_trigger=lambda episode_id: episode_id % video_log_freq == 0,
                    disable_logger=True,
                )
            except Exception as e:
                # If video recording fails (e.g., moviepy not installed), continue without it
                print(f"Warning: Video recording disabled - {e}")

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)
        elif mode == "halfcheetah-backflip":
            # Use backflip reward shaping for HalfCheetah
            env = prepare_halfcheetah_backflip_env(env)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def prepare_atari_env(env: gym.Env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env)
    env = FrameStack(env, k=4)
    return env


def prepare_mujoco_env(env: gym.Env):
    env = ClipAction(env)
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = NormalizeReward(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


class HalfCheetahBackflipReward(gym.RewardWrapper):
    """
    Custom reward wrapper for HalfCheetah to learn backflips.
    
    Rewards:
    - Rotation: Tracks cumulative rotation angle (rewards full 360° rotation)
    - Height: Rewards getting off the ground
    - Survival: Small bonus for staying alive
    - Penalizes forward velocity (we want backflip, not forward running)
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.last_angle = 0.0
        self.total_rotation = 0.0
        self.last_z = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract initial orientation and height from observation
        # HalfCheetah obs: [x, y, z, qx, qy, qz, qw, ...] or similar
        # For simplicity, we'll track rotation from body orientation
        self.last_angle = 0.0
        self.total_rotation = 0.0
        if len(obs) > 2:
            self.last_z = obs[2] if len(obs) > 2 else 0.0
        return obs, info
    
    def reward(self, reward):
        # Get current state from the environment
        # HalfCheetah observation typically includes body position and orientation
        obs = self.env.unwrapped._get_obs() if hasattr(self.env.unwrapped, '_get_obs') else None
        
        # Extract body height (z-coordinate) - typically index 2 in observation
        # Or get from sim if available
        current_z = 0.0
        if hasattr(self.env.unwrapped, 'sim'):
            # Get body z position from MuJoCo simulation
            body_id = self.env.unwrapped.sim.model.body_name2id('torso')
            current_z = self.env.unwrapped.sim.data.body_xpos[body_id][2]
        
        # Calculate rotation from body quaternion or angle
        rotation_reward = 0.0
        if hasattr(self.env.unwrapped, 'sim'):
            # Get body orientation
            body_id = self.env.unwrapped.sim.model.body_name2id('torso')
            quat = self.env.unwrapped.sim.data.body_xquat[body_id]
            # Convert quaternion to rotation angle (simplified)
            # For backflip, we care about rotation around the pitch axis
            # Using quaternion to extract pitch angle
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            # Pitch angle from quaternion
            sin_pitch = 2.0 * (qw * qy - qx * qz)
            current_angle = np.arcsin(np.clip(sin_pitch, -1, 1))
            
            # Track cumulative rotation
            angle_diff = current_angle - self.last_angle
            # Handle wrap-around
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            self.total_rotation += angle_diff
            self.last_angle = current_angle
            
            # Reward for completing rotations (360° = 2π)
            # Give bonus when completing full rotations
            if abs(self.total_rotation) >= 2 * np.pi:
                rotation_reward = 10.0  # Big bonus for full rotation
                self.total_rotation = self.total_rotation % (2 * np.pi)  # Reset for next rotation
            else:
                # Small reward for making progress
                rotation_reward = 0.1 * abs(angle_diff)
        
        # Height reward: encourage getting off the ground
        height_reward = max(0, current_z - 0.5) * 2.0  # Reward being above 0.5m
        
        # Penalize forward velocity (x-velocity) - we want backflip, not forward running
        forward_penalty = 0.0
        if hasattr(self.env.unwrapped, 'sim'):
            body_id = self.env.unwrapped.sim.model.body_name2id('torso')
            x_vel = self.env.unwrapped.sim.data.qvel[0]  # Forward velocity
            forward_penalty = -0.1 * abs(x_vel)  # Penalize forward/backward movement
        
        # Survival bonus
        survival_bonus = 0.1
        
        # Combine rewards
        backflip_reward = rotation_reward + height_reward + forward_penalty + survival_bonus
        
        # Update last_z for next step
        self.last_z = current_z
        
        return backflip_reward


def prepare_halfcheetah_backflip_env(env: gym.Env):
    """Prepare HalfCheetah environment with backflip reward shaping."""
    env = ClipAction(env)
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # Apply backflip reward wrapper BEFORE normalization
    env = HalfCheetahBackflipReward(env)
    env = NormalizeReward(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def generate_and_plot_trajectory(trainer, args, steps=500, fps=50):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    # Set up the environment and agent
    
    env = CartPole(env_count=1, device="cpu")
    obs, _ = env.reset()

    # Initialize a tensor to store images
    images = t.zeros((steps, *env.render().shape), dtype=t.uint8)

    # Run the environment for a single trajectory
    
    # Use tqdm to measure the number of steps
    for step_count in tqdm(range(steps), desc="Running trajectory"):

        # Render the environment, reduce the resolution, and store it
        img = env.render()
        images[step_count] = t.tensor(img, dtype=t.uint8)

        # Get action from the policy network
        obs_tensor = t.tensor(obs, dtype=t.float32).unsqueeze(0).to(args.device)
        with t.no_grad():
            action_logits = trainer.agent.policy_network(obs_tensor)
            action = t.argmax(action_logits, dim=-1).item()

        # Take the action in the environment
        obs, reward, done, _, _ = env.step(action)

    # Close the environment rendering
    env.close()

    # Plot the images as a GIF
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(images[0].numpy())

    def update(frame):
        im.set_array(images[frame].numpy())
        return [im]

    ani = FuncAnimation(fig, update, frames=range(step_count), blit=True, repeat=False, interval=1000/fps)
    
    # Save the animation as a GIF file
    import os
    from pathlib import Path
    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"cartpole_trajectory_{time.strftime('%Y%m%d_%H%M%S')}.gif"
    ani.save(str(gif_path), writer='pillow', fps=fps)
    print(f"Video saved to: {gif_path.absolute()}")
    
    # Also return HTML for Jupyter notebook display
    return HTML(ani.to_jshtml())