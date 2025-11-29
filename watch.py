"""!
@file watch.py
@brief Trained Agent Visualization Tool

@details
Loads a trained PPO model and visualizes its performance on FlappyBird-v0.
Displays both the game rendering and the agent's preprocessed observation.

Features:
- Dual display (game view + agent's CNN input)
- Synchronized environments for accurate visualization
- Episode statistics tracking
- Termination reason reporting

@author Adrian
@date 2025-11-29
@version 2.0

@usage
    python watch.py --model runs/run_XXXXXXXX/checkpoints/best_model.pth
    python watch.py --model checkpoint.pth --episodes 10 --delay 0.02
"""

import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import argparse
import os
import time
import cv2
from typing import Optional

from config import EnvironmentConfig, NetworkConfig, PPOConfig
from model import PPOActorCritic
from ppo_agent import PPOAgent
from utils import FrameStack, PixelObservationWrapper
from constants import (
    ENV_NAME, FRAME_STACK_SIZE, N_ACTIONS,
    WATCH_DISPLAY_SIZE, CHECKPOINT_BEST, DIR_CHECKPOINTS
)


def _create_agent_environment(env_config: EnvironmentConfig) -> gym.Env:
    """!
    @brief Create agent environment with preprocessing wrappers
    
    @param stack_size Number of frames to stack
    @return Wrapped environment
    """
    env = gym.make(env_config.env_name, render_mode='rgb_array', use_lidar=False)
    env = PixelObservationWrapper(env, env_config)
    env = FrameStack(env, env_config)
    return env


def _create_human_environment(render_mode: str, env_config: EnvironmentConfig) -> Optional[gym.Env]:
    """!
    @brief Create human-viewable environment
    
    @param render_mode Rendering mode
    @return Environment or None if not in human mode
    """
    if render_mode == 'human':
        return gym.make(env_config.env_name, render_mode='human', use_lidar=False)
    return None


def _load_trained_agent(
    model_path: str, 
    stack_size: int, 
    device: torch.device
) -> PPOAgent:
    """!
    @brief Load trained agent from checkpoint
    
    @param model_path Path to checkpoint file
    @param stack_size Number of stacked frames
    @param device Compute device
    @return Loaded PPO agent in evaluation mode
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer number of actions from checkpoint
    if 'policy_net_state_dict' in checkpoint:
        policy_head_weight = checkpoint['policy_net_state_dict']['policy_head.weight']
        n_actions = policy_head_weight.shape[0]
    else:
        n_actions = N_ACTIONS
    
    # Create and load agent
    policy_net = PPOActorCritic(config=NetworkConfig())
    agent = PPOAgent(n_actions=n_actions, policy_net=policy_net, config=PPOConfig(), device=device)
    agent.load(model_path)
    agent.policy_net.eval()
    
    return agent


def _display_agent_vision(frame: np.ndarray) -> None:
    """!
    @brief Display agent's preprocessed observation
    
    @param frame Single preprocessed frame (84x84)
    
    @details
    Scales frame to larger size for human viewing and displays in window.
    """
    frame_large = cv2.resize(
        frame, 
        (WATCH_DISPLAY_SIZE, WATCH_DISPLAY_SIZE), 
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imshow("Agent Vision (What CNN Sees)", frame_large)
    cv2.waitKey(1)


def _get_termination_reason(terminated: bool, truncated: bool) -> str:
    """!
    @brief Determine why episode ended
    
    @param terminated Whether episode ended due to crash
    @param truncated Whether episode ended due to time limit
    @return Human-readable termination reason
    """
    if terminated:
        return "CRASHED (bird died)"
    elif truncated:
        return "TIME LIMIT (max steps reached)"
    else:
        return "UNKNOWN"


def watch_agent(
    model_path: str, 
    n_episodes: int = 5, 
    render_mode: str = 'human', 
    delay: float = 0.01,
    env_config: EnvironmentConfig = EnvironmentConfig()
) -> None:
    """!
    @brief Visualize trained agent performance
    
    @param model_path Path to trained model checkpoint file
    @param n_episodes Number of episodes to watch
    @param render_mode Rendering mode ('human' for display)
    @param stack_size Number of stacked frames (must match training)
    @param delay Delay between frames in seconds (for speed control)
    
    @details
    Creates two synchronized environments:
    1. Agent environment: Uses same wrappers as training, provides observations
    2. Human environment: Renders game for human viewing
    
    Displays agent's preprocessed view (84x84 grayscale) in separate window.
    
    @note Environments must be synced with same seed for accurate visualization
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize agent environment (for observations)
    print("Initializing Agent Environment...")
    env = _create_agent_environment(env_config)
    print("  ✓ All wrappers applied\n")
    
    # Initialize human environment (for visualization)
    human_env = _create_human_environment(render_mode, env_config)
    if human_env:
        print("  ✓ Human display initialized\n")
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent = _load_trained_agent(model_path, env_config.stack_size, device)
    print("  ✓ Model loaded successfully\n")
    
    print(f"\nStarting to watch agent for {n_episodes} episode(s)...")
    print("Close the window or press Ctrl+C to stop.\n")
    
    try:
        for episode in range(1, n_episodes + 1):
            # SYNC SEEDS
            seed = np.random.randint(0, 1000000)
            state, info = env.reset(seed=seed)  # FrameStack wrapper returns stacked state
            
            if human_env is not None:
                human_env.reset(seed=seed)
            
            total_reward = 0
            steps = 0
            done = False
            
            print(f"Episode {episode}/{n_episodes} started...")
            
            while not done:
                # 1. Agent decides action using stacked STATE
                action, _, _ = agent.act(state, training=False)
                
                # 2. Step the Agent Environment (returns stacked state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Visualize agent's preprocessed observation
                _display_agent_vision(state[-1])
                
                # Sync human environment
                if human_env is not None:
                    human_env.step(action)
                    if delay > 0:
                        time.sleep(delay)
                if reward > 0 and reward < 0.5:
                    reward = 0
                total_reward += reward
                steps += 1
                state = next_state # Update state
                
                if steps % 100 == 0:
                    print(f"  Step {steps}, Reward: {total_reward:.2f}")
            
            # Report episode results
            end_reason = _get_termination_reason(terminated, truncated)
            print(f"Episode {episode} finished:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Reason: {end_reason}\n")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    finally:
        env.close()
        if human_env is not None:
            human_env.close()
        cv2.destroyAllWindows()
        print("Environment closed.")

def main() -> None:
    """!
    @brief Main entry point for watch script
    
    @details
    Parses command-line arguments and starts agent visualization.
    """
    parser = argparse.ArgumentParser(
        description='Watch trained PPO agent play FlappyBird'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=f'{DIR_CHECKPOINTS}/{CHECKPOINT_BEST}',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=5,
        help='Number of episodes to watch'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=0.01,
        help='Delay between frames (seconds)'
    )
    args = parser.parse_args()
    
    watch_agent(model_path=args.model, n_episodes=args.episodes, delay=args.delay, env_config=EnvironmentConfig())

if __name__ == "__main__":
    main()