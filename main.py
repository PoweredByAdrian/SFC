"""PPO Training Pipeline for FlappyBird-v0.

Complete training pipeline implementing Proximal Policy Optimization (PPO)
for the FlappyBird-v0 environment with pixel observations.

Training Pipeline:
1. Environment initialization with preprocessing wrappers
2. Actor-Critic network setup
3. Rollout collection (experience gathering)
4. PPO policy updates with minibatch SGD
5. Checkpoint saving and metrics logging

Features:
- Automated directory management for runs
- CSV logging for metrics tracking
- Periodic checkpoint saving
- Training visualization plots
- Resume training from checkpoints
- GPU acceleration support
- Reproducible seeding

Example:
    Command line usage:
        python main.py --episodes 1000
        python main.py --resume runs/run_XXXXXXXX/checkpoints/best_model.pth
    
    Programmatic usage with config object:
        from config import PPOTrainingConfig
        config = PPOTrainingConfig()
        trainer = Trainer(config=config)
        trainer.train()
"""

# HEADLESS MODE: Disable PyGame window for maximum training speed
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Suppress torch.compile warnings about limited GPU resources
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

from typing import Tuple, Optional
import argparse
import random
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import csv
from tqdm import tqdm

from model import PPOActorCritic
from ppo_agent import PPOAgent
from utils import FrameStack, PixelObservationWrapper
from config import PPOTrainingConfig, EnvironmentConfig, PPOConfig, NetworkConfig, TrainingConfig
from constants import (
    CHECKPOINT_FINAL, CHECKPOINT_INTERRUPTED,
    DIR_CHECKPOINTS, DIR_PLOTS, DIR_LOGS, LOG_CSV_NAME,
    SUMMARY_FILE_NAME, CSV_HEADERS, PROGRESS_BAR_FORMAT,
    SEPARATOR_LENGTH, PLOT_DPI, REWARD_THRESHOLD_PIPE, REWARD_SURVIVAL
)


class Trainer:
    """PPO Training Manager for FlappyBird-v0.
    
    Manages the complete training lifecycle including:
    - Environment setup with preprocessing wrappers
    - Network initialization
    - Training loop execution
    - Checkpoint management
    - Metrics logging and visualization
    - Reproducible seeding
    
    Note:
        Automatically creates timestamped directories for each run.
        Supports GPU acceleration when available.
    """
    
    def __init__(
        self,
        config: PPOTrainingConfig,
        device: Optional[str] = None
    ) -> None:
        """Initialize PPO trainer with hyperparameters.
        
        Sets up environment, network, optimizer, and logging infrastructure.
        Creates timestamped directories for organizing training outputs.
        If config object is provided, it takes precedence over individual parameters.
        
        Args:
            config: Optional PPOTrainingConfig object (overrides individual params).
            n_episodes: Number of training episodes (rollout-update cycles).
            rollout_steps: Steps to collect before each policy update.
            n_epochs: Number of PPO optimization epochs per update.
            batch_size: Minibatch size for gradient descent.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Discount factor for future rewards.
            gae_lambda: Lambda parameter for GAE advantage estimation.
            clip_range: PPO clipping parameter epsilon.
            value_coef: Coefficient for value loss in total loss.
            entropy_coef: Coefficient for entropy bonus.
            save_freq: Checkpoint saving frequency (in episodes).
            stack_size: Number of frames to stack for temporal info.
            device: Compute device ('cpu', 'cuda', or None for auto-detect).
        """

        config.validate()
        self.config = config
        device = config.training.device
        
        # Set random seeds for reproducibility
        if config.training.seed is not None:
            print(f"Setting random seed: {config.training.seed}")
            random.seed(config.training.seed)
            np.random.seed(config.training.seed)
            torch.manual_seed(config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.training.seed)
                # Additional reproducibility settings for CUDA
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            print(f"  ✓ Seed applied to random, numpy, torch")
        
        # Setup device (prioritize GPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.device = torch.device('cpu')
                print("WARNING: GPU not available, using CPU")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create environment
        print(f"Initializing {config.env.env_name} environment...")
        base_env = gym.make(config.env.env_name, render_mode='rgb_array', use_lidar=False)
        # no frame skip
        base_env = PixelObservationWrapper(base_env, config.env)
        self.env = FrameStack(base_env, config.env)
        print(f"  ✓ Environment created")
        print(f"  ✓ Wrappers applied: PixelObservation -> FrameStack(4)")
        print(f"  ✓ Steps per update: {self.config.ppo.rollout_steps}")
        
        # Initialize PPO Actor-Critic network
        print("Initializing PPO Actor-Critic network...")
        self.policy_net = PPOActorCritic(config.network)
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            n_actions=config.network.n_actions,
            policy_net=self.policy_net,
            config=config.ppo,
            device=self.device
        )
        
        # Create unique directories for this training run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = f'runs/run_{timestamp}'
        self.checkpoint_dir = os.path.join(self.run_dir, DIR_CHECKPOINTS)
        self.plots_dir = os.path.join(self.run_dir, DIR_PLOTS)
        self.logs_dir = os.path.join(self.run_dir, DIR_LOGS)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # CSV logging file
        self.csv_path = os.path.join(self.logs_dir, LOG_CSV_NAME)
        
        print(f"Saving results to: {self.run_dir}")
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_games = []  # Games played per episode
        self.episode_pipes = []  # Best pipes passed per episode
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        print("Initialization complete!")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def collect_rollout(self) -> Tuple[float, int, int]:
        """Collect a rollout of experience from environment.
        
        Executes rollout_steps in the environment, collecting transitions.
        Applies reward shaping: removes survival bonus, counts pipe passes.
        
        Returns:
            Tuple of (best_game_reward, games_played, best_pipes).
        """
        state, _ = self.env.reset()
        current_game_reward = 0.0
        best_game_reward = -float('inf')
        games_played = 0
        current_pipes = 0
        best_pipes = 0
        
        for step in range(self.config.ppo.rollout_steps):
            # Get action
            action, value, log_prob = self.agent.act(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            
            
            # Apply reward shaping
            if done:
                games_played += 1
                # Update best pipes for this episode
                if current_pipes > best_pipes:
                    best_pipes = current_pipes
                current_pipes = 0  # Reset for next game
            elif reward > REWARD_THRESHOLD_PIPE:  # Pipe passed
                current_pipes += 1
            elif reward > 0:  # Survival
                reward = REWARD_SURVIVAL
                
            
            current_game_reward += reward
            
            # Store in buffer
            self.agent.remember(state, action, reward, value, log_prob, done)
            
            # Update state
            state = next_state
            
            # Reset if done
            if done:
                # Track best game reward
                if current_game_reward > best_game_reward:
                    best_game_reward = current_game_reward
                current_game_reward = 0.0  # Reset for next game
                state, _ = self.env.reset()
        
        # If no game completed, use current ongoing game reward
        if best_game_reward == -float('inf'):
            best_game_reward = current_game_reward
        
        return best_game_reward, games_played, best_pipes
    
    def train_episode(self) -> Tuple[float, int, int, float, float, float]:
        """Execute one complete training episode (rollout + update).
        
        Performs one complete PPO training cycle:
        1. Collect rollout_steps of experience
        2. Compute advantages using GAE
        3. Update policy with PPO for n_epochs
        
        Returns:
            Tuple of (reward, games_played, best_pipes, policy_loss, value_loss, entropy).
        """
        # Collect rollout experience
        reward, games_played, best_pipes = self.collect_rollout()
        
        # Perform PPO update
        policy_loss, value_loss, entropy = self.agent.train_step()
        
        return reward, games_played, best_pipes, policy_loss, value_loss, entropy
    
    def train(self, start_episode: int = 1) -> None:
        """Execute main PPO training loop.
        
        Main training loop that:
        - Collects rollouts and updates policy repeatedly
        - Saves best model when new best reward achieved
        - Logs metrics to CSV and generates plots
        - Handles checkpointing at regular intervals
        
        Args:
            start_episode: Episode number to start from (for resuming).
        """
        print("\n" + "="*SEPARATOR_LENGTH)
        print(f"Starting Training - {self.config.env.env_name} PPO")
        print("="*SEPARATOR_LENGTH)
        print(f"Episodes: {self.config.training.n_episodes}")
        print(f"Rollout steps: {self.config.ppo.rollout_steps}")
        print(f"Batch size: {self.config.ppo.batch_size}")
        print(f"PPO epochs: {self.config.ppo.n_epochs}")
        print("="*SEPARATOR_LENGTH)
        
        # Calculate best reward from existing history
        best_reward = max(self.episode_rewards) if self.episode_rewards else -float('inf')
        print(f"Current best reward: {best_reward:.2f}")
        
        # Initialize CSV log file with header (only if new file)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(CSV_HEADERS)
            print(f"CSV log initialized: {self.csv_path}")
        else:
            print(f"Appending to existing CSV: {self.csv_path}")
        
        # Calculate episode range
        end_episode = start_episode + self.config.training.n_episodes
        
        # Create progress bar
        pbar = tqdm(range(start_episode, end_episode), desc="Training", 
                   bar_format=PROGRESS_BAR_FORMAT)
        
        for episode in pbar:
            # Train one episode (collect rollout + PPO update)
            reward, games_played, best_pipes, policy_loss, value_loss, entropy = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(reward)
            self.episode_games.append(games_played)
            self.episode_pipes.append(best_pipes)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.entropies.append(entropy)
            
            # Calculate metrics for logging
            avg_reward_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward
            avg_reward_50 = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else avg_reward_10
            
            # Log to CSV
            with open(self.csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([episode, f'{reward:.2f}', games_played, best_pipes, f'{policy_loss:.4f}', 
                                    f'{value_loss:.4f}', f'{entropy:.4f}',
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
            # Update progress bar
            pbar.set_postfix({
                'R': f'{reward:.1f}',
                'Pipes': best_pipes,
                'Games': games_played,
                'Avg10': f'{avg_reward_10:.1f}'
            })
            
            # Detailed print at regular intervals
            if episode % self.config.training.log_interval == 0:
                print(f"\n[Ep {episode}] Reward: {reward:.2f} | Pipes: {best_pipes} | Games: {games_played} | Avg(10): {avg_reward_10:.2f}")
                print(f"  Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Entropy: {entropy:.4f}")
            
            # Save best model
            if reward > best_reward:
                # Delete old best model if it exists
                if hasattr(self, 'best_model_path') and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                
                best_reward = reward
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, 
                    f'best_model_reward_{best_reward:.2f}.pth'
                )
                self.agent.save(self.best_model_path)
                print(f"[Episode {episode}] New best reward: {best_reward:.2f}")
            
            # Periodic checkpoint
            if episode % self.config.training.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'model_episode_{episode}.pth'
                )
                self.agent.save(checkpoint_path)
                self.plot_metrics(episode)
        
        print("\n" + "="*SEPARATOR_LENGTH)
        print("Training Complete!")
        print("="*SEPARATOR_LENGTH)
        
        # Final save
        final_path = os.path.join(self.checkpoint_dir, CHECKPOINT_FINAL)
        self.agent.save(final_path)
        self.plot_metrics(self.config.training.n_episodes)
        
        # Save training summary
        self.save_summary(best_reward)
        
        self.env.close()
    
    def save_summary(self, best_reward: float) -> None:
        """Save training run summary to text file.
        
        Creates a human-readable summary with full configuration and results.
        
        Args:
            best_reward: Best reward achieved during training.
        """
        summary_path = os.path.join(self.run_dir, SUMMARY_FILE_NAME)
        with open(summary_path, 'w') as f:
            f.write("="*SEPARATOR_LENGTH + "\n")
            f.write("Training Run Summary\n")
            f.write("="*SEPARATOR_LENGTH + "\n\n")
            
            # Run Information
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Full Training Configuration
            f.write("="*SEPARATOR_LENGTH + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("="*SEPARATOR_LENGTH + "\n\n")
            
            f.write("Environment Settings:\n")
            f.write(f"  Environment: {self.config.env.env_name}\n")
            f.write(f"  Frame Stack Size: {self.config.env.stack_size}\n")
            f.write(f"  Image Size: {self.config.env.img_height}x{self.config.env.img_width}\n")
            f.write(f"  Action Space: {self.config.network.n_actions}\n\n")
            
            f.write("PPO Hyperparameters:\n")
            f.write(f"  Learning Rate: {self.agent.optimizer.param_groups[0]['lr']}\n")
            f.write(f"  Gamma (Discount): {self.agent.gamma}\n")
            f.write(f"  GAE Lambda: {self.agent.gae_lambda}\n")
            f.write(f"  Clip Range: {self.agent.clip_range}\n")
            f.write(f"  Value Coefficient: {self.agent.value_coef}\n")
            f.write(f"  Entropy Coefficient: {self.agent.entropy_coef}\n")
            f.write(f"  Max Gradient Norm: {self.agent.max_grad_norm}\n\n")
            
            f.write("Training Parameters:\n")
            f.write(f"  Total Episodes: {self.config.training.n_episodes}\n")
            f.write(f"  Rollout Steps: {self.config.ppo.rollout_steps}\n")
            f.write(f"  PPO Epochs: {self.config.ppo.n_epochs}\n")
            f.write(f"  Batch Size: {self.config.ppo.batch_size}\n")
            f.write(f"  Save Frequency: {self.config.training.save_freq} episodes\n\n")
            
            f.write("Network Architecture:\n")
            f.write(f"  Input Channels: {self.policy_net.input_channels}\n")
            f.write(f"  Output Actions: {self.policy_net.n_actions}\n")
            f.write(f"  Total Parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}\n")
            f.write(f"  Trainable Parameters: {sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad):,}\n\n")
            
            # Training Results
            f.write("="*SEPARATOR_LENGTH + "\n")
            f.write("TRAINING RESULTS\n")
            f.write("="*SEPARATOR_LENGTH + "\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Best Reward: {best_reward:.2f}\n")
            f.write(f"  Final Reward: {self.episode_rewards[-1]:.2f}\n")
            f.write(f"  Average Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}\n")
            f.write(f"  Average Reward (last 50): {np.mean(self.episode_rewards[-50:]):.2f}\n")
            f.write(f"  Average Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}\n")
            f.write(f"  Average Reward (all): {np.mean(self.episode_rewards):.2f}\n\n")
            
            f.write("Episode Statistics:\n")
            f.write(f"  Total Episodes Completed: {len(self.episode_rewards)}\n")
            f.write(f"  Total Games Played: {sum(self.episode_games)}\n")
            f.write(f"  Average Games per Episode: {np.mean(self.episode_games):.2f}\n")
            f.write(f"  Best Pipes Passed (single game): {max(self.episode_pipes) if self.episode_pipes else 0}\n")
            f.write(f"  Average Pipes per Episode: {np.mean(self.episode_pipes):.2f}\n\n")
            
            f.write("Loss Statistics:\n")
            f.write(f"  Final Policy Loss: {self.policy_losses[-1]:.4f}\n")
            f.write(f"  Final Value Loss: {self.value_losses[-1]:.4f}\n")
            f.write(f"  Final Entropy: {self.entropies[-1]:.4f}\n")
            f.write(f"  Average Policy Loss: {np.mean(self.policy_losses):.4f}\n")
            f.write(f"  Average Value Loss: {np.mean(self.value_losses):.4f}\n")
            f.write(f"  Average Entropy: {np.mean(self.entropies):.4f}\n\n")
            
            f.write("="*SEPARATOR_LENGTH + "\n")
            f.write(f"Checkpoints saved in: {self.checkpoint_dir}\n")
            f.write(f"Training logs saved in: {self.logs_dir}\n")
            f.write(f"Plots saved in: {self.plots_dir}\n")
            f.write("="*SEPARATOR_LENGTH + "\n")
            
        print(f"\nTraining summary saved to: {summary_path}")
    
    def plot_metrics(self, episode: int) -> None:
        """Generate and save training metrics plots.
        
        Creates 4-panel plot showing:
        - Episode rewards with moving average
        - Policy loss over time
        - Value loss over time
        - Policy entropy over time
        
        Args:
            episode: Current episode number (for filename).
        """
        fig, axes = plt.subplots(4, 1, figsize=(10, 14))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(10)/10,
                mode='valid'
            )
            axes[0].plot(range(9, len(self.episode_rewards)), moving_avg, 
                        'r-', linewidth=2, label='Moving Avg (10)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot policy loss
        axes[1].plot(self.policy_losses, alpha=0.6, label='Policy Loss', color='blue')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Policy Loss (PPO Clipped Objective)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot value loss
        axes[2].plot(self.value_losses, alpha=0.6, label='Value Loss', color='green')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Value Loss (Critic MSE)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot entropy
        axes[3].plot(self.entropies, alpha=0.6, label='Entropy', color='purple')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Entropy')
        axes[3].set_title('Policy Entropy (Exploration)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f'metrics_episode_{episode}.png')
        plt.savefig(plot_path, dpi=PLOT_DPI)
        plt.close()


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for training.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Train PPO agent on FlappyBird-v0 with pixel observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --episodes 1000
  python main.py --resume runs/run_20231129/checkpoints/best_model.pth
  python main.py --config custom_config.py
        """
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    # Import defaults from config.py dataclasses
    from config import PPOConfig, TrainingConfig
    ppo_defaults = PPOConfig()
    training_defaults = TrainingConfig()
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=training_defaults.n_episodes,
        help=f'Number of episodes to train (default: {training_defaults.n_episodes})'
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Fine-tune mode: use new hyperparameters instead of saved ones'
    )
    

    # PPO Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=ppo_defaults.learning_rate, help=f'PPO learning rate (default: {ppo_defaults.learning_rate})')
    parser.add_argument('--gamma', type=float, default=ppo_defaults.gamma, help=f'PPO gamma (default: {ppo_defaults.gamma})')
    parser.add_argument('--gae-lambda', type=float, default=ppo_defaults.gae_lambda, help=f'PPO GAE lambda (default: {ppo_defaults.gae_lambda})')
    parser.add_argument('--clip-range', type=float, default=ppo_defaults.clip_range, help=f'PPO clip range (default: {ppo_defaults.clip_range})')
    parser.add_argument('--value-coef', type=float, default=ppo_defaults.value_coef, help=f'PPO value coefficient (default: {ppo_defaults.value_coef})')
    parser.add_argument('--entropy-coef', type=float, default=ppo_defaults.entropy_coef, help=f'PPO entropy coefficient (default: {ppo_defaults.entropy_coef})')
    parser.add_argument('--rollout-steps', type=int, default=ppo_defaults.rollout_steps, help=f'PPO rollout steps (default: {ppo_defaults.rollout_steps})')
    parser.add_argument('--batch-size', type=int, default=ppo_defaults.batch_size, help=f'PPO batch size (default: {ppo_defaults.batch_size})')
    parser.add_argument('--n-epochs', type=int, default=ppo_defaults.n_epochs, help=f'PPO epochs (default: {ppo_defaults.n_epochs})')
    # Training options
    parser.add_argument('--save-freq', type=int, default=training_defaults.save_freq, help=f'Checkpoint save frequency (default: {training_defaults.save_freq})')
    parser.add_argument('--device', type=str, default=training_defaults.device, help=f'Device (cpu/cuda, default: {training_defaults.device})')
    parser.add_argument('--seed', type=int, default=training_defaults.seed, help=f'Random seed (default: {training_defaults.seed})')
    parser.add_argument('--log-interval', type=int, default=training_defaults.log_interval, help=f'Log interval (default: {training_defaults.log_interval})')
    
    return parser

def _setup_resume_training(trainer: Trainer, checkpoint_path: str, fine_tune: bool = False) -> Tuple[int, str]:
    """Setup trainer for resuming from checkpoint.
    
    Extracts run directory, loads checkpoint, restores metrics from CSV.
    
    Args:
        trainer: Trainer instance to configure.
        checkpoint_path: Path to checkpoint file.
        fine_tune: If True, uses new hyperparameters (fine-tuning mode).
        
    Returns:
        Tuple of (start_episode, run_dir).
    """
    # Extract run directory from checkpoint path
    checkpoint_path = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up 2 levels
    
    print(f"Resuming in existing directory: {run_dir}\n")
    
    # Override run directories to use existing ones
    trainer.run_dir = run_dir
    trainer.checkpoint_dir = os.path.join(run_dir, DIR_CHECKPOINTS)
    trainer.plots_dir = os.path.join(run_dir, DIR_PLOTS)
    trainer.logs_dir = os.path.join(run_dir, DIR_LOGS)
    trainer.csv_path = os.path.join(trainer.logs_dir, LOG_CSV_NAME)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    if fine_tune:
        print("Fine-tuning mode: Using NEW hyperparameters")
        trainer.agent.load(checkpoint_path, load_hyperparameters=False)
    else:
        print("Resume mode: Using SAVED hyperparameters")
        trainer.agent.load(checkpoint_path, load_hyperparameters=True)
    print("Checkpoint loaded successfully!\n")
    
    # Load existing metrics if CSV exists
    start_episode = 1
    if os.path.exists(trainer.csv_path):
        start_episode = _load_training_history(trainer)
    
    return start_episode, run_dir


def _load_training_history(trainer: Trainer) -> int:
    """Load training history from CSV log.
    
    Reads CSV log and restores all metric lists.
    
    Args:
        trainer: Trainer instance to populate with history.
        
    Returns:
        Next episode number to resume from.
    """
    print("Loading existing training history from CSV...")
    try:
        with open(trainer.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Extract metrics (convert string to float/int)
        trainer.episode_rewards = [float(r['Reward']) for r in rows]
        trainer.episode_games = [int(r.get('Games', 0)) for r in rows]
        trainer.episode_pipes = [int(r.get('Best_Pipes', 0)) for r in rows]
        trainer.policy_losses = [float(r['Policy_Loss']) for r in rows]
        trainer.value_losses = [float(r['Value_Loss']) for r in rows]
        trainer.entropies = [float(r['Entropy']) for r in rows]
        
        # Start from next episode
        start_episode = len(trainer.episode_rewards) + 1
        
        print(f"✓ Loaded {len(trainer.episode_rewards)} episodes of history")
        print(f"  Last episode: {start_episode - 1}")
        print(f"  Last reward: {trainer.episode_rewards[-1]:.2f}")
        print(f"  Best reward: {max(trainer.episode_rewards):.2f}")
        print(f"  Resuming from episode: {start_episode}")
        
        return start_episode
    except Exception as e:
        print(f"Warning: Could not load CSV history: {e}")
        print("Starting fresh metric tracking...")
        return 1


def main() -> None:
    """Main entry point for PPO training.
    
    Parses arguments, creates trainer, handles training loop with
    interrupt handling and error reporting.
    """
    # Parse command line arguments
    parser = _setup_argument_parser()
    args = parser.parse_args()
    
    # Build PPOConfig with possible overrides
    ppo_kwargs = {}
    if args.learning_rate is not None:
        ppo_kwargs['learning_rate'] = args.learning_rate
    if args.gamma is not None:
        ppo_kwargs['gamma'] = args.gamma
    if args.gae_lambda is not None:
        ppo_kwargs['gae_lambda'] = args.gae_lambda
    if args.clip_range is not None:
        ppo_kwargs['clip_range'] = args.clip_range
    if args.value_coef is not None:
        ppo_kwargs['value_coef'] = args.value_coef
    if args.entropy_coef is not None:
        ppo_kwargs['entropy_coef'] = args.entropy_coef
    if args.rollout_steps is not None:
        ppo_kwargs['rollout_steps'] = args.rollout_steps
    if args.batch_size is not None:
        ppo_kwargs['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        ppo_kwargs['n_epochs'] = args.n_epochs

    training_kwargs = {'n_episodes': args.episodes}
    if args.save_freq is not None:
        training_kwargs['save_freq'] = args.save_freq
    if args.device is not None:
        training_kwargs['device'] = args.device
    if args.seed is not None:
        training_kwargs['seed'] = args.seed
    if args.log_interval is not None:
        training_kwargs['log_interval'] = args.log_interval

    config = PPOTrainingConfig(
        training=TrainingConfig(**training_kwargs),
        env=EnvironmentConfig(),
        ppo=PPOConfig(**ppo_kwargs),
        network=NetworkConfig()
    )
    
    print("\n" + "="*SEPARATOR_LENGTH)
    print(f"{config.env.env_name} PPO Training (Pixel Observations + CNN)")
    print("="*SEPARATOR_LENGTH + "\n")
    
    print("Training Configuration:")
    print(config.summary())
    print()
    
    # Create trainer and setup for resume if needed
    start_episode = 1
    if args.resume and os.path.exists(args.resume):
        trainer = Trainer(config)
        start_episode, _ = _setup_resume_training(trainer, args.resume, fine_tune=args.fine_tune)
    else:
        if args.resume:
            print(f"Warning: Checkpoint not found at {args.resume}")
            print("Starting training from scratch...\n")
        trainer = Trainer(config)
    
    try:
        # Start training (resume from start_episode if checkpoint loaded)
        trainer.train(start_episode=start_episode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        interrupt_path = os.path.join(trainer.checkpoint_dir, CHECKPOINT_INTERRUPTED)
        trainer.agent.save(interrupt_path)
        trainer.plot_metrics(len(trainer.episode_rewards))
        print(f"Model saved to: {interrupt_path}")
        print("Exiting...")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
