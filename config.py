"""Configuration Management for PPO Training.

Centralizes all hyperparameters and configuration settings for the PPO
training pipeline. Uses constants.py as the single source of truth for
default values.

Note:
    All default values are imported from constants.py to avoid duplication.
    This allows experimenting with different hyperparameters while maintaining
    a single source of truth.
"""

from dataclasses import dataclass
from typing import Optional

from constants import (
    ENV_NAME,
    N_ACTIONS,
    IMG_HEIGHT,
    IMG_WIDTH,
    FRAME_STACK_SIZE,
    GROUND_CROP_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_CLIP_RANGE,
    DEFAULT_VALUE_COEF,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_N_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_N_EPISODES,
    DEFAULT_SAVE_FREQ,
    DEFAULT_LOG_INTERVAL,
    CONV1_OUT_CHANNELS,
    CONV2_OUT_CHANNELS,
    CONV3_OUT_CHANNELS,
    FC_LAYER_SIZE,
)


@dataclass
class EnvironmentConfig:
    """Environment configuration parameters.
    
    All default values imported from constants.py for single source of truth.
    Override any parameter when instantiating for experiments.
    
    Attributes:
        env_name: Environment identifier from Gymnasium.
        render_mode: Render mode ('rgb_array' for training, 'human' for visualization).
        use_lidar: Whether to use LIDAR observations (False for pixel observations).
        img_height: Target height for preprocessed frames.
        img_width: Target width for preprocessed frames.
        stack_size: Number of frames to stack for temporal information.
        crop_ratio: Ratio of frame to keep (crops bottom portion for ground removal).
    """
    
    env_name: str = ENV_NAME
    render_mode: str = 'rgb_array'
    use_lidar: bool = False
    img_height: int = IMG_HEIGHT
    img_width: int = IMG_WIDTH
    stack_size: int = FRAME_STACK_SIZE
    crop_ratio: float = GROUND_CROP_RATIO


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters.
    
    All default values imported from constants.py for single source of truth.
    Override any parameter when instantiating for experiments.
    
    Attributes:
        learning_rate: Learning rate for Adam optimizer.
        gamma: Discount factor for future rewards (0 < γ ≤ 1).
        gae_lambda: GAE lambda parameter for advantage estimation (0 < λ ≤ 1).
        clip_range: PPO clipping parameter epsilon (typically 0.1-0.3).
        value_coef: Coefficient for value loss in total loss.
        entropy_coef: Coefficient for entropy bonus (exploration).
        max_grad_norm: Maximum gradient norm for gradient clipping.
        n_epochs: Number of optimization epochs per policy update.
        batch_size: Minibatch size for gradient descent.
        rollout_steps: Number of steps to collect before policy update.
    """
    
    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_range: float = DEFAULT_CLIP_RANGE
    value_coef: float = DEFAULT_VALUE_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS


@dataclass
class NetworkConfig:
    """Neural network architecture configuration.
    
    All default values imported from constants.py for single source of truth.
    Override any parameter when instantiating for experiments.
    
    Attributes:
        input_channels: Number of input channels (typically equals frame stack size).
        n_actions: Number of discrete actions in the action space.
        conv1_channels: Output channels for first convolutional layer.
        conv2_channels: Output channels for second convolutional layer.
        conv3_channels: Output channels for third convolutional layer.
        fc_size: Size of fully connected shared layer.
    """
    
    input_channels: int = FRAME_STACK_SIZE
    n_actions: int = N_ACTIONS
    conv1_channels: int = CONV1_OUT_CHANNELS
    conv2_channels: int = CONV2_OUT_CHANNELS
    conv3_channels: int = CONV3_OUT_CHANNELS
    fc_size: int = FC_LAYER_SIZE


@dataclass
class TrainingConfig:
    """Training loop configuration.
    
    All default values imported from constants.py for single source of truth.
    Override any parameter when instantiating for experiments.
    
    Attributes:
        n_episodes: Total number of training episodes.
        save_freq: Checkpoint saving frequency (in episodes).
        device: Compute device ('cpu', 'cuda', or None for auto-detect).
        seed: Random seed for reproducibility (None for random).
        log_interval: Interval for detailed logging (in episodes).
    
    Note:
        File paths and directory names are not included here as they are
        static constants managed in constants.py, not experiment parameters.
    """
    
    n_episodes: int = DEFAULT_N_EPISODES
    save_freq: int = DEFAULT_SAVE_FREQ
    device: Optional[str] = None
    seed: Optional[int] = None
    log_interval: int = DEFAULT_LOG_INTERVAL


@dataclass
class PPOTrainingConfig:
    """Complete training configuration.
    
    Combines all configuration components with defaults from constants.py.
    Use this for complete experiment configurations.
    
    Attributes:
        env: Environment configuration.
        ppo: PPO algorithm configuration.
        network: Network architecture configuration.
        training: Training loop configuration.
    
    Example:
        # Use defaults from constants.py
        config = PPOTrainingConfig()
        
        # Override specific parameters for experiments
        config = PPOTrainingConfig(
            ppo=PPOConfig(learning_rate=0.001, clip_range=0.1),
            training=TrainingConfig(n_episodes=5000, seed=42)
        )
    """
    
    env: EnvironmentConfig = EnvironmentConfig()
    ppo: PPOConfig = PPOConfig()
    network: NetworkConfig = NetworkConfig()
    training: TrainingConfig = TrainingConfig()
    
    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Checks that all hyperparameters are within valid ranges.
        
        Returns:
            True if configuration is valid.
            
        Raises:
            ValueError: If configuration parameters are invalid.
        """
        # Validate ranges
        assert 0 < self.ppo.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.ppo.gae_lambda <= 1, "gae_lambda must be in (0, 1]"
        assert 0 < self.ppo.clip_range < 1, "clip_range must be in (0, 1)"
        assert self.ppo.learning_rate > 0, "learning_rate must be positive"
        assert self.ppo.rollout_steps > 0, "rollout_steps must be positive"
        assert self.ppo.batch_size > 0, "batch_size must be positive"
        assert self.ppo.n_epochs > 0, "n_epochs must be positive"
        
        # Validate rollout_steps vs batch_size
        if self.ppo.rollout_steps < self.ppo.batch_size:
            raise ValueError("rollout_steps must be >= batch_size")
        
        return True
    
    def summary(self) -> str:
        """Generate configuration summary string.
        
        Returns:
            Formatted string with configuration details.
        """
        lines = [
            "=" * 60,
            "PPO Training Configuration",
            "=" * 60,
            "",
            "Environment:",
            f"  Name: {self.env.env_name}",
            f"  Image size: {self.env.img_height}x{self.env.img_width}",
            f"  Frame stack: {self.env.stack_size}",
            "",
            "PPO Hyperparameters:",
            f"  Learning rate: {self.ppo.learning_rate}",
            f"  Gamma: {self.ppo.gamma}",
            f"  GAE lambda: {self.ppo.gae_lambda}",
            f"  Clip range: {self.ppo.clip_range}",
            f"  Value coef: {self.ppo.value_coef}",
            f"  Entropy coef: {self.ppo.entropy_coef}",
            f"  Rollout steps: {self.ppo.rollout_steps}",
            f"  Batch size: {self.ppo.batch_size}",
            f"  Epochs: {self.ppo.n_epochs}",
            "",
            "Training:",
            f"  Episodes: {self.training.n_episodes}",
            f"  Save frequency: {self.training.save_freq}",
            f"  Device: {self.training.device or 'auto-detect'}",
            "=" * 60,
        ]
        return "\n".join(lines)
