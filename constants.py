"""!
@file constants.py
@brief Global Constants and Configuration Values

@details
Centralizes all magic numbers and constant values used throughout the project.
Improves maintainability by avoiding hardcoded values scattered across files.

@author Adrian
@date 2025-11-29
@version 2.0

@references
[1] Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    https://arxiv.org/abs/1707.06347
    Note: PPO hyperparameters (gamma, lambda, clip_range, etc.)

[2] Mnih et al. (2015) "Human-level control through deep reinforcement learning"
    https://www.nature.com/articles/nature14236
    Note: CNN architecture, frame preprocessing, image dimensions

[3] Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    https://arxiv.org/abs/1412.6980
    Note: Adam optimizer epsilon parameter

[4] ITU-R BT.601 - Grayscale conversion weights
    https://www.itu.int/rec/R-REC-BT.601/
"""

from typing import Final

# ============================================================================
# Environment Constants
# ============================================================================

ENV_NAME: Final[str] = 'FlappyBird-v0'
"""FlappyBird environment identifier"""

N_ACTIONS: Final[int] = 2
"""Number of discrete actions in FlappyBird (0: no-op, 1: flap)"""

# ============================================================================
# Image Processing Constants
# ============================================================================

IMG_HEIGHT: Final[int] = 84
"""Target height for preprocessed frames"""

IMG_WIDTH: Final[int] = 84
"""Target width for preprocessed frames"""

FRAME_STACK_SIZE: Final[int] = 4
"""Number of frames to stack for temporal information"""

GROUND_CROP_RATIO: Final[float] = 0.78
"""Ratio of frame height to keep (crops bottom 22%)"""

# RGB to Grayscale conversion weights (luminosity method)
GRAYSCALE_WEIGHTS: Final[list] = [0.299, 0.587, 0.114]
"""Weights for RGB to grayscale conversion (ITU-R BT.601)
Formula: Y = 0.299*R + 0.587*G + 0.114*B
Reference: https://www.itu.int/rec/R-REC-BT.601/"""

# ============================================================================
# Neural Network Constants
# ============================================================================
# Architecture based on Mnih et al. (2015) DQN paper:
# "Human-level control through deep reinforcement learning"
# https://www.nature.com/articles/nature14236

# CNN Architecture
CONV1_OUT_CHANNELS: Final[int] = 32
"""Output channels for first convolutional layer"""

CONV2_OUT_CHANNELS: Final[int] = 64
"""Output channels for second convolutional layer"""

CONV3_OUT_CHANNELS: Final[int] = 64
"""Output channels for third convolutional layer"""

CONV1_KERNEL_SIZE: Final[int] = 8
"""Kernel size for first convolution"""

CONV2_KERNEL_SIZE: Final[int] = 4
"""Kernel size for second convolution"""

CONV3_KERNEL_SIZE: Final[int] = 3
"""Kernel size for third convolution"""

CONV1_STRIDE: Final[int] = 4
"""Stride for first convolution"""

CONV2_STRIDE: Final[int] = 2
"""Stride for second convolution"""

CONV3_STRIDE: Final[int] = 1
"""Stride for third convolution"""

FC_LAYER_SIZE: Final[int] = 512
"""Size of fully connected shared layer"""

# Weight initialization (Orthogonal)
# Reference: Saxe et al. (2013) https://arxiv.org/abs/1312.6120
# Used in OpenAI Baselines: https://github.com/openai/baselines
POLICY_HEAD_GAIN: Final[float] = 0.01
"""Gain for policy head weight initialization (small gain for stable initial policy)"""

VALUE_HEAD_GAIN: Final[float] = 1.0
"""Gain for value head weight initialization (standard gain)"""

# ============================================================================
# PPO Algorithm Constants
# ============================================================================
# References:
# - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
#   https://arxiv.org/abs/1707.06347
# - OpenAI Baselines PPO2: https://github.com/openai/baselines

DEFAULT_LEARNING_RATE: Final[float] = 5e-5
"""Default learning rate for Adam optimizer (tuned for FlappyBird)"""

DEFAULT_GAMMA: Final[float] = 0.99
"""Default discount factor for future rewards (standard RL value)"""

DEFAULT_GAE_LAMBDA: Final[float] = 0.95
"""Default GAE lambda parameter (recommended by Schulman et al. 2015)"""

DEFAULT_CLIP_RANGE: Final[float] = 0.2
"""Default PPO clipping parameter (typical range: 0.1-0.3)"""

DEFAULT_VALUE_COEF: Final[float] = 0.5
"""Default coefficient for value loss"""

DEFAULT_ENTROPY_COEF: Final[float] = 0.03
"""Default coefficient for entropy bonus"""

DEFAULT_MAX_GRAD_NORM: Final[float] = 0.5
"""Default maximum gradient norm for clipping"""

# Training configuration
DEFAULT_ROLLOUT_STEPS: Final[int] = 2048
"""Default number of steps per rollout"""

DEFAULT_N_EPOCHS: Final[int] = 4
"""Default number of PPO epochs per update"""

DEFAULT_BATCH_SIZE: Final[int] = 256
"""Default minibatch size"""

# Adam optimizer epsilon
ADAM_EPSILON: Final[float] = 1e-5
"""Epsilon for Adam optimizer numerical stability
Reference: Kingma & Ba (2014) https://arxiv.org/abs/1412.6980"""

# ============================================================================
# Training Constants
# ============================================================================

DEFAULT_N_EPISODES: Final[int] = 1000
"""Default total number of training episodes"""

DEFAULT_SAVE_FREQ: Final[int] = 100
"""Default checkpoint saving frequency (episodes)"""

DEFAULT_LOG_INTERVAL: Final[int] = 50
"""Default detailed logging interval (episodes)"""

# Advantage normalization
ADVANTAGE_NORM_EPSILON: Final[float] = 1e-8
"""Epsilon for advantage normalization"""

# ============================================================================
# Reward Shaping Constants
# ============================================================================

REWARD_THRESHOLD_PIPE: Final[float] = 0.5
"""Threshold to detect pipe passing reward"""

REWARD_SURVIVAL: Final[float] = 0.0
"""Reward for survival (set to 0 to remove survival bonus)"""

# ============================================================================
# Visualization Constants
# ============================================================================

WATCH_DISPLAY_SIZE: Final[int] = 400
"""Size for agent vision display window"""

DEBUG_WARMUP_STEPS: Final[int] = 20
"""Number of warmup steps for debug visualization"""

PLOT_DPI: Final[int] = 150
"""DPI for saved plot images"""

# ============================================================================
# File and Directory Names
# ============================================================================

CHECKPOINT_BEST: Final[str] = 'best_model.pth'
"""Filename for best model checkpoint"""

CHECKPOINT_FINAL: Final[str] = 'final_model.pth'
"""Filename for final model checkpoint"""

CHECKPOINT_INTERRUPTED: Final[str] = 'interrupted_model.pth'
"""Filename for interrupted training checkpoint"""

DIR_CHECKPOINTS: Final[str] = 'checkpoints'
"""Directory name for model checkpoints"""

DIR_PLOTS: Final[str] = 'plots'
"""Directory name for training plots"""

DIR_LOGS: Final[str] = 'logs'
"""Directory name for training logs"""

LOG_CSV_NAME: Final[str] = 'training_log.csv'
"""CSV log filename"""

SUMMARY_FILE_NAME: Final[str] = 'training_summary.txt'
"""Training summary filename"""

# ============================================================================
# CSV Log Headers
# ============================================================================

CSV_HEADERS: Final[list] = [
    'Episode', 'Reward', 'Games', 'Best_Pipes', 
    'Policy_Loss', 'Value_Loss', 'Entropy', 'Timestamp'
]
"""Headers for CSV training log"""

# ============================================================================
# Display Constants
# ============================================================================

PROGRESS_BAR_FORMAT: Final[str] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
"""Format string for tqdm progress bar"""

SEPARATOR_LENGTH: Final[int] = 60
"""Length of separator lines in console output"""
