"""PPO Actor-Critic Neural Network Architecture.

This module implements the neural network architecture for Proximal Policy
Optimization (PPO) applied to FlappyBird-v0 with pixel observations.

The architecture consists of:
- Shared CNN feature extractor (3 convolutional layers)
- Policy head (actor) for action probability distribution
- Value head (critic) for state value estimation

References:
    [1] Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning"
        Nature, 518(7540), 529-533.
        https://www.nature.com/articles/nature14236
        Note: CNN architecture adapted from DQN for PPO

    [2] Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013).
        "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
        arXiv preprint arXiv:1312.6120
        https://arxiv.org/abs/1312.6120
        Note: Orthogonal initialization

    [3] OpenAI Baselines - PPO2 Network Architecture
        https://github.com/openai/baselines/tree/master/baselines/ppo2
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import NetworkConfig
from constants import (
    CONV1_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV3_KERNEL_SIZE,
    CONV1_STRIDE, CONV2_STRIDE, CONV3_STRIDE,
    POLICY_HEAD_GAIN, VALUE_HEAD_GAIN,
)


class PPOActorCritic(nn.Module):
    """Actor-Critic Neural Network for PPO.
    
    Implements a deep neural network with shared feature extraction and
    separate policy and value heads for the PPO algorithm.
    
    Architecture:
    - Conv1: (4, 84, 84) -> (32, 20, 20) [kernel=8, stride=4]
    - Conv2: (32, 20, 20) -> (64, 9, 9)  [kernel=4, stride=2]
    - Conv3: (64, 9, 9) -> (64, 7, 7)    [kernel=3, stride=1]
    - FC: 3136 -> 512 (shared features)
    - Policy head: 512 -> n_actions
    - Value head: 512 -> 1
    
    Note:
        Uses orthogonal initialization as recommended by PPO paper.
        All convolutional layers use ReLU activation.
    """
    
    def __init__(
        self,
        config: NetworkConfig
    ) -> None:
        """Initialize the PPO Actor-Critic network.
        
        Constructs the complete network architecture including convolutional
        layers, shared feature layer, and separate policy/value heads.
        Applies orthogonal weight initialization for better training stability.
        
        Args:
            config: NetworkConfig object (overrides individual params).
        """
        super(PPOActorCritic, self).__init__()
        
        input_channels = config.input_channels
        n_actions = config.n_actions
        
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.input_channels = config.input_channels
            self.n_actions = config.n_actions
        else:
            self.input_channels = input_channels
            self.n_actions = n_actions
        
        # ==========================================
        # Shared CNN Feature Extractor
        # ==========================================
        # Conv1: (4, 84, 84) -> (32, 20, 20)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=config.conv1_channels,
            kernel_size=CONV1_KERNEL_SIZE,
            stride=CONV1_STRIDE,
            padding=0
        )
        
        # Conv2: (32, 20, 20) -> (64, 9, 9)
        self.conv2 = nn.Conv2d(
            in_channels=config.conv1_channels,
            out_channels=config.conv2_channels,
            kernel_size=CONV2_KERNEL_SIZE,
            stride=CONV2_STRIDE,
            padding=0
        )
        
        # Conv3: (64, 9, 9) -> (64, 7, 7)
        self.conv3 = nn.Conv2d(
            in_channels=config.conv2_channels,
            out_channels=config.conv3_channels,
            kernel_size=CONV3_KERNEL_SIZE,
            stride=CONV3_STRIDE,
            padding=0
        )
        
        # Calculate flattened feature size
        # After conv1: (84 - 8) / 4 + 1 = 20
        # After conv2: (20 - 4) / 2 + 1 = 9
        # After conv3: (9 - 3) / 1 + 1 = 7
        self.flatten_size = config.conv3_channels * 7 * 7
        
        # Shared feature layer
        self.fc_shared = nn.Linear(self.flatten_size, config.fc_size)
        
        # ==========================================
        # Policy Head (Actor)
        # ==========================================
        self.policy_head = nn.Linear(config.fc_size, n_actions)
        
        # ==========================================
        # Value Head (Critic)
        # ==========================================
        self.value_head = nn.Linear(config.fc_size, 1)
        
        # Initialize weights using orthogonal initialization (PPO standard)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization.
        
        Applies orthogonal initialization to all convolutional and linear layers
        as recommended by the PPO paper. Policy head uses smaller gain (0.01)
        for more stable initial policy, while value head uses standard gain (1.0).
        
        Note:
            This follows PPO implementation best practices from OpenAI.
        
        References:
            Saxe et al. (2013) "Exact solutions to the nonlinear dynamics of learning"
            https://arxiv.org/abs/1312.6120
            Used in OpenAI Baselines: https://github.com/openai/baselines
        """
        # Orthogonal initialization for conv and linear layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for policy and value heads (PPO paper recommendations)
        nn.init.orthogonal_(self.policy_head.weight, gain=POLICY_HEAD_GAIN)
        nn.init.orthogonal_(self.value_head.weight, gain=VALUE_HEAD_GAIN)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Processes input through CNN layers, flattens, passes through shared FC layer,
        then splits into policy and value heads.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, 84, 84).
        
        Returns:
            Tuple containing:
                - action_logits: Raw logits for action distribution (batch_size, n_actions)
                - state_values: State value estimates V(s) (batch_size, 1)
        """
        # Shared CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # (batch_size, flatten_size)
        
        # Shared fully connected layer
        features = F.relu(self.fc_shared(x))
        
        # Policy head: action logits (no softmax, will use Categorical distribution)
        action_logits = self.policy_head(features)
        
        # Value head: state value estimate
        state_values = self.value_head(features)
        
        return action_logits, state_values
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value for PPO.
        
        This method is used during PPO training to sample actions and compute
        all necessary quantities for the PPO loss function.
        
        Args:
            x: Input state tensor.
            action: Optional action tensor for computing log_prob of specific actions.
        
        Returns:
            Tuple containing:
                - action: Sampled or provided action
                - log_prob: Log probability of the action
                - entropy: Entropy of the action distribution
                - value: State value estimate
        """
        action_logits, value = self.forward(x)
        
        # Create categorical distribution from logits
        dist = Categorical(logits=action_logits)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
