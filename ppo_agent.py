"""!
@file ppo_agent.py
@brief Proximal Policy Optimization Agent Implementation

@details
Implements the complete PPO algorithm with:
- RolloutBuffer: Trajectory storage for on-policy learning
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective for policy updates
- Value function optimization with MSE loss
- Entropy regularization for exploration
- Minibatch gradient descent with multiple epochs

@author Adrian
@date 2025-11-29
@version 2.0

@references
[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms"
    arXiv preprint arXiv:1707.06347
    https://arxiv.org/abs/1707.06347

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    arXiv preprint arXiv:1506.02438
    https://arxiv.org/abs/1506.02438

[3] OpenAI Baselines - PPO2 Implementation
    https://github.com/openai/baselines
"""

from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from config import PPOConfig
from constants import (ADAM_EPSILON, ADVANTAGE_NORM_EPSILON)


class RolloutBuffer:
    """!
    @brief Trajectory storage buffer for on-policy learning
    
    @details
    Stores complete trajectories (states, actions, rewards, values, log_probs)
    collected during rollout phase. Used by PPO for batch policy updates.
    
    @note Buffer is cleared after each policy update (on-policy learning)
    """
    
    def __init__(self):
        """!
        @brief Initialize empty trajectory buffer
        
        @details Creates empty lists for storing trajectory components
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        value: float, 
        log_prob: float, 
        done: bool
    ) -> None:
        """!
        @brief Add transition to buffer
        
        @param state Current state observation
        @param action Action taken
        @param reward Reward received
        @param value Value estimate V(s)
        @param log_prob Log probability of action
        @param done Episode termination flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """!
        @brief Get all stored transitions
        
        @return Tuple of (states, actions, rewards, values, log_probs, dones)
        
        @details
        Returns all buffered transitions as numpy arrays.
        Does not clear the buffer (call clear() separately).
        """
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        return states, actions, rewards, values, log_probs, dones
    
    def clear(self) -> None:
        """!
        @brief Clear all stored transitions
        
        @details
        Empties all internal lists. Should be called after policy update.
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.states)


class PPOAgent:
    """!
    @brief Complete PPO agent with training capabilities
    
    @details
    Implements the PPO algorithm for discrete action spaces:
    
    Key Features:
    - Clipped surrogate objective: L^{CLIP}(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
    - GAE for advantage estimation: Â = Σ(γλ)^t δ_t
    - Value function fitting with MSE loss
    - Entropy bonus for exploration
    - Gradient clipping for stability
    
    @note Uses Adam optimizer with default lr=3e-4
    @note Implements standard PPO hyperparameters from literature
    """
    
    def __init__(
        self,
        n_actions: int,
        policy_net: nn.Module,
        config: PPOConfig,
        device: str = 'cpu'
    ) -> None:
        """!
        @brief Initialize PPO agent with hyperparameters
        
        @param policy_net Actor-Critic neural network
        @param config PPOConfig object (overrides individual params)
        @param device Device to run computations ('cpu' or 'cuda')
        
        @details
        Initializes the policy network, optimizer, and rollout buffer.
        Sets up all hyperparameters for PPO training.
        """

        self.n_actions = n_actions
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_range = config.clip_range
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        learning_rate = config.learning_rate
        
        self.config = config
        
        self.device = torch.device(device)
        self.policy_net = policy_net.to(self.device)
        
        # Store initial learning rate for annealing
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        
        # Rollout buffer for collecting trajectories
        self.rollout_buffer = RolloutBuffer()
        
        # Optimizer (Adam is standard for PPO)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate, 
            eps=ADAM_EPSILON
        )
        
        print(f"PPO Agent initialized:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma: {self.gamma}")
        print(f"  GAE Lambda: {self.gae_lambda}")
        print(f"  Clip range: {self.clip_range}")
        print(f"  Value coefficient: {self.value_coef}")
        print(f"  Entropy coefficient: {self.entropy_coef}")
    
    def update_learning_rate(self, progress: float) -> None:
        """!
        @brief Update learning rate with linear annealing
        
        @param progress Training progress from 1.0 (start) to 0.0 (end)
        
        @details
        Linearly decays learning rate from initial value to 0 based on progress.
        Common technique in PPO for improved convergence.
        """
        self.current_lr = self.initial_lr * progress
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """!
        @brief Select action using current policy
        
        @param state Current state observation
        @param training Whether in training mode (affects action selection)
        
        @return Tuple of (action, value, log_prob)
                If not training, value and log_prob are 0.0
        
        @details
        Training mode: Samples action from policy distribution
        Evaluation mode: Uses greedy action (argmax)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob, _, value = self.policy_net.get_action_and_value(state_tensor)
                return action.item(), value.item(), log_prob.item()
            else:
                # Greedy action selection for evaluation
                action_logits, _ = self.policy_net(state_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.probs.argmax(dim=1)
                return action.item(), 0.0, 0.0
    
    def remember(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        value: float, 
        log_prob: float, 
        done: bool
    ) -> None:
        """!
        @brief Store transition in rollout buffer
        
        @param state Current state observation
        @param action Action taken
        @param reward Reward received
        @param value Value estimate V(s)
        @param log_prob Log probability of action
        @param done Episode termination flag
        """
        self.rollout_buffer.push(state, action, reward, value, log_prob, done)
    
    def compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray, 
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """!
        @brief Compute Generalized Advantage Estimation (GAE)
        
        @param rewards Array of rewards from trajectory
        @param values Array of value estimates V(s)
        @param dones Array of episode termination flags
        @param next_value Value estimate of final state
        
        @return Tuple of (advantages, returns)
        
        @details
        Implements GAE as described in Schulman et al. (2015):
        A_t = Σ(γλ)^l δ_{t+l} where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        # Add next_value to values array for computation
        values_with_next = np.append(values, next_value)
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values_with_next[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values_with_next[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # GAE: A_t = δ_t + (γ * λ) * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def train_step(self) -> Tuple[float, float, float]:
        """!
        @brief Perform PPO update on collected rollout data
        
        @param n_epochs Number of optimization epochs
        @param batch_size Minibatch size for SGD
        
        @return Tuple of (avg_policy_loss, avg_value_loss, avg_entropy)
        
        @details
        Performs multiple epochs of minibatch SGD on the collected rollout.
        Uses clipped surrogate objective for policy and MSE for value function.
        
        @reference
        Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
        Equation (7): https://arxiv.org/abs/1707.06347
        Implementation based on OpenAI Baselines PPO2
        """
        n_epochs: int = self.config.n_epochs
        batch_size: int = self.config.batch_size
        
        # Get rollout data
        states, actions, rewards, values, old_log_probs, dones = self.rollout_buffer.get()
        
        # Compute next value for GAE (use last state's value or 0 if done)
        with torch.no_grad():
            if dones[-1]:
                next_value = 0.0
            else:
                last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
                _, next_value_tensor = self.policy_net(last_state)
                next_value = next_value_tensor.item()
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + ADVANTAGE_NORM_EPSILON)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(n_epochs):
            # Shuffle indices for each epoch
            np.random.shuffle(indices)
            
            # Minibatch updates
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get minibatch
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # Forward pass
                action_logits, new_values = self.policy_net(batch_states)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # ==========================================
                # PPO Clipped Surrogate Objective
                # ==========================================
                # Ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                
                # Policy loss: -min(surr1, surr2)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ==========================================
                # Value Loss (MSE)
                # ==========================================
                new_values = new_values.squeeze(-1)
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # ==========================================
                # Total Loss
                # ==========================================
                # L_total = L_policy + c1 * L_value - c2 * entropy
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (important for stability)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Clear rollout buffer after update
        self.rollout_buffer.clear()
        
        # Return average losses
        avg_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0.0
        avg_value_loss = total_value_loss / n_updates if n_updates > 0 else 0.0
        avg_entropy = total_entropy / n_updates if n_updates > 0 else 0.0
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def save(self, filepath: str) -> None:
        """!
        @brief Save model checkpoint
        
        @param filepath Path to save checkpoint file (.pth)
        
        @details
        Saves policy network weights, optimizer state, and hyperparameters.
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load(self, filepath: str, load_optimizer: bool = True, load_hyperparameters: bool = True) -> None:
        """!
        @brief Load model checkpoint
        
        @param filepath Path to checkpoint file (.pth)
        @param load_optimizer If True, loads optimizer state (for continuing training)
        @param load_hyperparameters If True, loads saved hyperparameters (overrides current)
        
        @details
        Loads policy network weights and optionally optimizer state and hyperparameters.
        Set load_hyperparameters=False for fine-tuning with new hyperparameters.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
        # Load optimizer state (optional - for continuing training)
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✓ Optimizer state restored (Adam momentum preserved)")
        else:
            print(f"  ✓ Optimizer state NOT loaded (fresh start for fine-tuning)")
        
        # Load hyperparameters from checkpoint (optional)
        if load_hyperparameters:
            self.n_actions = checkpoint.get('n_actions', self.n_actions)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.gae_lambda = checkpoint.get('gae_lambda', self.gae_lambda)
            self.clip_range = checkpoint.get('clip_range', self.clip_range)
            self.value_coef = checkpoint.get('value_coef', self.value_coef)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.max_grad_norm = checkpoint.get('max_grad_norm', self.max_grad_norm)
            print(f"  ✓ Hyperparameters loaded from checkpoint")
        else:
            print(f"  ✓ Using NEW hyperparameters (fine-tuning mode)")
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Clip range: {self.clip_range}")
