"""Environment Wrappers and Preprocessing Utilities.

This module provides Gymnasium environment wrappers for preprocessing
FlappyBird observations for PPO training with CNNs.

Components:
- PixelObservationWrapper: Converts environment to pixel observations
- FramePreprocessor: RGB to grayscale conversion, cropping, normalization
- FrameStack: Temporal frame stacking for motion information

References:
    [1] Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning"
        Nature, 518(7540), 529-533.
        https://www.nature.com/articles/nature14236
        Note: Frame stacking and preprocessing pipeline inspired by DQN

    [2] ITU-R Recommendation BT.601-7 (2011).
        "Studio encoding parameters of digital television"
        https://www.itu.int/rec/R-REC-BT.601/
        Note: Luminosity method for RGB to grayscale conversion
"""

from typing import Optional, Tuple
import numpy as np
from collections import deque
import cv2
import gymnasium as gym

from config import EnvironmentConfig
from constants import (GRAYSCALE_WEIGHTS)


class PixelObservationWrapper(gym.Wrapper):
    """Wrapper for converting environment observations to pixels.
    
    Converts FlappyBird's default LIDAR/coordinate observations to preprocessed
    pixel observations suitable for CNN input. Renders the environment at each
    step and applies preprocessing pipeline.
    
    Note:
        Requires render_mode='rgb_array' in base environment.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        config: EnvironmentConfig,
    ) -> None:
        """Initialize the pixel observation wrapper.
        
        Args:
            env: Gymnasium environment (must have render_mode='rgb_array').
            config: Optional EnvironmentConfig object (overrides individual params).
            img_height: Target height for resized observations.
            img_width: Target width for resized observations.
            
        Raises:
            ValueError: If environment doesn't have render_mode='rgb_array'.
        """
        super().__init__(env)
        

        self.img_height = config.img_height
        self.img_width = config.img_width

        self.preprocessor = FramePreprocessor(self.img_height, self.img_width, config.crop_ratio)
        self.last_frame: Optional[np.ndarray] = None
        
        # Ensure environment is in rgb_array mode
        if self.env.render_mode != 'rgb_array':
            raise ValueError(f"PixelObservationWrapper requires render_mode='rgb_array', got '{self.env.render_mode}'")
        
        # Update observation space to reflect pixel observations
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(self.img_height, self.img_width), 
            dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and return pixel observation.
        
        Args:
            **kwargs: Additional arguments for environment reset.
            
        Returns:
            Tuple of (preprocessed_observation, info).
        """
        _, info = self.env.reset(**kwargs)
        
        # Get pixel observation by rendering
        pixel_obs = self.env.render()
        
        # Validate we got a valid frame
        if pixel_obs is None:
            raise RuntimeError("env.render() returned None. Ensure render_mode='rgb_array'")
        
        if not isinstance(pixel_obs, np.ndarray):
            raise RuntimeError(f"env.render() returned {type(pixel_obs)}, expected numpy array")
        
        # Store raw frame for debugging
        self.last_frame = pixel_obs.copy()
        
        # Preprocess to grayscale 84x84
        processed_obs = self.preprocessor.preprocess(pixel_obs)
        
        return processed_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take environment step and return pixel observation.
        
        Args:
            action: Action to take in environment.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        
        # Get pixel observation by rendering
        pixel_obs = self.env.render()
        
        # Validate frame
        if pixel_obs is None:
            raise RuntimeError("env.render() returned None during step")
        
        # Store raw frame for debugging
        self.last_frame = pixel_obs.copy()
        
        # Preprocess to grayscale 84x84
        processed_obs = self.preprocessor.preprocess(pixel_obs)
        
        return processed_obs, reward, terminated, truncated, info
    
    def save_debug_frame(self, filename: str = "debug_agent_vision.png") -> None:
        """Save current frame for debugging.
        
        Saves both raw RGB and preprocessed grayscale frames side-by-side.
        
        Args:
            filename: Output filename for the debug image.
        """
        if self.last_frame is None:
            print("WARNING: No frame available yet. Call reset() or step() first.")
            return
        
        import matplotlib.pyplot as plt
        
        # Get preprocessed frame
        processed = self.preprocessor.preprocess(self.last_frame)
        
        # Create debug visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw RGB frame
        axes[0].imshow(self.last_frame)
        axes[0].set_title(f"Raw Rendered Frame\n{self.last_frame.shape}")
        axes[0].axis('off')
        
        # Preprocessed grayscale frame
        axes[1].imshow(processed, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Preprocessed (Agent's View)\n{processed.shape} | Range: [{processed.min():.3f}, {processed.max():.3f}]")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Debug frame saved to: {filename}")
        print(f"  Raw shape: {self.last_frame.shape}, dtype: {self.last_frame.dtype}")
        print(f"  Processed shape: {processed.shape}, dtype: {processed.dtype}")
        print(f"  Processed range: [{processed.min():.3f}, {processed.max():.3f}]")


class FramePreprocessor:
    """Frame preprocessing pipeline for FlappyBird.
    
    Applies a series of preprocessing steps to raw RGB frames:
    1. RGB to grayscale conversion (luminosity method)
    2. Ground cropping (removes bottom 22% of frame)
    3. Resize to target dimensions (84x84)
    4. Normalize pixel values to [0, 1]
    
    Note:
        Ground cropping removes static, non-informative pixels.
    """
    
    def __init__(
        self, 
        img_height: int, 
        img_width: int,
        crop_ratio: float
    ) -> None:
        """Initialize the frame preprocessor.
        
        Args:
            img_height: Target height for resized frames.
            img_width: Target width for resized frames.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.crop_ratio = crop_ratio
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single RGB frame.
        
        Processing steps:
        1. RGB to grayscale using ITU-R BT.601 luminosity weights (0.299R + 0.587G + 0.114B)
        2. Crop bottom 22% (removes static ground - domain-specific for FlappyBird)
        3. Resize to target dimensions (84x84 standard from DQN paper)
        4. Normalize to [0, 1]
        
        Args:
            frame: RGB frame of shape (H, W, 3) with values in [0, 255].
            
        Returns:
            Preprocessed grayscale frame of shape (84, 84) with values in [0, 1].
            
        References:
            ITU-R BT.601: https://www.itu.int/rec/R-REC-BT.601/
            Grayscale weights: Y = 0.299R + 0.587G + 0.114B
        """
        # Convert RGB to grayscale using luminosity method
        gray = np.dot(frame[..., :3], GRAYSCALE_WEIGHTS)
        
        # Crop out the ground
        height = gray.shape[0]
        crop_height = int(height * self.crop_ratio)
        cropped = gray[:crop_height, :]
        
        # Resize to target dimensions
        resized = cv2.resize(
            cropped, 
            (self.img_width, self.img_height), 
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize to [0, 1] for neural network
        normalized = resized / 255.0
        
        return normalized.astype(np.float32)


class FrameStack(gym.Wrapper):
    """!
    @brief Wrapper for stacking multiple frames temporally
    
    @details
    Maintains a deque of the most recent frames and returns them stacked
    along the first dimension. Provides temporal information to the agent
    for better decision making (e.g., velocity estimation).
    
    Output shape: (stack_size, height, width)
    
    @note Uses collections.deque for efficient frame management
    
    @reference
    Mnih et al. (2015) "Human-level control through deep reinforcement learning"
    Section "Preprocessing": Frame stacking provides temporal information for
    estimating velocity and acceleration from static frames.
    https://www.nature.com/articles/nature14236
    """
    
    def __init__(
        self, 
        env: gym.Env,
        config: EnvironmentConfig
    ) -> None:
        """Initialize the frame stacking wrapper.
        
        Args:
            env: Environment to wrap (must return 2D observations).
            config: Optional EnvironmentConfig object (overrides stack_size).
            stack_size: Number of frames to stack.
            
        Raises:
            ValueError: If observation space is not Box or not 2D.
        """
        super().__init__(env)
        

        self.stack_size = config.stack_size

        self.frames: deque = deque(maxlen=self.stack_size)
        
        # Get original observation space shape
        orig_space = env.observation_space
        if not isinstance(orig_space, gym.spaces.Box):
            raise ValueError(f"FrameStack requires Box observation space, got {type(orig_space)}")
        
        # Verify observation is 2D (height, width) - single channel grayscale
        if len(orig_space.shape) != 2:
            raise ValueError(f"FrameStack expects 2D observations (H, W), got shape {orig_space.shape}")
        
        height, width = orig_space.shape
        
        # Update observation space to reflect stacked frames: (stack_size, height, width)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.stack_size, height, width), 
            dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and initialize frame stack.
        
        Args:
            **kwargs: Additional arguments for environment reset.
            
        Returns:
            Tuple of (stacked_observation, info).
        """
        obs, info = self.env.reset(**kwargs)
        
        # Fill stack with copies of the initial frame
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        
        return self._get_stacked_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take step and return stacked observation.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (stacked_observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add new frame to stack
        self.frames.append(obs)
        
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observation as numpy array.
        
        Returns:
            Stacked frames of shape (stack_size, height, width).
        """
        return np.array(self.frames, dtype=np.float32)