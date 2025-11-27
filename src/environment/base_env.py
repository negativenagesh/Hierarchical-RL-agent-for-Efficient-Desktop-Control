"""Base OS Environment Interface"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class TaskDifficulty(IntEnum):
    """Task difficulty levels for curriculum learning"""
    EASY = 0  # < 8 steps
    MEDIUM = 1  # 8-15 steps
    HARD = 2  # > 15 steps


@dataclass
class OSState:
    """OS State representation"""
    screenshot: np.ndarray  # RGB image [H, W, 3]
    instruction: str  # Task instruction
    task_id: int  # Unique task identifier
    step_count: int  # Current step in episode
    previous_action: Optional[int] = None  # Previous action type
    cursor_position: Tuple[int, int] = (0, 0)  # Current cursor position
    difficulty: TaskDifficulty = TaskDifficulty.EASY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for network input"""
        return {
            'image': self.screenshot,
            'instruction': self.instruction,
            'numeric': np.array([
                self.task_id,
                self.step_count,
                self.previous_action or -1,
                self.cursor_position[0],
                self.cursor_position[1],
                self.difficulty,
                0.0,  # Reserved
                0.0,  # Reserved
                0.0,  # Reserved
                0.0   # Reserved
            ], dtype=np.float32)
        }


@dataclass
class OSAction:
    """OS Action representation"""
    action_type: int  # ActionType enum
    coordinates: Tuple[float, float]  # Normalized [-1, 1]
    text: Optional[str] = None  # For TYPE action
    scroll_amount: int = 0  # For SCROLL action


class OSEnvironment(gym.Env):
    """
    Base class for OS Control Environment
    
    This is a Gym-compatible environment for training RL agents
    to control operating systems.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        max_steps: int = 50,
        reward_per_step: float = -0.1,
        reward_success: float = 10.0,
        reward_failure: float = -5.0
    ):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_steps = max_steps
        self.reward_per_step = reward_per_step
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0, high=255,
                shape=(screen_height, screen_width, 3),
                dtype=np.uint8
            ),
            'instruction': gym.spaces.Text(max_length=512),
            'numeric': gym.spaces.Box(
                low=-1e6, high=1e6,
                shape=(10,),
                dtype=np.float32
            )
        })
        
        # Define action space (hybrid)
        self.action_space = gym.spaces.Dict({
            'action_type': gym.spaces.Discrete(7),  # 7 action types
            'coordinates': gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        })
        
        # State tracking
        self.current_state: Optional[OSState] = None
        self.current_instruction: str = ""
        self.current_task_id: int = 0
        self.step_count: int = 0
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment
        
        Returns:
            - observation: Initial observation
            - info: Additional information
        """
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Get new task (to be implemented by subclasses)
        instruction, task_id, difficulty = self._get_new_task()
        
        # Capture initial screenshot
        screenshot = self._capture_screenshot()
        
        # Create initial state
        self.current_state = OSState(
            screenshot=screenshot,
            instruction=instruction,
            task_id=task_id,
            step_count=self.step_count,
            difficulty=difficulty
        )
        
        obs = self.current_state.to_dict()
        info = {
            'task_id': task_id,
            'instruction': instruction,
            'difficulty': difficulty.name
        }
        
        return obs, info
    
    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Dictionary with 'action_type' and 'coordinates'
        Returns:
            - observation: Next observation
            - reward: Reward for this step
            - terminated: Whether episode is done (success/failure)
            - truncated: Whether episode was truncated (max steps)
            - info: Additional information
        """
        self.step_count += 1
        
        # Execute action in the OS
        success = self._execute_action(action)
        
        # Capture new screenshot
        screenshot = self._capture_screenshot()
        
        # Check if task is complete
        task_done, task_success = self._check_task_completion()
        
        # Calculate reward
        reward = self.reward_per_step  # Step penalty
        
        terminated = False
        if action['action_type'] == 6:  # EARLY_STOP
            if task_success:
                reward += self.reward_success
                terminated = True
            else:
                reward += self.reward_failure
                terminated = True
        elif task_done:
            reward += self.reward_success if task_success else self.reward_failure
            terminated = True
        
        # Check truncation
        truncated = self.step_count >= self.max_steps
        
        # Update state
        self.current_state = OSState(
            screenshot=screenshot,
            instruction=self.current_state.instruction,
            task_id=self.current_state.task_id,
            step_count=self.step_count,
            previous_action=action['action_type'],
            cursor_position=self._get_cursor_position(),
            difficulty=self.current_state.difficulty
        )
        
        obs = self.current_state.to_dict()
        info = {
            'success': task_success if terminated else False,
            'step_count': self.step_count,
            'action_executed': success
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_new_task(self) -> Tuple[str, int, TaskDifficulty]:
        """Get a new task (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def _capture_screenshot(self) -> np.ndarray:
        """Capture current screenshot (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute action in OS (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def _check_task_completion(self) -> Tuple[bool, bool]:
        """
        Check if task is complete
        
        Returns:
            - done: Whether task is finished
            - success: Whether task was successful
        """
        raise NotImplementedError
    
    def _get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position"""
        raise NotImplementedError
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment"""
        if self.current_state is not None:
            return self.current_state.screenshot
        return None
    
    def close(self) -> None:
        """Clean up environment resources"""
        pass
