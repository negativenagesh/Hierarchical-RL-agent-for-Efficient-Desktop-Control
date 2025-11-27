"""Worker Policy - Low-Level Action Execution"""

import torch
import torch.nn as nn
import pyautogui
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WorkerAction:
    """Low-level action to be executed"""
    delta_x: float = 0.0
    delta_y: float = 0.0
    mouse_down: bool = False
    mouse_up: bool = False
    key_press: Optional[str] = None
    scroll_amount: int = 0


class WorkerPolicy(nn.Module):
    """
    Learned Worker Policy for executing Manager's commands
    (Can be used for end-to-end learning)
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        goal_dim: int = 4,  # target_x, target_y, action_type, etc.
        hidden_dim: int = 256,
        action_dim: int = 4  # delta_x, delta_y, click, scroll
    ):
        super().__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(
        self, state_embedding: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state_embedding: Current state [B, state_dim]
            goal: Manager's goal [B, goal_dim]
        Returns:
            Low-level actions [B, action_dim]
        """
        combined = torch.cat([state_embedding, goal], dim=-1)
        return self.policy(combined)


class HardcodedWorker:
    """
    Hardcoded Worker for executing Manager's commands
    
    This is the practical approach mentioned in the paper - using
    hardcoded macros to execute high-level commands.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        move_duration: float = 0.2,
        click_delay: float = 0.1
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.move_duration = move_duration
        self.click_delay = click_delay
        
        # Disable PyAutoGUI fail-safe for production
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        
    def denormalize_coordinates(
        self, normalized_coords: Tuple[float, float]
    ) -> Tuple[int, int]:
        """Convert normalized coordinates [-1, 1] to screen coordinates"""
        x_norm, y_norm = normalized_coords
        
        # Convert from [-1, 1] to [0, screen_size]
        x = int((x_norm + 1) / 2 * self.screen_width)
        y = int((y_norm + 1) / 2 * self.screen_height)
        
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        return x, y
    
    def execute_click(self, coordinates: Tuple[float, float]) -> bool:
        """Execute single click at coordinates"""
        try:
            x, y = self.denormalize_coordinates(coordinates)
            pyautogui.click(x, y, duration=self.move_duration)
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Click failed: {e}")
            return False
    
    def execute_double_click(self, coordinates: Tuple[float, float]) -> bool:
        """Execute double click at coordinates"""
        try:
            x, y = self.denormalize_coordinates(coordinates)
            pyautogui.doubleClick(x, y, duration=self.move_duration)
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Double click failed: {e}")
            return False
    
    def execute_right_click(self, coordinates: Tuple[float, float]) -> bool:
        """Execute right click at coordinates"""
        try:
            x, y = self.denormalize_coordinates(coordinates)
            pyautogui.rightClick(x, y, duration=self.move_duration)
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Right click failed: {e}")
            return False
    
    def execute_type(self, text: str, interval: float = 0.05) -> bool:
        """Type text"""
        try:
            pyautogui.write(text, interval=interval)
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Type failed: {e}")
            return False
    
    def execute_scroll(self, amount: int = 3) -> bool:
        """Execute scroll (positive = up, negative = down)"""
        try:
            pyautogui.scroll(amount * 120)  # 120 units per scroll tick
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Scroll failed: {e}")
            return False
    
    def execute_wait(self, duration: float = 0.5) -> bool:
        """Wait for specified duration"""
        time.sleep(duration)
        return True
    
    def execute_manager_action(
        self,
        action_type: int,
        coordinates: Optional[Tuple[float, float]] = None,
        text: Optional[str] = None,
        scroll_amount: int = 3
    ) -> bool:
        """
        Execute a high-level action from the Manager
        
        Args:
            action_type: ActionType enum value
            coordinates: Normalized coordinates [-1, 1]
            text: Text to type (for TYPE action)
            scroll_amount: Scroll amount (for SCROLL action)
        Returns:
            Success status
        """
        from .manager import ActionType
        
        if action_type == ActionType.CLICK:
            if coordinates is None:
                return False
            return self.execute_click(coordinates)
        
        elif action_type == ActionType.DOUBLE_CLICK:
            if coordinates is None:
                return False
            return self.execute_double_click(coordinates)
        
        elif action_type == ActionType.RIGHT_CLICK:
            if coordinates is None:
                return False
            return self.execute_right_click(coordinates)
        
        elif action_type == ActionType.TYPE:
            if text is None:
                return False
            return self.execute_type(text)
        
        elif action_type == ActionType.SCROLL:
            return self.execute_scroll(scroll_amount)
        
        elif action_type == ActionType.WAIT:
            return self.execute_wait()
        
        elif action_type == ActionType.EARLY_STOP:
            # Early stop doesn't execute anything
            return True
        
        return False
    
    def get_current_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()
    
    def move_mouse(self, coordinates: Tuple[float, float]) -> bool:
        """Move mouse to coordinates without clicking"""
        try:
            x, y = self.denormalize_coordinates(coordinates)
            pyautogui.moveTo(x, y, duration=self.move_duration)
            return True
        except Exception as e:
            print(f"Mouse move failed: {e}")
            return False
