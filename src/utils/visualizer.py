"""
Real-time Training Visualization for OS Control Agent.

Provides real-time GUI visualization during training to see:
1. Agent's screen observations
2. Current actions being taken
3. Training metrics (reward, success rate, etc.)
4. Task progress

Uses pygame for efficient rendering optimized for 16GB GPU.
"""

import pygame
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque
import time
from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class TrainingVisualizer:
    """
    Real-time visualization window for training.
    
    Shows:
    - Left panel: Agent's screen observation
    - Right top: Action history
    - Right middle: Reward curve
    - Right bottom: Success rate and metrics
    """
    
    def __init__(
        self,
        window_width: int = 1600,
        window_height: int = 900,
        screen_width: int = 1920,
        screen_height: int = 1080,
        update_fps: int = 4,
        history_length: int = 100
    ):
        """
        Args:
            window_width: Visualization window width
            window_height: Visualization window height
            screen_width: Agent's screen width
            screen_height: Agent's screen height
            update_fps: Update frequency (frames per second)
            history_length: Number of steps to keep in history
        """
        self.window_width = window_width
        self.window_height = window_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.update_fps = update_fps
        
        # Initialize pygame
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Hierarchical RL Agent Training - Real-time Visualization")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_large = pygame.font.Font(None, 48)
            self.active = True
            logger.info("Training visualizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame visualizer: {e}")
            self.active = False
            return
        
        # History tracking
        self.reward_history = deque(maxlen=history_length)
        self.success_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=10)  # Last 10 actions
        
        # Current state
        self.current_screenshot = None
        self.current_action = None
        self.current_reward = 0.0
        self.current_step = 0
        self.current_episode = 0
        self.total_success = 0
        self.total_episodes = 0
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (220, 50, 50)
        self.GREEN = (50, 220, 50)
        self.BLUE = (50, 150, 220)
        self.GRAY = (100, 100, 100)
        self.LIGHT_GRAY = (180, 180, 180)
        
        # Layout
        self.screenshot_rect = pygame.Rect(10, 10, 960, 540)  # 1920x1080 scaled to 960x540
        self.info_panel_rect = pygame.Rect(980, 10, 610, 880)
        
    def update(
        self,
        screenshot: Optional[np.ndarray] = None,
        action: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        step: Optional[int] = None,
        episode: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Update visualization with new information.
        
        Args:
            screenshot: Current screen observation [H, W, 3]
            action: Current action dict with action_type, coordinates, text
            reward: Current reward
            step: Current step number
            episode: Current episode number
            metrics: Additional metrics dict
        """
        if not self.active:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.active = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.active = False
                    return
        
        # Update data
        if screenshot is not None:
            self.current_screenshot = screenshot
        if action is not None:
            self.current_action = action
            self.action_history.append(action)
        if reward is not None:
            self.current_reward = reward
            self.reward_history.append(reward)
        if step is not None:
            self.current_step = step
        if episode is not None:
            self.current_episode = episode
        
        if metrics:
            if 'success' in metrics:
                self.success_history.append(metrics['success'])
                if metrics['success']:
                    self.total_success += 1
            if 'episode_end' in metrics and metrics['episode_end']:
                self.total_episodes += 1
        
        # Render
        self._render()
        
        # Control frame rate
        self.clock.tick(self.update_fps)
    
    def _render(self):
        """Render the visualization."""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw screenshot panel
        self._draw_screenshot()
        
        # Draw info panel
        self._draw_info_panel()
        
        # Draw metrics panel
        self._draw_metrics()
        
        # Draw reward curve
        self._draw_reward_curve()
        
        # Draw action history
        self._draw_action_history()
        
        # Update display
        pygame.display.flip()
    
    def _draw_screenshot(self):
        """Draw the agent's screen observation."""
        if self.current_screenshot is None:
            # Draw placeholder
            pygame.draw.rect(self.screen, self.GRAY, self.screenshot_rect)
            text = self.font_medium.render("Waiting for observation...", True, self.WHITE)
            text_rect = text.get_rect(center=self.screenshot_rect.center)
            self.screen.blit(text, text_rect)
            return
        
        # Resize screenshot to fit panel
        try:
            screenshot = self.current_screenshot
            if screenshot.shape[0] != 540 or screenshot.shape[1] != 960:
                # Resize using pygame
                screenshot_surface = pygame.surfarray.make_surface(
                    screenshot.transpose(1, 0, 2)
                )
                screenshot_surface = pygame.transform.scale(
                    screenshot_surface, 
                    (self.screenshot_rect.width, self.screenshot_rect.height)
                )
            else:
                screenshot_surface = pygame.surfarray.make_surface(
                    screenshot.transpose(1, 0, 2)
                )
            
            self.screen.blit(screenshot_surface, self.screenshot_rect.topleft)
            
            # Draw border
            pygame.draw.rect(self.screen, self.WHITE, self.screenshot_rect, 2)
            
            # Draw action overlay if available
            if self.current_action:
                self._draw_action_overlay()
                
        except Exception as e:
            logger.error(f"Failed to render screenshot: {e}")
            pygame.draw.rect(self.screen, self.RED, self.screenshot_rect)
    
    def _draw_action_overlay(self):
        """Draw current action on top of screenshot."""
        if not self.current_action:
            return
        
        action_type = self.current_action.get('action_type', 0)
        coords = self.current_action.get('coordinates', [0.5, 0.5])
        
        # Map coordinates to screenshot rect
        x = int(self.screenshot_rect.x + coords[0] * self.screenshot_rect.width)
        y = int(self.screenshot_rect.y + coords[1] * self.screenshot_rect.height)
        
        # Action names
        action_names = ['Click', 'DblClick', 'RightClick', 'Type', 'Scroll', 'Wait', 'Stop']
        action_name = action_names[action_type] if action_type < len(action_names) else 'Unknown'
        
        # Draw crosshair
        color = self.GREEN if action_type in [0, 1, 2] else self.BLUE
        pygame.draw.circle(self.screen, color, (x, y), 15, 3)
        pygame.draw.line(self.screen, color, (x-20, y), (x+20, y), 2)
        pygame.draw.line(self.screen, color, (x, y-20), (x, y+20), 2)
        
        # Draw action label
        label = self.font_small.render(action_name, True, self.WHITE)
        label_bg = pygame.Surface((label.get_width() + 10, label.get_height() + 6))
        label_bg.fill(color)
        label_bg.set_alpha(200)
        self.screen.blit(label_bg, (x + 20, y - 30))
        self.screen.blit(label, (x + 25, y - 27))
    
    def _draw_info_panel(self):
        """Draw information panel."""
        y_offset = 20
        x = self.info_panel_rect.x + 10
        
        # Title
        title = self.font_large.render("Training Info", True, self.WHITE)
        self.screen.blit(title, (x, y_offset))
        y_offset += 60
        
        # Episode and step
        info_text = [
            f"Episode: {self.current_episode}",
            f"Step: {self.current_step}",
            f"Current Reward: {self.current_reward:.3f}",
            f"Success Rate: {self._calculate_success_rate():.1%}",
            f"Total Episodes: {self.total_episodes}",
            f"Total Successes: {self.total_success}"
        ]
        
        for text in info_text:
            rendered = self.font_medium.render(text, True, self.LIGHT_GRAY)
            self.screen.blit(rendered, (x, y_offset))
            y_offset += 35
    
    def _draw_metrics(self):
        """Draw metrics panel."""
        y_offset = 340
        x = self.info_panel_rect.x + 10
        
        # Metrics title
        title = self.font_large.render("Metrics", True, self.WHITE)
        self.screen.blit(title, (x, y_offset))
        y_offset += 50
        
        # Average reward
        avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0.0
        text = self.font_medium.render(f"Avg Reward: {avg_reward:.3f}", True, self.LIGHT_GRAY)
        self.screen.blit(text, (x, y_offset))
        y_offset += 35
        
        # Recent success rate
        recent_success = self._calculate_recent_success_rate()
        color = self.GREEN if recent_success > 0.5 else self.RED if recent_success > 0 else self.GRAY
        text = self.font_medium.render(f"Recent Success: {recent_success:.1%}", True, color)
        self.screen.blit(text, (x, y_offset))
    
    def _draw_reward_curve(self):
        """Draw reward history curve."""
        if len(self.reward_history) < 2:
            return
        
        y_offset = 480
        x = self.info_panel_rect.x + 10
        width = self.info_panel_rect.width - 20
        height = 150
        
        # Title
        title = self.font_medium.render("Reward History", True, self.WHITE)
        self.screen.blit(title, (x, y_offset))
        y_offset += 35
        
        # Draw graph background
        graph_rect = pygame.Rect(x, y_offset, width, height)
        pygame.draw.rect(self.screen, self.GRAY, graph_rect, 1)
        
        # Plot rewards
        rewards = list(self.reward_history)
        if rewards:
            max_reward = max(rewards) if max(rewards) > 0 else 1.0
            min_reward = min(rewards) if min(rewards) < 0 else 0.0
            range_reward = max_reward - min_reward if max_reward != min_reward else 1.0
            
            points = []
            for i, reward in enumerate(rewards):
                x_pos = x + (i / len(rewards)) * width
                y_pos = y_offset + height - ((reward - min_reward) / range_reward) * height
                points.append((x_pos, y_pos))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
    
    def _draw_action_history(self):
        """Draw recent action history."""
        y_offset = 660
        x = self.info_panel_rect.x + 10
        
        # Title
        title = self.font_medium.render("Recent Actions", True, self.WHITE)
        self.screen.blit(title, (x, y_offset))
        y_offset += 35
        
        # Action names
        action_names = ['Click', 'DblClick', 'RightClick', 'Type', 'Scroll', 'Wait', 'Stop']
        
        # Draw recent actions
        for i, action in enumerate(reversed(list(self.action_history))):
            action_type = action.get('action_type', 0)
            action_name = action_names[action_type] if action_type < len(action_names) else 'Unknown'
            
            coords = action.get('coordinates', [0, 0])
            text = f"{i+1}. {action_name} ({coords[0]:.2f}, {coords[1]:.2f})"
            
            rendered = self.font_small.render(text, True, self.LIGHT_GRAY)
            self.screen.blit(rendered, (x, y_offset))
            y_offset += 25
            
            if y_offset > self.window_height - 20:
                break
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_success / self.total_episodes
    
    def _calculate_recent_success_rate(self, window: int = 20) -> float:
        """Calculate recent success rate."""
        if not self.success_history:
            return 0.0
        recent = list(self.success_history)[-window:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def close(self):
        """Close the visualization window."""
        if self.active:
            pygame.quit()
            self.active = False
            logger.info("Training visualizer closed")
    
    def is_active(self) -> bool:
        """Check if visualizer is still active."""
        return self.active


def create_visualizer(enabled: bool = True, **kwargs) -> Optional[TrainingVisualizer]:
    """
    Factory function to create training visualizer.
    
    Args:
        enabled: Whether to create visualizer
        **kwargs: Arguments for TrainingVisualizer
        
    Returns:
        TrainingVisualizer instance or None if disabled
    """
    if not enabled:
        return None
    
    try:
        return TrainingVisualizer(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create visualizer: {e}")
        return None


__all__ = ['TrainingVisualizer', 'create_visualizer']
