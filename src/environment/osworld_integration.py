"""
Real OSWorld Integration for OS Control Agent.

Integrates with OSWorld benchmark from: https://github.com/xlang-ai/OSWorld
OSWorld provides real Ubuntu desktop environments with GUI access.

This module handles:
1. OSWorld repository setup and installation
2. Desktop environment initialization via desktop_env
3. Task loading and evaluation
4. Action execution through OSWorld API

Note: OSWorld uses desktop_env package, not a Docker image.
Install: pip install git+https://github.com/xlang-ai/OSWorld.git
"""

import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
from loguru import logger
import json
import os
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class OSWorldManager:
    """
    Manages OSWorld setup and desktop environments.
    
    OSWorld is installed from GitHub and uses desktop_env for VM management.
    """
    
    def __init__(
        self,
        osworld_path: Optional[str] = None,
        provider: str = "docker",
        base_port: int = 5900,
        max_environments: int = 4
    ):
        """
        Args:
            osworld_path: Path to OSWorld repository (auto-installs if None)
            provider: VM provider ('docker', 'vmware', 'virtualbox', 'aws')
            base_port: Base VNC port (will use base_port, base_port+1, ...)
            max_environments: Maximum number of parallel environments
        """
        self.provider = provider
        self.base_port = base_port
        self.max_environments = max_environments
        
        # Setup OSWorld
        if osworld_path is None:
            osworld_path = os.path.expanduser("~/.osworld")
        self.osworld_path = Path(osworld_path)
        
        # Ensure OSWorld is installed
        self._ensure_osworld_installed()
        
        # Track running environments
        self.environments: Dict[str, Any] = {}
        self.env_ports: Dict[str, int] = {}
        
    def _ensure_osworld_installed(self):
        """Ensure OSWorld repository is cloned and desktop_env is installed."""
        try:
            # Try importing desktop_env first
            import desktop_env
            logger.info("desktop_env already installed")
            return
        except ImportError:
            logger.info("desktop_env not found, installing OSWorld...")
            
        # Clone OSWorld if not exists
        if not self.osworld_path.exists():
            logger.info(f"Cloning OSWorld to {self.osworld_path}...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/xlang-ai/OSWorld.git", str(self.osworld_path)],
                    check=True,
                    capture_output=True
                )
                logger.info("OSWorld repository cloned successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone OSWorld: {e.stderr.decode()}")
                raise
        
        # Install desktop_env
        logger.info("Installing desktop_env package...")
        try:
            subprocess.run(
                ["pip", "install", "-e", str(self.osworld_path)],
                check=True,
                capture_output=True
            )
            logger.info("desktop_env installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install desktop_env: {e.stderr.decode()}")
            # Try alternative installation
            try:
                subprocess.run(
                    ["pip", "install", "git+https://github.com/xlang-ai/OSWorld.git"],
                    check=True,
                    capture_output=True
                )
                logger.info("desktop_env installed via git URL")
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to install desktop_env: {e2.stderr.decode()}")
                raise
    
    def start_environment(self, env_id: str, task_config: Optional[Dict] = None) -> Tuple[str, int]:
        """
        Start an OSWorld desktop environment.
        
        Args:
            env_id: Unique identifier for this environment
            task_config: Optional task configuration
            
        Returns:
            env_id: Environment identifier
            vnc_port: VNC port number for this environment
        """
        if env_id in self.environments:
            logger.warning(f"Environment {env_id} already running")
            return env_id, self.env_ports[env_id]
        
        # Find available port
        vnc_port = self.base_port + len(self.environments)
        
        try:
            # Import desktop_env
            from desktop_env.desktop_env import DesktopEnv
            
            # Initialize desktop environment
            env = DesktopEnv(
                provider_name=self.provider,
                os_type="Ubuntu",  # Can be made configurable
                headless=False,  # Show GUI
                require_terminal=True,
                action_space="pyautogui"
            )
            
            logger.info(f"Started OSWorld environment {env_id} with provider {self.provider}")
            
            self.environments[env_id] = env
            self.env_ports[env_id] = vnc_port
            
            return env_id, vnc_port
            
        except ImportError as e:
            logger.error(f"desktop_env not installed: {e}")
            logger.error("Run: pip install git+https://github.com/xlang-ai/OSWorld.git")
            raise
        except Exception as e:
            logger.error(f"Failed to start environment {env_id}: {e}")
            raise
    
    def stop_environment(self, env_id: str):
        """Stop and close a desktop environment."""
        if env_id not in self.environments:
            logger.warning(f"Environment {env_id} not found")
            return
        
        try:
            env = self.environments[env_id]
            env.close()
            
            del self.environments[env_id]
            del self.env_ports[env_id]
            
            logger.info(f"Stopped environment {env_id}")
        except Exception as e:
            logger.error(f"Failed to stop environment {env_id}: {e}")
    
    def stop_all(self):
        """Stop all running environments."""
        env_ids = list(self.environments.keys())
        for env_id in env_ids:
            self.stop_environment(env_id)
    
    def get_environment(self, env_id: str):
        """Get a desktop environment by ID."""
        return self.environments.get(env_id)


class OSWorldEnvironment(gym.Env):
    """
    Gymnasium environment interface for OSWorld.
    
    Provides:
    - Visual observations from VNC screen capture
    - Action execution through OSWorld API
    - Task loading and evaluation
    - Real-time GUI visualization support
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        task_name: str = "basic_web_browsing",
        render_mode: Optional[str] = None,
        visualize: bool = True,
        osworld_manager: Optional[OSWorldManager] = None
    ):
        """
        Args:
            task_name: OSWorld task to run
            render_mode: 'human' for GUI window, 'rgb_array' for numpy array
            visualize: Whether to show real-time GUI during training
            osworld_manager: Shared OSWorld manager (creates new if None)
        """
        super().__init__()
        
        self.task_name = task_name
        self.render_mode = render_mode
        self.visualize = visualize
        
        # OSWorld manager
        if osworld_manager is None:
            self.manager = OSWorldManager(provider="docker")
        else:
            self.manager = osworld_manager
        
        self.env_id = f"env_{id(self)}"
        self.desktop_env = None
        self.vnc_port = None
        
        # Observation and action spaces
        self.observation_space = spaces.Dict({
            'screenshot': spaces.Box(
                low=0, high=255, 
                shape=(1080, 1920, 3), 
                dtype=np.uint8
            ),
            'task_description': spaces.Text(max_length=1000),
            'step_count': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32)
        })
        
        # Action space: [action_type, x, y, scroll_amount, text]
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(7),  # click, dblclick, rightclick, type, scroll, wait, stop
            'x': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'y': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'scroll': spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32),
            'text': spaces.Text(max_length=500)
        })
        
        self.current_step = 0
        self.max_steps = 100
        self.task_data = None
        
        # Visualization window
        if self.visualize:
            try:
                import pygame
                pygame.init()
                self.screen = pygame.display.set_mode((1920, 1080))
                pygame.display.set_caption(f"OSWorld Agent - {task_name}")
                logger.info("Pygame visualization window initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pygame: {e}")
                self.visualize = False
    
    def _capture_screen(self) -> np.ndarray:
        """Capture screenshot from desktop environment."""
        try:
            if self.desktop_env is None:
                logger.warning("Desktop environment not initialized")
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Get observation from desktop_env
            obs = self.desktop_env.observe()
            
            # desktop_env returns screenshot as PIL Image or numpy array
            if isinstance(obs, dict) and 'screenshot' in obs:
                screenshot = obs['screenshot']
            elif hasattr(obs, 'screenshot'):
                screenshot = obs.screenshot
            else:
                screenshot = obs
            
            # Convert to numpy array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            # Ensure correct shape (H, W, C)
            if screenshot.ndim == 2:
                screenshot = np.stack([screenshot] * 3, axis=-1)
            
            # Resize to expected dimensions if needed
            if screenshot.shape[:2] != (1080, 1920):
                screenshot = np.array(Image.fromarray(screenshot).resize((1920, 1080)))
            
            return screenshot
            
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            # Return black screen as fallback
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute action in OSWorld desktop environment.
        
        Args:
            action: Action dictionary with type, coordinates, text
            
        Returns:
            success: Whether action was executed successfully
        """
        try:
            if self.desktop_env is None:
                logger.error("Desktop environment not initialized")
                return False
            
            action_type = action['action_type']
            x = int(action['x'] * 1920)
            y = int(action['y'] * 1080)
            
            # Map action types to desktop_env actions
            if action_type == 0:  # Click
                desktop_action = f"pyautogui.click({x}, {y})"
            elif action_type == 1:  # Double click
                desktop_action = f"pyautogui.doubleClick({x}, {y})"
            elif action_type == 2:  # Right click
                desktop_action = f"pyautogui.rightClick({x}, {y})"
            elif action_type == 3:  # Type
                text = action.get('text', '')
                desktop_action = f"pyautogui.typewrite('{text}')"
            elif action_type == 4:  # Scroll
                scroll_amount = int(action['scroll'] * 100)
                desktop_action = f"pyautogui.scroll({scroll_amount}, x={x}, y={y})"
            elif action_type == 5:  # Wait
                time.sleep(0.5)
                return True
            elif action_type == 6:  # Early stop
                return True
            else:
                return False
            
            # Execute action via desktop_env
            obs, reward, done, info = self.desktop_env.step(desktop_action)
            
            time.sleep(0.1)  # Small delay for action to take effect
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and start new task."""
        super().reset(seed=seed)
        
        # Start OSWorld desktop environment if not running
        if self.desktop_env is None:
            _, self.vnc_port = self.manager.start_environment(self.env_id)
            self.desktop_env = self.manager.get_environment(self.env_id)
            
            # Reset the desktop environment
            if self.desktop_env:
                self.desktop_env.reset()
        
        self.current_step = 0
        
        # Load task data
        # In real implementation, load from OSWorld task repository
        self.task_data = {
            'description': f'Task: {self.task_name}',
            'goal': 'Complete the specified task',
            'max_steps': self.max_steps
        }
        
        # Get initial observation
        screenshot = self._capture_screen()
        
        observation = {
            'screenshot': screenshot,
            'task_description': self.task_data['description'],
            'step_count': np.array([0], dtype=np.int32)
        }
        
        info = {
            'task_name': self.task_name,
            'vnc_port': self.vnc_port
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return next observation."""
        self.current_step += 1
        
        # Execute action
        success = self._execute_action(action)
        
        # Capture new state
        screenshot = self._capture_screen()
        
        # Visualize if enabled
        if self.visualize and self.screen:
            import pygame
            # Convert to pygame surface and display
            surf = pygame.surfarray.make_surface(screenshot.transpose(1, 0, 2))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.visualize = False
        
        observation = {
            'screenshot': screenshot,
            'task_description': self.task_data['description'],
            'step_count': np.array([self.current_step], dtype=np.int32)
        }
        
        # Calculate reward (simplified - real OSWorld has evaluation metrics)
        reward = 0.1 if success else 0.0
        
        # Check termination
        terminated = (action['action_type'] == 6)  # Early stop
        truncated = (self.current_step >= self.max_steps)
        
        info = {
            'success': success,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (handled by visualize flag)."""
        if self.render_mode == 'rgb_array':
            return self._capture_screen()
        return None
    
    def close(self):
        """Clean up resources."""
        if self.visualize and hasattr(self, 'screen'):
            import pygame
            pygame.quit()
        
        if self.desktop_env:
            self.manager.stop_environment(self.env_id)
            self.desktop_env = None
        
        logger.info(f"Closed environment {self.env_id}")


# Utility functions
def list_osworld_tasks() -> List[str]:
    """List available OSWorld tasks."""
    # This would load from OSWorld task repository
    return [
        "basic_web_browsing",
        "file_management",
        "text_editing",
        "email_tasks",
        "system_settings",
        "application_usage"
    ]


def create_osworld_env(
    task_name: str,
    visualize: bool = True,
    **kwargs
) -> OSWorldEnvironment:
    """
    Factory function to create OSWorld environment.
    
    Args:
        task_name: Task to run
        visualize: Show GUI during execution
        **kwargs: Additional arguments for OSWorldEnvironment
        
    Returns:
        Configured OSWorld environment
    """
    env = OSWorldEnvironment(
        task_name=task_name,
        visualize=visualize,
        render_mode='human' if visualize else None,
        **kwargs
    )
    
    return env


__all__ = [
    'OSWorldManager',
    'OSWorldEnvironment',
    'list_osworld_tasks',
    'create_osworld_env'
]
