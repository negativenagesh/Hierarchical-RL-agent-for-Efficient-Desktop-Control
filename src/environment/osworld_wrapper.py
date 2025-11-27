"""OSWorld Environment Wrapper"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json
import random
from pathlib import Path

from .base_env import OSEnvironment, TaskDifficulty
from .screenshot import ScreenCapture
from ..agent.worker import HardcodedWorker


class OSWorldEnv(OSEnvironment):
    """
    OSWorld Environment Wrapper
    
    This wraps the OSWorld benchmark for training and evaluation.
    OSWorld provides a Dockerized desktop environment with predefined tasks.
    """
    
    def __init__(
        self,
        tasks_file: str = "tasks.json",
        difficulty_filter: Optional[TaskDifficulty] = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
        capture_width: int = 640,
        capture_height: int = 480,
        **kwargs
    ):
        """
        Args:
            tasks_file: Path to JSON file with task definitions
            difficulty_filter: Filter tasks by difficulty (for curriculum learning)
            screen_width: Screen width
            screen_height: Screen height
            capture_width: Width to resize screenshots to
            capture_height: Height to resize screenshots to
        """
        super().__init__(screen_width, screen_height, **kwargs)
        
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.difficulty_filter = difficulty_filter
        
        # Load tasks
        self.tasks = self._load_tasks(tasks_file)
        if difficulty_filter is not None:
            self.tasks = [t for t in self.tasks if t['difficulty'] == difficulty_filter]
        
        # Initialize screen capture
        self.screen_capture = ScreenCapture(
            monitor_index=1,
            target_width=capture_width,
            target_height=capture_height
        )
        
        # Initialize worker for action execution
        self.worker = HardcodedWorker(
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        # Current task
        self.current_task: Optional[Dict[str, Any]] = None
        
    def _load_tasks(self, tasks_file: str) -> List[Dict[str, Any]]:
        """
        Load tasks from JSON file
        
        Expected format:
        [
            {
                "id": 1,
                "instruction": "Open Calculator",
                "difficulty": "EASY",
                "num_steps": 3,
                "success_criteria": {...}
            },
            ...
        ]
        """
        tasks_path = Path(tasks_file)
        
        if not tasks_path.exists():
            # Create dummy tasks for testing
            return self._create_dummy_tasks()
        
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)
        
        # Convert difficulty strings to enum
        for task in tasks:
            task['difficulty'] = TaskDifficulty[task['difficulty']]
        
        return tasks
    
    def _create_dummy_tasks(self) -> List[Dict[str, Any]]:
        """Create dummy tasks for testing"""
        return [
            {
                'id': 1,
                'instruction': 'Open the calculator application',
                'difficulty': TaskDifficulty.EASY,
                'num_steps': 5,
                'success_criteria': {'window_title': 'Calculator'}
            },
            {
                'id': 2,
                'instruction': 'Create a new text file named test.txt',
                'difficulty': TaskDifficulty.EASY,
                'num_steps': 7,
                'success_criteria': {'file_exists': 'test.txt'}
            },
            {
                'id': 3,
                'instruction': 'Open browser and navigate to example.com',
                'difficulty': TaskDifficulty.MEDIUM,
                'num_steps': 10,
                'success_criteria': {'url_contains': 'example.com'}
            },
            {
                'id': 4,
                'instruction': 'Find invoice.pdf, rename it to invoice_2024.pdf, and move to Documents',
                'difficulty': TaskDifficulty.HARD,
                'num_steps': 20,
                'success_criteria': {
                    'file_exists': 'Documents/invoice_2024.pdf',
                    'file_not_exists': 'invoice.pdf'
                }
            }
        ]
    
    def _get_new_task(self) -> Tuple[str, int, TaskDifficulty]:
        """Get a new random task"""
        if not self.tasks:
            raise ValueError("No tasks available")
        
        self.current_task = random.choice(self.tasks)
        
        return (
            self.current_task['instruction'],
            self.current_task['id'],
            self.current_task['difficulty']
        )
    
    def _capture_screenshot(self) -> np.ndarray:
        """Capture current screenshot"""
        return self.screen_capture.capture(resize=True)
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute action using the worker"""
        return self.worker.execute_manager_action(
            action_type=action['action_type'],
            coordinates=tuple(action['coordinates']),
            text=action.get('text'),
            scroll_amount=action.get('scroll_amount', 3)
        )
    
    def _check_task_completion(self) -> Tuple[bool, bool]:
        """
        Check if current task is complete
        
        This is a placeholder - real implementation would check
        the success criteria defined in the task.
        """
        if self.current_task is None:
            return False, False
        
        # In real implementation, this would check:
        # - Window titles
        # - File existence
        # - URL navigation
        # - Application state
        # etc.
        
        # For now, return False (not done) - requires OSWorld integration
        return False, False
    
    def _get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position"""
        return self.worker.get_current_mouse_position()
    
    def set_difficulty_filter(self, difficulty: Optional[TaskDifficulty]) -> None:
        """
        Change difficulty filter (for curriculum learning)
        
        Args:
            difficulty: New difficulty filter, or None for all tasks
        """
        self.difficulty_filter = difficulty
        self.tasks = self._load_tasks("tasks.json")
        
        if difficulty is not None:
            self.tasks = [t for t in self.tasks if t['difficulty'] == difficulty]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about available tasks"""
        if not self.tasks:
            return {}
        
        stats = {
            'total_tasks': len(self.tasks),
            'easy_tasks': sum(1 for t in self.tasks if t['difficulty'] == TaskDifficulty.EASY),
            'medium_tasks': sum(1 for t in self.tasks if t['difficulty'] == TaskDifficulty.MEDIUM),
            'hard_tasks': sum(1 for t in self.tasks if t['difficulty'] == TaskDifficulty.HARD),
            'avg_steps': np.mean([t['num_steps'] for t in self.tasks])
        }
        
        return stats
    
    def close(self) -> None:
        """Clean up resources"""
        self.screen_capture.close()
        super().close()
