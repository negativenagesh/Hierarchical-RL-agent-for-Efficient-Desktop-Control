"""Curriculum Learning Manager"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..environment.base_env import TaskDifficulty


@dataclass
class CurriculumStage:
    """Curriculum learning stage"""
    name: str
    difficulty: TaskDifficulty
    min_steps: int
    success_threshold: float
    duration_steps: int


class CurriculumManager:
    """
    Curriculum Learning Manager
    
    Implements staged curriculum learning as described in the paper:
    1. Easy tasks (< 8 steps) first
    2. Progressively harder tasks
    3. Transition based on success rate
    """
    
    def __init__(
        self,
        easy_threshold: float = 0.6,
        medium_threshold: float = 0.5,
        min_steps_per_stage: int = 50000
    ):
        """
        Args:
            easy_threshold: Success rate to move from EASY to MEDIUM
            medium_threshold: Success rate to move from MEDIUM to HARD
            min_steps_per_stage: Minimum steps before allowing stage transition
        """
        self.easy_threshold = easy_threshold
        self.medium_threshold = medium_threshold
        self.min_steps_per_stage = min_steps_per_stage
        
        # Define curriculum stages
        self.stages = [
            CurriculumStage(
                name="Easy",
                difficulty=TaskDifficulty.EASY,
                min_steps=0,
                success_threshold=easy_threshold,
                duration_steps=100000
            ),
            CurriculumStage(
                name="Medium",
                difficulty=TaskDifficulty.MEDIUM,
                min_steps=8,
                success_threshold=medium_threshold,
                duration_steps=150000
            ),
            CurriculumStage(
                name="Hard",
                difficulty=TaskDifficulty.HARD,
                min_steps=15,
                success_threshold=0.0,  # No threshold for final stage
                duration_steps=float('inf')
            )
        ]
        
        # Current stage
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.successes_in_stage: list[bool] = []
        
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    @property
    def current_difficulty(self) -> TaskDifficulty:
        """Get current task difficulty"""
        return self.current_stage.difficulty
    
    def update(self, success: bool, steps_taken: int) -> bool:
        """
        Update curriculum based on episode result
        
        Args:
            success: Whether episode was successful
            steps_taken: Number of steps in episode
        Returns:
            Whether stage was advanced
        """
        self.successes_in_stage.append(success)
        self.steps_in_stage += steps_taken
        
        # Check if we should advance stage
        if self._should_advance():
            return self.advance_stage()
        
        return False
    
    def _should_advance(self) -> bool:
        """Check if we should advance to next stage"""
        # Can't advance from final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        # Need minimum steps in current stage
        if self.steps_in_stage < self.min_steps_per_stage:
            return False
        
        # Need enough episodes for statistics
        if len(self.successes_in_stage) < 50:
            return False
        
        # Check success rate (last 100 episodes)
        recent_successes = self.successes_in_stage[-100:]
        success_rate = np.mean(recent_successes)
        
        return success_rate >= self.current_stage.success_threshold
    
    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage
        
        Returns:
            Whether advancement was successful
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.steps_in_stage = 0
            self.successes_in_stage = []
            
            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCED TO STAGE {self.current_stage_idx + 1}: {self.current_stage.name}")
            print(f"Difficulty: {self.current_stage.difficulty.name}")
            print(f"{'='*60}\n")
            
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        if len(self.successes_in_stage) == 0:
            success_rate = 0.0
        else:
            success_rate = np.mean(self.successes_in_stage[-100:])
        
        return {
            'stage': self.current_stage.name,
            'stage_idx': self.current_stage_idx,
            'difficulty': self.current_stage.difficulty.name,
            'steps_in_stage': self.steps_in_stage,
            'episodes_in_stage': len(self.successes_in_stage),
            'success_rate': success_rate,
            'next_threshold': self.current_stage.success_threshold if self.current_stage_idx < len(self.stages) - 1 else None
        }
    
    def reset(self) -> None:
        """Reset curriculum to first stage"""
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.successes_in_stage = []
    
    def set_stage(self, stage_idx: int) -> None:
        """Manually set curriculum stage"""
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.steps_in_stage = 0
            self.successes_in_stage = []
