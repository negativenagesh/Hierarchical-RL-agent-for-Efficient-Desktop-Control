"""Replay Buffer for PPO"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class Transition:
    """Single transition"""
    state: Dict[str, Any]
    action_type: int
    coordinates: np.ndarray
    reward: float
    value: float
    log_prob: float
    done: bool
    advantage: float = 0.0
    return_: float = 0.0


class ReplayBuffer:
    """
    Replay Buffer for storing and processing PPO rollouts
    
    Stores transitions and computes advantages using GAE.
    """
    
    def __init__(self, capacity: int = 2048):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.clear()
    
    def add(
        self,
        state: Dict[str, Any],
        action_type: int,
        coordinates: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Add transition to buffer"""
        self.states.append(state)
        self.action_types.append(action_type)
        self.coordinates.append(coordinates)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        returns = []
        
        advantage = 0
        next_value = 0
        
        # Iterate backwards
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0
                advantage = 0
            
            # TD error
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            
            # GAE
            advantage = delta + gamma * gae_lambda * advantage
            
            # Store
            advantages.insert(0, advantage)
            returns.insert(0, advantage + self.values[t])
            
            # Update for next iteration
            next_value = self.values[t]
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages.tolist()
        self.returns = returns
    
    def get(self) -> Dict[str, List]:
        """
        Get all data from buffer
        
        Returns:
            Dictionary with all transitions
        """
        return {
            'states': self.states,
            'action_types': self.action_types,
            'coordinates': self.coordinates,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
            'advantages': self.advantages,
            'returns': self.returns
        }
    
    def clear(self) -> None:
        """Clear buffer"""
        self.states: List[Dict[str, Any]] = []
        self.action_types: List[int] = []
        self.coordinates: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []
    
    def __len__(self) -> int:
        """Get buffer size"""
        return len(self.states)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self) >= self.capacity
