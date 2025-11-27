"""Manager Policy - High-Level Decision Making"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Any
from enum import IntEnum


class ActionType(IntEnum):
    """Discrete action types for the Manager"""
    CLICK = 0
    DOUBLE_CLICK = 1
    RIGHT_CLICK = 2
    TYPE = 3
    SCROLL = 4
    WAIT = 5
    EARLY_STOP = 6


class ManagerPolicy(nn.Module):
    """
    High-Level Manager Policy
    
    Outputs:
    - Action Type (discrete): Click, DoubleClick, RightClick, Type, Scroll, Wait, EarlyStop
    - Action Arguments (continuous): Coordinates (x, y) or text
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 512,
        num_action_types: int = 7,
        coord_output_dim: int = 2,
        max_coord: float = 1.0
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_action_types = num_action_types
        self.max_coord = max_coord
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action type head (discrete)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_action_types)
        )
        
        # Coordinate head (continuous) - outputs mean and log_std
        self.coord_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, coord_output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self.coord_log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, coord_output_dim)
        )
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self, state_embedding: torch.Tensor
    ) -> Tuple[Categorical, Normal, torch.Tensor]:
        """
        Args:
            state_embedding: State representation [B, state_dim]
        Returns:
            - action_type_dist: Categorical distribution over action types
            - coord_dist: Normal distribution for coordinates
            - value: State value estimate [B, 1]
        """
        shared_features = self.shared(state_embedding)
        
        # Action type distribution
        action_logits = self.action_type_head(shared_features)
        action_type_dist = Categorical(logits=action_logits)
        
        # Coordinate distribution
        coord_mean = self.coord_mean_head(shared_features) * self.max_coord
        coord_log_std = torch.clamp(self.coord_log_std_head(shared_features), -20, 2)
        coord_std = torch.exp(coord_log_std)
        coord_dist = Normal(coord_mean, coord_std)
        
        # Value estimate
        value = self.value_head(shared_features)
        
        return action_type_dist, coord_dist, value
    
    def sample_action(
        self, state_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy
        
        Returns:
            - action_type: Sampled action type [B]
            - coordinates: Sampled coordinates [B, 2]
            - log_prob: Log probability of the action
            - value: State value estimate
        """
        action_type_dist, coord_dist, value = self.forward(state_embedding)
        
        action_type = action_type_dist.sample()
        coordinates = coord_dist.sample()
        
        # Calculate total log probability
        action_type_log_prob = action_type_dist.log_prob(action_type)
        coord_log_prob = coord_dist.log_prob(coordinates).sum(dim=-1)
        total_log_prob = action_type_log_prob + coord_log_prob
        
        return action_type, coordinates, total_log_prob, value
    
    def evaluate_action(
        self,
        state_embedding: torch.Tensor,
        action_type: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate an action for PPO update
        
        Returns:
            - log_prob: Log probability of the action
            - entropy: Entropy of the distributions
            - value: State value estimate
        """
        action_type_dist, coord_dist, value = self.forward(state_embedding)
        
        action_type_log_prob = action_type_dist.log_prob(action_type)
        coord_log_prob = coord_dist.log_prob(coordinates).sum(dim=-1)
        total_log_prob = action_type_log_prob + coord_log_prob
        
        # Calculate entropy
        action_type_entropy = action_type_dist.entropy()
        coord_entropy = coord_dist.entropy().sum(dim=-1)
        total_entropy = action_type_entropy + coord_entropy
        
        return total_log_prob, total_entropy, value
    
    def get_action_deterministic(
        self, state_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get deterministic action (for evaluation)
        
        Returns:
            - action_type: Most likely action type [B]
            - coordinates: Mean coordinates [B, 2]
        """
        action_type_dist, coord_dist, _ = self.forward(state_embedding)
        
        action_type = action_type_dist.probs.argmax(dim=-1)
        coordinates = coord_dist.mean
        
        return action_type, coordinates
