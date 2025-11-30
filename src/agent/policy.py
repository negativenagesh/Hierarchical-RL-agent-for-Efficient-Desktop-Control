"""Hierarchical Policy - Combines Manager and Worker"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from .encoder import StateEncoder
from .manager import ManagerPolicy, ActionType
from .worker import HardcodedWorker


class HierarchicalPolicy(nn.Module):
    """
    Complete Hierarchical RL Policy
    
    Combines:
    - Triple-Modal State Encoder
    - Manager Policy (High-Level)
    - Worker Policy (Low-Level, hardcoded for now)
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 256,
        numeric_dim: int = 64,
        state_dim: int = 512,
        manager_hidden_dim: int = 512,
        num_action_types: int = 7,
        screen_width: int = 1920,
        screen_height: int = 1080
    ):
        super().__init__()
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        
        # State encoder
        logger.info("Initializing state encoder...")
        self.encoder = StateEncoder(
            visual_dim=visual_dim,
            text_dim=text_dim,
            numeric_dim=numeric_dim,
            output_dim=state_dim
        )
        logger.info("State encoder initialized")
        
        # Manager policy
        logger.info("Initializing manager policy...")
        self.manager = ManagerPolicy(
            state_dim=state_dim,
            hidden_dim=manager_hidden_dim,
            num_action_types=num_action_types
        )
        logger.info("Manager policy initialized")
        
        # Worker (hardcoded for execution)
        logger.info("Initializing hardcoded worker...")
        self.worker = HardcodedWorker(
            screen_width=screen_width,
            screen_height=screen_height
        )
        logger.info("Worker initialized")
        
        self.state_dim = state_dim
        logger.info("HierarchicalPolicy initialization complete")
        
    def encode_state(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """Encode raw state to embedding"""
        return self.encoder(state_dict)
    
    def forward(self, state_dict: Dict[str, Any]) -> Tuple:
        """
        Forward pass through the hierarchical policy
        
        Args:
            state_dict: Dictionary with 'image', 'instruction', 'numeric'
        Returns:
            Manager's output distributions and value
        """
        state_embedding = self.encode_state(state_dict)
        return self.manager(state_embedding)
    
    def sample_action(
        self, state_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the hierarchical policy
        
        Returns:
            - action_type: Sampled action type
            - coordinates: Sampled coordinates
            - log_prob: Log probability
            - value: State value
        """
        state_embedding = self.encode_state(state_dict)
        return self.manager.sample_action(state_embedding)
    
    def get_action(
        self, state_dict: Dict[str, Any], deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Get action from the policy (high-level interface)
        
        Args:
            state_dict: State information
            deterministic: If True, use deterministic policy
        Returns:
            Action dictionary
        """
        state_embedding = self.encode_state(state_dict)
        
        if deterministic:
            action_type, coordinates = self.manager.get_action_deterministic(state_embedding)
        else:
            action_type, coordinates, log_prob, value = self.manager.sample_action(state_embedding)
        
        # Convert to numpy for execution
        action_type_int = action_type.item() if torch.is_tensor(action_type) else action_type
        coords = coordinates.detach().cpu().numpy() if torch.is_tensor(coordinates) else coordinates
        
        return {
            'action_type': action_type_int,
            'coordinates': tuple(coords[0]) if len(coords.shape) > 1 else tuple(coords),
            'action_name': ActionType(action_type_int).name
        }
    
    def execute_action(
        self,
        action_dict: Dict[str, Any],
        text: Optional[str] = None,
        scroll_amount: int = 3
    ) -> bool:
        """
        Execute action using the hardcoded worker
        
        Args:
            action_dict: Action dictionary from get_action()
            text: Text to type (for TYPE action)
            scroll_amount: Scroll amount (for SCROLL action)
        Returns:
            Success status
        """
        return self.worker.execute_manager_action(
            action_type=action_dict['action_type'],
            coordinates=action_dict.get('coordinates'),
            text=text,
            scroll_amount=scroll_amount
        )
    
    def act_and_execute(
        self,
        state_dict: Dict[str, Any],
        text: Optional[str] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Get action and execute it immediately
        
        Returns:
            - action_dict: Action information
            - success: Execution status
        """
        action_dict = self.get_action(state_dict, deterministic=deterministic)
        success = self.execute_action(action_dict, text=text)
        return action_dict, success
    
    def evaluate_actions(
        self,
        state_dict: Dict[str, Any],
        action_type: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Returns:
            - log_prob: Log probability
            - entropy: Entropy
            - value: State value
        """
        state_embedding = self.encode_state(state_dict)
        return self.manager.evaluate_action(state_embedding, action_type, coordinates)
    
    def get_value(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """Get value estimate for a state"""
        state_embedding = self.encode_state(state_dict)
        _, _, value = self.manager(state_embedding)
        return value
    
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'manager': self.manager.state_dict(),
        }, path)
    
    def load(self, path: str, device: torch.device = torch.device('cpu')) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        # Handle both checkpoint formats:
        # 1. Training checkpoint: {'policy_state_dict': ..., 'optimizer_state_dict': ..., ...}
        # 2. Direct save: {'encoder': ..., 'manager': ...}
        if 'policy_state_dict' in checkpoint:
            # Training checkpoint format - load full policy state
            self.load_state_dict(checkpoint['policy_state_dict'])
        elif 'encoder' in checkpoint and 'manager' in checkpoint:
            # Direct save format - load components separately
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.manager.load_state_dict(checkpoint['manager'])
        else:
            raise ValueError(
                "Invalid checkpoint format. Expected either 'policy_state_dict' or 'encoder'/'manager' keys."
            )
