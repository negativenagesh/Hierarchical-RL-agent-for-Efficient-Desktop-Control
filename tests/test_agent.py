"""Test agent modules"""

import pytest
import torch
import numpy as np

from src.agent.encoder import TripleModalEncoder, StateEncoder
from src.agent.manager import ManagerPolicy, ActionType
from src.agent.worker import HardcodedWorker
from src.agent.policy import HierarchicalPolicy


class TestTripleModalEncoder:
    """Test triple-modal encoder"""
    
    def test_initialization(self):
        """Test encoder initialization"""
        encoder = TripleModalEncoder()
        assert encoder is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        encoder = TripleModalEncoder()
        
        # Create dummy inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 480, 640)
        input_ids = torch.randint(0, 1000, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        numeric = torch.randn(batch_size, 10)
        
        # Forward pass
        output = encoder(images, input_ids, attention_mask, numeric)
        
        assert output.shape == (batch_size, 512)


class TestManagerPolicy:
    """Test manager policy"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = ManagerPolicy()
        assert manager is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        manager = ManagerPolicy()
        
        state = torch.randn(2, 512)
        action_dist, coord_dist, value = manager(state)
        
        assert action_dist.logits.shape == (2, 7)
        assert coord_dist.mean.shape == (2, 2)
        assert value.shape == (2, 1)
    
    def test_sample_action(self):
        """Test action sampling"""
        manager = ManagerPolicy()
        
        state = torch.randn(1, 512)
        action_type, coords, log_prob, value = manager.sample_action(state)
        
        assert 0 <= action_type.item() < 7
        assert coords.shape == (1, 2)


class TestHardcodedWorker:
    """Test hardcoded worker"""
    
    def test_initialization(self):
        """Test worker initialization"""
        worker = HardcodedWorker()
        assert worker is not None
    
    def test_coordinate_conversion(self):
        """Test coordinate denormalization"""
        worker = HardcodedWorker(screen_width=1920, screen_height=1080)
        
        # Test center
        x, y = worker.denormalize_coordinates((0.0, 0.0))
        assert x == 960
        assert y == 540
        
        # Test corners
        x, y = worker.denormalize_coordinates((-1.0, -1.0))
        assert x == 0
        assert y == 0


class TestHierarchicalPolicy:
    """Test hierarchical policy"""
    
    def test_initialization(self):
        """Test policy initialization"""
        policy = HierarchicalPolicy()
        assert policy is not None
    
    def test_get_action(self):
        """Test getting action"""
        policy = HierarchicalPolicy()
        
        state_dict = {
            'image': torch.randn(1, 3, 480, 640),
            'instruction': ["Open calculator"],
            'numeric': torch.zeros(1, 10)
        }
        
        action = policy.get_action(state_dict, deterministic=True)
        
        assert 'action_type' in action
        assert 'coordinates' in action
        assert 'action_name' in action
