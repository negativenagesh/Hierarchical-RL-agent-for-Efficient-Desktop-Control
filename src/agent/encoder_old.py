"""Triple-Modal State Encoder Implementation"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models
from typing import Dict, Any, Tuple


class VisualEncoder(nn.Module):
    """Lightweight Visual Encoder using EfficientNet"""
    
    def __init__(self, visual_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # Use EfficientNet-B0 for lightweight visual encoding
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(1280, visual_dim),  # EfficientNet-B0 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Batch of screenshots [B, C, H, W]
        Returns:
            Visual features [B, visual_dim]
        """
        features = self.backbone(images)
        return self.projection(features)


class TextEncoder(nn.Module):
    """Lightweight Text Encoder using BERT-tiny"""
    
    def __init__(self, text_dim: int = 256, model_name: str = "prajjwal1/bert-tiny"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Freeze BERT weights for efficiency
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tokenized instruction IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
        Returns:
            Text features [B, text_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)
    
    def encode_text(self, texts: list[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method to tokenize and encode text"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)


class NumericStateEncoder(nn.Module):
    """Encoder for numeric state information (task_id, step_count, etc.)"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, numeric_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            numeric_state: Numeric features [B, input_dim]
        Returns:
            Encoded features [B, output_dim]
        """
        return self.mlp(numeric_state)


class TripleModalEncoder(nn.Module):
    """Combined Triple-Modal State Encoder (Vision + Text + Numeric)"""
    
    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 256,
        numeric_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 512
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder(visual_dim)
        self.text_encoder = TextEncoder(text_dim)
        self.numeric_encoder = NumericStateEncoder(output_dim=numeric_dim)
        
        # Fusion layer
        total_dim = visual_dim + text_dim + numeric_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: Screenshots [B, C, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            numeric_state: Numeric features [B, numeric_dim]
        Returns:
            Fused state representation [B, output_dim]
        """
        visual_features = self.visual_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        numeric_features = self.numeric_encoder(numeric_state)
        
        # Concatenate all modalities
        combined = torch.cat([visual_features, text_features, numeric_features], dim=1)
        
        # Fusion
        state_embedding = self.fusion(combined)
        return state_embedding


class StateEncoder(nn.Module):
    """Wrapper for backward compatibility"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = TripleModalEncoder(**kwargs)
        
    def forward(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            state_dict: Dictionary containing:
                - 'image': Screenshots [B, C, H, W]
                - 'instruction': Text instructions (list of strings)
                - 'numeric': Numeric state [B, N]
        Returns:
            State embedding [B, output_dim]
        """
        images = state_dict['image']
        instructions = state_dict['instruction']
        numeric = state_dict['numeric']
        
        # Encode text
        device = images.device
        input_ids, attention_mask = self.encoder.text_encoder.encode_text(
            instructions, device
        )
        
        return self.encoder(images, input_ids, attention_mask, numeric)
