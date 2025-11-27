"""
UPGRADED Triple-Modal State Encoder for OS Control Agent.

Uses powerful pre-trained models optimized for 16GB GPU:
- CLIP ViT-B/16 for visual encoding (OpenAI's vision-language model)
- OpenAI text-embedding-3-large OR sentence-transformers for text encoding
- MLP for numeric state encoding

Based on ComputerAgent paper's triple-modal architecture with SOTA models.
"""

import torch
import torch.nn as nn
import clip
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from typing import Dict, Tuple, Optional, Any
import numpy as np
from loguru import logger
import os


class VisualEncoder(nn.Module):
    """
    Visual encoder using CLIP ViT-B/16 (OpenAI).
    Much more powerful than EfficientNet-B0, trained on 400M image-text pairs.
    Output: 512-dimensional visual embeddings.
    """
    
    def __init__(self, device: str = "cuda", freeze_backbone: bool = True):
        super().__init__()
        self.device = device
        
        # Load CLIP model - more powerful than EfficientNet
        logger.info("Loading CLIP ViT-B/16 model for visual encoding...")
        logger.info("NOTE: First run will download ~350MB model from OpenAI servers")
        logger.info("Model will be cached in ~/.cache/clip for future use")
        
        try:
            import os
            cache_path = os.path.expanduser("~/.cache/clip")
            if os.path.exists(cache_path):
                logger.info(f"Using cached CLIP model from {cache_path}")
            else:
                logger.info("Downloading CLIP model (this may take a few minutes)...")
            
            self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device, download_root=cache_path)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            logger.error("Please check your internet connection or manually download the model")
            raise
        
        if freeze_backbone:
            # Freeze CLIP backbone to save memory and training time
            for param in self.clip_model.parameters():
                param.requires_grad = False
            logger.info("CLIP visual encoder frozen")
        
        # CLIP outputs 512-dim embeddings, add projection layer
        self.projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512)
        ).to(device)
        
        self.visual_dim = 512
        logger.info(f"VisualEncoder initialized (output_dim={self.visual_dim})")
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W) in [0, 1] range
            
        Returns:
            visual_features: Tensor of shape (batch_size, visual_dim)
        """
        # CLIP expects images in specific format
        batch_size = images.size(0)
        
        # Ensure images are on correct device
        images = images.to(self.device)
        
        # CLIP preprocessing is already applied in environment
        # Extract visual features using CLIP image encoder
        with torch.no_grad() if not self.training else torch.enable_grad():
            visual_features = self.clip_model.encode_image(images)
            visual_features = visual_features.float()
        
        # Project to desired dimension
        visual_features = self.projection(visual_features)
        
        return visual_features


class TextEncoder(nn.Module):
    """
    Text encoder using powerful transformer models.
    Two options:
    1. OpenAI text-embedding-3-large (3072-dim, most powerful, requires API)
    2. sentence-transformers/all-mpnet-base-v2 (768-dim, open-source, excellent)
    
    Choose based on whether you want to use OpenAI API or run fully local.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.device = device
        self.use_openai = use_openai
        
        if use_openai:
            # OpenAI embedding API - most powerful option
            try:
                import openai
                if openai_api_key is None:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
                else:
                    logger.warning("No OpenAI API key provided, falling back to local model")
                    self.use_openai = False
            except ImportError:
                logger.warning("openai package not installed, falling back to local model")
                self.use_openai = False
        
        if self.use_openai:
            self.model_name = "text-embedding-3-large"
            self.text_dim = 3072  # OpenAI embedding size
            logger.info(f"Using OpenAI {self.model_name} for text encoding")
            
            # Projection to match other modalities
            self.projection = nn.Sequential(
                nn.Linear(3072, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 512)
            ).to(device)
            
        else:
            # Use sentence-transformers all-mpnet-base-v2 (open-source, powerful)
            model_name = "sentence-transformers/all-mpnet-base-v2"
            logger.info(f"Loading {model_name} for text encoding...")
            logger.info("NOTE: First run will download model from HuggingFace")
            
            try:
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("Loading text model...")
                self.model = AutoModel.from_pretrained(model_name).to(device)
                logger.info("Text model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                logger.error("Please check your internet connection or manually download the model")
                raise
            
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("Text encoder frozen")
            
            self.text_dim = 768  # MPNet embedding size
            
            # Projection to match other modalities
            self.projection = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            ).to(device)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take average of all token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: List of strings (batch_size,)
            
        Returns:
            text_features: Tensor of shape (batch_size, 512)
        """
        if self.use_openai:
            # Use OpenAI API for embeddings
            try:
                embeddings = []
                for text in texts:
                    response = self.openai_client.embeddings.create(
                        input=text,
                        model=self.model_name
                    )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                
                text_features = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
                text_features = self.projection(text_features)
            except Exception as e:
                logger.error(f"OpenAI API error: {e}, falling back to zeros")
                # Fallback to zeros if API fails
                text_features = torch.zeros(len(texts), 512, device=self.device)
            
        else:
            # Use local transformer model
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.model(**encoded)
            
            # Mean pooling
            text_features = self._mean_pooling(outputs, encoded['attention_mask'])
            text_features = self.projection(text_features)
        
        return text_features
    
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Helper method for backward compatibility"""
        return self.forward(texts)


class NumericStateEncoder(nn.Module):
    """Encoder for numeric state information (task_id, step_count, mouse position, etc.)"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU()
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
    """
    UPGRADED Combined Triple-Modal State Encoder (Vision + Text + Numeric).
    
    Uses CLIP + sentence-transformers/OpenAI for much more powerful encoding.
    Optimized for 16GB GPU with mixed precision support.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        visual_dim: int = 512,
        text_dim: int = 512,
        numeric_dim: int = 64,
        hidden_dim: int = 768,
        output_dim: int = 512,
        freeze_backbones: bool = True
    ):
        super().__init__()
        self.device = device
        
        # Initialize powerful encoders
        self.visual_encoder = VisualEncoder(
            device=device, 
            freeze_backbone=freeze_backbones
        )
        
        self.text_encoder = TextEncoder(
            device=device,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            freeze_backbone=freeze_backbones
        )
        
        self.numeric_encoder = NumericStateEncoder(
            output_dim=numeric_dim
        ).to(device)
        
        # Multi-modal fusion with attention mechanism
        total_dim = visual_dim + text_dim + numeric_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ).to(device)
        
        logger.info(f"TripleModalEncoder initialized on {device}")
        logger.info(f"  Visual: CLIP ViT-B/16 ({visual_dim}D)")
        logger.info(f"  Text: {'OpenAI API' if use_openai else 'MPNet'} ({text_dim}D)")
        logger.info(f"  Numeric: MLP ({numeric_dim}D)")
        logger.info(f"  Output: {output_dim}D")
        
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        numeric_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: Screenshots [B, C, H, W]
            texts: List of instruction strings (length B)
            numeric_state: Numeric features [B, numeric_dim]
        Returns:
            Fused state representation [B, output_dim]
        """
        # Encode each modality
        visual_features = self.visual_encoder(images)
        text_features = self.text_encoder(texts)
        numeric_features = self.numeric_encoder(numeric_state)
        
        # Concatenate all modalities
        combined = torch.cat([visual_features, text_features, numeric_features], dim=1)
        
        # Fusion
        state_embedding = self.fusion(combined)
        return state_embedding


class StateEncoder(nn.Module):
    """
    Wrapper for backward compatibility with existing codebase.
    Translates old state_dict format to new encoder format.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        use_openai: bool = False,
        **kwargs
    ):
        super().__init__()
        self.encoder = TripleModalEncoder(
            device=device,
            use_openai=use_openai,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
        
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
        
        return self.encoder(images, instructions, numeric)


# For easy import
__all__ = [
    'VisualEncoder',
    'TextEncoder',
    'NumericStateEncoder',
    'TripleModalEncoder',
    'StateEncoder'
]
