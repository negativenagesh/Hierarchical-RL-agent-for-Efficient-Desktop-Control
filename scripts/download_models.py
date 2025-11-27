#!/usr/bin/env python3
"""
Script to pre-download all required pretrained models.

This script downloads:
1. CLIP ViT-B/16 model (~350MB)
2. sentence-transformers/all-mpnet-base-v2 model (~420MB)

Total download size: ~770MB

Run this before training to avoid delays during model initialization.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def download_clip_model():
    """Download CLIP ViT-B/16 model"""
    logger.info("=" * 60)
    logger.info("Downloading CLIP ViT-B/16 model...")
    logger.info("=" * 60)
    
    try:
        import clip
        import torch
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Set cache directory
        cache_path = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_path, exist_ok=True)
        
        # Check if already cached
        model_file = os.path.join(cache_path, "ViT-B-16.pt")
        if os.path.exists(model_file):
            logger.info(f"✓ CLIP model already cached at: {model_file}")
            logger.info(f"  File size: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")
        else:
            logger.info("Downloading CLIP model (this may take a few minutes)...")
            logger.info("Model will be saved to: " + cache_path)
        
        # Load model (will download if not cached)
        model, preprocess = clip.load("ViT-B/16", device=device, download_root=cache_path)
        
        logger.success("✓ CLIP model downloaded and verified successfully!")
        logger.info(f"  Cache location: {cache_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download CLIP model: {e}")
        return False


def download_text_model():
    """Download sentence-transformers model"""
    logger.info("=" * 60)
    logger.info("Downloading sentence-transformers/all-mpnet-base-v2 model...")
    logger.info("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Check cache
        from transformers import cached_file
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        logger.info(f"HuggingFace cache directory: {cache_dir}")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.success("✓ Tokenizer downloaded")
        
        # Download model
        logger.info("Downloading model weights (this may take a few minutes)...")
        model = AutoModel.from_pretrained(model_name)
        logger.success("✓ Text model downloaded and verified successfully!")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download text model: {e}")
        return False


def check_disk_space():
    """Check available disk space"""
    import shutil
    
    # Check home directory
    home = os.path.expanduser("~")
    total, used, free = shutil.disk_usage(home)
    
    free_gb = free / (1024 ** 3)
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 2.0:
        logger.warning("⚠ Low disk space! At least 2GB recommended for model downloads")
        return False
    
    return True


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Model Download Script")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This script will download pretrained models required for training:")
    logger.info("  1. CLIP ViT-B/16 (~350MB)")
    logger.info("  2. sentence-transformers/all-mpnet-base-v2 (~420MB)")
    logger.info("")
    logger.info("Total download size: ~770MB")
    logger.info("")
    
    # Check disk space
    if not check_disk_space():
        logger.error("Insufficient disk space. Please free up space and try again.")
        return 1
    
    logger.info("Starting downloads...")
    logger.info("")
    
    # Download models
    success = True
    
    # Download CLIP
    if not download_clip_model():
        success = False
        logger.error("Failed to download CLIP model")
    logger.info("")
    
    # Download text model
    if not download_text_model():
        success = False
        logger.error("Failed to download text model")
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    if success:
        logger.success("✓ All models downloaded successfully!")
        logger.info("")
        logger.info("You can now run training without delays:")
        logger.info("  python src/training/train.py --visualize")
        logger.info("")
        return 0
    else:
        logger.error("✗ Some models failed to download")
        logger.info("")
        logger.info("Please check:")
        logger.info("  1. Internet connection")
        logger.info("  2. Firewall settings")
        logger.info("  3. Disk space")
        logger.info("")
        logger.info("You can also manually download models:")
        logger.info("  CLIP: https://openaipublic.azureedge.net/clip/models/")
        logger.info("  Text: https://huggingface.co/sentence-transformers/all-mpnet-base-v2")
        logger.info("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
