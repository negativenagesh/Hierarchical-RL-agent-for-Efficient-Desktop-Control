"""API Configuration"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings - Simplified for Academic Use"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Model Settings - Upgraded for 16GB GPU
    MODEL_CHECKPOINT_PATH: str = "checkpoints/final_model.pt"
    DEVICE: str = "cuda"  # cuda or cpu
    USE_OPENAI: bool = False  # Set to True to use OpenAI embeddings
    OPENAI_API_KEY: str = ""
    
    # Screen Settings
    SCREEN_WIDTH: int = 1920
    SCREEN_HEIGHT: int = 1080
    CAPTURE_WIDTH: int = 640
    CAPTURE_HEIGHT: int = 480
    
    # Training Settings - Optimized for 16GB GPU
    TRAINING_ENABLED: bool = True
    DEFAULT_LEARNING_RATE: float = 3e-4
    DEFAULT_ROLLOUT_STEPS: int = 2048
    VISUALIZE_TRAINING: bool = True  # Show real-time GUI during training
    MIXED_PRECISION: bool = True  # Use automatic mixed precision for efficiency
    
    # OSWorld Settings
    OSWORLD_REPO_PATH: str = "~/.osworld"  # Path to clone OSWorld repo
    OSWORLD_PROVIDER: str = "docker"  # VM provider: docker, vmware, virtualbox, aws
    OSWORLD_OS_TYPE: str = "Ubuntu"  # OS type: Ubuntu or Windows
    OSWORLD_BASE_PORT: int = 5900
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

