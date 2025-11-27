"""Logging utilities"""

import sys
from loguru import logger
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_path / "app.log",
        rotation="500 MB",
        retention="10 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    # Add error file handler
    logger.add(
        log_path / "error.log",
        rotation="500 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def get_logger(name: str):
    """
    Get logger instance
    
    Args:
        name: Logger name (usually __name__)
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
