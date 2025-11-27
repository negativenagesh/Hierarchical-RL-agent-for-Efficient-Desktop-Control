"""Screenshot Capture Utilities"""

import mss
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional


class ScreenCapture:
    """Fast screenshot capture using MSS"""
    
    def __init__(
        self,
        monitor_index: int = 1,
        target_width: int = 640,
        target_height: int = 480
    ):
        """
        Args:
            monitor_index: Monitor to capture (1 = primary)
            target_width: Width to resize to
            target_height: Height to resize to
        """
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.target_width = target_width
        self.target_height = target_height
        
        # Get monitor info
        self.monitor = self.sct.monitors[monitor_index]
        self.screen_width = self.monitor['width']
        self.screen_height = self.monitor['height']
        
    def capture(self, resize: bool = True) -> np.ndarray:
        """
        Capture screenshot
        
        Args:
            resize: Whether to resize to target dimensions
        Returns:
            Screenshot as numpy array [H, W, 3] in RGB
        """
        # Capture screenshot
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Resize if requested
        if resize and (img.shape[0] != self.target_height or img.shape[1] != self.target_width):
            img = cv2.resize(
                img,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        return img
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        resize: bool = False
    ) -> np.ndarray:
        """
        Capture specific region of screen
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height
            resize: Whether to resize to target dimensions
        Returns:
            Screenshot region as numpy array [H, W, 3] in RGB
        """
        # Define region
        region = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
        
        # Capture
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        if resize:
            img = cv2.resize(
                img,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        return img
    
    def save_screenshot(self, path: str) -> None:
        """Save screenshot to file"""
        img = self.capture(resize=False)
        Image.fromarray(img).save(path)
    
    def preprocess_for_model(
        self,
        img: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess screenshot for model input
        
        Args:
            img: Screenshot [H, W, 3]
            normalize: Whether to normalize to [0, 1]
        Returns:
            Preprocessed image [C, H, W]
        """
        # Convert to float
        img = img.astype(np.float32)
        
        # Normalize
        if normalize:
            img = img / 255.0
        
        # Transpose to [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def close(self) -> None:
        """Close MSS instance"""
        self.sct.close()
    
    def __del__(self):
        """Cleanup"""
        self.close()
