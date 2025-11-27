"""Example usage of the Hierarchical RL Agent"""

import torch
import numpy as np
from PIL import Image
import time

from src.agent.policy import HierarchicalPolicy
from src.environment.screenshot import ScreenCapture


def example_inference():
    """Example: Run inference with the trained model"""
    print("=" * 60)
    print("Example: Model Inference")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    policy = HierarchicalPolicy(
        screen_width=1920,
        screen_height=1080
    ).to(device)
    
    # Load checkpoint if available
    checkpoint_path = "checkpoints/final_model.pt"
    try:
        policy.load(checkpoint_path, device)
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print("No checkpoint found, using random weights")
    
    policy.eval()
    
    # Capture screenshot
    screen_capture = ScreenCapture(monitor_index=1, target_width=640, target_height=480)
    screenshot = screen_capture.capture()
    print(f"Screenshot captured: {screenshot.shape}")
    
    # Prepare state
    instruction = "Open the calculator application"
    state_dict = {
        'image': torch.tensor(screenshot, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
        'instruction': [instruction],
        'numeric': torch.zeros((1, 10), dtype=torch.float32, device=device)
    }
    
    # Get action
    with torch.no_grad():
        action = policy.get_action(state_dict, deterministic=True)
    
    print(f"\nInstruction: {instruction}")
    print(f"Predicted Action: {action['action_name']}")
    print(f"Coordinates (normalized): {action['coordinates']}")
    
    # Clean up
    screen_capture.close()
    print("\nInference complete!")


def example_task_execution():
    """Example: Execute a complete task"""
    print("\n" + "=" * 60)
    print("Example: Task Execution")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = HierarchicalPolicy().to(device)
    
    try:
        policy.load("checkpoints/final_model.pt", device)
        print("Model loaded")
    except FileNotFoundError:
        print("No checkpoint found, using random weights")
    
    policy.eval()
    
    # Setup
    screen_capture = ScreenCapture(monitor_index=1, target_width=640, target_height=480)
    instruction = "Open the calculator application"
    max_steps = 10
    
    print(f"\nTask: {instruction}")
    print(f"Max steps: {max_steps}\n")
    
    # Execute task
    for step in range(max_steps):
        # Capture screenshot
        screenshot = screen_capture.capture()
        
        # Prepare state
        state_dict = {
            'image': torch.tensor(screenshot, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
            'instruction': [instruction],
            'numeric': torch.tensor([[0, step, -1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)
        }
        
        # Get action
        with torch.no_grad():
            action_dict, success = policy.act_and_execute(state_dict, deterministic=True)
        
        print(f"Step {step + 1}: {action_dict['action_name']} at {action_dict['coordinates']} - {'Success' if success else 'Failed'}")
        
        # Check for early stop
        if action_dict['action_type'] == 6:  # EARLY_STOP
            print("Task completed (early stop)")
            break
        
        time.sleep(0.5)  # Small delay
    
    screen_capture.close()
    print("\nTask execution complete!")


def example_api_usage():
    """Example: Using the API"""
    print("\n" + "=" * 60)
    print("Example: API Usage")
    print("=" * 60)
    
    import requests
    
    api_url = "http://localhost:8000"
    
    # Health check
    response = requests.get(f"{api_url}/health")
    print(f"Health check: {response.json()}")
    
    # Predict action
    predict_data = {
        "instruction": "Open the calculator application",
        "use_live_screen": True,
        "deterministic": False
    }
    
    response = requests.post(f"{api_url}/api/v1/agent/predict", json=predict_data)
    if response.status_code == 200:
        action = response.json()
        print(f"\nPredicted action: {action['action_type']}")
        print(f"Coordinates: {action['coordinates']}")
    else:
        print(f"Error: {response.status_code}")
    
    print("\nAPI usage example complete!")


if __name__ == "__main__":
    # Run examples
    print("Hierarchical RL Agent - Examples\n")
    
    try:
        example_inference()
    except Exception as e:
        print(f"Inference example failed: {e}")
    
    # Uncomment to run task execution
    # try:
    #     example_task_execution()
    # except Exception as e:
    #     print(f"Task execution example failed: {e}")
    
    # Uncomment to test API (make sure API is running)
    # try:
    #     example_api_usage()
    # except Exception as e:
    #     print(f"API example failed: {e}")
