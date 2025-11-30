"""Demo script to see the trained agent control the desktop"""

import torch
import time
from pathlib import Path

from src.agent.policy import HierarchicalPolicy
from src.environment.screenshot import ScreenCapture


def run_demo(
    checkpoint_path: str = "checkpoints/final_model.pt",
    instruction: str = "Open the calculator application",
    max_steps: int = 10,
    deterministic: bool = True,
    delay_between_actions: float = 1.0
):
    """
    Run a live demo of the trained agent
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        instruction: Task instruction to execute
        max_steps: Maximum number of steps to take
        deterministic: Use deterministic policy (True) or stochastic (False)
        delay_between_actions: Delay in seconds between actions
    """
    
    print("=" * 80)
    print("HIERARCHICAL RL AGENT - LIVE DEMO")
    print("=" * 80)
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure training completed and the checkpoint was saved.")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load model
    print(f"📦 Loading model from {checkpoint_path}...")
    policy = HierarchicalPolicy(
        screen_width=1920,
        screen_height=1080
    ).to(device)
    
    policy.load(checkpoint_path, device)
    policy.eval()
    print("✅ Model loaded successfully!")
    
    # Setup screen capture
    print("📸 Initializing screen capture...")
    screen_capture = ScreenCapture(monitor_index=1, target_width=640, target_height=480)
    print("✅ Screen capture ready!")
    
    # Display task
    print("\n" + "=" * 80)
    print(f"🎯 TASK: {instruction}")
    print(f"🔧 Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"📊 Max steps: {max_steps}")
    print("=" * 80)
    
    print("\n⚠️  WARNING: The agent will now control your mouse and keyboard!")
    print("⚠️  Move your mouse to the top-left corner to trigger failsafe if needed.")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Execute task
    for step in range(max_steps):
        print(f"\n{'─' * 80}")
        print(f"STEP {step + 1}/{max_steps}")
        print(f"{'─' * 80}")
        
        # Capture current screen state
        screenshot = screen_capture.capture()
        
        # Prepare state dictionary
        state_dict = {
            'image': torch.tensor(screenshot, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
            'instruction': [instruction],
            'numeric': torch.zeros((1, 10), dtype=torch.float32, device=device)
        }
        
        # Get action from policy
        with torch.no_grad():
            action = policy.get_action(state_dict, deterministic=deterministic)
        
        # Display action
        print(f"🎬 Action: {action['action_name']}")
        print(f"📍 Coordinates (normalized): {action['coordinates']}")
        
        # Convert to screen coordinates
        screen_coords = policy.worker.denormalize_coordinates(action['coordinates'])
        print(f"📍 Screen coordinates: ({screen_coords[0]:.0f}, {screen_coords[1]:.0f})")
        
        # Execute action
        print("⚡ Executing action...")
        success = policy.execute_action(action, text="test" if action['action_name'] == 'TYPE' else None)
        
        if success:
            print("✅ Action executed successfully")
        else:
            print("❌ Action execution failed")
        
        # Check if task is complete
        if action['action_name'] == 'EARLY_STOP':
            print("\n🏁 Agent signaled task completion (EARLY_STOP)")
            break
        
        # Delay before next action
        if step < max_steps - 1:
            print(f"⏳ Waiting {delay_between_actions}s before next action...")
            time.sleep(delay_between_actions)
    
    # Cleanup
    screen_capture.close()
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained agent demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Open the calculator application",
        help="Task instruction"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps to execute"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between actions in seconds"
    )
    
    args = parser.parse_args()
    
    run_demo(
        checkpoint_path=args.checkpoint,
        instruction=args.instruction,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        delay_between_actions=args.delay
    )


if __name__ == "__main__":
    main()
