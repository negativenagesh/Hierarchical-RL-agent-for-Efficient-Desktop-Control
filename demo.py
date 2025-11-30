"""Demo script to see the trained agent control the desktop"""

import torch
import time
import subprocess
import platform
from pathlib import Path

from src.agent.policy import HierarchicalPolicy
from src.environment.screenshot import ScreenCapture


def check_application_running(app_name: str) -> bool:
    """Check if an application is currently running"""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            result = subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to get name of (processes where name contains "{app_name}")'],
                capture_output=True, text=True, timeout=2
            )
            return app_name.lower() in result.stdout.lower()
        elif system == "Linux":
            result = subprocess.run(["pgrep", "-f", app_name], capture_output=True, timeout=2)
            return result.returncode == 0
        elif system == "Windows":
            result = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=2)
            return app_name.lower() in result.stdout.lower()
    except:
        pass
    return False


def get_task_validation_keywords(instruction: str) -> dict:
    """Extract validation keywords from task instruction"""
    validation = {
        'apps_to_check': [],
        'keywords': []
    }
    
    instruction_lower = instruction.lower()
    
    # Application mappings
    app_mappings = {
        'calculator': ['calculator', 'calc'],
        'notepad': ['notepad', 'textedit', 'gedit'],
        'file manager': ['finder', 'nautilus', 'explorer'],
        'browser': ['chrome', 'firefox', 'safari', 'edge'],
        'terminal': ['terminal', 'cmd', 'powershell'],
    }
    
    for app_type, keywords in app_mappings.items():
        for keyword in keywords:
            if keyword in instruction_lower:
                validation['apps_to_check'].append(keyword)
                validation['keywords'].append(app_type)
                break
    
    return validation


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
    
    # Track execution statistics
    execution_log = []
    total_reward = 0.0
    successful_actions = 0
    failed_actions = 0
    action_type_counts = {}
    start_time = time.time()
    task_completed = False
    
    # Get validation info for task
    validation_info = get_task_validation_keywords(instruction)
    
    # Check initial state of applications
    initial_apps_running = {}
    if validation_info['apps_to_check']:
        print("\n🔍 Checking initial application states...")
        for app in validation_info['apps_to_check']:
            is_running = check_application_running(app)
            initial_apps_running[app] = is_running
            status = "✅ Running" if is_running else "❌ Not running"
            print(f"  • {app}: {status}")
    
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
        action_start_time = time.time()
        success = policy.execute_action(action, text="test" if action['action_name'] == 'TYPE' else None)
        action_duration = time.time() - action_start_time
        
        # Update statistics
        action_name = action['action_name']
        action_type_counts[action_name] = action_type_counts.get(action_name, 0) + 1
        
        if success:
            print("✅ Action executed successfully")
            successful_actions += 1
            step_reward = 0.1  # Positive reward for successful action
        else:
            print("❌ Action execution failed")
            failed_actions += 1
            step_reward = -0.5  # Penalty for failed action
        
        total_reward += step_reward
        
        # Log this step
        execution_log.append({
            'step': step + 1,
            'action': action_name,
            'coordinates': action['coordinates'],
            'screen_coords': screen_coords,
            'success': success,
            'duration': action_duration,
            'reward': step_reward
        })
    # Cleanup
    screen_capture.close()
    
    # Check final state of applications and validate task completion
    final_apps_running = {}
    apps_successfully_opened = []
    
    if validation_info['apps_to_check']:
        print("\n\n🔍 Validating task completion...")
        print("─" * 80)
        for app in validation_info['apps_to_check']:
            was_running = initial_apps_running.get(app, False)
            is_running = check_application_running(app)
            final_apps_running[app] = is_running
            
            if not was_running and is_running:
                status = "✅ Successfully opened"
                apps_successfully_opened.append(app)
                task_completed = True  # Override if we detected app opened
            elif was_running and is_running:
                status = "ℹ️  Was already running"
            elif not was_running and not is_running:
                status = "❌ Not opened"
            else:
                status = "⚠️  Was closed"
            
            print(f"  • {app}: {status}")
    print(f"\n🎯 Task: {instruction}")
    
    # Determine actual task completion
    if apps_successfully_opened:
        print(f"✅ TASK COMPLETED SUCCESSFULLY!")
        print(f"   Applications opened: {', '.join(apps_successfully_opened)}")
        task_completed = True
        total_reward += 10.0  # Add completion bonus
    elif task_completed:
        print(f"✅ AGENT SIGNALED COMPLETION (EARLY_STOP)")
    else:
        print(f"❌ TASK NOT COMPLETED")
    
    print(f"\n📈 Performance Metrics:")me
    total_steps = len(execution_log)
    success_rate = (successful_actions / total_steps * 100) if total_steps > 0 else 0
    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    
    # Print final resultsps - 1:
            print(f"⏳ Waiting {delay_between_actions}s before next action...")
            time.sleep(delay_between_actions)
    
    # Cleanup
    screen_capture.close()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    total_steps = len(execution_log)
    success_rate = (successful_actions / total_steps * 100) if total_steps > 0 else 0
    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    
    # Print final results
    print("\n" + "=" * 80)
    print("📊 EXECUTION RESULTS")
    print("=" * 80)
    
    print(f"\n🎯 Task: {instruction}")
    print(f"{'✅ COMPLETED' if task_completed else '❌ NOT COMPLETED'}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"  • Total Steps: {total_steps}/{max_steps}")
    print(f"  • Successful Actions: {successful_actions}")
    print(f"  • Failed Actions: {failed_actions}")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Total Reward: {total_reward:.2f}")
    print(f"  • Average Reward: {avg_reward:.2f}")
    print(f"  • Total Time: {total_time:.2f}s")
    print(f"  • Avg Time per Action: {(total_time/total_steps):.2f}s" if total_steps > 0 else "  • Avg Time per Action: N/A")
    
    print(f"\n🎬 Action Distribution:")
    if action_type_counts:
        max_count = max(action_type_counts.values())
        for action_type, count in sorted(action_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_steps * 100) if total_steps > 0 else 0
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            print(f"  {action_type:15s} {count:3d} ({percentage:5.1f}%) {bar}")
    else:
        print("  No actions executed")
    
    print(f"\n📝 Step-by-Step Summary:")
    print(f"{'Step':<6} {'Action':<15} {'Screen Coords':<20} {'Status':<10} {'Reward':<8} {'Time':<8}")
    print("─" * 80)
    for log in execution_log:
        status = "✅ Success" if log['success'] else "❌ Failed"
        coords = f"({log['screen_coords'][0]:.0f}, {log['screen_coords'][1]:.0f})"
        print(f"{log['step']:<6} {log['action']:<15} {coords:<20} {status:<10} {log['reward']:>7.2f} {log['duration']:>7.2f}s")
    
    print("\n" + "=" * 80)
    print("💡 Agent Behavior Analysis:")
    print("=" * 80)
    
    # Analyze behavior patterns
    if execution_log:
        # Check for repetitive behavior
        actions_sequence = [log['action'] for log in execution_log]
        if len(actions_sequence) >= 3:
            repetitive = any(
                actions_sequence[i] == actions_sequence[i+1] == actions_sequence[i+2]
                for i in range(len(actions_sequence) - 2)
            )
            if repetitive:
                print("⚠️  Warning: Detected repetitive action patterns - agent may be stuck")
            else:
                print("✅ Good: Agent showed diverse action selection")
        
        # Check for spatial clustering
        click_actions = [log for log in execution_log if 'CLICK' in log['action']]
        if len(click_actions) >= 2:
            coords_variance = sum(
    print("\n" + "=" * 80)
    print(f"{'🎉 DEMO COMPLETE!' if task_completed else '⚠️  DEMO FINISHED (Task not completed)'}")
    print("=" * 80)
    
    # Final verdict
    if apps_successfully_opened:
        print("\n" + "🎊" * 40)
        print("✅ SUCCESS! The agent completed the task!")
        print(f"✅ Verified: {', '.join(apps_successfully_opened)} {'is' if len(apps_successfully_opened) == 1 else 'are'} now running")
        print("🎊" * 40)
        print("\n💡 The agent demonstrated:")
        print(f"  • Successful interaction with the desktop environment")
        print(f"  • Effective action execution ({success_rate:.1f}% success rate)")
        print(f"  • {total_steps} coordinated steps to achieve the goal")
        print("\nConsider trying more complex tasks to further test capabilities!")
    elif task_completed:
        print("\n⚠️  Agent signaled completion but couldn't verify application state")
        print("   This might be due to:")
        print("   • Application detection limitations")
        print("   • Task not involving application launching")
        print("   • Application launched but not detected")
    else:
        print("\n💡 Suggestions for improvement:")
        print("  • Increase --max-steps to allow more time for task completion")
        print("  • Try --stochastic mode for more exploration")
        print("  • Adjust --delay for better UI responsiveness")
        print("  • Continue training the model for better performance")
        print("  • Ensure the target application is installed and accessible")
            print(f"✅ Excellent: Task completed in {completion_percentage:.0f}% of max steps")
        elif 'EARLY_STOP' in action_type_counts and not task_completed:
            print("⚠️  Warning: Premature early stopping - agent may need more training")
        
        # Check success rate
        if success_rate > 80:
            print("✅ Excellent: High action success rate indicates good execution")
        elif success_rate > 50:
            print("⚠️  Moderate: Some actions failed - environment may be challenging")
        else:
            print("❌ Poor: Many actions failed - agent needs more training or environment issues")
        
        # Check reward trend
        if total_steps >= 3:
            early_reward = sum(log['reward'] for log in execution_log[:total_steps//2]) / (total_steps//2)
            late_reward = sum(log['reward'] for log in execution_log[total_steps//2:]) / (total_steps - total_steps//2)
            if late_reward > early_reward:
                print("📈 Positive: Reward improved over time - agent making progress")
            elif late_reward < early_reward:
                print("📉 Concerning: Reward decreased over time - agent may be degrading")
            else:
                print("➡️  Stable: Consistent reward throughout execution")
    
    print("\n" + "=" * 80)
    print(f"{'🎉 DEMO COMPLETE!' if task_completed else '⚠️  DEMO FINISHED (Task not completed)'}")
    print("=" * 80)
    
    if not task_completed:
        print("\n💡 Suggestions for improvement:")
        print("  • Increase --max-steps to allow more time for task completion")
        print("  • Try --stochastic mode for more exploration")
        print("  • Adjust --delay for better UI responsiveness")
        print("  • Continue training the model for better performance")
    else:
        print("\n🎊 The agent successfully completed the task!")
        print("   Consider trying more complex instructions to test capabilities.")


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
