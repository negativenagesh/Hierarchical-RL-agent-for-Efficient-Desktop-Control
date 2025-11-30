# Demo Guide: Running Your Trained Agent

Your model has been successfully trained and saved to `checkpoints/final_model.pt`. Here's how to see it in action!

## Quick Start - Watch the Agent Work

### Option 1: Live Demo (Recommended)

The easiest way to see your agent control the desktop:

```bash
uv run python demo.py
```

This will:
1. Load your trained model
2. Capture your current screen
3. Execute actions to complete the task
4. Show each step with detailed output

**Example output:**
```
================================================================================
HIERARCHICAL RL AGENT - LIVE DEMO
================================================================================

📍 Device: cuda
📦 Loading model from checkpoints/final_model.pt...
✅ Model loaded successfully!
📸 Initializing screen capture...
✅ Screen capture ready!

================================================================================
🎯 TASK: Open the calculator application
🔧 Mode: Deterministic
📊 Max steps: 10
================================================================================

⚠️  WARNING: The agent will now control your mouse and keyboard!

────────────────────────────────────────────────────────────────────────────────
STEP 1/10
────────────────────────────────────────────────────────────────────────────────
🎬 Action: CLICK
📍 Coordinates (normalized): (-0.85, 0.92)
📍 Screen coordinates: (144, 1037)
⚡ Executing action...
✅ Action executed successfully
```

### Option 2: Custom Tasks

Try different tasks:

```bash
# Open calculator (default)
uv run python demo.py

# Open file manager
uv run python demo.py --instruction "Open file manager and navigate to Documents"

# Create a file
uv run python demo.py --instruction "Create a new text file named test.txt"

# Longer execution with more steps
uv run python demo.py --max-steps 20 --delay 0.5
```

### Option 3: Use the API

Start the API server:

```bash
# Start server
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or use the script
bash scripts/run_api.sh
```

Then test it:

```bash
# Check if model is loaded
curl http://localhost:8000/api/v1/agent/model-info

# Predict a single action
curl -X POST http://localhost:8000/api/v1/agent/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Open calculator",
    "use_live_screen": true,
    "deterministic": true
  }'

# Execute a complete task
curl -X POST http://localhost:8000/api/v1/agent/complete-task \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Open the calculator application",
    "max_steps": 10,
    "deterministic": true
  }'
```

### Option 4: Python Integration

Use the model in your own Python scripts:

```python
import torch
from src.agent.policy import HierarchicalPolicy
from src.environment.screenshot import ScreenCapture

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = HierarchicalPolicy().to(device)
policy.load("checkpoints/final_model.pt", device)
policy.eval()

# Setup screen capture
screen_capture = ScreenCapture(monitor_index=1, target_width=640, target_height=480)

# Execute task loop
instruction = "Open the calculator application"
for step in range(10):
    # Capture screen
    screenshot = screen_capture.capture()
    
    # Prepare state
    state_dict = {
        'image': torch.tensor(screenshot, dtype=torch.float32, device=device)
                 .unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
        'instruction': [instruction],
        'numeric': torch.zeros((1, 10), dtype=torch.float32, device=device)
    }
    
    # Get action
    with torch.no_grad():
        action = policy.get_action(state_dict, deterministic=True)
    
    print(f"Step {step}: {action['action_name']} at {action['coordinates']}")
    
    # Execute
    success = policy.execute_action(action)
    
    # Stop if task complete
    if action['action_name'] == 'EARLY_STOP':
        break
    
    time.sleep(1)  # Delay between actions

screen_capture.close()
```

## Demo Parameters

### Command Line Arguments

```bash
uv run python demo.py [OPTIONS]
```

**Options:**

- `--checkpoint PATH`: Path to model checkpoint (default: `checkpoints/final_model.pt`)
- `--instruction TEXT`: Task instruction (default: `"Open the calculator application"`)
- `--max-steps N`: Maximum steps to execute (default: `10`)
- `--stochastic`: Use stochastic policy instead of deterministic
- `--delay SECONDS`: Delay between actions (default: `1.0`)

**Examples:**

```bash
# Fast execution
uv run python demo.py --delay 0.3 --max-steps 20

# Exploratory behavior
uv run python demo.py --stochastic

# Different checkpoint
uv run python demo.py --checkpoint checkpoints/checkpoint_50000.pt

# Complex task
uv run python demo.py \
  --instruction "Open file manager, create new folder, name it 'test'" \
  --max-steps 30 \
  --delay 1.5
```

## Understanding the Output

### Action Types

The agent can perform 7 types of actions:

1. **CLICK**: Single left-click at coordinates
2. **DOUBLE_CLICK**: Double-click for opening/selecting
3. **RIGHT_CLICK**: Open context menu
4. **TYPE**: Type text via keyboard
5. **SCROLL**: Scroll mouse wheel
6. **WAIT**: Pause for UI to load
7. **EARLY_STOP**: Signal task completion

### Coordinate System

- **Normalized coordinates**: Range [-1, 1] (what the model outputs)
  - (-1, -1) = top-left corner
  - (1, 1) = bottom-right corner
  - (0, 0) = center of screen

- **Screen coordinates**: Absolute pixel positions (what gets executed)
  - Depends on your screen resolution
  - Example: (960, 540) on 1920x1080 screen

## Safety Features

### Failsafe

PyAutoGUI's failsafe is enabled:
- Move mouse to **top-left corner** to abort
- Emergency stop mechanism
- Prevents unintended actions

### Controlled Execution

- **Deterministic mode**: Same input → same output (repeatable)
- **Stochastic mode**: Adds exploration (variable behavior)
- **Step delays**: Prevents action flooding
- **Max steps**: Prevents infinite loops

## Troubleshooting

### Model Not Found

```
❌ Error: Checkpoint not found at checkpoints/final_model.pt
```

**Solution:**
- Check if training completed successfully
- Verify the checkpoint path
- Look for other checkpoints: `ls checkpoints/`

### Display/DISPLAY Errors

```
KeyError: 'DISPLAY'
```

**Solution (headless environments):**
```bash
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
```

### CUDA Errors

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
uv run python demo.py  # Will auto-detect and use CPU if CUDA fails
```

### Action Execution Failures

```
❌ Action execution failed
```

**Possible causes:**
- UI element not found
- Application not responding
- Coordinates out of bounds
- Permission issues

**Solution:**
- Increase delay between actions: `--delay 2.0`
- Reduce max steps to debug: `--max-steps 5`
- Check if target application is available

## Next Steps

### Evaluate Performance

Run evaluation to get metrics:

```python
from src.training.ppo_trainer import PPOTrainer

trainer = PPOTrainer(env, policy, device)
trainer.load_checkpoint("checkpoints/final_model.pt")
eval_stats = trainer.evaluate(num_episodes=10)

print(f"Success Rate: {eval_stats['success_rate']:.2%}")
print(f"Avg Reward: {eval_stats['avg_reward']:.2f}")
print(f"Avg Length: {eval_stats['avg_length']:.1f}")
```

### Fine-tune on Specific Tasks

Continue training on specific tasks:

```bash
uv run python src/training/train.py \
  --total-timesteps 10000 \
  --difficulty EASY \
  --device cuda
```

The model will load from the last checkpoint and continue training.

### Deploy as Service

Use the API for production deployment:

```bash
# Start API
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yml up -d
```

### Export for Different Environments

Save model for different devices:

```python
# Export for CPU deployment
policy.load("checkpoints/final_model.pt", torch.device('cuda'))
policy.save("checkpoints/model_cpu.pt")

# Load on CPU-only machine
cpu_policy = HierarchicalPolicy()
cpu_policy.load("checkpoints/model_cpu.pt", torch.device('cpu'))
```

## Tips for Better Results

### Task Design

- **Start simple**: "Open calculator" before "Open calculator and compute 15+27"
- **Be specific**: "Click the Start button" vs "Do something"
- **Sequential**: Break complex tasks into steps

### Model Configuration

- **Deterministic mode**: Use for consistent, repeatable behavior
- **Stochastic mode**: Use for exploration and training data collection
- **Temperature**: Adjust in code for exploration vs exploitation balance

### Execution Tuning

- **Longer delays**: More reliable but slower (`--delay 2.0`)
- **More steps**: Handle complex tasks (`--max-steps 30`)
- **Screen resolution**: Match training resolution for best results

## Examples Library

### Basic Tasks

```bash
# Calculator
uv run python demo.py --instruction "Open the calculator application"

# File manager
uv run python demo.py --instruction "Open file manager"

# Text editor
uv run python demo.py --instruction "Open text editor"
```

### Medium Tasks

```bash
# Navigate folders
uv run python demo.py \
  --instruction "Open file manager and navigate to Documents" \
  --max-steps 15

# Create file
uv run python demo.py \
  --instruction "Create a new text file named test.txt" \
  --max-steps 20
```

### Advanced Tasks

```bash
# Multi-step workflow
uv run python demo.py \
  --instruction "Open calculator, compute 15+27, save result" \
  --max-steps 30 \
  --delay 1.5
```

## Getting Help

- **Documentation**: See `README.md` and `docs/` folder
- **API Reference**: See `docs/API.md`
- **Training Guide**: See `docs/TRAINING.md`
- **Examples**: See `examples.py`

Enjoy watching your trained agent in action! 🚀
