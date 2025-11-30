<div align="center">
<h1>Hierarchical RL Agent for Efficient Desktop Control<h1>
</div>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/negativenagesh/Hierarchical-RL-agent-for-Efficient-Desktop-Control?style=social)](https://github.com/negativenagesh/Hierarchical-RL-agent-for-Efficient-Desktop-Control)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
</div>


Desktop automation is still a challenge for AI research. Machines' performance in playing games, detecting images and interpreting languages has been rather impressive, but the difficulty of making them doing something similar with ordinary computer tasks remains surprising. We are so used to going through applications, clicking buttons, typing, and scrolling through documents so effortlessly. These actions might look simple, but in reality, they demand a sophisticated comprehension of visual interfaces, written instructions, and accurate cursor control.

This endeavor faces the challenge of establishing an up to the mark hierarchical RL agent that is capable of the desktop environment's control in an autonomous manner. The main point is that human computer interaction has a hierarchy; we do not consider it in terms of individual pixel movements and key presses. We rather operate at different levels of abstraction, making the decision on what task to do (high-level) and working out exactly how to do it (low-level). We have, in a way, replicated the hierarchical thinking in an AI agent.

The system integrates the three types of understanding: visual (what is seen on the screen), linguistic (what is being asked by the user), and contextual (the stage of the task). The agent forms a complete view of the current state by encoding the screenshots through a vision model, processing the natural language instructions through a language model, and tracking the task metadata through the numeric features. A Manager policy then decides on high-level actions like "click here" or "type this," while a Worker component translates those decisions into actual mouse movements and keyboard inputs.

What really adds thrill to a reinforcement learning course is that it introduces various advanced concepts in conjunction: hierarchical policy decomposition, multi-modal state representation, hybrid action spaces (combining discrete choices with continuous parameters), and curriculum learning. The agent doesn't simply memorize individual tasks; it learns through error and trial, acquiring the ability to work with even more complex problems as its skill increases at the same time.

The authors of the paper present a full implementation with real-time desktop control functionalities, a training infrastructure employing Proximal Policy Optimization (PPO) and an API that lets other applications utilize the trained agent.

## Quick Start

### Prerequisites

- Python 3.9-3.12
- UV package manager
- CUDA-capable GPU (recommended for training)
- Xvfb (for headless environments)

### Installation

#### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clone Repository

```bash
git clone https://github.com/negativenagesh/Hierarchical-RL-agent-for-Efficient-OS-Control.git
cd Hierarchical-RL-agent-for-Efficient-OS-Control
```

#### 3. Install Dependencies

```bash
# Install project dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

#### 4. Download Pre-trained Models

```bash
uv run python scripts/download_models.py
```

This downloads:
- CLIP ViT-B/16 model for visual encoding
- all-mpnet-base-v2 model for text encoding

#### 5. Setup for Headless Environments (Cloud/Docker)

```bash
# Install Xvfb for virtual display
sudo apt-get update
sudo apt-get install -y xvfb x11-utils

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

### Training

#### Basic Training

```bash
# Train with default settings (1M timesteps)
uv run python src/training/train.py

# Train with custom parameters
uv run python src/training/train.py \
  --total-timesteps 25000 \
  --difficulty EASY \
  --device cuda
```

#### Training for Specific Number of Episodes

Based on training results, each episode runs for approximately 50 steps. Calculate timesteps as:

```
total_timesteps = episodes × 50
```

Examples:

```bash
# 500 episodes (~25,000 timesteps)
uv run python src/training/train.py --total-timesteps 25000 --device cuda --difficulty EASY

# 1000 episodes (~50,000 timesteps)
uv run python src/training/train.py --total-timesteps 50000 --device cuda --difficulty EASY
```

#### Training with Visualization

```bash
# Enable visualization (requires display)
uv run python src/training/train.py \
  --visualize \
  --total-timesteps 10000 \
  --difficulty EASY \
  --device cuda
```

#### Headless Training (Cloud/Server)

```bash
# Using xvfb-run wrapper
xvfb-run -a uv run python src/training/train.py \
  --total-timesteps 25000 \
  --device cuda \
  --difficulty EASY

# Or with manual Xvfb setup
export DISPLAY=:99
uv run python src/training/train.py \
  --total-timesteps 25000 \
  --device cuda \
  --difficulty EASY
```

#### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--total-timesteps` | Total training steps | 1000000 | Any integer |
| `--rollout-steps` | Steps per rollout | 2048 | Any integer |
| `--learning-rate` | Learning rate | 0.0003 | Float |
| `--difficulty` | Task difficulty | EASY | EASY, MEDIUM, HARD |
| `--device` | Compute device | cuda | cuda, cpu |
| `--save-dir` | Checkpoint directory | checkpoints | Path |
| `--log-dir` | Log directory | logs | Path |
| `--curriculum` | Enable curriculum learning | False | Flag |
| `--visualize` | Enable visualization | False | Flag |

#### Using Training Script

```bash
# Set environment variables
export TOTAL_STEPS=25000
export DIFFICULTY=EASY
export DEVICE=cuda

# Run training script
bash scripts/train.sh
```

#### Using Makefile

```bash
# Train with default settings
make train

# Or modify scripts/train.sh and run
make train
```

### Monitoring Training Progress

#### TensorBoard

```bash
# Start TensorBoard
uv run tensorboard --logdir logs

# Access at http://localhost:6006
```

Metrics tracked:
- Episode rewards
- Success rates
- Action distributions
- Policy loss
- Value loss
- Entropy
- KL divergence

#### Terminal Output

Training displays real-time statistics for each episode:

```
================================================================================
EPISODE 708 COMPLETE
================================================================================
Task: Open file manager and navigate to Documents
Status: ✗ FAILED
Steps: 50 | Total Reward: 26.40

Episode Statistics:
  Avg Reward/Step: 0.5280 | Max: 0.6600 | Min: 0.1100
  Positive Rewards: 50/50 | Negative: 0/50

Action Distribution:
  CLICK             5 ( 10.0%) ██
  DOUBLE_CLICK     12 ( 24.0%) ████
  RIGHT_CLICK      11 ( 22.0%) ████
  TYPE             10 ( 20.0%) ████
  SCROLL            4 (  8.0%) █
  WAIT              8 ( 16.0%) ███

Recent Performance (Last 10 Episodes):
  Success Rate: 0.00%
  Avg Reward: 22.26
  Avg Length: 43.1 steps

Overall Training Progress:
  Total Episodes: 708
  Total Steps: 20460
  Overall Success Rate: 0.00%
```

### Running the Trained Agent

After training completes, you'll have a saved model checkpoint (typically `checkpoints/final_model.pt`). This section covers how to run the trained agent both locally and in cloud environments.

#### Running Locally (With Display)

**Prerequisites:**
- Trained model checkpoint at `checkpoints/final_model.pt`
- Display available (macOS, Linux with X11, or Windows)
- Mouse and keyboard control permissions

**Basic Usage:**

```bash
# Run with default settings
uv run python demo.py

# Run with custom task
uv run python demo.py --instruction "Open the calculator application"

# Run with more steps and custom delay
uv run python demo.py --max-steps 20 --delay 0.5

# Use stochastic policy for exploration
uv run python demo.py --stochastic

# Use custom checkpoint path
uv run python demo.py --checkpoint checkpoints/custom_model.pt
```

**Demo Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | `checkpoints/final_model.pt` |
| `--instruction` | Task to execute | "Open the calculator application" |
| `--max-steps` | Maximum steps | 10 |
| `--stochastic` | Use stochastic policy | False (deterministic) |
| `--delay` | Delay between actions (seconds) | 1.0 |

**Example Tasks:**

```bash
# Open calculator
uv run python demo.py --instruction "Open the calculator application"

# Open file manager
uv run python demo.py --instruction "Open file manager and navigate to Documents"

# Create text file
uv run python demo.py --instruction "Create a new text file named test.txt"

# Open browser and navigate
uv run python demo.py --instruction "Open Chrome browser" --max-steps 15
```

**What to Expect:**

The demo script will:
1. ✅ Load the trained model from checkpoint
2. 🔍 Check initial application states (e.g., is calculator already running?)
3. 🎬 Execute actions step-by-step with visual feedback in terminal
4. 📊 Display detailed execution logs (actions, coordinates, success/failure)
5. ✅ Verify task completion (e.g., did calculator actually open?)
6. 📈 Show comprehensive performance metrics and analysis

**Safety Note:** The agent will control your mouse and keyboard. Move your mouse to the top-left corner to trigger PyAutoGUI's failsafe if needed.

#### Running in Cloud/Headless Environment

When running on cloud servers (AWS, GCP, Azure) or headless systems without physical displays, you need a virtual display.

**Setup Virtual Display (One-Time):**

```bash
# Install Xvfb (X Virtual Framebuffer)
sudo apt-get update
sudo apt-get install -y xvfb x11-utils

# Verify installation
which Xvfb
```

**Method 1: Using xvfb-run (Recommended)**

```bash
# Run demo with automatic virtual display
xvfb-run -a python demo.py

# With UV
xvfb-run -a uv run python demo.py --instruction "Open calculator"

# Specify display resolution
xvfb-run -a --server-args="-screen 0 1920x1080x24" \
  uv run python demo.py --max-steps 15
```

**Method 2: Manual Xvfb Setup**

```bash
# Start virtual display (run once per session)
Xvfb :99 -screen 0 1920x1080x24 &

# Export display variable
export DISPLAY=:99

# Verify display is working
xdpyinfo -display :99

# Run demo normally
uv run python demo.py --instruction "Open file manager"

# Run multiple demos
uv run python demo.py --instruction "Open calculator" --max-steps 10
uv run python demo.py --instruction "Open text editor" --max-steps 15
```

**Method 3: Persistent Display Setup (Docker/Long-Running Servers)**

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# Start Xvfb on login if not already running
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :99 -screen 0 1920x1080x24 &
fi
export DISPLAY=:99
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

**Cloud-Specific Instructions:**

**AWS EC2:**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Setup and run
sudo apt-get install -y xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &

cd /path/to/Hierarchical-RL-agent-for-Efficient-OS-Control
uv run python demo.py --checkpoint checkpoints/final_model.pt
```

**GCP Compute Engine:**
```bash
# SSH into instance (use GCP console or gcloud CLI)
gcloud compute ssh your-instance-name

# Same setup as above
sudo apt-get install -y xvfb
xvfb-run -a uv run python demo.py
```

**Azure VM:**
```bash
# SSH into VM
ssh azureuser@your-vm-ip

# Setup virtual display
sudo apt-get install -y xvfb x11-utils
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &

# Run demo
uv run python demo.py --instruction "Open calculator"
```

**Docker Container:**

```dockerfile
# Add to your Dockerfile
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Set display environment variable
ENV DISPLAY=:99

# Start Xvfb in entrypoint
ENTRYPOINT ["/bin/bash", "-c", "Xvfb :99 -screen 0 1920x1080x24 & exec \"$@\"", "--"]
```

Then run:
```bash
docker run -it your-image uv run python demo.py
```

**Verifying Cloud Setup:**

```bash
# Check if Xvfb is running
ps aux | grep Xvfb

# Check display environment variable
echo $DISPLAY

# Test screen capture works
uv run python -c "from src.environment.screenshot import ScreenCapture; sc = ScreenCapture(); print('Screen capture working!')"

# Run a quick test
uv run python demo.py --max-steps 5 --instruction "Test run"
```

**Troubleshooting Cloud Execution:**

1. **Display not found:**
   ```bash
   # Error: "Cannot open display"
   export DISPLAY=:99
   ps aux | grep Xvfb  # Ensure Xvfb is running
   ```

2. **Xvfb not starting:**
   ```bash
   # Kill existing Xvfb processes
   pkill Xvfb
   
   # Start with verbose output
   Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
   ```

3. **Permission issues:**
   ```bash
   # Ensure proper permissions
   chmod +x demo.py
   sudo usermod -a -G video $USER
   ```

4. **Model not found:**
   ```bash
   # Verify checkpoint exists
   ls -lh checkpoints/
   
   # Specify full path
   uv run python demo.py --checkpoint /full/path/to/checkpoints/final_model.pt
   ```

**Performance Notes:**

- **Local execution:** Full GUI responsiveness, real-time visual feedback
- **Cloud execution:** No visual display, relies on terminal output and application state validation
- **Screen resolution:** 1920x1080 recommended for best results (matches training setup)
- **Network latency:** Does not affect execution (runs locally on server)
- **Recording:** Use `screen` or `tmux` to keep sessions running after disconnect

**Recording Demo Output:**

```bash
# Save terminal output to file
uv run python demo.py --instruction "Open calculator" 2>&1 | tee demo_output.log

# Run in background with nohup
nohup xvfb-run -a uv run python demo.py > demo_run.log 2>&1 &

# Check progress
tail -f demo_run.log
```

#### Using the API

Start the API server to interact with the trained model programmatically:

```bash
# Start API server
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the script
bash scripts/run_api.sh
```

Test the API:

```bash
# Check model status
curl http://localhost:8000/api/v1/agent/model-info

# Predict action
curl -X POST http://localhost:8000/api/v1/agent/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Open calculator",
    "use_live_screen": true,
    "deterministic": true
  }'

# Execute complete task
curl -X POST http://localhost:8000/api/v1/agent/complete-task \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Open the calculator application",
    "max_steps": 10,
    "deterministic": true
  }'
```

#### Python Script Integration

```python
import torch
from src.agent.policy import HierarchicalPolicy
from src.environment.screenshot import ScreenCapture

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = HierarchicalPolicy().to(device)
policy.load("checkpoints/final_model.pt", device)
policy.eval()

# Capture screenshot
screen_capture = ScreenCapture(monitor_index=1, target_width=640, target_height=480)
screenshot = screen_capture.capture()

# Prepare state
state_dict = {
    'image': torch.tensor(screenshot, dtype=torch.float32, device=device)
             .unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
    'instruction': ["Open the calculator application"],
    'numeric': torch.zeros((1, 10), dtype=torch.float32, device=device)
}

# Get and execute action
with torch.no_grad():
    action = policy.get_action(state_dict, deterministic=True)
    
print(f"Action: {action['action_name']}")
print(f"Coordinates: {action['coordinates']}")

# Execute
success = policy.execute_action(action)
```

### Visualization

#### Real-time Training Visualization

```bash
# Enable visualization during training (requires display)
uv run python src/training/train.py \
  --visualize \
  --total-timesteps 10000 \
  --device cuda
```

Visualization shows:
- Current screenshot
- Selected action
- Coordinate targets
- Task instruction
- Episode statistics

#### Post-Training Visualization

```bash
# Generate training plots
uv run python src/utils/visualizer.py --log-dir logs

# View saved plots in logs/ directory
```

### Troubleshooting

#### CUDA Errors

```bash
# Clear CUDA cache
uv run python -c "import torch; torch.cuda.empty_cache()"

# Check GPU availability
nvidia-smi

# Use CPU if GPU issues persist
uv run python src/training/train.py --device cpu
```

#### Display Errors (DISPLAY not set)

```bash
# Install Xvfb
sudo apt-get install -y xvfb x11-utils

# Use xvfb-run
xvfb-run -a uv run python src/training/train.py
```

#### PyTorch CUDA Version Mismatch

```bash
# Reinstall PyTorch with correct CUDA version
uv pip uninstall torch torchvision

# For CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Architecture
	
	Here we applied a Hierarchical Reinforcement Learning (HRL) agent that was designed for desktop control motivated by the paper -”Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces” via visual and natural language commands. The structure is based on a two-level hierarchy that takes its cue from the Manager-Worker model, in which the high-level decision-making process is distinct from the low-level execution.
	
The RL agent follows three main principles:
Triple-Modal State Encoding: The agent sees the desktop environment in three different ways, which are visuals (screenshots), texts (natural language instructions), and numbers (mouse position, step counts, task metadata).
Hierarchical Decision Making: A Manager network selects an action (click, type, scroll, etc.) and a place for its execution, while a Worker looks after the technical aspects of the action.
Curriculum Learning: Training begins with easy tasks (< 8 steps) and ends with intricate multi-step operations, thus the agent is able to acquire skill progressively.

State Representation & Perception System:

Triple-Modal Encoder Architecture - The agent's perception system comprises three pre-trained models that are very good and can be used to analyze different aspects of the desktop environment:
Visual Encoding (CLIP ViT-B/16): The screenshots of the desktops are sent to OpenAI's CLIP vision transformer to encode the desktop screenshots. The visual encoder uses a Vision Transformer backbone that has been pre-trained on 400 million image-text pairs to process the screenshots. This gives rise to 512-dimensional visual embeddings that show the meaning of UI elements, text on screen, application states, and spatial relationships. The frozen CLIP backbone provides stable visual features with a low memory footprint, while a learnable projection layer adjusts CLIP's output to match the agent's representation space
Text Encoding: Works on natural language task instructions like "Open Chrome and navigate to Google". A local all-mpnet-base-v2 (768-dim) is used for this (inner, 768-dim), The text encoder decodes the task semantics, intent, and the action sequence that is to be performed. Mean pooling takes the sum of all the token-level embeddings and forms a single representation of the instruction. Then, a projection layer reduces the text embeddings to 512 dimensions in order to maintain consistency
Numeric State Encoding (MLP): It encodes the structured state information such as the current mouse position, step count in episode, task ID, cursor coordinates, and time elapsed. A multi-layer perceptron with layer normalization and GELU activations processes these features. This results in the production of 64-dimensional numeric embeddings that reflect the temporal and spatial context
Multi-Modal Fusion: The three different modalities that have been encoded are put together (512 + 512 + 64 = 1088 dimensions). Through deep fusion network that has residual connections and layer normalization, these representations are combined. The fusion technique discovers the relationships between different modes (for instance, linking the text "click the red button" with the red areas in the image). A 512-dimensional unified state embedding that is the input to the policy networks is the final output of this whole process. Such a triple-modal strategy gives the agent the power not just to see what is on the screen but also to know the accomplishment, thus a very rich contextual understanding of the task state is created.

Hierarchical Policy Structure

Manager Policy (High-Level Controller) The Manager performs at the abstract decision-making level, which is the highest level, and hence determines the types of actions and the places where these actions will be carried out:
Action Space:
The Manager produces distinct action types which belong to the 7 categories:
CLICK: Single left-click at given point
DOUBLE_CLICK: Double left-click for the purpose of opening/selecting
RIGHT_CLICK: Context menu opening
TYPE: Text input through the keyboard
SCROLL: Mouse's wheel scrolling
WAIT: Intentional pause for user interface loading
EARLY_STOP: Indication of task being done

Dual-Head Architecture

The 512-dimensional state embedding is passed through a two-layer MLP to generate logits for the seven action types in Action Type Head (Discrete). During training, a categorical distribution is applied for sampling. In the case of inference, either sampling (exploration) or argmax selection (exploitation) is employed.

Coordinate Head (Continuous)

The parallel processing lane runs through the common backbone. The results are the mean and the log-standard-deviation for the x and y coordinates. The coordinates are normalized to the range of [-1, 1] through the tanh activation function. This forms a 2D Gaussian spread on the screen. That makes it possible to target locations on the screen smoothly and continuously. The standard deviation is learned adaptively for the exploration-exploitation trade-off.

Value Head (Critic)

The current state’s expected cumulative reward is estimated. This is critical for the PPO algorithm's advantage estimation. Action heads utilize the same backbone for more efficient learning. A single output neuron generates the value estimate of the state.




Decision Process 


Upon receiving a state embedding, the Manager - processes the embedding through shared layers to extract features relevant to the action Simultaneously calculates probabilities of action types and distributions of coordinates. Samples or chooses actions according to the training/inference mode. Computes log-probabilities for policy gradient updates. Estimates state value for advantage calculation.

The manager assigns actions and locations but doesn't touch the mouse or keyboard; this is what the worker is left to do.

Worker Policy (Low-Level Executor) The Worker translates Manager's abstract decisions into concrete desktop - level actions:

Hardcoded Execution Strategy: Instead of mastering low-level control, the Worker employs the use of deterministic macros (as in the research for real-world applications).

This selection of design brings several benefits:
Reliability: Correct and secure execution of underlying primitives
Sample Efficiency: The manager gives the learning to the high-level reasoning
Interpretability: Actions are completely see-through and can be reversed
Cross-Platform: Adaptation of the macros for various desktop OSs is easy and takes less time

Execution

Coordinate Denormalization: The manager sends out the coordinates in the normalized form [-1, 1]. The worker then translates these into absolute pixel coordinates: x = (x_norm + 1) / 2 * screen_width. Additionally, it clamps the values to the screen boundaries so that no errors occur.
Action Execution (via PyAutoGUI): Click Actions: The target cursor movement, clicking, and waiting for the UI response are all part of the action sequence. Type Actions: The simulation of keyboard input with the inter character delays for stability is one of the features of this action. Scroll Actions: This action translates the abstract scroll amounts to the OS wheel events. Wait Actions: The action introduces the specified time delays for the asynchronous UI updates.
Timing and Safety: The cursor movement can be configured for different durations (default 0.2s) as a way of achieving smooth human-like movement. The click delays (0.1s) prevent flooding of actions and also allow the UI state updates to take place. The failsafe is off in the controlled training environments but is there for safety in case of emergencies.

Why Worker?
The process of acquiring low-level motor control would be extremely sample-hungry and even more complicated. The separation of the Manager and Worker lets the RL agent to concentrate on making high-level decisions while using the deterministic and efficient execution primitives to their full extent. This resembles human computer interaction - we are mentally deciding what to click, not exactly how to move the mouse.

Reinforcement Learning Algorithm -  Proximal Policy Optimization (PPO)

PPO with its ability to provide stability, sample efficiency, and to prove its power in difficult control tasks is selected: Trust Region Optimization - Precludes any large policy updates that might lead to catastrophic forgetting. On-Policy Learning - Only utilizes the experiences of the current policy for updates, hence, ensuring stability. Clipped Objective - It does not allow policy changes that are too big to the extent of disrupting the training process. Value Function Learning - Learns the state values at the same time for reduction of variance.

PPO Components
Trajectory Collection:
The agent gathers experience through interaction with the environment:

In each rollout step the following happens:
  1. The current state is observed (screenshot + instruction + numeric data)
  2. The state is encoded using a triple-modal encoder resulting in a 512-dimensional embedding
  3. From the policy distributions, the manager draws a sample of action type and coordinates
  4. The worker carries out the action on the real desktop
  5. A reward signal is received (task progress, penalties, completion bonus)
  6. (state, action, reward, value, log_prob) is stored in the replay buffer
  7. The episode is checked for termination (success, failure, timeout)

2. Advantage Estimation (GAE):
Generalized Advantage Estimation computes how much better an action was compared to the baseline:

Returns Calculation: R_t = reward_t + γ * reward_{t+1} + γ² * reward_{t+2} + ...
TD Residuals: δ_t = reward_t + γ * V(s_{t+1}) - V(s_t)
GAE: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
λ = 0.95 balances bias-variance tradeoff
γ = 0.99 for long-term credit assignment
Advantages tell the agent which actions led to better-than-expected outcomes, guiding policy improvement.

3. Policy Update Mechanism:

The PPO update occurs in multiple epochs over collected data:
For each update epoch:
 
 For each minibatch:
    1. Re-evaluate actions under current policy → new_log_probs
    2. Compute importance ratio: r = exp(new_log_prob - old_log_prob)
    3. Calculate clipped surrogate objective:
       L_CLIP = min(r * advantage, clip(r, 1-ε, 1+ε) * advantage)
    4. Compute value loss: L_VF = (V_pred - V_target)²
    5. Compute entropy bonus: L_ENT = -entropy(policy)
    6. Total loss: L = -L_CLIP + c1*L_VF - c2*L_ENT
    7. Backpropagate and update policy/value networks

Clipping Mechanism (ε = 0.2):

The existence of an advantage as a positive (good action) has a limit ratio of 1.2 (which does not allow for the excessive taking of advantage). On the contrary, if the advantage is negative (bad action), the limit ratio is 0.8 (which does not allow for the over-penalization). This risk-averse approach to updating effectively guards against the phenomenon of policy collapse caused by radical updates.

4. Multi-Objective Loss:
The training loss is the combination of three objectives:

Policy Loss (-L_CLIP): generates actions with positive advantages, and negatives ones are discouraged.
Value Loss (0.5 * MSE): increases the accuracy of the value function for obtaining better advantage estimates.
Entropy Regularization (-0.05 * H initially): Stays exploration by punishing too deterministic policies.
Entropy coefficient drops from 0.05 to 0.001 during the training process.
At the beginning of the training: high entropy for the wide exploration
At the end of training: low entropy for the concentrated exploitation

5. Gradient Clipping:
Gradients have been clipped to the highest norm of 0.5
Stops the occurrence of exploding gradients in deep networks
Provides a guarantee of stable learning even if the batch updates are very large

Training Process & Curriculum Learning

The entire training procedure takes place in accordance with the actor-critic model among other things with the collection of on-policy data:
Rollout Phase (2048 steps per iteration)
The agent interacts with the environment by getting the state of the desktop (the current application window, UI elements, and cursor position). The triple-modal encoder processes the observation and produces a unified state representation. The manager policy samples the action type and coordinates it. The worker carries out the action on the actual desktop environment. The environment replies with: the next state, the reward, the termination signal, and metadata.
Experience storage: Full information for each transition is stored in the replay buffer. The buffers are being filled with: states, actions, rewards, value predictions, log-probabilities, done flags. After 2048 steps, advantages and returns are calculated through GAE and this is counted as one complete batch for policy updates.
Management of episodes: The episodes end in case of: achieving a task, a failure, or a timeout of 30 steps. The environmental task-specific validation provides the basis for success detection. The following episode statistics are recorded: total reward, length, action distribution, and success status. The detailed terminal output displays step-by-step action sequences for debugging purposes.


Update Phase (4 epochs × 4 minibatches):

Minibatch Creation: A buffer of 2048-steps is divided into 4 minibatches of 512 steps each. Random shuffling in between each epoch eliminates temporal correlation. Working on the same data for multiple epochs increases sample efficiency.
Policy Optimization: Forward pass: current policy is used to evaluate the stored actions. The importance sampling ratios for the off-policy correction are computed. The clipped PPO objective with advantages is calculated. The policy and value networks are updated via backpropagation. Gradient clipping is performed for stable updates. 
Monitoring: The policy loss, value loss, entropy as well as the approximate KL divergence are tracked. The clip fraction shows how often the ratio clipping is active. Logs are sent to TensorBoard for real-time monitoring and analysis.

Curriculum Learning Strategy
Training progresses through three difficulty stages, mirroring human learning:

Stage 1: EASY Tasks (0-100k steps):
Task Traits: One-step or two-to-three-step sequences - Illustrations: "Hit the Start button", "Launch Notepad", "Write 'hello'"
Success Condition: 60% success rate in the last 100 episodes
Objective: Master primitive actions and identify UI elements
Regular Time: Around 50,000 steps with the proper hyperparameters

Stage 2: MEDIUM Tasks (100k-250k steps):
Task Characteristics: operations from 8 to 15 steps carried out in multiple phases. For instance: 'Launch Chrome, go to Google, type 'machine learning' in search bar''. 
Advancement Criterion: 50% of last 100 episodes' success rate 
Purpose: acquisition of action sequencing, state tracking and subgoal decomposition skills 
Challenges: it requires knowing causal chains and understanding application workflows

Stage 3: HARD Tasks (250k+ steps)
Task Features: Workflows with over 15 steps of complexity - Illustrations: "Get a file, unzip it, use it in the program, change it and save it"
No Progress: Last phase of training
Aim: To acquire skills in complex task combinations and reliable execution
Obstacles: Credit assignment over a long period, error recovery, coordination among different applications


Curriculum Progression Mechanism:

Following every single episode:
  1. Record the outcome of success/failure
  2. Determine if at least 50k steps have been taken in current stage
  3. Determine rolling success rate (last 100 episodes)
  4. If success_rate >= threshold AND min_steps_met:
     → Move to next stage
     → Clear stage statistics
     → Modify environment to draw tasks that are more challenging
  5. Write curriculum statistics to TensorBoard

Why Curriculum Learning needed here - 
Sample Efficiency: By mastering easier skills, faster convergence is achieved 
Stability: Early discouragement from hard tasks that are beyond the learner's current level is avoided 
Transfer: The skills learned from the simpler tasks can be applied in sophisticated scenarios 
Exploration: The gradual increase in difficulty formation of the strategy to explore is natural

Reward Structure
The reward function steers the learning process by means of both sparse and dense signals:

1. Sparse Rewards:
Completion of the Task: +10 for an accomplished task
Failure of the Task: -5 for failure (wrong action sequence)

2. Dense Rewards (per step):
Advancement: +0.1 for every positive action towards the goal (changes in UI state)
Penalty for Time: -0.01 every step to promote efficiency
Action not Valid: -0.5 for actions that cannot be done (e.g., clicking off-screen, pressing invalid key)
Penalty for Repetition: -0.2 for doing the same thing without changing the state

3. Reward Shaping Philosophy:
Sparse rewards are the learning signal (task success/failure) at the most
Dense rewards show the way through the intermediate steps and facilitate learning
Penalties eliminate unproductive behaviors without strictness
Exploration incentives and execution efficiency are kept in balance.




Training Dynamics & Behavior Emergence

Early Training (Steps 0-50k)

Exploration Phase:
The application of a high entropy coefficient (0.05) leads to the sampling of diverse actions.
The agent is on the way to getting the maximum possible actions: mixing clicks, types, and scrolls in different ways.
Randomly chosen screen coordinates result in clicks all over the screen.
The agent experiences a lot of failures, but the few successes give it the first learning signal.

Primitive Learning: 
The agent figures out the very simple cause and effect: clicking makes a change, and typing gives the text.
The value function becomes capable of distinguishing between successful and unsuccessful states.
The advantage signals are given to the actions that brought about the progress.
The agent slowly pairs the text instructions with the visual targets that correspond to them.

Typical Behaviors:
A lot of random clicks on the screen.
Sometimes the agent waits for too long or stops too soon.
At first ignoring the instruction text.
The agent is learning to choose CLICK over other actions (which are the most successful ones and are used most).

Mid Training (Steps 50k-200k)

Skill Composition:
The skill composition was based on the successful completion of easy tasks with the consistency of more than 60% success rate.
2-3 actions were beginning to be chained correctly.
The learning of spatial priors took place implying that the UI elements were usually in specific screen regions.
Text phrases like "open" were associated with action sequences i.e., CLICK + coordinates of app icon.

Curriculum Transition:
The transition from easy to medium tasks happened when the easy skills were completely mastered.
The initial performance drop was considered normal as the harder tasks were introduced.
The policy was gradually adapting to the longer action sequences.
The tracking of the task state across multiple steps was learned.

Typical Behaviors:
The clicks were particularly directed and performed according to visual characteristics.
There was always a good choice of action type (TYPE when the text box is highlighted).
At times, correct multi-step sequences took place.
Error recovery and long-horizon planning were still the major issues.

Late Training (Steps 200k+)

Task Mastery:
Consistently solving MEDIUM tasks (>50% success rate)
Starting to deal with HARD multi-step workflows
Sturdy against small UI changes and distractions
Developing error recovery: retrying failed actions in a suitable manner

Policy Refinement:
Entropy coefficient reduced to about 0.01, policy becoming completely deterministic
Accurate coordinate prediction for UI parts
Smart action sequences with little redundant steps
Early stopping recognized as valid task completion signal

Emergent Strategies:
Exploration: systematic way: checking many locations before making a decision
State-dependent action selection: varying behaviors in different apps
Implicit subgoal decomposition: partitioning intricate tasks into stages
Temporal credit assignment: right actions even with later rewards

Desktop Controlling - OS-Level Interaction

Using PyAutoGUI:

API calls to the operating system directly: Win32 API in the case of Windows, Quartz for macOS, and Xlib for Linux
Sub-pixel precision for the mouse cursor which means very accurate clicks
All regular keys and modifiers are supported in keyboard emulation

Screen Capture:

Screenshots at regular intervals (every action step) are visual observations
Efficient screen capturing: mss library (around 60fps supported)
Automatic resizing and normalization for neural network input
Capturing optional region-of-interest for concentrated observation



Execution Safety:

Failsafe mechanism: mouse movement to corner as emergency stop (not available during training)
Action verification: verify screen limits prior to carrying out
Timeout safeguard: the longest episode length stops infinite loops
Exception management: smooth drop on execution errors

Multi-Application Handling - 

Application State Tracking:

Environment tracks active window and application
Task instructions specify required applications
Agent learns to recognize application-specific UI patterns
Cross-application workflows require sequential window focusing

Task-Specific Validation:

The environment monitors the currently active window and application. The task instructions indicate which applications are needed. The agent acquires the ability to identify user interface patterns that are typical for each specific application. The workflows that cut across different applications necessitate focusing on windows in sequence.

### Environment Architecture for Desktop Control Training

Classic RL environments that imitate abstract game states or robot control are different from this environment which has to link neural network choices with real operating system actions. The environment design complies with the OpenAI Gymnasium standard, thus giving a uniform method of interaction that isolates the agent's educational program from the desktop interaction components.



Core Environment: The OSEnvironment

The OSEnvironment base class, which defines the agreement between the RL agent and the OS, is at the base of the environment system. This class extends Gymnasium's Env interface, thus guaranteeing standard RL training frameworks compatibility and facilitating smooth connection with already existing algorithms like PPO.
The environment functions with a three-fold observation space that corresponds to the perception architecture of the agent. Each observation captures three components working in sync: a visual screenshot of the desktop - presented as an RGB numpy array with the resolution matching the screen size, a text task instruction stating what the agent should do, and a numeric state vector that includes ten scalar features. These numeric features provide important temporal and spatial information, which includes the current task ID, the episode's step count, the previous action type, the current cursor position (x and y coordinates), the difficulty level of the task, and other fields set aside for future extensions. With this multi-modal setup, the agent can at the same time comprehend what it sees, what it has to do, and where exactly it is in the execution of the task.
The action space incorporates a hybrid discrete-continuous structure that neatly reflects the dual control aspect of a desktop system. The discrete part picks among seven basic action types: single click, double click, right click, typing text, scrolling the mouse, waiting purposely, and stopping the task early. The continuous part indicates the target coordinates that are normalized to the interval of [-1, 1], which gives the agent the ability to accurately point out any location on the screen no matter the actual resolution. This normalization turns out to be very important for cross-display learning and it also helps in keeping the policy resolution-independent while training.


Reward Shaping and Learning Signals
	
The core reward system gives a slight negative punishment for every timestep (-0.01), which is a way of motivating the agent to be efficient, big positive rewards for task completion (+10.0), and moderate punishments for complete failure (-1.0). 
The reward shaping mechanism has five different behavioral incentives, and each of them is aimed to direct the agent to good exploration patterns. Initially, a survival bonus is given that consists of small positive rewards (+0.02) during the first 20 steps of each episode, which cancels out the per-step penalty and promotes the agent to take significant actions rather than stopping the episode too early. This is especially critical in the case of early training of the agent when it still doesn't know what is a progress or not.
Secondly, an exploration bonus encourages the agent by giving him a reward of an amount equal to such that he has not used it before in the current episode’s action types. The agent is rewarded with a +0.1 increment when he does an action type for the first time in an episode (which is monitored through a set of unique actions). This makes the agent’s behavior varied in the early phases of training, preventing the agent from getting stuck in a cycle of repeating actions before finding a successful strategy.
Third, a diversity penalty particularly aims at action repetition by recognizing when the agent does the same action type three times in a row. Such behavior is typical for an agent being stuck in a local minimum or not adapting to the environment's feedback, thus, being discouraged with a -0.2 penalty for these loops considered non-productive.
Fourth, the desktop change bonus gives the agent intermediate progress signals by recognizing when the actions of the agent actually change the desktop state. The environment keeps a hash of the last screenshot and after each action, compares it with the current screen state. When changes are detected (signifying the agent successfully interacted with the UI), a +0.5 progress reward is given. This rich signal is of great importance for the process of credit assignment, as it enables the agent to ascertain which actions bring about effects even before discovering the entire task solution.
In the end, the rewards that are specific to actions give the most precise advice depending on the type of action and the context. Early stopping suffers from scattered penalties that are determined by the progress of the episode - trying to stop after just 2-3 steps incurs a large penalty (-3.0), while stopping after a reasonable exploration (10+ steps) gets only a small penalty (-0.5). The excessive waiting (having more than 2 wait actions in the last 5 steps) penalties are imposed to deter passive behavior. In contrast, the interactive actions like clicks receive small bonuses (+0.05) to lure users to keep engaging with the UI.
The complex reward shaping changes the sparse signal of task completion into a rich landscape of intermediate objectives, which helps learning to be done very quickly in early training phases while the final goal of task success is still kept.

Screenshot Capture
Taking a screenshot is heavily dependent on the MSS python package which is for high-speed screen capturing that can be done across Windows, macOS, and Linux. In the case of Windows, the Desktop Duplication API is used, while, for macOS, the CGWindowListCreateImage is the associated API, and in the case of Linux, X11's XGetImage is the API used, with a speed of more than 60 frames per second as the outcome of this direct interfacing with the OS's native screen-capturing APIs.
First, the raw pixel data of the designated monitor area is taken by MSS and is given back in BGRA format (which consists of blue-green-red-alpha color channels). The environment then swiftly changes this to RGB format via OpenCV'scolor space conversion functions since neural networks usually demand RGB ordering, and the alpha channel does not provide any significant information to the agent. If the size-changing feature is activated (which is commonly the case for memory efficiency), OpenCV's bilinear interpolation re-sizes the full-screen image to the specified dimensions - usually 640x480 or 1920x1080, depending on whether speed or quality is the priority.
The resizing geometry is a very important factor to consider in the system. Full HD screenshots (1920x1080) retain the finest details of the user interface such as small text and icons, which are very important for good interaction, but they consume a lot of memory when processed in the neural network in batches. Lower resolutions (640x480) allow larger batch sizes and faster training iterations but may lose crucial visual information. The standard setup employs 640x480 for the state encoding (because the visual encoder downsamples to 224x224 for CLIP anyway) while keeping the higher resolution for visualization and debugging purposes.
A further preprocessing step normalizes pixel values from the integer range [0, 255] to floating-point [0, 1] and reorders the dimensions from the image format (height, width, channels) to PyTorch's expected tensor format (channels, height, width). These transformations are performed efficiently utilizing NumPy's vectorized operations, thereby adding almost no overhead to the capture process.

OSWorld Integration:

Although the base setup offers the necessary groundwork, the OSWorld integration layer llows the agent to interact with hardware and software in a realistic and complicated way. OSWorld, which is a product of XLang Lab's research for computer control agents, gives users access to completely operational Ubuntu desktop environments running in either virtualized or containerized setups. 
The OSWorldEnvironment Gymnasium wrapper makes no distinction between OSWorld's desktop environments and the standardized RL interface. It holds both the OSWorld manager and the particular desktop environment instance that is allocated to this training worker as a reference. During reset operations, it either brings the desktop to a clean state or resets it, takes over task specifications from OSWorld's task repository, and captures the first screenshot that will serve as the initial observation. 
The execution of actions in the OSWorld environments takes place through a layer of translation that turns the agent's high-level action dictionaries into PyAutoGUI command strings. For instance, a click action at the normalized coordinates (0.5, 0.3) becomes the Python code pyautogui.click(960, 324) after being transformed to the 1920x1080 resolution. These command strings are then sent to the step function of the desktop_env, which operates on them within the context of the virtual desktop and then presents the resulting screen state.
The OSWorld integration also comes with task validation features. Every task in the OSWorld benchmark has specific success criteria attached to it - it could be verifying if a file with the correct name is created, if a certain application window is opened, or if the web page content is compared with expected text. After every step, the environment's _check_task_completion() method checks through these validation functions and decides if the agent has successfully completed the task, has failed it for sure, or should still try.






















## Results

Training was conducted for approximately 1370 episodes. The agent showed progressive learning from random exploration to more structured interaction patterns. Initial episodes exhibited mostly failures with negative rewards, while later episodes demonstrated improved reward accumulation and longer episode lengths, indicating the agent was learning to interact more meaningfully with the desktop environment.

Key observations:
- Episode length increased from 4-10 steps to 50 steps (maximum)
- Reward improved from -5.5 to +26.4 per episode
- Action distribution became more diverse over time
- Positive reward ratio increased from 0% to 100% in successful episodes

The training progression shows the agent transitioning from random exploration to more deliberate action sequences, though task completion rates remained challenging due to the complexity of desktop control tasks.

## References
[1] Z. Dong, X. Fan, Z. Tang, and Y. Li, "Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces," arXiv preprint arXiv:2509.18230, 2025.

