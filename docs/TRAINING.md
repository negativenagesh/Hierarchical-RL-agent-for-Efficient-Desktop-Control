# Training Guide

## Overview

This guide covers training the Hierarchical RL Agent from scratch.

## Prerequisites

- NVIDIA GPU with 12GB+ VRAM (or CPU for testing)
- Python 3.9+
- UV package manager installed

## Quick Start

```bash
# Start training with default settings
bash scripts/train.sh

# Or with custom parameters
TOTAL_STEPS=500000 DIFFICULTY=EASY DEVICE=cuda bash scripts/train.sh
```

## Training Configuration

### Basic Parameters

Edit `config/config.yaml`:

```yaml
training:
  learning_rate: 0.0003    # Adam learning rate
  clip_epsilon: 0.2        # PPO clipping parameter
  gamma: 0.99              # Discount factor
  gae_lambda: 0.95         # GAE lambda
  buffer_size: 2048        # Replay buffer size
  update_epochs: 4         # Epochs per update
```

### Curriculum Learning

The system automatically progresses through difficulty stages:

**Stage 1: Easy Tasks (< 8 steps)**
- Duration: ~100k steps
- Success threshold: 60%
- Examples: Open app, create file

**Stage 2: Medium Tasks (8-15 steps)**
- Duration: ~150k steps  
- Success threshold: 50%
- Examples: Navigate folders, open browser

**Stage 3: Hard Tasks (> 15 steps)**
- Duration: Unlimited
- Examples: Complex multi-step operations

Configure thresholds in `config/config.yaml`:

```yaml
curriculum:
  easy_threshold: 0.6
  medium_threshold: 0.5
  min_steps_per_stage: 50000
```

## Training from Scratch

### 1. Prepare Tasks

Create or edit `config/tasks.json`:

```json
[
  {
    "id": 1,
    "instruction": "Open the calculator",
    "difficulty": "EASY",
    "num_steps": 5,
    "success_criteria": {
      "window_title": "Calculator"
    }
  }
]
```

### 2. Set Environment Variables

```bash
export DEVICE=cuda
export LOG_LEVEL=INFO
export TRAINING_ENABLED=true
```

### 3. Start Training

```bash
python src/training/train.py \
    --total-timesteps 1000000 \
    --rollout-steps 2048 \
    --learning-rate 0.0003 \
    --difficulty EASY \
    --device cuda \
    --save-dir checkpoints \
    --log-dir logs \
    --curriculum
```

### 4. Monitor Progress

**Tensorboard:**
```bash
tensorboard --logdir=logs
# Visit http://localhost:6006
```

**Console Output:**
```
Update 10/488 | Step 20480/1000000
FPS: 156.23 | Time: 131.12s
Mean Reward: 8.42
Success Rate: 45.00%
Policy Loss: 0.0234
Value Loss: 0.1156
```

## Training Parameters Explained

### Learning Rate (`learning_rate`)
- **Default**: 3e-4
- **Range**: 1e-5 to 1e-3
- **Effect**: Controls update step size
- **Recommendation**: Start with 3e-4, decrease if unstable

### Clip Epsilon (`clip_epsilon`)
- **Default**: 0.2
- **Range**: 0.1 to 0.3
- **Effect**: PPO clipping range
- **Recommendation**: 0.2 works well for most cases

### Gamma (`gamma`)
- **Default**: 0.99
- **Range**: 0.95 to 0.999
- **Effect**: Discount factor for future rewards
- **Recommendation**: 0.99 for long-horizon tasks

### GAE Lambda (`gae_lambda`)
- **Default**: 0.95
- **Range**: 0.9 to 0.99
- **Effect**: Advantage estimation bias-variance tradeoff
- **Recommendation**: 0.95 balances bias and variance

### Buffer Size (`buffer_size`)
- **Default**: 2048
- **Range**: 1024 to 8192
- **Effect**: Steps collected before update
- **Recommendation**: 2048 for 12GB VRAM, 4096 for 24GB

## Advanced Training

### Fine-tuning from Checkpoint

```python
from src.agent.policy import HierarchicalPolicy
from src.training.ppo_trainer import PPOTrainer

# Load pre-trained model
policy = HierarchicalPolicy()
policy.load("checkpoints/pretrained.pt")

# Continue training
trainer = PPOTrainer(policy, env, learning_rate=1e-4)
trainer.train(total_timesteps=100000)
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
import torch.nn as nn
policy = nn.DataParallel(policy, device_ids=[0, 1])
```

### Custom Reward Shaping

Edit `src/environment/base_env.py`:

```python
def step(self, action):
    # ... existing code ...
    
    # Custom reward shaping
    if action_successful:
        reward += 0.5
    
    if cursor_moved_towards_target:
        reward += 0.1
    
    return obs, reward, done, info
```

## Troubleshooting

### Low Success Rate

1. **Check curriculum stage**: May need more training at current stage
2. **Reduce learning rate**: Try 1e-4 instead of 3e-4
3. **Increase buffer size**: More diverse experiences
4. **Add reward shaping**: Guide agent with intermediate rewards

### Training Unstable

1. **Decrease learning rate**: 1e-4 or 3e-5
2. **Increase clip epsilon**: 0.3 for more exploration
3. **Add gradient clipping**: Default is 0.5
4. **Check task definitions**: Ensure tasks are achievable

### Out of Memory

1. **Reduce buffer size**: 1024 or 512
2. **Reduce batch size**: Fewer minibatches
3. **Use smaller encoder**: Replace EfficientNet-B0 with MobileNet
4. **Enable gradient checkpointing**

## Checkpoints

Checkpoints are saved every 10,000 steps to `checkpoints/`:

```
checkpoints/
├── checkpoint_10000.pt
├── checkpoint_20000.pt
├── ...
└── final_model.pt
```

Load checkpoint:

```python
policy.load("checkpoints/checkpoint_50000.pt")
```

## Performance Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Increase rollout workers**: Parallel environment collection
3. **Optimize screenshot capture**: Use MSS instead of PIL
4. **Profile code**: Identify bottlenecks with cProfile
5. **Mixed precision**: Use torch.cuda.amp for faster training

## Next Steps

After training:

1. Evaluate on test tasks
2. Export for deployment
3. Fine-tune on specific use cases
4. Monitor in production with API
