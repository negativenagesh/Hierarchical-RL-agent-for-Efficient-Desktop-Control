#!/bin/bash

# Training script for Hierarchical RL Agent

set -e

echo "Starting training..."

# Activate environment if needed
# source venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default parameters
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-2048}
DIFFICULTY=${DIFFICULTY:-EASY}
DEVICE=${DEVICE:-cuda}

echo "Configuration:"
echo "  Total Steps: $TOTAL_STEPS"
echo "  Rollout Steps: $ROLLOUT_STEPS"
echo "  Difficulty: $DIFFICULTY"
echo "  Device: $DEVICE"

# Run training
python -m src.training.train \
    --total-timesteps $TOTAL_STEPS \
    --rollout-steps $ROLLOUT_STEPS \
    --difficulty $DIFFICULTY \
    --device $DEVICE \
    --save-dir checkpoints \
    --log-dir logs

echo "Training complete!"
