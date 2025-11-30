# Installation Fix for Python 3.13 Compatibility

## Problem
OSWorld has dependency conflicts with Python 3.13:
- Requires old `opencv-python-headless<=4.5.4.60` 
- Requires `numpy==1.21.2` which doesn't support Python 3.13
- Dependency resolution takes forever and fails

## Solution

### Option 1: Use Python 3.9-3.12 (Recommended)

**If you're using conda/miniconda:**
```bash
# Create a new environment with Python 3.11
conda create -n rl-agent python=3.11 -y
conda activate rl-agent

# Now run the setup script
bash scripts/complete_setup.sh
```

**If you're using pyenv:**
```bash
# Install Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9

# Now run the setup script
bash scripts/complete_setup.sh
```

### Option 2: Manual Installation (if stuck on Python 3.13)

Install packages in this exact order:

```bash
# 1. Core dependencies
pip install fastapi uvicorn[standard] pydantic pydantic-settings
pip install torch torchvision transformers timm openai
pip install gymnasium stable-baselines3 tensorboard

# 2. NumPy FIRST (before opencv)
pip install "numpy<2.0"

# 3. OpenCV (without version constraints)
pip install opencv-python

# 4. Other dependencies
pip install pyautogui mss pillow pynput pygame matplotlib
pip install pandas pyyaml python-dotenv loguru httpx python-multipart docker requests

# 5. CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# 6. OSWorld WITHOUT dependencies (skip conflicting deps)
pip install --no-deps git+https://github.com/xlang-ai/OSWorld.git

# 7. Verify
python -c "import torch; print('✅ PyTorch works')"
python -c "import clip; print('✅ CLIP works')"
python -c "import desktop_env; print('✅ OSWorld works')"
```

### Option 3: Use Docker

If installation still fails, run everything in Docker:
```bash
# Pull OSWorld's official Docker image
docker pull xlanglab/osworld:latest

# Run training inside Docker
docker run -it --gpus all -v $(pwd):/workspace xlanglab/osworld:latest bash
cd /workspace
pip install -e .
python src/training/train.py
```

## What Changed in `pyproject.toml`

1. **Python version constraint**: `>=3.9,<3.13` (was `>=3.12`)
2. **Removed OSWorld and CLIP from dependencies** - install them separately
3. **Reordered dependencies** - numpy and opencv-python come before other packages

## What Changed in `complete_setup.sh`

The script now:
1. Checks Python version and warns if 3.13+
2. Installs core dependencies first
3. Installs numpy and opencv BEFORE other packages
4. Installs CLIP separately
5. Installs OSWorld with `--no-deps` flag to skip conflicting dependencies

## Testing Installation

After installation, verify everything works:
```bash
bash scripts/verify_setup.sh
```

Or manually test:
```bash
python -c "
import torch
import clip
import pygame
import desktop_env
print('✅ All imports successful!')
"
```

## Common Issues

### Issue: Still getting numpy version conflicts
```bash
# Force compatible numpy
pip install --force-reinstall "numpy<2.0,>=1.26"
```

### Issue: opencv-python-headless build fails
```bash
# Use regular opencv-python instead
pip uninstall opencv-python-headless
pip install opencv-python
```

### Issue: CLIP import fails
```bash
# Reinstall CLIP
pip uninstall clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Issue: Desktop-env not found
```bash
# Reinstall without deps
pip uninstall desktop-env osworld
pip install --no-deps git+https://github.com/xlang-ai/OSWorld.git
```

## Recommended: Fresh Virtual Environment

Best practice is to start clean:
```bash
# Remove old environment
conda env remove -n rl-agent

# Create fresh environment with Python 3.11
conda create -n rl-agent python=3.11 -y
conda activate rl-agent

# Install from scratch
cd /path/to/Hierarchical-RL-agent-for-Efficient-OS-Control
bash scripts/complete_setup.sh
```

## Next Steps

Once installation succeeds:
```bash
# Test the API
uvicorn src.api.main:app --reload

# Or start training
python src/training/train.py --visualize
```
