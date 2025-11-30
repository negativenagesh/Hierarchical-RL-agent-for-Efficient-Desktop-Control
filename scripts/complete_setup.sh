#!/bin/bash

# Complete Setup Script - Run This to Get Everything Working!

echo "=================================================="
echo "  🚀 Complete Setup for Hierarchical RL Agent"
echo "=================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if [[ $(python -c 'import sys; print(sys.version_info >= (3, 13))') == "True" ]]; then
    echo "⚠️  Warning: Python 3.13+ detected. OSWorld may have compatibility issues."
    echo "   Recommended: Use Python 3.9-3.12 for best compatibility"
    echo ""
fi

# 1. Install core dependencies first
echo "📦 Step 1: Installing core Python dependencies..."
pip install -e . --no-deps
pip install fastapi uvicorn[standard] pydantic pydantic-settings
pip install torch torchvision transformers timm openai
pip install gymnasium stable-baselines3 tensorboard

# 2. Install numpy and opencv BEFORE OSWorld
echo ""
echo "📦 Step 2: Installing numpy and opencv (pre-requisites)..."
pip install "numpy<2.0"
pip install opencv-python

# 3. Install other computer control dependencies
echo ""
echo "📦 Step 3: Installing computer control dependencies..."
pip install pyautogui mss pillow pynput pygame matplotlib
pip install pandas pyyaml python-dotenv loguru httpx python-multipart docker requests

# 4. Install CLIP
echo ""
echo "📦 Step 4: Installing CLIP..."
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# 5. Install OSWorld (desktop-env) with constraints
echo ""
echo "📦 Step 5: Installing OSWorld (desktop-env)..."
echo "   This may take a while and show warnings - this is normal..."
pip install --no-deps git+https://github.com/xlang-ai/OSWorld.git || echo "⚠️  OSWorld installation had issues, but continuing..."

# 6. Setup environment file
echo ""
echo "⚙️  Step 6: Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# 7. Verify Docker
echo ""
echo "🐳 Step 7: Verifying Docker..."
if docker ps &> /dev/null; then
    echo "✅ Docker is working!"
else
    echo "⚠️  Docker permission issue"
    echo "   Run: sudo chmod 666 /var/run/docker.sock"
fi

# 8. Test imports
echo ""
echo "🔍 Step 8: Testing imports..."

python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch not installed"
python -c "import clip; print('✅ CLIP installed')" 2>/dev/null || echo "⚠️  CLIP not installed"
python -c "import pygame; print('✅ Pygame installed')" 2>/dev/null || echo "⚠️  Pygame not installed"
python -c "import desktop_env; print('✅ OSWorld (desktop-env) installed')" 2>/dev/null || echo "⚠️  desktop-env not installed"

echo ""
echo "=================================================="
echo "  ✅ SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "🎯 What to do next:"
echo ""
echo "1. Edit .env file if needed:"
echo "   nano .env"
echo ""
echo "2. Start training with visualization:"
echo "   python src/training/train.py --visualize"
echo ""
echo "3. Or test the API:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "📚 Read the docs:"
echo "   - DOCKER_FIXED.md - Docker setup & OSWorld explanation"
echo "   - UPGRADE_COMPLETE.md - What was upgraded"
echo "   - SETUP_UPGRADED.md - Detailed setup guide"
echo ""
echo "🎉 Happy training!"
echo ""