#!/bin/bash

# Complete Setup Script - Run This to Get Everything Working!

echo "=================================================="
echo "  🚀 Complete Setup for Hierarchical RL Agent"
echo "=================================================="
echo ""

# 1. Install Python dependencies
echo "📦 Step 1: Installing Python dependencies..."
uv pip install -e .

# 2. Install CLIP
echo ""
echo "📦 Step 2: Installing CLIP..."
uv pip install git+https://github.com/openai/CLIP.git

# 3. Install OSWorld (desktop-env)
echo ""
echo "📦 Step 3: Installing OSWorld (desktop-env)..."
pip install git+https://github.com/xlang-ai/OSWorld.git

# 4. Setup environment file
echo ""
echo "⚙️  Step 4: Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# 5. Verify Docker
echo ""
echo "🐳 Step 5: Verifying Docker..."
if docker ps &> /dev/null; then
    echo "✅ Docker is working!"
else
    echo "⚠️  Docker permission issue"
    echo "   Run: sudo chmod 666 /var/run/docker.sock"
fi

# 6. Test imports
echo ""
echo "🔍 Step 6: Testing imports..."

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
