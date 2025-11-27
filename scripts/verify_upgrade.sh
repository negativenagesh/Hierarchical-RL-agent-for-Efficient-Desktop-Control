#!/bin/bash

# Verification Script for Upgraded Hierarchical RL Agent
# Checks all new components and dependencies

echo "=============================================="
echo "  UPGRADED AGENT VERIFICATION"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 ${RED}(MISSING)${NC}"
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 ${RED}(MISSING)${NC}"
        return 1
    fi
}

# Check new/upgraded files
echo "📦 Checking New/Upgraded Files..."
echo "--------------------------------"
check_file "src/agent/encoder.py"
check_file "src/agent/encoder_old.py"
check_file "src/environment/osworld_integration.py"
check_file "src/utils/visualizer.py"
check_file "src/utils/metrics_simple.py"
check_file "SETUP_UPGRADED.md"
check_file "UPGRADE_SUMMARY.md"
echo ""

# Check simplified files
echo "🔧 Checking Simplified Configuration..."
echo "---------------------------------------"
check_file ".env.example"
check_file "docker/docker-compose.yml"
check_file "pyproject.toml"
check_file "src/api/config.py"
echo ""

# Check Docker
echo "🐳 Checking Docker Setup..."
echo "--------------------------"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker installed"
    
    # Check if OSWorld image exists
    if docker images | grep -q "osworld"; then
        echo -e "${GREEN}✓${NC} OSWorld image found"
    else
        echo -e "${YELLOW}⚠${NC} OSWorld image not found"
        echo "  Run: docker pull xlanglab/osworld:latest"
    fi
else
    echo -e "${RED}✗${NC} Docker not installed"
    echo "  Install from: https://docs.docker.com/get-docker/"
fi
echo ""

# Check CUDA/GPU
echo "🎮 Checking GPU Setup..."
echo "----------------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA drivers installed"
    
    # Get GPU info
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1)
    GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2)
    
    echo "  GPU: $GPU_NAME"
    echo "  Memory: ${GPU_MEMORY}MB"
    
    if (( $(echo "$GPU_MEMORY >= 16000" | bc -l) )); then
        echo -e "${GREEN}✓${NC} GPU has sufficient memory (≥16GB)"
    else
        echo -e "${YELLOW}⚠${NC} GPU has less than 16GB memory"
        echo "  Recommendation: Reduce batch size in config"
    fi
else
    echo -e "${RED}✗${NC} NVIDIA GPU not detected"
    echo "  This agent is optimized for 16GB GPU"
fi
echo ""

# Check Python packages
echo "🐍 Checking Python Dependencies..."
echo "---------------------------------"

check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 ${RED}(NOT INSTALLED)${NC}"
        return 1
    fi
}

# Critical new packages
check_python_package "clip"
check_python_package "pygame"
check_python_package "docker"
check_python_package "transformers"
check_python_package "torch"
echo ""

# Check removed packages (should NOT be installed for clean setup)
echo "🗑️  Checking Removed Dependencies..."
echo "------------------------------------"

check_removed_package() {
    if python -c "import $1" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} $1 ${YELLOW}(still installed, not needed)${NC}"
        return 1
    else
        echo -e "${GREEN}✓${NC} $1 (not installed - good!)"
        return 0
    fi
}

check_removed_package "redis"
check_removed_package "prometheus_client"
echo ""

# Check configuration
echo "⚙️  Checking Configuration..."
echo "----------------------------"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"
    
    # Check key settings
    if grep -q "VISUALIZE_TRAINING=true" .env; then
        echo -e "${GREEN}✓${NC} Visualization enabled"
    else
        echo -e "${YELLOW}⚠${NC} Visualization not enabled"
    fi
    
    if grep -q "DEVICE=cuda" .env; then
        echo -e "${GREEN}✓${NC} CUDA device configured"
    else
        echo -e "${YELLOW}⚠${NC} CUDA device not configured"
    fi
else
    echo -e "${YELLOW}⚠${NC} .env file not found"
    echo "  Run: cp .env.example .env"
fi
echo ""

# Count files
echo "📊 Project Statistics..."
echo "----------------------"
PYTHON_FILES=$(find src -name "*.py" | wc -l)
LOC=$(find src -name "*.py" -exec wc -l {} + | tail -n 1 | awk '{print $1}')
echo "  Python files: $PYTHON_FILES"
echo "  Lines of code: ~$LOC"
echo ""

# Summary
echo "=============================================="
echo "  VERIFICATION COMPLETE"
echo "=============================================="
echo ""
echo "📋 Next Steps:"
echo "  1. Install dependencies: uv pip install -e ."
echo "  2. Install CLIP: uv pip install git+https://github.com/openai/CLIP.git"
echo "  3. Pull OSWorld: docker pull xlanglab/osworld:latest"
echo "  4. Configure .env: cp .env.example .env && nano .env"
echo "  5. Start training: python src/training/train.py --visualize"
echo ""
echo "📚 Documentation:"
echo "  - UPGRADE_SUMMARY.md - What changed"
echo "  - SETUP_UPGRADED.md - Complete setup guide"
echo "  - docs/TRAINING.md - Training guide"
echo ""
echo "🎉 Your upgraded agent is ready!"
echo ""
