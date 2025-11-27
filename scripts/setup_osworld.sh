#!/bin/bash

# Quick Setup Script for OSWorld Integration
# This installs OSWorld from GitHub and sets up the environment

echo "=================================================="
echo "  OSWorld Setup for Hierarchical RL Agent"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install OSWorld from GitHub
echo "📦 Installing OSWorld from GitHub..."
echo "   This will install the desktop-env package with Docker provider support"
echo ""

pip install git+https://github.com/xlang-ai/OSWorld.git

if [ $? -eq 0 ]; then
    echo "✅ OSWorld installed successfully"
else
    echo "❌ Failed to install OSWorld"
    exit 1
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import desktop_env; print('✅ desktop_env imported successfully')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  desktop_env import failed, but this might be OK if running headless"
fi

# Check Docker
echo ""
echo "🐳 Checking Docker setup..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    
    # Test Docker permissions
    if docker ps &> /dev/null; then
        echo "✅ Docker permissions OK"
    else
        echo "⚠️  Docker permission issue detected"
        echo "   Run: sudo chmod 666 /var/run/docker.sock"
    fi
else
    echo "❌ Docker is not installed"
    echo "   Install from: https://docs.docker.com/engine/install/"
fi

echo ""
echo "=================================================="
echo "  Setup Instructions"
echo "=================================================="
echo ""
echo "OSWorld does NOT provide pre-built Docker images."
echo "Instead, it uses Docker as a VM provider."
echo ""
echo "To use OSWorld:"
echo "1. Install desktop-env: pip install desktop-env"
echo "2. Use Docker provider: provider_name='docker'"
echo "3. OSWorld will manage containers automatically"
echo ""
echo "See: https://github.com/xlang-ai/OSWorld"
echo ""
echo "For this project, the OSWorldEnvironment class"
echo "will handle all Docker container management."
echo ""
