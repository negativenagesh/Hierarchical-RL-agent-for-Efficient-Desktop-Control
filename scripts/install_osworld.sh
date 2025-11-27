#!/bin/bash

# OSWorld Installation Script
# Clones OSWorld from GitHub and installs desktop_env

echo "=================================================="
echo "  Installing OSWorld from GitHub"
echo "=================================================="
echo ""

# Define installation path
OSWORLD_PATH="${HOME}/.osworld"

# Check if already cloned
if [ -d "$OSWORLD_PATH" ]; then
    echo "✓ OSWorld repository already exists at $OSWORLD_PATH"
    echo "  To reinstall, remove the directory first:"
    echo "  rm -rf $OSWORLD_PATH"
else
    # Clone OSWorld
    echo "📦 Cloning OSWorld from GitHub..."
    git clone https://github.com/xlang-ai/OSWorld.git "$OSWORLD_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ OSWorld cloned successfully"
    else
        echo "✗ Failed to clone OSWorld"
        exit 1
    fi
fi

# Install desktop_env
echo ""
echo "📦 Installing desktop_env package..."

# Try installing from local clone first
if [ -d "$OSWORLD_PATH" ]; then
    pip install -e "$OSWORLD_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ desktop_env installed from local clone"
    else
        echo "⚠ Local installation failed, trying direct from GitHub..."
        pip install git+https://github.com/xlang-ai/OSWorld.git
        
        if [ $? -eq 0 ]; then
            echo "✓ desktop_env installed from GitHub"
        else
            echo "✗ Failed to install desktop_env"
            exit 1
        fi
    fi
else
    # Install directly from GitHub
    pip install git+https://github.com/xlang-ai/OSWorld.git
    
    if [ $? -eq 0 ]; then
        echo "✓ desktop_env installed from GitHub"
    else
        echo "✗ Failed to install desktop_env"
        exit 1
    fi
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import desktop_env; print('✓ desktop_env imported successfully')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✅ OSWorld Installation Complete!"
    echo "=================================================="
    echo ""
    echo "OSWorld repository: $OSWORLD_PATH"
    echo "Package: desktop_env"
    echo ""
    echo "You can now use OSWorld in your code:"
    echo "  from desktop_env.desktop_env import DesktopEnv"
    echo ""
else
    echo ""
    echo "⚠ desktop_env installed but import failed"
    echo "This might be OK if you're in a different Python environment"
    echo ""
fi
