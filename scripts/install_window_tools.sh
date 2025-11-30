#!/bin/bash
# Install window management tools for task completion checking

echo "Installing window management tools..."

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS"
    exit 1
fi

# Install based on OS
case "$OS" in
    ubuntu|debian)
        echo "Installing wmctrl and xdotool for Ubuntu/Debian..."
        sudo apt-get update
        sudo apt-get install -y wmctrl xdotool
        ;;
    centos|rhel|fedora)
        echo "Installing wmctrl and xdotool for CentOS/RHEL/Fedora..."
        sudo yum install -y wmctrl xdotool
        ;;
    arch)
        echo "Installing wmctrl and xdotool for Arch..."
        sudo pacman -S --noconfirm wmctrl xdotool
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install wmctrl or xdotool manually"
        exit 1
        ;;
esac

# Verify installation
if command -v wmctrl &> /dev/null; then
    echo "✓ wmctrl installed successfully"
    wmctrl -m
elif command -v xdotool &> /dev/null; then
    echo "✓ xdotool installed successfully"
    xdotool --version
else
    echo "✗ Installation failed"
    exit 1
fi

echo ""
echo "Window management tools installed!"
echo "Task completion checking for window titles is now enabled."
