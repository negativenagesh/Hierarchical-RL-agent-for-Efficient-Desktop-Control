#!/bin/bash
# CUDA Installation Script for Ubuntu 24.04 with NVIDIA RTX 4000 Ada
# This script installs NVIDIA drivers and CUDA Toolkit 12.4

set -e  # Exit on error

echo "=========================================="
echo "CUDA Installation for Ubuntu 24.04"
echo "GPU: NVIDIA RTX 4000 Ada Generation"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please do not run as root. Run with sudo when prompted."
    exit 1
fi

echo ""
echo "Step 1: Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "Step 2: Removing old NVIDIA drivers (if any)..."
sudo apt remove --purge '^nvidia-.*' -y || true
sudo apt remove --purge '^libnvidia-.*' -y || true
sudo apt autoremove -y
sudo apt autoclean

echo ""
echo "Step 3: Installing prerequisites..."
sudo apt install -y \
    build-essential \
    dkms \
    linux-headers-$(uname -r) \
    pkg-config \
    libglvnd-dev \
    ubuntu-drivers-common

echo ""
echo "Step 4: Detecting recommended NVIDIA driver..."
ubuntu-drivers devices

echo ""
echo "Step 5: Installing NVIDIA driver (recommended)..."
sudo ubuntu-drivers install

echo ""
echo "Step 6: Adding NVIDIA CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

echo ""
echo "Step 7: Installing CUDA Toolkit 12.4..."
sudo apt install -y cuda-toolkit-12-4

echo ""
echo "Step 8: Installing cuDNN (for deep learning)..."
sudo apt install -y cudnn9-cuda-12

echo ""
echo "Step 9: Setting up environment variables..."
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "Added CUDA environment variables to ~/.bashrc"
else
    echo "CUDA environment variables already in ~/.bashrc"
fi

echo ""
echo "Step 10: Cleaning up..."
rm -f cuda-keyring_1.1-1_all.deb

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: You must REBOOT your system now!"
echo ""
echo "After reboot, verify installation with:"
echo "  nvidia-smi"
echo "  nvcc --version"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "To reboot now, run: sudo reboot"
echo "=========================================="
