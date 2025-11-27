# 🚀 CUDA Installation Guide

## Your Hardware
- **GPU**: NVIDIA RTX 4000 Ada Generation (16GB)
- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 6.14.0-36-generic

## Quick Installation

I've created an automated installation script for you. Here's how to use it:

### Option 1: Automated Installation (Recommended)

```bash
# Run the installation script
./install_cuda.sh
```

This will:
1. Update your system
2. Remove old NVIDIA drivers
3. Install prerequisites
4. Install recommended NVIDIA driver (555+ for RTX 4000 Ada)
5. Install CUDA Toolkit 12.4
6. Install cuDNN 9 for deep learning
7. Set up environment variables

**Time required**: 10-15 minutes

### Option 2: Manual Installation

If you prefer to do it step by step:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install prerequisites
sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# 3. Install NVIDIA driver (recommended for your GPU)
sudo ubuntu-drivers install

# 4. Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 5. Install CUDA Toolkit
sudo apt install -y cuda-toolkit-12-4

# 6. Install cuDNN
sudo apt install -y cudnn9-cuda-12

# 7. Add to ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## After Installation

### 1. REBOOT YOUR SYSTEM (REQUIRED!)
```bash
sudo reboot
```

### 2. Verify Installation

After reboot, check everything is working:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA RTX 4000 Ada Generation
```

### 3. Reinstall PyTorch with CUDA support (if needed)

If PyTorch doesn't detect CUDA after reboot:

```bash
# Uninstall CPU-only version
uv pip uninstall torch torchvision

# Install CUDA version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Troubleshooting

### Problem: `nvidia-smi` not found after reboot
**Solution**: Driver didn't load properly
```bash
sudo modprobe nvidia
sudo nvidia-modprobe
```

### Problem: CUDA not detected by PyTorch
**Solution**: Reinstall PyTorch with CUDA
```bash
uv pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

### Problem: Driver conflicts
**Solution**: Purge all NVIDIA packages and start fresh
```bash
sudo apt remove --purge '^nvidia-.*' -y
sudo apt autoremove -y
./install_cuda.sh  # Run script again
```

## What Gets Installed

- **NVIDIA Driver**: Latest recommended for RTX 4000 Ada (~555.x series)
- **CUDA Toolkit 12.4**: Compiler, libraries, tools
- **cuDNN 9**: Deep learning primitives
- **Size**: ~3-4 GB total

## Important Notes

1. **Reboot is MANDATORY** - Driver won't load until you reboot
2. **Secure Boot**: If enabled, you may need to disable it or sign the driver
3. **Wayland vs X11**: Some display issues may occur; switch if needed
4. **Backup**: No system files are deleted, but backup is always good practice

## After CUDA is Working

Run the test again to verify everything:
```bash
python test_setup.py
```

You should now see:
```
✓ PyTorch 2.5.1+cu124
  CUDA available: True
  GPU: NVIDIA RTX 4000 Ada Generation
  GPU Memory: 16.0 GB
```

Then you're ready to train with full GPU acceleration! 🚀

## Training with GPU

Once CUDA is working:

```bash
# Training will automatically use GPU
python src/training/train.py --visualize

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

Your 16GB RTX 4000 Ada is perfect for:
- CLIP ViT-B/16 (151M params) ✅
- MPNet-base-v2 (109M params) ✅
- PPO training with large batch sizes ✅
- Multiple parallel environments ✅

Happy training! 🎉
