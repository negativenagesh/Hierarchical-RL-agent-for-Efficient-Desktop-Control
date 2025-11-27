# Pretrained Models Download Guide

## Issue

The training script requires two large pretrained models:

1. **CLIP ViT-B/16** (~350MB) - For visual encoding
2. **sentence-transformers/all-mpnet-base-v2** (~420MB) - For text encoding

On first run, these models are downloaded automatically from the internet. If you're experiencing:
- Long delays during training startup
- Connection timeouts
- Network errors

This guide will help you resolve the issue.

## Quick Diagnosis

Run this to check if models are already cached:

```bash
# Check CLIP cache
ls -lh ~/.cache/clip/

# Check HuggingFace cache
ls -lh ~/.cache/huggingface/hub/
```

## Solution 1: Automatic Download (Recommended)

Use our download script:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run download script
python scripts/download_models.py
```

This will:
- Check available disk space
- Download both models
- Verify successful download
- Cache models for future use

## Solution 2: Manual Download

If automatic download fails due to network issues:

### Option A: Download via wget/curl

```bash
# Create cache directory
mkdir -p ~/.cache/clip

# Download CLIP ViT-B/16
wget -O ~/.cache/clip/ViT-B-16.pt \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

# Verify download
ls -lh ~/.cache/clip/ViT-B-16.pt
```

For the text model:
```bash
# Use transformers-cli
pip install transformers
transformers-cli download sentence-transformers/all-mpnet-base-v2
```

### Option B: Use Proxy/VPN

If you're behind a firewall or have network restrictions:

```bash
# Set proxy (if needed)
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# Then run download script
python scripts/download_models.py
```

### Option C: Download on Another Machine

1. Download models on a machine with good internet
2. Copy the cache directories:
   - `~/.cache/clip/` 
   - `~/.cache/huggingface/`
3. Transfer to your training machine

## Solution 3: Use Smaller Models (Fallback)

If you can't download these models, you can modify the code to use smaller alternatives:

Edit `src/agent/encoder.py`:

```python
# Instead of CLIP ViT-B/16, use RN50 (much smaller)
self.clip_model, self.preprocess = clip.load("RN50", device=device)

# Instead of all-mpnet-base-v2, use distilbert
model_name = "distilbert-base-uncased"
```

Note: This will reduce model quality but allow you to start training.

## Troubleshooting

### Problem: Connection Timeout

**Symptoms:**
```
ERROR - ✗ Failed to download CLIP model: <urlopen error [Errno 110] Connection timed out>
```

**Solutions:**
1. Check internet connection: `ping openaipublic.azureedge.net`
2. Try at different time (server may be busy)
3. Use VPN or proxy
4. Download manually (see Option A above)

### Problem: Insufficient Disk Space

**Symptoms:**
```
WARNING - ⚠ Low disk space! At least 2GB recommended
```

**Solution:**
```bash
# Check disk space
df -h ~

# Clear old caches if needed
rm -rf ~/.cache/pip
```

### Problem: Training Hangs at "Loading CLIP"

**Symptoms:**
```
INFO - Loading CLIP ViT-B/16 model for visual encoding...
[hangs for 10+ minutes]
```

**Cause:** Model is being downloaded in background

**Solutions:**
1. Wait for download to complete (may take 10-30 minutes on slow connection)
2. Pre-download using script: `python scripts/download_models.py`
3. Check download progress: `watch -n 1 ls -lh ~/.cache/clip/`

## Verification

After successful download, verify models are cached:

```bash
# Check CLIP
ls -lh ~/.cache/clip/ViT-B-16.pt
# Should show: ~338MB file

# Check text model
ls ~/.cache/huggingface/hub/ | grep mpnet
# Should show: models--sentence-transformers--all-mpnet-base-v2
```

## Training Without Models

If all else fails, you can train without the pretrained encoders:

```bash
# Use simple CNN encoder (implemented in code)
python src/training/train.py --simple-encoder --visualize
```

Note: This requires code modification to add `--simple-encoder` flag.

## Support

If you continue to have issues:

1. Check the logs: `tail -f logs/training.log`
2. Run with verbose logging: `python src/training/train.py --visualize --verbose`
3. Check model cache: `du -sh ~/.cache/clip ~/.cache/huggingface`

## Model Details

| Model | Size | URL | Purpose |
|-------|------|-----|---------|
| CLIP ViT-B/16 | 338 MB | [Link](https://openaipublic.azureedge.net/clip/models/) | Visual encoding |
| all-mpnet-base-v2 | 420 MB | [Link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Text encoding |

Total: ~760 MB

## Next Steps

Once models are downloaded successfully:

```bash
# Start training
python src/training/train.py --visualize

# Or run examples
python examples.py
```

The first run after download will be much faster (~10-30 seconds to load models into GPU memory).
