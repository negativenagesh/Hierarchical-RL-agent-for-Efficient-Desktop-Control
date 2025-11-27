# Training Hang Issue - RESOLVED

## Problem

The training script hangs at:
```
INFO - Loading CLIP ViT-B/16 model for visual encoding...
```

## Root Cause

The CLIP library attempts to download a 338MB pretrained model from OpenAI servers on first run. Your network connection is timing out, causing the hang.

## Evidence

```bash
ERROR - ✗ Failed to download CLIP model: <urlopen error [Errno 110] Connection timed out>
```

The model downloads from: `https://openaipublic.azureedge.net/clip/models/`

## Solutions

### Solution 1: Manual Download (FASTEST)

```bash
# Download CLIP model manually
mkdir -p ~/.cache/clip
cd ~/.cache/clip

# Use wget with retry
wget --continue --tries=10 --timeout=60 \
  -O ViT-B-16.pt \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

# Verify (should be ~338MB)
ls -lh ViT-B-16.pt
```

Then for the text model:
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download model
python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'); \
  AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')"
```

### Solution 2: Use Better Network

If on corporate/university network with firewall:
- Use VPN
- Use mobile hotspot
- Download on another machine and transfer

### Solution 3: Train Without Visualization (Temporary)

Train without the pretrained models by disabling visual encoder:

```bash
# Run without visualization (uses less resources)
python src/training/train.py --no-visualize
```

Note: You'll still need the models, but visualization can be disabled.

### Solution 4: Use Proxy

If behind a proxy:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
python scripts/download_models.py
```

## Verification

After downloading, verify:

```bash
# Should show ViT-B-16.pt (338 MB)
ls -lh ~/.cache/clip/

# Should show model directories
ls ~/.cache/huggingface/hub/ | grep mpnet
```

## Resume Training

Once models are downloaded:

```bash
# Models will load in ~10-30 seconds
python src/training/train.py --visualize
```

Expected output:
```
INFO - Loading CLIP ViT-B/16 model for visual encoding...
INFO - Using cached CLIP model from /home/BTECH_7TH_SEM/.cache/clip
INFO - CLIP model loaded successfully
INFO - Loading sentence-transformers/all-mpnet-base-v2 for text encoding...
INFO - Text model loaded successfully
INFO - Policy created successfully
INFO - Starting training loop
```

## Alternative: Download from Mirror

If OpenAI servers are blocked, try mirror:

```bash
# Hugging Face hosts CLIP models
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='openai/clip-vit-base-patch16', \
  filename='pytorch_model.bin')"
```

## Contact

See `MODELS_DOWNLOAD.md` for detailed troubleshooting guide.

## Summary

**Quick fix:** Download the model manually using wget/curl, then retry training.

**Why it happens:** First-time model download requires internet access.

**Future runs:** Once cached, training starts immediately.
