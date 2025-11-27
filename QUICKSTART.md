# 🚀 Quick Start Guide

Get up and running with the Hierarchical RL Agent in 5 minutes!

## Prerequisites

- Python 3.9+
- UV package manager
- (Optional) CUDA-capable GPU for training

## Step 1: Clone & Install (2 minutes)

```bash
# Navigate to the project
cd /home/BTECH_7TH_SEM/Downloads/Hierarchical-RL-agent-for-Efficient-OS-Control

# Install dependencies with UV
uv pip install -e .

# Verify installation
python -c "import torch; print('✓ PyTorch installed')"
python -c "import fastapi; print('✓ FastAPI installed')"
```

## Step 2: Configure (1 minute)

```bash
# Copy environment template
cp .env.example .env

# (Optional) Edit configuration
nano .env
```

**Default settings work out-of-the-box!**

## Step 3: Run API (1 minute)

```bash
# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the script
bash scripts/run_api.sh

# Or use Make
make run-api
```

**API will be available at: http://localhost:8000**

## Step 4: Test It (1 minute)

### Option A: Use cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict an action
curl -X POST http://localhost:8000/api/v1/agent/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Open the calculator application",
    "use_live_screen": true,
    "deterministic": false
  }'
```

### Option B: Use Python

```python
import requests

# Predict action
response = requests.post(
    "http://localhost:8000/api/v1/agent/predict",
    json={
        "instruction": "Open calculator",
        "use_live_screen": True
    }
)

action = response.json()
print(f"Action: {action['action_type']}")
print(f"Coordinates: {action['coordinates']}")
```

### Option C: Run Examples

```bash
python examples.py
```

## Step 5: Explore! (<1 minute)

### View API Documentation
Open your browser: http://localhost:8000/docs

### Check Metrics
```bash
curl http://localhost:8000/api/v1/monitoring/metrics
```

### View Model Info
```bash
curl http://localhost:8000/api/v1/agent/model-info
```

---

## 🎓 Next Steps

### Start Training

```bash
# Quick training test (Easy tasks, 10k steps)
python src/training/train.py \
    --total-timesteps 10000 \
    --difficulty EASY \
    --device cuda

# Full training with curriculum
python src/training/train.py \
    --total-timesteps 1000000 \
    --curriculum \
    --device cuda
```

### Use Docker

```bash
# Build image
docker build -t hierarchical-rl-agent -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 hierarchical-rl-agent

# Or use Docker Compose (includes monitoring)
docker-compose -f docker/docker-compose.yml up -d
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📚 Documentation

- **Full README**: See `README.md`
- **API Reference**: See `docs/API.md`
- **Training Guide**: See `docs/TRAINING.md`
- **Architecture**: See `docs/PROJECT_STRUCTURE.md`

---

## ❓ Common Issues

### "Import torch could not be resolved"

The imports will show as errors in the IDE until you install dependencies. This is normal. After running `uv pip install -e .`, everything will work.

### "Model not loaded"

The API will start even without a trained model. You'll get a 503 error when trying to predict. This is expected. Train a model first or skip model-dependent endpoints.

### "Permission denied" on scripts

```bash
chmod +x scripts/*.sh
```

### GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, training will use CPU (slower but works)
```

---

## 🎯 What Can You Do?

### Without Training (Random Model)
- ✅ API endpoints work
- ✅ Screenshot capture works
- ✅ Action execution works
- ❌ Actions will be random/poor

### With Training
- ✅ Intelligent action prediction
- ✅ Task completion
- ✅ Curriculum progression
- ✅ Production-ready agent

---

## 💡 Tips

1. **Start small**: Test API first, then train on Easy tasks
2. **Monitor training**: Use TensorBoard (`tensorboard --logdir=logs`)
3. **Check logs**: Application logs in `logs/app.log`
4. **GPU memory**: Reduce buffer size if OOM errors occur
5. **Tasks**: Edit `config/tasks.json` to add custom tasks

---

## 🆘 Getting Help

1. Check error logs: `tail -f logs/app.log`
2. Verify setup: `bash scripts/verify_setup.sh`
3. Read docs: `docs/` folder
4. Check examples: `examples.py`

---

**You're all set! 🎉**

The microservice is now running and ready to use. Start with the API documentation to explore endpoints, or jump into training to build your own agent!
