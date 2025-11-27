# Project Structure Overview

## Complete Directory Structure

```
Hierarchical-RL-agent-for-Efficient-OS-Control/
│
├── src/                              # Main source code
│   ├── agent/                        # RL agent components
│   │   ├── __init__.py
│   │   ├── encoder.py               # Triple-modal state encoder
│   │   ├── manager.py               # High-level Manager policy
│   │   ├── worker.py                # Low-level Worker execution
│   │   └── policy.py                # Hierarchical policy wrapper
│   │
│   ├── environment/                  # OS environment interfaces
│   │   ├── __init__.py
│   │   ├── base_env.py              # Base environment class
│   │   ├── screenshot.py            # Fast screenshot capture (MSS)
│   │   └── osworld_wrapper.py       # OSWorld benchmark integration
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   ├── ppo_trainer.py           # PPO trainer implementation
│   │   ├── replay_buffer.py         # Experience replay buffer
│   │   ├── curriculum.py            # Curriculum learning manager
│   │   └── train.py                 # Training script
│   │
│   ├── api/                          # FastAPI microservice
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   ├── models.py                # Pydantic models
│   │   └── config.py                # API configuration
│   │
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py                # Loguru logging setup
│   │   └── metrics.py               # Prometheus metrics
│   │
│   └── __init__.py
│
├── config/                           # Configuration files
│   ├── config.yaml                  # Main configuration
│   └── tasks.json                   # Task definitions
│
├── docker/                           # Docker setup
│   ├── Dockerfile                   # Main Dockerfile
│   ├── docker-compose.yml           # Multi-service composition
│   └── prometheus.yml               # Prometheus config
│
├── docs/                             # Documentation
│   ├── API.md                       # API documentation
│   └── TRAINING.md                  # Training guide
│
├── scripts/                          # Utility scripts
│   ├── train.sh                     # Training script
│   └── run_api.sh                   # API startup script
│
├── tests/                            # Unit tests
│   └── test_agent.py                # Agent module tests
│
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
├── Makefile                          # Build automation
├── README.md                         # Project README
├── pyproject.toml                    # UV dependencies
├── main.py                           # Entry point (optional)
└── examples.py                       # Usage examples
```

## Key File Descriptions

### Agent Components (`src/agent/`)

**encoder.py** (496 lines)
- `VisualEncoder`: EfficientNet-B0 for screenshot encoding
- `TextEncoder`: BERT-tiny for instruction encoding
- `NumericStateEncoder`: MLP for task metadata
- `TripleModalEncoder`: Combines all three modalities
- Output: 512-dim state embedding

**manager.py** (217 lines)
- `ManagerPolicy`: High-level decision making
- Outputs: Action type (7 discrete) + Coordinates (2D continuous)
- Uses hybrid action space (Categorical + Normal distributions)
- Includes value head for PPO

**worker.py** (216 lines)
- `WorkerPolicy`: Learnable low-level policy (optional)
- `HardcodedWorker`: PyAutoGUI-based execution
- Handles: Click, DoubleClick, RightClick, Type, Scroll, Wait
- Coordinate denormalization [-1,1] → screen pixels

**policy.py** (194 lines)
- `HierarchicalPolicy`: Complete agent wrapper
- Combines encoder + manager + worker
- High-level API for inference and execution
- Model save/load functionality

### Environment (`src/environment/`)

**base_env.py** (232 lines)
- `OSEnvironment`: Gymnasium-compatible base class
- Observation space: Screenshot + Instruction + Metadata
- Action space: Hybrid (discrete type + continuous params)
- Reward shaping: Step penalty + Success/failure bonuses

**screenshot.py** (132 lines)
- `ScreenCapture`: Fast MSS-based capture
- Real-time screenshot at 640x480 (configurable)
- Region capture support
- Preprocessing for model input

**osworld_wrapper.py** (209 lines)
- `OSWorldEnv`: OSWorld benchmark wrapper
- Task loading from JSON
- Difficulty filtering for curriculum
- Success criteria checking (placeholder)

### Training (`src/training/`)

**ppo_trainer.py** (368 lines)
- `PPOTrainer`: Complete PPO implementation
- Features: GAE, clipped objective, value loss
- Rollout collection and policy updates
- Tensorboard logging

**replay_buffer.py** (117 lines)
- `ReplayBuffer`: Experience storage
- GAE advantage computation
- Advantage normalization

**curriculum.py** (147 lines)
- `CurriculumManager`: Staged learning
- Auto-progression based on success rate
- 3 difficulty stages (Easy → Medium → Hard)

**train.py** (70 lines)
- Main training script with CLI arguments
- Environment and trainer setup
- Training loop execution

### API (`src/api/`)

**main.py** (123 lines)
- FastAPI application setup
- Model loading on startup
- CORS middleware
- Health check endpoints

**routes.py** (285 lines)
- `/api/v1/agent/*`: Agent control endpoints
- `/api/v1/monitoring/*`: Metrics endpoints
- `/api/v1/training/*`: Training control
- Request/response handling

**models.py** (118 lines)
- Pydantic models for validation
- Request/response schemas
- Type safety and documentation

**config.py** (43 lines)
- `Settings`: Application configuration
- Environment variable loading
- Default values

### Utilities (`src/utils/`)

**logger.py** (54 lines)
- Loguru-based logging
- Console + file handlers
- Error log separation

**metrics.py** (145 lines)
- Prometheus metrics
- Counter, Gauge, Histogram
- Training and inference tracking

## Installation & Setup

### 1. Install Dependencies (UV)

```bash
# Clone repository
git clone https://github.com/negativenagesh/Hierarchical-RL-agent-for-Efficient-OS-Control.git
cd Hierarchical-RL-agent-for-Efficient-OS-Control

# Install with UV
uv pip install -e .
uv pip install -e ".[dev]"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit as needed
nano .env
```

### 3. Run API

```bash
# Using Makefile
make run-api

# Or directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start Training

```bash
# Using Makefile
make train

# Or with parameters
TOTAL_STEPS=100000 DIFFICULTY=EASY make train

# Or directly
python src/training/train.py --total-timesteps 100000 --device cuda
```

### 5. Run Tests

```bash
make test
# or
pytest tests/ -v
```

## Docker Deployment

```bash
# Build image
make docker-build

# Start all services (API + Redis + Prometheus + Grafana)
make docker-up

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## API Usage Examples

### Python Client

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

# Execute task
response = requests.post(
    "http://localhost:8000/api/v1/agent/complete-task",
    json={
        "instruction": "Create file test.txt",
        "max_steps": 20
    }
)
result = response.json()
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict action
curl -X POST http://localhost:8000/api/v1/agent/predict \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Open calculator", "use_live_screen": true}'

# Get metrics
curl http://localhost:8000/api/v1/monitoring/metrics
```

## Development Workflow

1. **Make changes** to source code
2. **Run tests**: `make test`
3. **Test API**: `make run-api` and test endpoints
4. **Train model**: `make train` with desired config
5. **Monitor**: Check Tensorboard/Grafana
6. **Deploy**: Build and run Docker container

## Performance Optimization

- **GPU Training**: 10-20x faster than CPU
- **Batch Size**: Larger = more stable but slower
- **Rollout Steps**: 2048 default, 4096 for 24GB VRAM
- **Workers**: Parallel environment collection
- **Mixed Precision**: Use torch.cuda.amp

## Monitoring Stack

1. **Tensorboard** (`logs/`): Training metrics
2. **Prometheus** (`:9090`): System metrics
3. **Grafana** (`:3000`): Visualization
4. **Loguru** (`logs/`): Application logs

## Next Steps

1. ✅ Complete microservice implementation
2. ⏳ Integrate full OSWorld benchmark
3. ⏳ Train on diverse task set
4. ⏳ Deploy pre-trained models
5. ⏳ Build web dashboard
6. ⏳ Add multi-monitor support
