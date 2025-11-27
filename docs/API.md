# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required. Future versions will support API keys.

## Endpoints

### Health Check

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### Agent Endpoints

#### POST /api/v1/agent/predict
Predict action for given instruction and screenshot.

**Request Body:**
```json
{
  "instruction": "Open the calculator application",
  "screenshot_base64": null,
  "use_live_screen": true,
  "deterministic": false
}
```

**Response:**
```json
{
  "action_type": "CLICK",
  "action_type_id": 0,
  "coordinates": [0.5, 0.3],
  "screen_coordinates": [960, 324],
  "confidence": 0.95
}
```

#### POST /api/v1/agent/execute
Execute a specific action on the OS.

**Request Body:**
```json
{
  "action_type": "CLICK",
  "coordinates": [0.5, 0.3],
  "text": null,
  "scroll_amount": 3
}
```

**Response:**
```json
{
  "success": true,
  "message": "Action executed successfully"
}
```

#### POST /api/v1/agent/complete-task
Execute a complete multi-step task.

**Request Body:**
```json
{
  "instruction": "Open calculator and compute 15 + 27",
  "max_steps": 20,
  "deterministic": false
}
```

**Response:**
```json
{
  "success": true,
  "steps_taken": 12,
  "total_reward": 8.5,
  "actions": [
    {
      "step": 0,
      "action": "CLICK",
      "coordinates": [0.1, 0.9],
      "success": true
    }
  ],
  "message": "Task completed in 12 steps"
}
```

#### GET /api/v1/agent/model-info
Get information about the loaded model.

**Response:**
```json
{
  "model_loaded": true,
  "device": "cuda",
  "parameters": 45231892,
  "checkpoint_path": "checkpoints/final_model.pt"
}
```

---

### Monitoring Endpoints

#### GET /api/v1/monitoring/metrics
Get system metrics.

**Response:**
```json
{
  "total_inferences": 1542,
  "total_executions": 1398,
  "avg_inference_time": 47.3,
  "success_rate": 0.78
}
```

#### POST /api/v1/monitoring/reset-metrics
Reset all metrics to zero.

**Response:**
```json
{
  "message": "Metrics reset successfully"
}
```

---

### Training Endpoints

#### POST /api/v1/training/start
Start training in background.

**Request Body:**
```json
{
  "total_timesteps": 100000,
  "rollout_steps": 2048,
  "learning_rate": 0.0003,
  "difficulty_filter": "EASY",
  "save_dir": "checkpoints"
}
```

**Response:**
```json
{
  "message": "Training started",
  "total_timesteps": 100000,
  "note": "Training implementation requires full environment setup"
}
```

#### GET /api/v1/training/status
Get current training status.

**Response:**
```json
{
  "is_training": true,
  "current_step": 45632,
  "total_steps": 100000,
  "success_rate": 0.65,
  "mean_reward": 12.4
}
```

#### POST /api/v1/training/stop
Stop ongoing training.

**Response:**
```json
{
  "message": "Training stopped"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

---

## Python Client Example

```python
import requests

class HierarchicalRLClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, instruction, deterministic=False):
        """Predict action for instruction"""
        response = requests.post(
            f"{self.base_url}/api/v1/agent/predict",
            json={
                "instruction": instruction,
                "use_live_screen": True,
                "deterministic": deterministic
            }
        )
        return response.json()
    
    def execute_task(self, instruction, max_steps=50):
        """Execute complete task"""
        response = requests.post(
            f"{self.base_url}/api/v1/agent/complete-task",
            json={
                "instruction": instruction,
                "max_steps": max_steps
            }
        )
        return response.json()
    
    def get_metrics(self):
        """Get system metrics"""
        response = requests.get(
            f"{self.base_url}/api/v1/monitoring/metrics"
        )
        return response.json()

# Usage
client = HierarchicalRLClient()
action = client.predict("Open calculator")
print(f"Predicted: {action['action_type']}")
```
