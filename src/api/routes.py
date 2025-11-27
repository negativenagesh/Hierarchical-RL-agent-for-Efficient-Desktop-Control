"""API Routes"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import base64
import io
import numpy as np
from PIL import Image
import torch
import time

from .models import *
from .config import settings
from ..environment.screenshot import ScreenCapture
from ..agent.manager import ActionType

# Router instances
agent_router = APIRouter()
monitoring_router = APIRouter()
training_router = APIRouter()

# Global metrics
metrics = {
    "total_inferences": 0,
    "total_executions": 0,
    "inference_times": [],
    "successes": []
}

# Screen capture instance
screen_capture = None
training_job = None


def get_screen_capture():
    """Get or create screen capture instance"""
    global screen_capture
    if screen_capture is None:
        screen_capture = ScreenCapture(
            monitor_index=1,
            target_width=settings.CAPTURE_WIDTH,
            target_height=settings.CAPTURE_HEIGHT
        )
    return screen_capture


# ==================== Agent Routes ====================

@agent_router.post("/predict", response_model=ActionResponse)
async def predict_action(request: InferenceRequest):
    """
    Predict action for given instruction and screenshot
    """
    from .main import get_model
    
    start_time = time.time()
    
    try:
        model = get_model()
        
        # Get screenshot
        if request.use_live_screen or request.screenshot_base64 is None:
            capture = get_screen_capture()
            screenshot = capture.capture(resize=True)
        else:
            # Decode base64 screenshot
            img_bytes = base64.b64decode(request.screenshot_base64)
            img = Image.open(io.BytesIO(img_bytes))
            screenshot = np.array(img)
        
        # Prepare state
        state_dict = {
            'image': torch.tensor(screenshot, dtype=torch.float32, device=model.device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
            'instruction': [request.instruction],
            'numeric': torch.zeros((1, 10), dtype=torch.float32, device=model.device)
        }
        
        # Get action
        with torch.no_grad():
            action = model.get_action(state_dict, deterministic=request.deterministic)
        
        # Convert to screen coordinates
        worker = model.worker
        screen_coords = worker.denormalize_coordinates(action['coordinates'])
        
        # Update metrics
        inference_time = (time.time() - start_time) * 1000
        metrics["total_inferences"] += 1
        metrics["inference_times"].append(inference_time)
        
        return ActionResponse(
            action_type=ActionTypeEnum(action['action_name']),
            action_type_id=action['action_type'],
            coordinates=list(action['coordinates']),
            screen_coordinates=list(screen_coords)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@agent_router.post("/execute", response_model=ExecuteResponse)
async def execute_action(request: ExecuteRequest):
    """
    Execute a specific action
    """
    from .main import get_model
    
    try:
        model = get_model()
        
        # Convert action type to ID
        action_type_map = {
            "CLICK": 0,
            "DOUBLE_CLICK": 1,
            "RIGHT_CLICK": 2,
            "TYPE": 3,
            "SCROLL": 4,
            "WAIT": 5,
            "EARLY_STOP": 6
        }
        
        action_dict = {
            'action_type': action_type_map[request.action_type],
            'coordinates': tuple(request.coordinates) if request.coordinates else None
        }
        
        # Execute
        success = model.execute_action(
            action_dict,
            text=request.text,
            scroll_amount=request.scroll_amount
        )
        
        # Update metrics
        metrics["total_executions"] += 1
        
        return ExecuteResponse(
            success=success,
            message="Action executed successfully" if success else "Action execution failed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@agent_router.post("/complete-task", response_model=TaskResponse)
async def complete_task(request: TaskRequest):
    """
    Execute a complete task with multiple steps
    """
    from .main import get_model
    
    try:
        model = get_model()
        capture = get_screen_capture()
        
        actions_taken = []
        total_reward = 0
        
        for step in range(request.max_steps):
            # Capture screenshot
            screenshot = capture.capture(resize=True)
            
            # Prepare state
            state_dict = {
                'image': torch.tensor(screenshot, dtype=torch.float32, device=model.device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
                'instruction': [request.instruction],
                'numeric': torch.tensor([[0, step, -1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=model.device)
            }
            
            # Get action
            with torch.no_grad():
                action = model.get_action(state_dict, deterministic=request.deterministic)
            
            # Execute
            success = model.execute_action({
                'action_type': action['action_type'],
                'coordinates': action['coordinates']
            })
            
            # Log action
            actions_taken.append({
                'step': step,
                'action': action['action_name'],
                'coordinates': action['coordinates'],
                'success': success
            })
            
            total_reward += 1.0 if success else -0.5
            
            # Check for early stop
            if action['action_type'] == 6:  # EARLY_STOP
                break
            
            time.sleep(0.1)  # Small delay between actions
        
        task_success = len(actions_taken) < request.max_steps
        
        return TaskResponse(
            success=task_success,
            steps_taken=len(actions_taken),
            total_reward=total_reward,
            actions=actions_taken,
            message=f"Task {'completed' if task_success else 'failed'} in {len(actions_taken)} steps"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@agent_router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    from .main import model_instance
    
    if model_instance is None:
        return ModelInfoResponse(
            model_loaded=False,
            device=settings.DEVICE,
            checkpoint_path=None
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in model_instance.parameters())
    
    return ModelInfoResponse(
        model_loaded=True,
        device=str(model_instance.device),
        parameters=num_params,
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH
    )


# ==================== Monitoring Routes ====================

@monitoring_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    avg_inference_time = np.mean(metrics["inference_times"][-100:]) if metrics["inference_times"] else 0
    success_rate = np.mean(metrics["successes"][-100:]) if metrics["successes"] else None
    
    return MetricsResponse(
        total_inferences=metrics["total_inferences"],
        total_executions=metrics["total_executions"],
        avg_inference_time=avg_inference_time,
        success_rate=success_rate
    )


@monitoring_router.post("/reset-metrics")
async def reset_metrics():
    """Reset metrics"""
    global metrics
    metrics = {
        "total_inferences": 0,
        "total_executions": 0,
        "inference_times": [],
        "successes": []
    }
    return {"message": "Metrics reset successfully"}


# ==================== Training Routes ====================

@training_router.post("/start")
async def start_training(request: TrainingStartRequest, background_tasks: BackgroundTasks):
    """Start training in background"""
    global training_job
    
    if not settings.TRAINING_ENABLED:
        raise HTTPException(status_code=403, detail="Training is disabled")
    
    if training_job is not None and training_job.get("is_running", False):
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # This would start training in background
    # For now, return placeholder response
    return {
        "message": "Training started",
        "total_timesteps": request.total_timesteps,
        "note": "Training implementation requires full environment setup"
    }


@training_router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get training status"""
    global training_job
    
    if training_job is None:
        return TrainingStatusResponse(is_training=False)
    
    return TrainingStatusResponse(
        is_training=training_job.get("is_running", False),
        current_step=training_job.get("current_step"),
        total_steps=training_job.get("total_steps"),
        success_rate=training_job.get("success_rate"),
        mean_reward=training_job.get("mean_reward")
    )


@training_router.post("/stop")
async def stop_training():
    """Stop training"""
    global training_job
    
    if training_job is None or not training_job.get("is_running", False):
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # Stop training
    training_job["is_running"] = False
    
    return {"message": "Training stopped"}
