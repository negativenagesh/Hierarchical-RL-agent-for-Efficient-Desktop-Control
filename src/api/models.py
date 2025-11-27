"""Pydantic models for API"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionTypeEnum(str, Enum):
    """Action types"""
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    TYPE = "TYPE"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    EARLY_STOP = "EARLY_STOP"


class TaskDifficultyEnum(str, Enum):
    """Task difficulty levels"""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class InferenceRequest(BaseModel):
    """Request for agent inference"""
    instruction: str = Field(..., description="Task instruction")
    screenshot_base64: Optional[str] = Field(None, description="Base64 encoded screenshot")
    use_live_screen: bool = Field(True, description="Capture live screenshot if not provided")
    deterministic: bool = Field(False, description="Use deterministic policy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Open the calculator application",
                "use_live_screen": True,
                "deterministic": False
            }
        }


class ActionResponse(BaseModel):
    """Response with predicted action"""
    action_type: ActionTypeEnum = Field(..., description="Predicted action type")
    action_type_id: int = Field(..., description="Action type ID")
    coordinates: List[float] = Field(..., description="Normalized coordinates [-1, 1]")
    screen_coordinates: Optional[List[int]] = Field(None, description="Actual screen coordinates")
    confidence: Optional[float] = Field(None, description="Action confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "CLICK",
                "action_type_id": 0,
                "coordinates": [0.5, 0.3],
                "screen_coordinates": [960, 324],
                "confidence": 0.95
            }
        }


class ExecuteRequest(BaseModel):
    """Request to execute an action"""
    action_type: ActionTypeEnum = Field(..., description="Action to execute")
    coordinates: Optional[List[float]] = Field(None, description="Normalized coordinates [-1, 1]")
    text: Optional[str] = Field(None, description="Text to type")
    scroll_amount: int = Field(3, description="Scroll amount")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "CLICK",
                "coordinates": [0.5, 0.3]
            }
        }


class ExecuteResponse(BaseModel):
    """Response from action execution"""
    success: bool = Field(..., description="Execution success")
    message: str = Field(..., description="Execution message")


class TaskRequest(BaseModel):
    """Request to complete a task"""
    instruction: str = Field(..., description="Task instruction")
    max_steps: int = Field(50, description="Maximum steps allowed")
    deterministic: bool = Field(False, description="Use deterministic policy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Open calculator and compute 15 + 27",
                "max_steps": 20,
                "deterministic": False
            }
        }


class TaskResponse(BaseModel):
    """Response from task execution"""
    success: bool = Field(..., description="Task completion success")
    steps_taken: int = Field(..., description="Number of steps taken")
    total_reward: float = Field(..., description="Total reward accumulated")
    actions: List[Dict[str, Any]] = Field(..., description="List of actions taken")
    message: str = Field(..., description="Completion message")


class ModelInfoResponse(BaseModel):
    """Model information"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device (cpu/cuda)")
    parameters: Optional[int] = Field(None, description="Number of parameters")
    checkpoint_path: Optional[str] = Field(None, description="Loaded checkpoint path")


class MetricsResponse(BaseModel):
    """Training/inference metrics"""
    total_inferences: int = Field(..., description="Total inference requests")
    total_executions: int = Field(..., description="Total action executions")
    avg_inference_time: float = Field(..., description="Average inference time (ms)")
    success_rate: Optional[float] = Field(None, description="Success rate")


class TrainingStartRequest(BaseModel):
    """Request to start training"""
    total_timesteps: int = Field(1000000, description="Total training timesteps")
    rollout_steps: int = Field(2048, description="Steps per rollout")
    learning_rate: float = Field(3e-4, description="Learning rate")
    difficulty_filter: Optional[TaskDifficultyEnum] = Field(None, description="Task difficulty filter")
    save_dir: str = Field("checkpoints", description="Checkpoint save directory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_timesteps": 100000,
                "rollout_steps": 2048,
                "learning_rate": 0.0003,
                "difficulty_filter": "EASY"
            }
        }


class TrainingStatusResponse(BaseModel):
    """Training status"""
    is_training: bool = Field(..., description="Whether training is active")
    current_step: Optional[int] = Field(None, description="Current training step")
    total_steps: Optional[int] = Field(None, description="Total training steps")
    success_rate: Optional[float] = Field(None, description="Current success rate")
    mean_reward: Optional[float] = Field(None, description="Mean episode reward")
