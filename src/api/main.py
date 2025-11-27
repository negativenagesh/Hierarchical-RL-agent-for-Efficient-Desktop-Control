"""FastAPI Application"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
from pathlib import Path

from .routes import agent_router, monitoring_router, training_router
from .config import settings
from ..agent.policy import HierarchicalPolicy
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Global model instance
model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model_instance
    
    # Startup
    logger.info("Starting Hierarchical RL Agent API")
    logger.info(f"Model checkpoint: {settings.MODEL_CHECKPOINT_PATH}")
    
    # Load model if checkpoint exists
    if Path(settings.MODEL_CHECKPOINT_PATH).exists():
        try:
            device = torch.device(settings.DEVICE)
            model_instance = HierarchicalPolicy(
                screen_width=settings.SCREEN_WIDTH,
                screen_height=settings.SCREEN_HEIGHT
            ).to(device)
            model_instance.load(settings.MODEL_CHECKPOINT_PATH, device)
            model_instance.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model_instance = None
    else:
        logger.warning("No model checkpoint found. API will run without loaded model.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Hierarchical RL Agent API",
    description="REST API for OS Control using Hierarchical Reinforcement Learning",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hierarchical RL Agent API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": model_instance is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": settings.DEVICE
    }


# Include routers
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
app.include_router(training_router, prefix="/api/v1/training", tags=["Training"])


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def get_model():
    """Get the global model instance"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance
