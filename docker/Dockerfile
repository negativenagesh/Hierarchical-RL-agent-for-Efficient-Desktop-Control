# Hierarchical-RL-agent-for-Efficient-OS-Control

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    libxi-dev \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

# Install dependencies using UV
RUN uv pip install --system -e .

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV MODEL_CHECKPOINT_PATH=/app/checkpoints/final_model.pt

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
