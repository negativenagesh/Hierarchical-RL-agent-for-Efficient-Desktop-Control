#!/bin/bash

# Start the FastAPI microservice

set -e

echo "Starting Hierarchical RL Agent API..."

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default parameters
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${WORKERS:-1}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"

# Run uvicorn
uvicorn src.api.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --reload

echo "API started!"
