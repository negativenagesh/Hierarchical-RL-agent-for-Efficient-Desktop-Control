# Makefile for Hierarchical RL Agent

.PHONY: help install run-api train test clean docker-build docker-up

help:
	@echo "Hierarchical RL Agent - Available commands:"
	@echo "  make install      - Install dependencies with UV"
	@echo "  make run-api      - Start FastAPI server"
	@echo "  make train        - Start training"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start Docker Compose stack"

install:
	uv pip install -e .
	uv pip install -e ".[dev]"

run-api:
	bash scripts/run_api.sh

train:
	bash scripts/train.sh

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ **/__pycache__ .pytest_cache
	rm -rf logs/* checkpoints/*
	rm -rf build/ dist/ *.egg-info

docker-build:
	docker build -t hierarchical-rl-agent -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
