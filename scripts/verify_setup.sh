#!/bin/bash

# Setup verification script for Hierarchical RL Agent

echo "========================================="
echo "Hierarchical RL Agent - Setup Verification"
echo "========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# Check UV installation
echo "2. Checking UV installation..."
if command -v uv &> /dev/null; then
    echo "✓ UV is installed"
    uv --version
else
    echo "✗ UV is not installed"
    echo "  Install with: pip install uv"
fi
echo ""

# Check directory structure
echo "3. Checking directory structure..."
required_dirs=("src" "src/agent" "src/api" "src/environment" "src/training" "src/utils" "config" "docker" "docs" "scripts" "tests")
all_present=true

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "✗ $dir (missing)"
        all_present=false
    fi
done
echo ""

# Check key files
echo "4. Checking key files..."
required_files=(
    "pyproject.toml"
    "README.md"
    "Makefile"
    ".env.example"
    "src/agent/encoder.py"
    "src/agent/manager.py"
    "src/agent/worker.py"
    "src/agent/policy.py"
    "src/training/ppo_trainer.py"
    "src/api/main.py"
    "config/config.yaml"
    "config/tasks.json"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
    fi
done
echo ""

# Count Python files
echo "5. Project statistics..."
py_files=$(find . -name "*.py" -not -path "./.venv/*" | wc -l)
echo "  Python files: $py_files"
echo "  Lines of code: $(find . -name "*.py" -not -path "./.venv/*" -exec cat {} \; | wc -l)"
echo ""

# Check if .env exists
echo "6. Configuration check..."
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "✗ .env file missing"
    echo "  Copy from: cp .env.example .env"
fi
echo ""

# Summary
echo "========================================="
echo "Setup Summary"
echo "========================================="
echo ""
if [ "$all_present" = true ]; then
    echo "✓ All directories present"
else
    echo "✗ Some directories missing"
fi
echo ""
echo "Next steps:"
echo "  1. Install dependencies: uv pip install -e ."
echo "  2. Configure .env: cp .env.example .env && nano .env"
echo "  3. Run API: make run-api"
echo "  4. Start training: make train"
echo "  5. Run tests: make test"
echo ""
echo "Documentation:"
echo "  - README.md - Project overview"
echo "  - docs/API.md - API documentation"
echo "  - docs/TRAINING.md - Training guide"
echo "  - docs/PROJECT_STRUCTURE.md - Structure details"
echo ""
echo "========================================="
