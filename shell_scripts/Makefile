# Makefile for Docker build and test workflow
# Supports local development and Azure ML environment preparation

# Configuration
IMAGE_NAME ?= ml-segmentation-local
TAG ?= latest
FULL_IMAGE_NAME = $(IMAGE_NAME):$(TAG)

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Docker Build and Test Workflow for Azure ML"
	@echo "==========================================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: check-docker
check-docker: ## Check if Docker is running
	@echo "🔹 Checking Docker availability..."
	@docker version > /dev/null 2>&1 || (echo "❌ Docker is not running. Please start Docker Desktop or Docker Engine." && exit 1)
	@echo "✅ Docker is running"

.PHONY: check-files
check-files: ## Validate required project files
	@echo "🔹 Validating project structure..."
	@test -f Dockerfile || (echo "❌ Dockerfile not found" && exit 1)
	@test -f pyproject.toml || (echo "❌ pyproject.toml not found" && exit 1)
	@test -f poetry.lock || (echo "❌ poetry.lock not found" && exit 1)
	@test -d src/ml_segmentation || (echo "❌ src/ml_segmentation not found" && exit 1)
	@echo "✅ Project structure validated"

.PHONY: clean
clean: ## Remove existing Docker image
	@echo "🔹 Removing existing image if present..."
	@docker rmi $(FULL_IMAGE_NAME) 2>/dev/null || echo "No existing image to remove"

.PHONY: build
build: check-docker check-files ## Build Docker image
	@echo "🔹 Building Docker image: $(FULL_IMAGE_NAME)"
	@echo "This may take 5-15 minutes..."
	@docker build -t $(FULL_IMAGE_NAME) -f Dockerfile .
	@echo "✅ Build completed"
	@docker images $(FULL_IMAGE_NAME)

.PHONY: test
test: ## Test the built container
	@echo "🔹 Testing container functionality..."
	@docker run --rm $(FULL_IMAGE_NAME) python --version
	@docker run --rm $(FULL_IMAGE_NAME) python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
	@docker run --rm $(FULL_IMAGE_NAME) python -c "import ml_segmentation; print('Package imported successfully')"
	@docker run --rm $(FULL_IMAGE_NAME) python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
	@echo "✅ All container tests passed"

.PHONY: all
all: build test ## Build and test the Docker image (full workflow)
	@echo ""
	@echo "✅ Local Docker build and test completed successfully!"
	@echo ""
	@echo "Next Steps:"
	@echo "1. Register environment with Azure ML:"
	@echo "   python scripts/azure_manage_data_assets.py build-environment --name my-ml-env --description 'My ML environment'"
	@echo ""
	@echo "2. Submit training job:"
	@echo "   python scripts/azure_submit_job.py --environment my-ml-env"

.PHONY: rebuild
rebuild: clean build test ## Clean, build, and test (full rebuild)

.PHONY: quick-test
quick-test: ## Quick test without full build (assumes image exists)
	@echo "🔹 Quick test of existing image..."
	@docker run --rm $(FULL_IMAGE_NAME) python -c "import ml_segmentation; print('✅ Package working')"

# Development helpers
.PHONY: shell
shell: ## Open interactive shell in the container
	@docker run --rm -it $(FULL_IMAGE_NAME) bash

.PHONY: info
info: ## Show image information
	@echo "Image: $(FULL_IMAGE_NAME)"
	@docker images $(FULL_IMAGE_NAME) 2>/dev/null || echo "Image not built yet. Run 'make build' first."

# Variables for customization
.PHONY: vars
vars: ## Show current variable values
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
	@echo "TAG: $(TAG)"
	@echo "FULL_IMAGE_NAME: $(FULL_IMAGE_NAME)"
