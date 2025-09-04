# CUDA + cuDNN, no framework pre-installed
# The nvidia/cuda image keeps the base slim and lets Poetry pull exactly the wheels you pinned in pyproject.toml, matching your local set-up and avoiding hidden mismatches.
# Note regarding base image: You can always install Python on top of a CUDA image, but adding CUDA to a Python image (e.g. python:3.11-slim) is more complex and error-prone.
# For Azure Machine Learning, AKS, or Azure Container Instances with GPU support, starting from a CUDA image is the recommended approach for GPU workloads.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- system + Python 3.11 ----------
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip build-essential git curl && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip && \
    pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# ---------- project code and dependencies ----------
WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
# Copy source code first since Poetry needs it to install the package
COPY src/ ./src
# does not install development libraries - PyTorch Lightning is included in main dependencies
RUN poetry install --without dev --no-ansi --no-interaction
# pulls torch-2.3.1+cu121 and pytorch-lightning

ENV PYTHONPATH=/app/src

# When I call "python azure_submit_job.py" in the scripts folder the following happens at runtime:

# 1. Azure ML mounts the code= folder (./scripts in your submission) under the working directory inside the container.

# 2. Python starts in that directory and evaluates import ml_segmentation … inside azure_train_eval.py.

# 3. The interpreter looks through its import search path (sys.path). Because of line ②, /app/src is on that list, so the copy of ml_segmentation that you baked into the image (line ①) is found immediately.
