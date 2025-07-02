# CUDA + cuDNN, no framework pre-installed
# The nvidia/cuda image keeps the base slim and lets Poetry pull exactly the wheels you pinned in pyproject.toml, matching your local set-up and avoiding hidden mismatches.
# Note regarding base image: You can always install Python on top of a CUDA image, but adding CUDA to a Python image (e.g. python:3.11-slim) is more complex and error-prone.
# For Azure Machine Learning, AKS, or Azure Container Instances with GPU support, starting from a CUDA image is the recommended approach for GPU workloads.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- system + Python 3.11 ----------
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential git && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip && \
    pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# ---------- runtime dependencies ----------
WORKDIR /app
COPY pyproject.toml poetry.lock ./
# does not install development libraries
RUN poetry install --without dev --no-ansi --no-interaction
# pulls torch-2.3.1+cu121

# ---------- project code ----------
# Scripts run from scripts folder (e.g azure_train_eval.py) will be uploaded to Azure ML on runtime and use the package. Hence scripts should not be included in the docker image
COPY src/ ./src
# Alternatively move the copy line before poetry install. Then the package will be installed as well, but this leads to longer docker build since typically the package code changes more frequently
ENV PYTHONPATH=/app/src

# When I call "python azure_submit_job.py" in the scripts folder the following happens at runtime:

# 1. Azure ML mounts the code= folder (./scripts in your submission) under the working directory inside the container.

# 2. Python starts in that directory and evaluates import ml_segmentation … inside azure_train_eval.py.

# 3. The interpreter looks through its import search path (sys.path). Because of line ②, /app/src is on that list, so the copy of ml_segmentation that you baked into the image (line ①) is found immediately.
