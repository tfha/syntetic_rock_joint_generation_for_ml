# CUDA + cuDNN, no framework pre-installed
# The nvidia/cuda image keeps the base slim and lets Poetry pull exactly the wheels you pinned in pyproject.toml, matching your local set-up and avoiding hidden mismatches.
# Note regarding base image: You can always install Python on top of a CUDA image, but adding CUDA to a Python image (e.g. python:3.11-slim) is more complex and error-prone.
# For Azure Machine Learning, AKS, or Azure Container Instances with GPU support, starting from a CUDA image is the recommended approach for GPU workloads.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- system + Python 3.11 ----------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential git curl ca-certificates && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir poetry && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MPLBACKEND=Agg

# ---------- project code and dependencies ----------
WORKDIR /app

# --- dependency layer (no project code yet for better caching) ---
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --without dev --no-root --no-ansi --no-interaction \
    && rm -rf ~/.cache/pip

# --- project source layer ---
COPY src/ ./src
RUN poetry install --without dev --no-ansi --no-interaction \
    && rm -rf ~/.cache/pip

# (Intentionally skip importing the package during build to keep the layer light
#  and avoid triggering heavy native loads early. A runtime smoke test exists in
#  the local build script.)

ENV PYTHONPATH=/app/src

# When I call "python azure_submit_job.py" in the scripts folder the following happens at runtime:

# 1. Azure ML mounts the code= folder (./scripts in your submission) under the working directory inside the container.

# 2. Python starts in that directory and evaluates import ml_segmentation … inside azure_train_eval.py.

# 3. The interpreter looks through its import search path (sys.path). Because of line ②, /app/src is on that list, so the copy of ml_segmentation that you baked into the image (line ①) is found immediately.
