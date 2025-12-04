## Base image: Azure ML curated PyTorch image (stable CUDA, torch, numpy, python)
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:42

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Keep pip tooling modern but avoid upgrading core scientific stack beyond curated pins
RUN python -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# Core compatibility pins to match curated stack and avoid ABI churn
# - numpy/scipy pinned to versions compatible with AML dataset/runtime and PyTorch 2.2
# - marshmallow pinned <4 due to azure-ai-ml constraint
# - azure-ai-ml pinned to a version known to work with marshmallow 3.x and its storage deps
RUN python -m pip install \
    "numpy==1.23.5" \
    "scipy==1.10.1" \
    "marshmallow==3.26.1" \
    "azure-ai-ml==1.29.0" \
    "azure-storage-file-datalake==12.21.0" \
    "azure-storage-file-share==12.22.0" \
    "azure-monitor-opentelemetry==1.8.1"

# Training-time libraries (pure Python or compatible with the pinned stack)
# Note: segmentation-models-pytorch and timm rely on torch already in curated image.
RUN python -m pip install \
    "segmentation-models-pytorch==0.5.0" \
    "timm==1.0.22" \
    "torchmetrics==1.6.0" \
    "torchinfo==1.8.0" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "mlflow==2.18.0" \
    "pandas==2.2.3" \
    "pillow==11.0.0" \
    "matplotlib==3.9.2" \
    "opencv-python==4.10.0.84" \
    "colorama==0.4.6" \
    "rich==13.4.2" \
    "toml==0.10.2"

# Only include scikit-learn / scikit-image if your pipeline truly needs them
# Versions below keep compatibility with numpy==1.23.5
RUN python -m pip install \
    "scikit-learn==1.7.2" \
    "scikit-image==0.24.0"

WORKDIR /workspace
COPY . /workspace

# If you need a specific torch wheel different from curated, install here (otherwise rely on curated)
# RUN python -m pip install "torch==2.3.1+cu121" --find-links /wheels

# Install the project in editable mode (pyproject.toml at repo root). Use --no-deps: all deps are pinned above.
RUN python -m pip install --no-deps -e /workspace

# Smoke tests during build to catch ABI/import issues early
RUN python - <<'PY'
import faulthandler, importlib, sys
faulthandler.enable()
checks = (
    "numpy","scipy","torch","matplotlib","pandas",
    "segmentation_models_pytorch","timm",
    "azure.ai.ml","marshmallow","cv2",
)
for m in checks:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print("IMPORT-ERROR", m, e)
        sys.exit(1)
print("SMOKE-OK")
PY
