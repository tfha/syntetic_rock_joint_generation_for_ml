# Start from the Azure ML curated image (explicit curated URI)
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:42

# Keep pip tooling modern but avoid runtime ABI churn
RUN python -m pip install --upgrade pip setuptools wheel

# Pin high-risk binaries to avoid runtime ABI upgrades. Adjust versions if you have validated different pins.
RUN python -m pip install --no-cache-dir \
    "numpy==1.23.5" \
    "scipy==1.10.1" \
    "marshmallow==3.26.1" \
    "azure-ai-ml==1.29.0" \
    "azure-storage-file-datalake==12.21.0" \
    "azure-storage-file-share==12.22.0" \
    "azure-monitor-opentelemetry==1.8.1" \
    "scikit-learn==1.7.2" \
    "scikit-image==0.24.0" \
    "colorama==0.4.6" \
    "rich==13.4.2" \
    "hydra-core>=1.3,<2.0" \
    "toml>=0.10.2"

WORKDIR /workspace
COPY . /workspace

# If the curated base doesn't already contain the exact torch wheel you need,
# install it here explicitly (replace with your validated wheel or package spec).
# Example (uncomment/adjust if you have a wheel or index):
# RUN python -m pip install --no-cache-dir "torch==2.3.1+cu121" --find-links /wheels

# Install the project in editable mode (pyproject.toml is at the repo root)
RUN python -m pip install --no-deps -e /workspace

# Smoke tests during build to catch ABI/import issues early
RUN python - <<'PY'
import faulthandler, importlib, sys
faulthandler.enable()
checks = ("numpy","scipy","matplotlib","azure.ai.ml","marshmallow","torch")
for m in checks:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print("IMPORT-ERROR", m, e)
        sys.exit(1)
print("SMOKE-OK")
PY
