"""
Install missing Python packages for curated Azure ML environments.

This script is intentionally small and uses the current Python interpreter's pip
so packages are installed into the active environment (works in Azure curated
environments where a minimal image is used).
"""

import subprocess
import sys

# Packages used by scripts/azure_train_eval.py and other entry scripts.
REQUIRED_PACKAGES: list[str] = [
    "hydra-core",
    "omegaconf",
    "pyyaml",
    "pydantic",
    "mlflow",
    "azure-ai-ml",
    "torch",
    "torchvision",
    "torchaudio",
    "tensorboard",
    "torchinfo",
    "torchmetrics",
    "segmentation-models-pytorch",
    "timm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "opencv-python",
    "Pillow",
    "matplotlib",
    "tqdm",
    "toml",
    "strictyaml",
    "rich",
]


def main() -> None:
    cmd = [sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
