"""
Install missing Python packages for curated Azure ML environments.

This script is intentionally small and installs only safe, missing packages.
It intentionally avoids reinstalling or upgrading heavy binary packages (numpy,
torch, scipy, opencv, etc.) because replacing them in a curated image can
change ABI and break preinstalled compiled extensions (see numpy 2.x / multiarray errors).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterable

# Packages used by scripts/azure_train_eval.py and other entry scripts.
# Keep this list conservative: avoid heavy binary packages that the curated
# image likely provides. If you need a different binary, prefer building a
# new custom image or environment.
REQUIRED_PACKAGES: list[str] = [
    # config / CLI / structured config
    "hydra-core",
    "omegaconf",
    "pyyaml",
    "pydantic",
    # ML infra & tracking (pure-python parts are useful)
    "mlflow",
    # Azure CLI / SDK pieces (some may be present in curated images)
    "azure-ai-ml",
    # Lightweight helpers (prefer to skip heavy binaries below)
    "torchinfo",
    "torchmetrics",
    # Model helpers that are pure-python wheels or small
    "segmentation-models-pytorch",
    "timm",
    # Data science utils (do not include numpy/scipy here; curated env often has them)
    "pandas",
    "scikit-learn",
    "scikit-image",
    "Pillow",
    "matplotlib",
    "tqdm",
    # misc / config / file formats
    "toml",
    "strictyaml",
    "rich",
]

# Packages that are heavy/binary and can break ABI if replaced in-place.
# We will NOT install these here; if missing, user should prefer a custom image.
HEAVY_BINARY_DISTS: set[str] = {
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "torchaudio",
    "opencv-python",
    "opencv_python",
}

# Map distribution name -> importable module name
_IMPORT_NAME_MAP: dict[str, str] = {
    "pyyaml": "yaml",
    "Pillow": "PIL",
    "opencv-python": "cv2",
    "opencv_python": "cv2",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "torchinfo": "torchinfo",
    "strictyaml": "strictyaml",
    "timm": "timm",
    "torchmetrics": "torchmetrics",
    "azure-ai-ml": "azure.ai.ml",
}


def _import_name_for_pkg(dist: str) -> str:
    return _IMPORT_NAME_MAP.get(
        dist,
        dist.split("==")[0]
        .split(">=")[0]
        .split("<=")[0]
        .split("~=")[0]
        .split(">")[0]
        .split("<")[0]
        .replace("-", "_"),
    )


def _print(msg: str) -> None:
    print(msg, flush=True)


def _is_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _filter_required(pkgs: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Return (to_install, skipped_heavy). 'to_install' contains safe packages that
    are not importable. 'skipped_heavy' lists heavy packages that would be
    installed but are intentionally skipped.
    """
    to_install: list[str] = []
    skipped_heavy: list[str] = []
    for dist in pkgs:
        # If distribution itself is heavy binary, skip it explicitly
        if dist in HEAVY_BINARY_DISTS:
            skipped_heavy.append(dist)
            continue

        import_name = _import_name_for_pkg(dist)
        if _is_importable(import_name):
            _print(f"Already importable: {import_name} (skipping '{dist}')")
        else:
            to_install.append(dist)
    return to_install, skipped_heavy


def main() -> None:
    _print("Preparing to ensure required packages are importable...")

    # Upgrade pip tools only (safe). This helps wheel handling but won't force binary changes.
    try:
        _print("Upgrading pip, setuptools, wheel (no other packages)...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            check=True,
        )
    except Exception as e:
        _print(f"Warning: pip upgrade failed: {e!r} -- continuing")

    to_install, skipped_heavy = _filter_required(REQUIRED_PACKAGES)

    if skipped_heavy:
        _print("")
        _print("NOTE: The installer intentionally skipped these heavy binary packages:")
        for h in skipped_heavy:
            _print(f"  - {h}")
        _print("If any of these are missing or need a different ABI, prefer creating a")
        _print("custom environment/image rather than installing them at runtime.")
        _print("Installing or upgrading heavy binaries in-place can downgrade/upgrade")
        _print("numpy or other compiled extensions and cause errors such as:")
        _print("  'ImportError: numpy.core.multiarray failed to import' or")
        _print(
            "  'A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x'."
        )
        _print("If you must change them, do it in a controlled environment build step.")

    if not to_install:
        _print(
            "No safe packages to install; environment looks good for pure-python dependencies."
        )
        return

    # Conservative set of packages known to often pull binary dependencies (numpy/scipy/torch).
    # We will install these with --no-deps to avoid upgrading system binaries at runtime.
    BINARY_RISK_PKGS: set[str] = {
        "segmentation-models-pytorch",
        "scikit-learn",
        "scikit-image",
        "timm",
        "azure-ai-ml",
    }

    pure_install: list[str] = []
    no_deps_install: list[str] = []
    for pkg in to_install:
        if pkg in BINARY_RISK_PKGS:
            no_deps_install.append(pkg)
        else:
            pure_install.append(pkg)

    if pure_install:
        _print(f"Installing pure-Python safe packages: {pure_install}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *pure_install], check=True
            )
            _print("Pure-Python installation complete.")
        except subprocess.CalledProcessError as e:
            _print(
                f"pip install failed for pure-Python packages with exit {e.returncode}; inspect logs."
            )
            raise

    if no_deps_install:
        _print(
            "The following packages may pull binary dependencies. Installing them with --no-deps to avoid "
            "upgrading compiled libraries (numpy/scipy/torch) at runtime:"
        )
        for p in no_deps_install:
            _print(f"  - {p}")
        _print(
            "If these packages require compiled deps not present in the image, prefer building a custom image "
            "or environment that includes the correct binary versions."
        )
        # Install with --no-deps to avoid triggering dependency resolution that may upgrade numpy/scipy/etc.
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", *no_deps_install],
                check=True,
            )
            _print("Installation of binary-risk packages (no-deps) complete.")
        except subprocess.CalledProcessError as e:
            _print(
                f"pip install --no-deps failed for {no_deps_install} with exit {e.returncode}; inspect logs."
            )
            raise


if __name__ == "__main__":
    main()
