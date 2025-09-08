#!/usr/bin/env python3
"""
Install missing packages not included in Azure ML curated environment.

This script installs packages required for the rock joint segmentation project
that are not included in the Azure ML curated environment 'acpt-pytorch-2.2-cuda12.1'.

The curated environment includes:
- PyTorch 2.2 with CUDA 12.1
- Basic scientific computing packages (numpy, scipy, pandas, matplotlib, etc.)
- Standard Python libraries

This script installs the additional packages we need from curated_env_requirements.txt.
"""

import importlib.util  # added
import subprocess
import sys
from pathlib import Path


def install_from_requirements_file() -> None:
    """Install packages from curated_env_requirements.txt file."""
    requirements_file = Path(__file__).parent / "curated_env_requirements.txt"

    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    print(f"📦 Installing packages from {requirements_file.name}...")

    try:
        # Use pip install with requirements file
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            text=True,
        )
        print("✅ Successfully installed all required packages!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("This might be due to package conflicts or network issues.")
        print("Trying to install packages individually...")

        # Fallback: install packages one by one
        install_packages_individually(requirements_file)


def install_packages_individually(requirements_file: Path) -> None:
    """Install packages one by one from requirements file."""
    with open(requirements_file) as f:
        lines = f.readlines()

    packages = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            packages.append(line)

    failed_packages = []
    for pkg in packages:
        try:
            print(f"Installing {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Successfully installed {pkg}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {pkg}")
            failed_packages.append(pkg)

    if failed_packages:
        print(f"\n⚠️  Failed to install {len(failed_packages)} packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\nContinuing with available packages...")
    else:
        print(f"\n✅ Successfully installed all {len(packages)} packages!")


def install_local_package(editable: bool = True) -> None:
    """Install the local ml_segmentation package if not already importable."""
    pkg_name = "ml_segmentation"
    if importlib.util.find_spec(pkg_name) is not None:
        print(f"✅ Local package '{pkg_name}' already importable (skip install)")
        return

    project_root = Path(__file__).resolve().parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("⚠️  pyproject.toml not found; cannot install local package")
        return

    print(f"📦 Installing local package '{pkg_name}' from source...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e" if editable else ".",
    ]
    if editable:
        cmd.append(project_root.as_posix())
    else:
        cmd[-1] = project_root.as_posix()

    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"✅ Installed local package '{pkg_name}'")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install local package '{pkg_name}': {e}")
        print("    Check pyproject.toml or dependency conflicts.")


def main():
    """Main entry point."""
    print("🔍 Installing additional packages for Azure ML curated environment...")
    print("   Curated environment: acpt-pytorch-2.2-cuda12.1")

    try:
        install_from_requirements_file()
        # NEW: install the local package
        install_local_package(editable=True)
        print("\n🚀 Package installation completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
