"""
Azure ML environment and dependency management utilities.

This module handles environment setup, Poetry/Conda exports,
and Azure ML environment configuration.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

import toml
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from rich.console import Console


def export_poetry_to_environment_yml(
    output_file: str = "environment.yml",
    default_python_version: str = "3.10",
    exclude_pytorch_packages: bool = True,
) -> str:
    """
    Export Poetry dependencies to environment.yml format for Azure ML,
    prioritizing conda packages over pip packages where possible.

    When using PyTorch base images in Azure ML, this function can automatically
    exclude core PyTorch packages to prevent version conflicts. Excluded packages
    include:
    - torch, torchvision, torchaudio, pytorch (core PyTorch)
    - NVIDIA CUDA runtime libraries and tools
    - MAGMA and Triton (PyTorch dependencies)

    PyTorch ecosystem packages like torchinfo, torchmetrics,
    segmentation-models-pytorch, timm, etc. are NOT excluded as they are typically
    not pre-installed in base images.

    Args:
        output_file (str): The path where the environment.yml file will be saved.
        default_python_version (str): Default Python version to use if detection fails.
        exclude_pytorch_packages (bool): Whether to exclude PyTorch-related packages
            (useful when using PyTorch base images in Azure ML).

    Returns:
        str: The path to the created environment.yml file.
    """
    print("Exporting Poetry environment to environment.yml...")

    # Get dependency information
    try:
        result = subprocess.run(
            ["poetry", "export", "--format=requirements.txt", "--without-hashes"],
            capture_output=True,
            text=True,
            check=True,
        )
        requirements_raw = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running poetry export: {e}")
        print("Make sure Poetry is installed and you're in a Poetry project directory")
        return ""

    # Get pyproject.toml content to analyze sources
    pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")
    custom_sources = {}
    cuda_packages = set()

    if os.path.exists(pyproject_path):
        with open(pyproject_path) as f:
            pyproject_data = toml.load(f)

        # Check for custom package sources (like PyTorch index)
        sources = pyproject_data.get("tool", {}).get("poetry", {}).get("source", [])
        for source in sources:
            if source.get("name") == "pytorch":
                custom_sources["pytorch"] = source.get("url", "")

        # Identify CUDA packages from dependencies
        poetry_tool = pyproject_data.get("tool", {})
        poetry_config = poetry_tool.get("poetry", {})
        poetry_dependencies = poetry_config.get("dependencies", {})
        for dep_name in poetry_dependencies:
            if "cuda" in dep_name.lower() or "torch" in dep_name.lower():
                cuda_packages.add(dep_name)

    # Clean up and parse requirements
    requirements = []
    for req in requirements_raw:
        if req.strip() and not req.startswith("#"):
            requirements.append(req.strip())

    # Use the default Python version for Azure ML compatibility
    # Azure ML PyTorch base images use Python 3.10
    python_version = default_python_version
    print(f"Using Python version {python_version} for Azure ML compatibility")

    # Define common packages that should be installed via conda
    # This list can be expanded based on project needs
    conda_preferred_packages = {
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pillow",
        "pyyaml",
        "requests",
        "tqdm",
        "jupyter",
        "ipython",
        "notebook",
        "seaborn",
        "plotly",
        "pytest",
        "flake8",
        "black",
        "isort",
        "mypy",
        "tensorboard",
        "opencv",
        "hydra-core",
        "rich",
    }

    # Separate conda and pip packages
    conda_packages = [f"python={python_version}", "pip"]
    pip_only_packages = []

    # Define platform-specific packages that should be excluded from Linux environments
    windows_only_packages = {
        "pywin32",
        "wmi",
        "pypiwin32",
        "windows-curses",
    }

    # Define PyTorch packages to exclude when using base images
    pytorch_core_packages = {
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch",
        "torchtext",
        "pytorch-lightning",
        "lightning",
    }

    # CUDA and related packages that come with PyTorch base images
    cuda_related_packages = {
        "nvidia-cublas-cu11",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu11",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu11",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu11",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu11",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu11",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu11",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu11",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu11",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu11",
        "nvidia-nccl-cu12",
        "nvidia-nvtx-cu11",
        "nvidia-nvtx-cu12",
        "triton",
        "nvidia-ml-py",
    }

    # Process each requirement
    for req in requirements:
        if not req or req.startswith("-"):
            continue

        # Extract package name (handle different requirement formats)
        package_name = (
            req.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
        )

        # Skip platform-specific packages
        if package_name.lower() in windows_only_packages:
            print(f"Skipping Windows-specific package: {package_name}")
            continue

        # Skip PyTorch packages if requested
        if exclude_pytorch_packages and package_name.lower() in pytorch_core_packages:
            print(
                f"Excluding PyTorch core package (assuming base image): {package_name}"
            )
            continue

        # Skip CUDA packages if requested
        if exclude_pytorch_packages and package_name.lower() in cuda_related_packages:
            print(f"Excluding CUDA package (assuming base image): {package_name}")
            continue

        # Prefer conda for known packages
        if package_name.lower().replace("-", "_") in conda_preferred_packages:
            conda_packages.append(req)
        else:
            pip_only_packages.append(req)

    # Create a dependencies list that may include strings and a pip section dict
    env_dependencies: list[str | dict[str, list[str]]] = list(conda_packages)

    # Create environment.yml structure
    env_data: dict[str, Any] = {
        "name": "ml-segmentation",
        "channels": ["conda-forge", "defaults"],
        "dependencies": env_dependencies,
    }

    # Add pip dependencies if any
    if pip_only_packages:
        env_dependencies.append({"pip": pip_only_packages})

    # Write environment.yml
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        yaml.dump(env_data, f, default_flow_style=False, sort_keys=False)

    print(f"Environment exported to {output_path}")
    print(f"Conda packages: {len(conda_packages)}")
    print(f"Pip packages: {len(pip_only_packages)}")

    if exclude_pytorch_packages:
        print("Note: PyTorch core packages excluded (assuming PyTorch base image)")

    return str(output_path)


def test_environment_export(output_path: str | None = None) -> None:
    """
    Test the environment export functionality.

    Args:
        output_path: Optional path for the test output file
    """
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        print("No pyproject.toml found. Please run this in a Poetry project directory.")
        return

    # Run the export function
    if output_path is None:
        output_path = "test_environment.yml"

    try:
        result_path = export_poetry_to_environment_yml(
            output_file=output_path,
            exclude_pytorch_packages=True,
        )

        if result_path and os.path.exists(result_path):
            print(f"✓ Environment export test successful: {result_path}")

            # Show first few lines
            with open(result_path) as f:
                lines = f.readlines()[:10]
                print("Preview:")
                for line in lines:
                    print(f"  {line.rstrip()}")
        else:
            print("✗ Environment export test failed")

    except Exception as e:
        print(f"✗ Environment export test failed: {e}")


def build_and_register_environment(
    ml_client: MLClient,
    console: Console,
    environment_name: str,
    dockerfile_path: str = "./Dockerfile",
    context_path: str = "./",
) -> Environment:
    """
    Build and register a custom environment in Azure ML.

    Args:
        ml_client: Azure ML client
        console: Console for logging
        environment_name: Name for the environment
        dockerfile_path: Path to the Dockerfile
        context_path: Build context path

    Returns:
        Azure ML Environment object
    """
    try:
        console.print(f"Building environment: {environment_name}", style="info")

        # Create environment with build context
        env = Environment(
            name=environment_name,
            description=f"Custom environment for {environment_name}",
            build=BuildContext(
                path=context_path,
                dockerfile_path=dockerfile_path,
            ),
        )

        # Register the environment
        environment = ml_client.environments.create_or_update(env)

        console.print(
            f"✓ Environment registered: {environment.name} (v{environment.version})",
            style="success",
        )

        return environment

    except Exception as e:
        console.print(f"Failed to build environment: {str(e)}", style="error")
        raise
