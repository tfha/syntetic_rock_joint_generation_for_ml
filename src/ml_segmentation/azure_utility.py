"""
Azure utility functions for environment setup and common operations.

This module provides reusable utilities for Azure environments,
including environment variable handling, credential management,
and common Azure configuration functionality.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Any, Callable, TypeVar

import toml
import yaml
from azure.ai.ml import MLClient
from azure.core.exceptions import AzureError, ServiceRequestError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from rich.console import Console

from ml_segmentation.utility import get_custom_console

# Type variable for generic retry function
T = TypeVar("T")


def check_compute_permissions(
    ml_client: MLClient, compute_name: str, console: Console
) -> dict[str, Any]:
    """
    Check if the compute cluster has the necessary permissions to access data.

    Args:
        ml_client: Azure ML client
        compute_name: Name of the compute cluster
        console: Console for logging

    Returns:
        Dictionary with results of permission checks
    """
    permissions = {
        "has_identity": False,
        "identity_type": None,
        "storage_blob_access": False,
        "workspace_access": False,
        "log_streaming_access": False,  # New field to track streaming access
    }

    try:
        # Get compute cluster details with refresh
        console.print(f"Checking compute cluster: {compute_name}", style="info")

        # First try to get compute with refresh to ensure we have latest data
        compute = refresh_compute_cluster(ml_client, compute_name, console)

        if compute is None:
            # Fall back to standard get if refresh fails
            compute = ml_client.compute.get(compute_name)
            console.print(
                "Using regular compute.get() as refresh failed", style="warning"
            )

        # Enhanced identity detection - try multiple ways to access identity
        identity = None

        # First attempt - check if compute has an identity attribute
        identity = getattr(compute, "identity", None)
        console.print(f"Identity attribute access: {type(identity)}", style="info")

        # If identity is not found or is None, try accessing it as a dictionary
        if identity is None and hasattr(compute, "properties"):
            # Second attempt - try to access through properties dictionary
            props = getattr(compute, "properties", {})
            if isinstance(props, dict) and "identity" in props:
                identity = props["identity"]
                console.print("Found identity in compute properties", style="info")

        # In case we get a serialization wrapper, try to get the actual data
        if hasattr(identity, "_attribute_map") and hasattr(identity, "as_dict"):
            identity = identity.as_dict()
            console.print(
                "Using identity.as_dict() to access identity data", style="info"
            )

        # Print identity for debugging
        console.print(f"Identity object: {identity}", style="info")

        if identity is not None:
            permissions["has_identity"] = True

            # Handle different ways the type might be structured
            if isinstance(identity, dict):
                # Check standard location for type
                identity_type = identity.get("type", None)

                # If not found, try alternative locations
                if not identity_type:
                    if "systemAssignedIdentity" in identity:
                        identity_type = "SystemAssigned"
                    elif "userAssignedIdentities" in identity:
                        identity_type = "UserAssigned"

                # Check for dual mode - both system and user assigned
                has_system = "systemAssignedIdentity" in identity
                has_user = "userAssignedIdentities" in identity
                if has_system and has_user:
                    identity_type = "SystemAssigned,UserAssigned"

                permissions["identity_type"] = identity_type
                console.print(f"Detected identity type: {identity_type}", style="info")
            elif hasattr(identity, "type"):
                # Handle case where identity is an object with a type attribute
                permissions["identity_type"] = identity.type
                console.print(
                    f"Identity object has type: {identity.type}", style="info"
                )

        # Try to list datastores to check workspace access
        try:
            datastores = list(ml_client.datastores.list())
            if datastores:
                permissions["workspace_access"] = True
                console.print(
                    f"Compute has access to {len(datastores)} datastores", style="info"
                )

                # Try to check default datastore access
                default_datastore = None
                for ds in datastores:
                    if getattr(ds, "is_default", False):
                        default_datastore = ds
                        break

                if default_datastore:
                    console.print(
                        f"Default datastore: {default_datastore.name}", style="info"
                    )

                    # Check if compute has a managed identity
                    identity_types = ["SystemAssigned", "system_assigned"]
                    identity_types += ["UserAssigned", "user_assigned"]

                    has_identity = permissions["has_identity"]
                    id_type = permissions["identity_type"]
                    if has_identity and id_type in identity_types:
                        # We can't directly check storage permissions
                        console.print(
                            "Compute has a managed identity that may have storage "
                            "access.",
                            style="info",
                        )
                        console.print(
                            "If job fails with NoIdentityOnCompute error, ensure "
                            "identity has Storage Blob Data Contributor role on the "
                            "storage.",
                            style="warning",
                        )
                        # Try to verify log streaming access permissions (new check)
                        try:
                            # Check for log streaming permissions
                            console.print(
                                "Checking log streaming permissions...", style="info"
                            )

                            # We can't fully verify streaming permissions without
                            # running a job but we can check identity type
                            id_types = ["SystemAssigned", "system_assigned"]
                            if permissions["identity_type"] in id_types:
                                console.print(
                                    "Compute has system-assigned identity, which is "
                                    "required for log streaming. Ensure it has:",
                                    style="info",
                                )
                                console.print(
                                    "1. 'Storage Blob Data Reader' role on the "
                                    "workspace storage account",
                                    style="info",
                                )
                                console.print(
                                    "2. 'AzureML Data Scientist' on the workspace",
                                    style="info",
                                )

                                # Mark as potentially having streaming access
                                # We can't be 100% sure without actually trying
                                permissions["log_streaming_access"] = True
                        except Exception as stream_error:
                            console.print(
                                f"Could not verify log streaming permissions: "
                                f"{str(stream_error)}",
                                style="warning",
                            )
        except Exception as e:
            console.print(
                f"Could not verify datastore access for compute: {str(e)}",
                style="warning",
            )

    except Exception as e:
        console.print(f"Error checking compute permissions: {str(e)}", style="error")

    return permissions


def retry_azure_operation(
    operation: Callable[..., T],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_catch: tuple = (AzureError, ServiceRequestError),
    operation_name: str = "Azure operation",
) -> Callable[..., T]:
    """
    Decorator for retrying Azure operations with exponential backoff.

    Azure operations can fail due to transient issues such as network connectivity,
    service throttling, or temporary service unavailability. This retry mechanism
    helps improve resilience and reliability when working with cloud services.
    Using this functionality replaces the need for manual retry logic or error handling
    in your Azure operations, allowing you to focus on the core logic of your
    application.

    Exponential backoff is a standard retry strategy where the wait time between
    retry attempts increases exponentially. This approach prevents overwhelming
    the service with rapid-fire retries and gives the service time to recover
    from any issues. For example:
        - First retry: Wait 1 second
        - Second retry: Wait 2 seconds (1 * 2.0)
        - Third retry: Wait 4 seconds (2 * 2.0)
        - And so on...

    This pattern is essential for cloud applications as it:
        1. Reduces load on potentially stressed services
        2. Increases probability of eventual success
        3. Follows Azure service rate-limiting best practices
        4. Handles intermittent network issues gracefully

    Args:
        operation: The function to retry
        max_retries: Maximum number of retry attempts before giving up
        initial_backoff: Initial backoff time in seconds before first retry
        backoff_factor: Multiplicative factor to increase backoff time with each retry
        exceptions_to_catch: Tuple of exception classes that should trigger a retry
        operation_name: Name of the operation for logging purposes

    Returns:
        A wrapped function that implements retry logic

    Example:
        >>> @retry_azure_operation(max_retries=5, operation_name="Get data asset")
        >>> def get_data_asset_wrapper(client, name):
        >>>     return client.data.get_asset(name)
    """

    def wrapper(*args, **kwargs):
        console = Console()

        retries = 0
        current_backoff = initial_backoff

        while True:
            try:
                return operation(*args, **kwargs)
            except exceptions_to_catch as e:
                retries += 1
                if retries > max_retries:
                    console.print(
                        f"[bold red]Failed {operation_name} after "
                        f"{max_retries} attempts: {str(e)}"
                    )
                    raise

                console.print(
                    f"[yellow]Attempt {retries}/{max_retries} for {operation_name} "
                    f"failed: {str(e)}. Retrying in {current_backoff} seconds..."
                )
                time.sleep(current_backoff)
                current_backoff *= backoff_factor

    return wrapper


def configure_azure_logging():
    """Configure logging to reduce verbose Azure client output.

    This function sets the logging level for various Azure client libraries
    to WARNING, reducing noise in the console output. It should be called
    at the beginning of any script that interacts with Azure services.
    """
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)


def setup_azure_environment_variables(console=None):
    """Set up the Azure environment and return a console for pretty printing.

    This function loads environment variables from .env file,
    validates Azure credentials, and returns a console object.

    Args:
        console: Optional console object for pretty printing. If None, a new
        console is created.

    Returns:
        tuple: (console, subscription_id, resource_group, workspace_name)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Create a console for pretty printing if not provided
    if console is None:
        console = get_custom_console()

    # Check if Azure environment variables are set
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id:
        console.print(
            "Error: AZURE_SUBSCRIPTION_ID environment variable not set", style="error"
        )
        console.print(
            "Please set it in your .env file:"
            "AZURE_SUBSCRIPTION_ID='your-subscription-id'",
            style="error",
        )
        sys.exit(1)

    if not resource_group:
        console.print(
            "Warning: AZURE_RESOURCE_GROUP environment variable not set",
            style="warning",
        )

    if not workspace_name:
        console.print(
            "Warning: AZURE_ML_WORKSPACE environment variable not set", style="warning"
        )

    return console, subscription_id, resource_group, workspace_name


def export_poetry_to_environment_yml(
    output_file: str = "environment.yml",
    default_python_version: str = "3.10",
    exclude_pytorch_packages: bool = True,
) -> str:
    """
    Export Poetry dependencies to environment.yml format for Azure ML,
    prioritizing conda packages over pip packages where possible.

    When using PyTorch base images in Azure ML, this function can automatically exclude
    core PyTorch packages to prevent version conflicts. Excluded packages include:
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
        # No need to get lock information if we're not using it
        # Just export requirements directly
        result = subprocess.run(
            ["poetry", "export", "--format", "requirements.txt", "--without-hashes"],
            capture_output=True,
            text=True,
            check=True,
        )
        requirements_raw = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting Poetry environment: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

    # Get pyproject.toml content to analyze sources
    pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")
    custom_sources = {}
    cuda_packages = set()

    if os.path.exists(pyproject_path):
        try:
            with open(pyproject_path, "r") as f:
                pyproject = toml.load(f)

            # Extract custom sources
            if "tool" in pyproject and "poetry" in pyproject["tool"]:
                if "source" in pyproject["tool"]["poetry"]:
                    for source in pyproject["tool"]["poetry"]["source"]:
                        custom_sources[source["name"]] = source["url"]

                # Check for packages with custom sources
                if "dependencies" in pyproject["tool"]["poetry"]:
                    for pkg, info in pyproject["tool"]["poetry"][
                        "dependencies"
                    ].items():
                        if isinstance(info, dict) and "source" in info:
                            # If package has a custom source, check if it's likely CUDA
                            if (
                                "cu" in info.get("version", "")
                                or "cuda"
                                in custom_sources.get(info["source"], "").lower()
                            ):
                                cuda_packages.add(pkg.lower())
        except (ImportError, Exception) as e:
            print(f"Warning: Could not parse pyproject.toml: {e}")

    # Clean up and parse requirements
    requirements = []
    for req in requirements_raw:
        if req and not req.startswith("#"):
            # Extract package name without version constraints
            if ";" in req:  # Handle environment markers
                req = req.split(";")[0].strip()

            if "==" in req:
                pkg_name = req.split("==")[0].strip()
                version = req.split("==")[1].strip()
                requirements.append((pkg_name, version, req))
            elif ">=" in req:
                pkg_name = req.split(">=")[0].strip()
                requirements.append((pkg_name, None, req))
            else:
                pkg_name = req.split("[")[0].strip() if "[" in req else req.strip()
                requirements.append((pkg_name, None, req))

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
    windows_specific_packages = {
        "pywin32",
        "pypiwin32",
        "pywinpty",
        "winrt",
        "windows-curses",
    }

    # Define packages that conflict with PyTorch base images in Azure ML
    # These packages are typically pre-installed in PyTorch base images
    # Only exclude packages that are ACTUALLY pre-installed in the base image
    pytorch_base_image_packages = {
        # Core PyTorch packages (definitely pre-installed)
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch",
        "--extra-index-url https://download.pytorch.org/whl/cu121",
        # NVIDIA/CUDA packages (included in CUDA base images)
        "nvidia-cuda-runtime-cu11",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu11",
        "nvidia-cublas-cu12",
        "nvidia-curand-cu11",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu11",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu11",
        "nvidia-cusparse-cu12",
        "nvidia-cudnn-cu11",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu11",
        "nvidia-cufft-cu12",
        "nvidia-nvtx-cu11",
        "nvidia-nvtx-cu12",
        "nvidia-ml-py",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "cuda-toolkit",
        "cudatoolkit",
        # MAGMA (used by PyTorch, included in base images)
        "magma-cuda110",
        "magma-cuda111",
        "magma-cuda112",
        "magma-cuda113",
        "magma-cuda114",
        "magma-cuda115",
        "magma-cuda116",
        "magma-cuda117",
        "magma-cuda118",
        "magma-cuda121",
        # Triton (PyTorch JIT compiler, included in newer base images)
        "triton",
    }

    # Define version mappings for Python compatibility
    python_compat_versions = {
        "ipython": {
            "3.10": "8.18.1",  # Last version compatible with Python 3.10
            "3.11": "8.18.1",  # Can use newer versions but this is safe
            "3.12": "8.18.1",  # Can use newer versions but this is safe
        }
    }

    for pkg_name, version, req_str in requirements:
        pkg_lower = pkg_name.lower()

        # Skip Windows-specific packages for Azure ML environments
        if pkg_lower in windows_specific_packages:
            print(
                f"Skipping Windows-specific package {pkg_name} for Azure ML environment"
            )
            continue

        # Skip packages that conflict with PyTorch base images (if enabled)
        if exclude_pytorch_packages and pkg_lower in pytorch_base_image_packages:
            print(
                f"Skipping PyTorch base image package {pkg_name} "
                f"(pre-installed in base image)"
            )
            continue

        # Handle Python version compatibility issues
        if (
            pkg_lower in python_compat_versions
            and python_version in python_compat_versions[pkg_lower]
        ):
            compatible_version = python_compat_versions[pkg_lower][python_version]
            print(
                f"Using compatible version {compatible_version} for {pkg_name} "
                f"with Python {python_version}"
            )
            if pkg_lower in conda_preferred_packages:
                conda_packages.append(f"{pkg_name}={compatible_version}")
            else:
                pip_only_packages.append(f"{pkg_name}=={compatible_version}")
            continue

        # Check if this is a package we prefer to install via conda
        if pkg_lower in conda_preferred_packages:
            if version:
                conda_packages.append(f"{pkg_name}={version}")
            else:
                conda_packages.append(pkg_name)
        else:
            # Add to pip_only_packages if not in conda preferred list
            pip_only_packages.append(req_str)

    # Create environment.yml content
    env_yaml = {
        "name": "rock-segmentation",
        "channels": ["conda-forge", "defaults"],
        "dependencies": conda_packages,
    }

    # Add pip packages if there are any
    all_pip_packages = pip_only_packages.copy()

    if all_pip_packages:
        env_yaml["dependencies"].append({"pip": all_pip_packages})

    # Note: Local package installation (-e .) is handled by setup_azure_environment.py
    # at runtime instead of in environment.yml to avoid Docker build context issues

    # Write to environment.yml
    with open(output_file, "w") as f:
        yaml.dump(env_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully exported Poetry environment to {output_file}")
    print(f"- Conda packages: {len(conda_packages) - 2}")  # Subtract python and pip
    print(f"- Pip-only packages: {len(all_pip_packages)}")
    print("- Local package installation: Handled by setup script at runtime")
    if exclude_pytorch_packages:
        print("- Core PyTorch packages excluded (torch, torchvision, torchaudio)")
        print("- PyTorch ecosystem packages included (torchinfo, torchmetrics, etc.)")
    return output_file


def test_environment_export(output_path: str = None) -> None:
    """
    Test the export_poetry_to_environment_yml function by running it
    and displaying information about the generated environment.yml file.

    Args:
        output_path (str, optional): Custom path for the output file.
            Defaults to environment.yml in the current directory.
    """
    console = get_custom_console()

    console.print(
        "\n[bold blue]Testing Poetry to Azure ML environment export[/bold blue]"
    )
    console.print("=" * 60)

    # Get the current working directory
    cwd = os.getcwd()
    console.print(f"Current working directory: [green]{cwd}[/green]")

    # Check if pyproject.toml exists
    pyproject_path = os.path.join(cwd, "pyproject.toml")
    if not os.path.exists(pyproject_path):
        console.print(
            f"[bold red]Error:[/bold red] pyproject.toml not found at {pyproject_path}"
        )
        console.print("Please run this command from the root of your Python project.")
        return

    # Run the export function
    if output_path is None:
        output_path = os.path.join(cwd, "environment.yml")

    try:
        console.print(f"Exporting environment to: [green]{output_path}[/green]")
        result_path = export_poetry_to_environment_yml(output_path)

        # Validate the generated file
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            console.print(
                (
                    f"\n[bold green]Success![/bold green] environment.yml generated "
                    f"({file_size} bytes)"
                )
            )

            # Read and display a summary of the environment.yml
            with open(result_path, "r") as f:
                env_data = yaml.safe_load(f)

            # Display a summary
            console.print("\n[bold]Environment Summary:[/bold]")
            console.print(f"Name: [cyan]{env_data.get('name', 'unnamed')}[/cyan]")
            console.print(
                f"Channels: [cyan]{', '.join(env_data.get('channels', []))}[/cyan]"
            )

            dependencies = env_data.get("dependencies", [])
            conda_deps = [d for d in dependencies if isinstance(d, str)]
            pip_deps = []

            for dep in dependencies:
                if isinstance(dep, dict) and "pip" in dep:
                    pip_deps = dep["pip"]

            console.print(f"Conda packages: [cyan]{len(conda_deps)}[/cyan]")
            console.print(f"Pip packages: [cyan]{len(pip_deps)}[/cyan]")

            # Check for PyTorch CUDA packages
            cuda_packages = [
                p
                for p in pip_deps
                if "torch" in p.lower()
                and ("cu" in p.lower() or "--index-url" in p.lower())
            ]
            if cuda_packages:
                console.print(
                    "[bold green]✓[/bold green] Found PyTorch CUDA configurations:"
                )
                for pkg in cuda_packages:
                    console.print(f"  - [cyan]{pkg}[/cyan]")

            # Provide instructions
            console.print("\n[bold yellow]Next Steps:[/bold yellow]")
            console.print("1. Inspect the environment.yml file for accuracy")
            console.print(
                "2. Use this file with Azure ML for compute environment configuration"
            )
            console.print(
                "3. Test the environment with: [green]conda env create -f "
                "environment.yml[/green]"
            )
        else:
            console.print(
                "[bold red]Error:[/bold red] Failed to generate environment.yml"
                f" at {result_path}"
            )

    except Exception as e:
        console.print(f"[bold red]Error during export:[/bold red] {str(e)}")
        import traceback

        console.print(traceback.format_exc())


def connect_to_azure_ml(
    subscription_id: str, resource_group: str = None, workspace_name: str = None
) -> MLClient:
    """
    Connect to Azure ML workspace with proper authentication.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Azure resource group name
        workspace_name: Azure ML workspace name

    Returns:
        Azure ML client
    """
    # Get values from environment if not provided
    resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = workspace_name or os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id or not resource_group or not workspace_name:
        raise ValueError(
            "Missing Azure ML configuration. Provide subscription_id, resource_group, "
            "and workspace_name as parameters or set them as environment variables."
        )

    console = get_custom_console()
    console.print(f"Connecting to Azure ML workspace: {workspace_name}", style="info")

    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        # Test connection
        _ = ml_client.workspaces.get(workspace_name)
        console.print(
            f"Successfully connected to workspace: {workspace_name}", style="success"
        )
        return ml_client

    except Exception as e:
        console.print(
            f"Error connecting to Azure ML workspace: {str(e)}", style="error"
        )
        raise


def refresh_compute_cluster(
    ml_client: MLClient, compute_name: str, console: Console
) -> Any:
    """
    Forcibly refresh the compute cluster information from Azure.
    Sometimes after making changes to compute in the portal, the local client cache
    needs to be refreshed to see the changes.

    Args:
        ml_client: Azure ML client
        compute_name: Name of the compute cluster
        console: Console for logging

    Returns:
        The refreshed compute cluster object
    """
    try:
        console.print(
            f"Refreshing compute cluster information for: {compute_name}", style="info"
        )

        # Force refresh by doing a list operation first
        all_computes = list(ml_client.compute.list())
        console.print(
            f"Found {len(all_computes)} compute resource(s) in workspace", style="info"
        )

        # Now get the specific compute we need - should be fresh
        compute = ml_client.compute.get(
            name=compute_name,
            resource_group_name=None,  # Use default from client
        )

        console.print(
            f"Successfully refreshed compute cluster: {compute_name}", style="success"
        )
        return compute

    except Exception as e:
        console.print(f"Error refreshing compute cluster: {str(e)}", style="error")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Poetry environment to Azure ML environment.yml"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="environment.yml",
        help="Path for the output environment.yml file",
    )
    args = parser.parse_args()

    test_environment_export(args.output)
