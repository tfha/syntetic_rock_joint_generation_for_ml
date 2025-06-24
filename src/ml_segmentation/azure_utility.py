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
from typing import Callable, TypeVar

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


def setup_azure_environment(console=None):
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
    output_file: str = "environment.yml", default_python_version: str = "3.11"
) -> str:
    """
    Export Poetry dependencies to environment.yml format for Azure ML,
    prioritizing conda packages over pip packages where possible.

    Args:
        output_file (str): The path where the environment.yml file will be saved.
        default_python_version (str): Default Python version to use if detection fails.

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

    # Get Python version from Poetry
    try:
        result = subprocess.run(
            ["python", "--version"], capture_output=True, text=True, check=True
        )
        python_version = result.stdout.strip().split(" ")[1]
    except subprocess.CalledProcessError:
        python_version = default_python_version
        print(f"Could not determine Python version, defaulting to {python_version}")

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
        "mlflow",
        "opencv",
        "hydra-core",
        "rich",
    }

    # Special handling for PyTorch-related packages
    pytorch_related = {"torch", "torchvision", "torchaudio", "pytorch"}

    # Separate conda and pip packages
    conda_packages = [f"python={python_version}", "pip"]
    pip_only_packages = []
    cuda_pip_packages = []

    for pkg_name, version, req_str in requirements:
        pkg_lower = pkg_name.lower()

        # Special handling for PyTorch with CUDA
        if pkg_lower in pytorch_related and (
            pkg_lower in cuda_packages or "torch" in cuda_packages
        ):
            cuda_pip_packages.append(req_str)
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

    # For CUDA packages, add special handling
    if cuda_pip_packages:
        for pkg in cuda_pip_packages:
            # Replace with just the package name to allow pip to resolve from custom
            # index
            for cuda_pkg in cuda_packages:
                if cuda_pkg in pkg.lower():
                    print(f"Adding CUDA package: {pkg}")
                    all_pip_packages.append(pkg)

    if all_pip_packages:
        # For PyTorch CUDA, add specific pip installation command if needed
        if cuda_packages and any("torch" in pkg.lower() for pkg in cuda_pip_packages):
            torch_indexes = [
                url
                for name, url in custom_sources.items()
                if "pytorch" in name.lower() or "torch" in name.lower()
            ]
            if torch_indexes:
                all_pip_packages.append(f"--index-url {torch_indexes[0]}")
                print(f"Adding PyTorch CUDA custom index: {torch_indexes[0]}")

        env_yaml["dependencies"].append({"pip": all_pip_packages})

    # Write to environment.yml
    with open(output_file, "w") as f:
        yaml.dump(env_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully exported Poetry environment to {output_file}")
    print(f"- Conda packages: {len(conda_packages) - 2}")  # Subtract python and pip
    print(f"- Pip-only packages: {len(all_pip_packages)}")
    if cuda_pip_packages:
        print(f"- CUDA-enabled packages: {len(cuda_pip_packages)}")
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
