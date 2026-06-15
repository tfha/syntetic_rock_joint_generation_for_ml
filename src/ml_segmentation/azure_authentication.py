"""
Azure authentication and workspace connection utilities.

This module handles Azure ML workspace connections, credential management,
and basic authentication operations.
"""

import os
import sys

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from rich.console import Console

from ml_segmentation.utility import get_custom_console


def setup_azure_environment_variables(
    console: Console | None = None,
) -> tuple[Console, str, str, str]:
    """
    Set up the Azure environment and return a console for pretty printing.

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
            "AZURE_SUBSCRIPTION_ID environment variable is not set", style="red"
        )
        console.print(
            "Please set this variable in your .env file or environment", style="blue"
        )
        sys.exit(1)

    if not resource_group:
        console.print(
            "AZURE_RESOURCE_GROUP environment variable is not set", style="red"
        )
        console.print(
            "Please set this variable in your .env file or environment", style="blue"
        )
        sys.exit(1)

    if not workspace_name:
        console.print("AZURE_ML_WORKSPACE environment variable is not set", style="red")
        console.print(
            "Please set this variable in your .env file or environment", style="blue"
        )
        sys.exit(1)

    return console, subscription_id, resource_group, workspace_name


def connect_to_azure_ml(
    subscription_id: str,
    console: Console,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> MLClient:
    """
    Connect to Azure ML workspace with proper authentication.

    Args:
        subscription_id: Azure subscription ID
        console: Rich Console instance for logging
        resource_group: Azure resource group name
        workspace_name: Azure ML workspace name

    Returns:
        Azure ML client

    Raises:
        SystemExit: If connection fails
    """
    # Get values from environment if not provided
    resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = workspace_name or os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id or not resource_group or not workspace_name:
        console.print("Missing required Azure configuration parameters", style="red")
        sys.exit(1)

    console.print(f"Connecting to Azure ML workspace: {workspace_name}", style="blue")

    try:
        # Use DefaultAzureCredential for authentication
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        console.print(
            "[OK] Successfully connected to Azure ML workspace", style="green"
        )
        return ml_client

    except Exception as e:
        console.print(f"Failed to connect to Azure ML: {str(e)}", style="red")
        console.print(
            "Ensure you are authenticated with Azure CLI or have "
            "appropriate credentials",
            style="blue",
        )
        sys.exit(1)


def validate_workspace_permissions(ml_client: MLClient, console: Console) -> bool:
    """
    Validate that the current user/identity has basic workspace permissions.

    This function performs basic operations to verify workspace access and
    provides guidance if permissions are insufficient.

    Args:
        ml_client: Azure ML client
        console: Console for logging

    Returns:
        bool: True if workspace access is confirmed, False otherwise
    """
    try:
        console.print("Verifying workspace permissions...", style="blue")

        # Test basic workspace operations
        _ = list(ml_client.datastores.list())
        console.print("[OK] Workspace access verified", style="green")
        return True

    except Exception as perm_error:
        console.print(
            f"Warning: Connected to workspace but may have limited permissions. "
            f"Error: {str(perm_error)}",
            style="yellow",
        )
        console.print(
            "What this means: your user or managed identity can authenticate to the "
            "Azure ML workspace, but is missing RBAC permissions to perform standard "
            "operations (e.g., list datastores, jobs, computes, environments, or read the "
            "workspace's default storage).",
            style="blue",
        )
        console.print(
            "How to fix (assign roles at the workspace or resource group scope):\n"
            "- Azure Machine Learning Contributor (or AzureML Data Scientist)\n"
            "- Storage Blob Data Reader on the workspace's default Storage account to read datastores; "
            "Storage Blob Data Contributor if you need to write outputs/artifacts\n"
            "- Key Vault Secrets User on the workspace Key Vault if your workflows access secrets\n"
            "- AzureML Compute Operator if you need to create or refresh compute",
            style="blue",
        )
        console.print(
            "Also verify you're using the intended subscription/tenant and that role assignments "
            "have propagated (can take a few minutes). If you're using a managed identity, ensure "
            "the above roles are assigned to that identity as well.",
            style="blue",
        )
        return False
