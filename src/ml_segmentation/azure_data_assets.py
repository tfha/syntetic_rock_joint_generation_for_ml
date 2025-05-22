"""
Azure ML data asset management for rock mass segmentation.

This module provides advanced utilities for working with Azure ML data assets,
including versioning, metadata management, and data lineage tracking.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from ml_segmentation.utility import get_custom_console


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


def register_data_asset(
    ml_client: MLClient,
    name: str,
    version: str,
    description: str,
    path: str,
    asset_type: str = AssetTypes.URI_FOLDER,
    tags: dict[str, str] = None,
    metadata: dict[str, str] = None,
    create_new_version: bool = True,
) -> Data:
    """
    Uploads and register a data asset with Azure ML with proper versioning and metadata.

    Args:
        ml_client: Azure ML client
        name: Name of the data asset
        version: Version of the data asset
        description: Description of the data asset
        path: Path to the data (could be local or cloud storage)
        asset_type: Type of asset (URI_FILE or URI_FOLDER)
        tags: Tags to attach to the data asset
        metadata: Additional metadata to attach to the data asset
        create_new_version: Whether to create a new version or update existing

    Returns:
        Azure ML Data asset
    """
    console = get_custom_console()

    # Prepare metadata
    if metadata is None:
        metadata = {}

    # Add creation date to metadata
    metadata["creation_date"] = datetime.now().isoformat()

    # Add basic statistics if it's a folder (and accessible)
    if asset_type == AssetTypes.URI_FOLDER and path.startswith(("./", "/")):
        try:
            local_path = Path(path)
            if local_path.exists():
                num_files = len(list(local_path.glob("*")))
                metadata["file_count"] = str(num_files)

                # Add more specific stats for images if appropriate
                if name.endswith(("_images", "_masks")):
                    img_extensions = [".jpg", ".png", ".tif", ".tiff"]
                    image_count = sum(
                        1
                        for f in local_path.glob("**/*")
                        if f.suffix.lower() in img_extensions
                    )
                    metadata["image_count"] = str(image_count)
        except Exception as e:
            console.print(
                f"Couldn't analyze local directory stats: {str(e)}", style="warning"
            )

    try:
        # Create the data asset directly with all required parameters
        my_data = Data(
            name=name,
            version=version,
            description=description,
            path=path,
            type=asset_type,
        )

        # Set optional properties if provided
        if tags:
            my_data.tags = tags
        if metadata:
            my_data.metadata = metadata

        # Check if dataset already exists with this version
        try:
            existing_data = ml_client.data.get(name=name, version=version)
            console.print(
                f"Data asset already exists. Name: {name}, version: {version}",
                style="info",
            )
            if not create_new_version:
                return existing_data
        except Exception:
            pass  # Data doesn't exist with this version, continue with creation

        # Register the data asset
        result = ml_client.data.create_or_update(my_data)
        console.print(
            f"Successfully registered data asset '{name}'"
            " with version {result.version}",
            style="success",
        )
        return result

    except Exception as e:
        console.print(f"Error registering data asset: {str(e)}", style="error")
        raise


def get_data_asset(
    ml_client: MLClient, name: str, version: str = None, label: str = "latest"
) -> Data:
    """
    Get a data asset from Azure ML workspace.

    Args:
        ml_client: Azure ML client
        name: Name of the data asset
        version: Version of the data asset (if specific version required)
        label: Label to use if version not provided (default: "latest")

    Returns:
        Azure ML Data asset
    """
    console = get_custom_console()

    try:
        if version:
            data = ml_client.data.get(name=name, version=version)
        else:
            data = ml_client.data.get(name=name, label=label)

        console.print(
            f"Retrieved data asset '{name}' (version {data.version})", style="info"
        )
        return data

    except Exception as e:
        console.print(f"Error retrieving data asset '{name}': {str(e)}", style="error")
        raise


def create_data_asset_from_split(
    ml_client: MLClient,
    base_name: str,
    source_path: str,
    split_info_file: str,
    version: str = None,
    description: str = "Dataset split for ML training",
) -> dict[str, Data]:
    """
    Create data assets from a dataset split (train/val/test).

    Args:
        ml_client: Azure ML client
        base_name: Base name for the dataset (e.g., "rockmass")
        source_path: Source path for the dataset files
        split_info_file: JSON file containing train/val/test split information
        version: Version string for the assets (default: timestamp)
        description: Description for the data assets

    Returns:
        Dictionary of data assets (train, val, test)
    """
    console = get_custom_console()

    # Generate version if not provided
    if not version:
        version = datetime.now().strftime("%Y%m%d.%H%M")

    # Load split information
    console.print(f"Loading split information from {split_info_file}", style="info")
    try:
        with open(split_info_file, "r") as f:
            split_data = json.load(f)
    except Exception as e:
        console.print(
            f"Error loading split file {split_info_file}: {str(e)}", style="error"
        )
        raise

    # Create data assets for each split
    result = {}

    for split_name in ["train", "val", "test"]:
        if split_name in split_data:
            metadata = {
                "source_dataset": base_name,
                "split_type": split_name,
                "num_samples": str(len(split_data[split_name])),
            }

            # Register the split as a data asset
            asset_name = f"{base_name}_{split_name}"

            # For now, we'll just register the split file itself
            # In a more advanced implementation, you could create filtered datasets
            asset = register_data_asset(
                ml_client=ml_client,
                name=asset_name,
                version=version,
                description=f"{description} - {split_name.capitalize()} split",
                path=split_info_file,  # Here we're just registering the JSON file
                asset_type=AssetTypes.URI_FILE,
                metadata=metadata,
            )

            result[split_name] = asset

    console.print(
        f"Created {len(result)} data assets for dataset splits", style="success"
    )
    return result


def list_data_asset_versions(ml_client: MLClient, name: str) -> pd.DataFrame:
    """
    List all versions of a data asset with their metadata.

    Args:
        ml_client: Azure ML client
        name: Name of the data asset

    Returns:
        DataFrame with version information
    """
    console = get_custom_console()

    try:
        assets = list(ml_client.data.list(name=name))

        if not assets:
            console.print(f"No data assets found with name '{name}'", style="warning")
            return pd.DataFrame()

        # Extract relevant information
        versions_info = []
        for asset in assets:
            info = {
                "version": asset.version,
                "creation_date": asset.creation_context.created_at,
                "created_by": asset.creation_context.created_by,
                "path": asset.path,
            }

            # Add metadata if available
            if hasattr(asset, "metadata") and asset.metadata:
                for k, v in asset.metadata.items():
                    info[f"metadata_{k}"] = v

            versions_info.append(info)

        # Create DataFrame and sort by version
        df = pd.DataFrame(versions_info)
        if not df.empty:
            df = df.sort_values("version", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        console.print(f"Error listing data asset versions: {str(e)}", style="error")
        return pd.DataFrame()


def compare_data_asset_versions(
    ml_client: MLClient, name: str, version1: str, version2: str
) -> dict[str, str | list[str]]:
    """
    Compare two versions of a data asset.

    Args:
        ml_client: Azure ML client
        name: Name of the data asset
        version1: First version to compare
        version2: Second version to compare

    Returns:
        Dictionary with comparison results
    """
    console = get_custom_console()

    try:
        asset1 = ml_client.data.get(name=name, version=version1)
        asset2 = ml_client.data.get(name=name, version=version2)

        # Basic comparison of metadata
        comparison = {
            "name": name,
            "version1": version1,
            "version2": version2,
            "metadata_differences": [],
            "path_changed": asset1.path != asset2.path,
        }

        # Compare metadata
        metadata1 = asset1.metadata or {}
        metadata2 = asset2.metadata or {}

        all_keys = set(metadata1.keys()).union(set(metadata2.keys()))

        for key in all_keys:
            if key not in metadata1:
                comparison["metadata_differences"].append(
                    f"Key '{key}' only in version {version2}"
                )
            elif key not in metadata2:
                comparison["metadata_differences"].append(
                    f"Key '{key}' only in version {version1}"
                )
            elif metadata1[key] != metadata2[key]:
                comparison["metadata_differences"].append(
                    f"Key '{key}' differs: {metadata1[key]} vs {metadata2[key]}"
                )

        return comparison

    except Exception as e:
        console.print(f"Error comparing data asset versions: {str(e)}", style="error")
        return {"error": str(e)}
