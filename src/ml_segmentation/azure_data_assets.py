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
from typing import Any, Callable

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from rich.console import Console
from tqdm import tqdm

from ml_segmentation.azure_core import retry_azure_operation
from ml_segmentation.data_loading import (
    get_data_files,
    get_datasets_prefixes,
    split_data,
)

# Imports all required modules above


def register_data_asset(
    ml_client: MLClient,
    console: Console,
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
        console: Console object for pretty printing
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
            f" with version {result.version}",
            style="success",
        )
        return result

    except Exception as e:
        console.print(f"Error registering data asset: {str(e)}", style="error")
        raise


def get_data_asset(
    ml_client: MLClient,
    console: Console,
    name: str,
    version: str = None,
    label: str = "latest",
) -> Data:
    """
    Get a data asset from Azure ML workspace.

    Note: This function should be wrapped with retry_azure_operation when used,
    as Azure operations may fail due to transient issues.

    Args:
        ml_client: Azure ML client
        console: Console object for pretty printing
        name: Name of the data asset
        version: Version of the data asset (if specific version required)
        label: Label to use if version not provided (default: "latest")

    Returns:
        Azure ML Data asset
    """

    if version:
        data = ml_client.data.get(name=name, version=version)
    else:
        data = ml_client.data.get(name=name, label=label)

    console.print(
        f"Retrieved data asset '{name}' (version {data.version})", style="info"
    )
    return data


def list_data_asset_versions(
    ml_client: MLClient, console: Console, name: str
) -> pd.DataFrame:
    """
    List all versions of a data asset with their metadata.

    Args:
        ml_client: Azure ML client
        console: Console object for pretty printing
        name: Name of the data asset

    Returns:
        DataFrame with version information
    """

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


def get_azure_storage_client(
    console: Console,
    require_key: bool = False,
) -> tuple[BlobServiceClient, str, str, str | None]:
    """Get Azure storage configuration from environment variables and connect to
    storage.

    This function retrieves Azure storage credentials from environment variables
    and optionally establishes a connection to the Azure Blob storage.

    Args:
        console: Console object for pretty printing
        require_key: Whether to require the storage key (default: False)

    Returns:
        tuple: (container_client, container_name, storage_account, storage_key)
            or (None, container_name, storage_account, storage_key) if connection fails

    Raises:
        SystemExit: If required configuration is missing
    """
    # Get storage config from environment variables
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    container_name = os.environ.get("AZURE_BLOB_DATASTORE")
    storage_key = os.environ.get("AZURE_STORAGE_KEY") if require_key else None

    # Validate required configuration
    missing_vars = []
    if not storage_account:
        missing_vars.append("AZURE_STORAGE_ACCOUNT")
    if not container_name:
        missing_vars.append("AZURE_BLOB_DATASTORE")
    if require_key and not storage_key:
        missing_vars.append("AZURE_STORAGE_KEY")

    if missing_vars:
        console.print("Error: Missing Azure storage configuration.", style="error")
        console.print(
            "Please set the following environment variables: "
            f"{', '.join(missing_vars)}",
            style="error",
        )
        sys.exit(1)

    # If connection is not required, return config only
    if not require_key:
        return None, container_name, storage_account, storage_key

    # Build connection string
    endpoint_protocol = "DefaultEndpointsProtocol=https"
    account_name = f"AccountName={storage_account}"
    account_key = f"AccountKey={storage_key}"
    endpoint_suffix = "EndpointSuffix=core.windows.net"
    connection_string = (
        f"{endpoint_protocol};{account_name};{account_key};{endpoint_suffix}"
    )

    # Connect to Azure Blob storage
    console.print("Connecting to Azure Blob storage...", style="info")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)
        console.print(
            f"Successfully connected to container: {container_name}", style="success"
        )
        return container_client, container_name, storage_account, storage_key
    except Exception as e:
        console.print(
            f"Error connecting to Azure Blob storage: {str(e)}", style="error"
        )
        return None, container_name, storage_account, storage_key


def upload_files(
    container_client: BlobServiceClient,
    console: Console,
    local_folder_path: Path,
    blob_folder: str,
) -> None:
    """Upload files from a local folder to Azure Blob storage.

    Args:
        container_client: Azure Blob container client
        console: Console object for pretty printing
        local_folder_path: Path to the local folder
        blob_folder: Base folder path in the blob container
    """
    # Get list of files to upload
    files = list(local_folder_path.glob("**/*"))
    files = [f for f in files if f.is_file()]

    if not files:
        console.print(f"No files found in {local_folder_path}", style="warning")
        return

    console.print(f"Uploading {len(files)} files to {blob_folder}/...", style="info")

    # Set content types based on file extensions
    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    # Upload files with progress bar
    for file in tqdm(files, desc=f"Uploading to {blob_folder}"):
        # Get relative path to maintain folder structure
        rel_path = file.relative_to(local_folder_path)
        blob_path = f"{blob_folder}/{rel_path}".replace("\\", "/")

        # Set content type based on file extension
        content_type = content_types.get(
            file.suffix.lower(), "application/octet-stream"
        )
        content_settings = ContentSettings(content_type=content_type)

        # Upload the file
        try:
            blob_client = container_client.get_blob_client(blob_path)
            with open(file, "rb") as data:
                blob_client.upload_blob(
                    data, overwrite=True, content_settings=content_settings
                )
        except Exception as e:
            console.print(f"Error uploading {file}: {str(e)}", style="error")


def upload_base_data_to_azure_blob(
    console: Console, path_images: str, path_raw_masks: str, path_processed_masks: str
):
    """Upload local data to Azure Blob storage.

    Args:
        console: Console object for pretty printing
        path_images: Path to the images directory (may contain config variables)
        path_raw_masks: Path to the raw mask labels directory
            (may contain config variables)
        path_processed_masks: Path to the processed mask labels directory
            (may contain config variables)
    """

    console.print("Preparing to upload data to Azure Blob storage...", style="info")

    # Check if paths exist
    paths_to_check = {
        "images": Path(path_images),
        "raw_masks": Path(path_raw_masks),
        "processed_masks": Path(path_processed_masks),
    }

    for name, path in paths_to_check.items():
        if not path.exists():
            console.print(f"Error: {name} path does not exist: {path}", style="error")
            return

    console.print("Local data paths:", style="info")
    console.print(f"  Images: {path_images}", style="info")
    console.print(f"  Raw masks: {path_raw_masks}", style="info")
    console.print(f"  Processed masks: {path_processed_masks}", style="info")

    # Confirm before uploading
    console.print("\nReady to upload data to Azure Blob storage", style="warning")
    console.print(
        "This may take a while depending on the size of your data.", style="warning"
    )

    confirmation = input("Do you want to continue? (y/n): ")

    if confirmation.strip().lower() != "y":
        console.print("Upload canceled.", style="warning")
        return

    # Get storage config and connect to Azure Blob storage in one step
    container_client, _, _, _ = get_azure_storage_client(console, require_key=True)

    if not container_client:
        console.print("Failed to connect to Azure Blob storage", style="error")
        return

    # Upload images
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["images"],
        blob_folder="rockmass",
    )

    # Upload processed masks
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["processed_masks"],
        blob_folder="label/binary",
    )

    # Upload raw masks
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["raw_masks"],
        blob_folder="label/raw_data",
    )

    console.print("\nUpload complete!", style="success")
    console.print("You can now register the data assets using:", style="info")
    console.print(
        "python scripts/azure_manage_data_assets.py "
        "azure_data_assets.command=register-base-datasets",
        style="info",
    )


def register_base_datasets(ml_client: MLClient, console: Console):
    """Register base datasets in Azure ML.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
    """

    console.print("Registering base datasets in Azure ML", style="info")

    # Get storage config and connect to Azure Blob storage in one step
    _, container_name, storage_account, _ = get_azure_storage_client(
        console, require_key=False
    )

    console.print("Reading dataset paths...", style="info")
    # Define the dataset paths using proper Azure Blob storage URL format
    # wasbs is a protocol identifier for Azure Blob Storage secure connection
    # (with SSL/TLS)
    dataset_paths = {
        "images": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/rockmass",
        "masks": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/binary",
        "raw": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/raw_data",
    }

    console.print("Using dataset paths:", style="info")
    console.print(f"  Images: {dataset_paths['images']}", style="info")
    console.print(f"  Masks: {dataset_paths['masks']}", style="info")
    console.print(f"  Raw: {dataset_paths['raw']}", style="info")

    # Generate a version based on current timestamp for dataset versioning in Azure ML
    version = datetime.now().strftime("%Y%m%d.%H%M")

    # Register the datasets with proper metadata
    assets = {}

    # 1. Register the images dataset
    #########################################################
    console.print("Registering images dataset...", style="info")
    images_metadata = {
        "content_type": "image/jpeg, image/png",
        "description": "Rock mass joint images",
        "domain": "geology",
        "purpose": "segmentation",
    }
    images_asset = register_data_asset(
        ml_client=ml_client,
        console=console,
        name="rock_images",
        version=version,
        description="Rock mass joint images for segmentation",
        path=dataset_paths["images"],
        asset_type=AssetTypes.URI_FOLDER,
        tags={"domain": "geology", "type": "image"},
        metadata=images_metadata,
    )
    assets["images"] = images_asset

    # 2. Register the masks dataset
    #########################################################
    console.print("Registering masks dataset...", style="info")
    masks_metadata = {
        "content_type": "image/png",
        "description": "Binary masks for rock mass joints",
        "domain": "geology",
        "purpose": "segmentation",
        "related_dataset": f"rock_images:{version}",
    }
    masks_asset = register_data_asset(
        ml_client=ml_client,
        console=console,
        name="rock_masks",
        version=version,
        description="Binary masks for rock mass joint segmentation",
        path=dataset_paths["masks"],
        asset_type=AssetTypes.URI_FOLDER,
        tags={"domain": "geology", "type": "mask"},
        metadata=masks_metadata,
    )
    assets["masks"] = masks_asset

    # 3. Register raw data
    #########################################################
    console.print("Registering raw data...", style="info")
    raw_metadata = {
        "description": "Raw unprocessed data for rock mass joints",
        "domain": "geology",
        "purpose": "archive",
        "processed_datasets": f"rock_images:{version},rock_masks:{version}",
    }
    raw_asset = register_data_asset(
        ml_client=ml_client,
        console=console,
        name="rock_raw_data",
        version=version,
        description="Raw unprocessed rock mass joint data",
        path=dataset_paths["raw"],
        asset_type=AssetTypes.URI_FOLDER,
        tags={"domain": "geology", "type": "raw"},
        metadata=raw_metadata,
    )
    assets["raw"] = raw_asset

    # Print summary of registered assets
    console.print("\nSummary of registered data assets:", style="success")
    for name, asset in assets.items():
        console.print(f"- {name}: {asset.name} (version {asset.version})", style="info")

    console.print(
        "\nThese data assets can now be used in your Azure ML pipelines!",
        style="success",
    )


def upload_split_data_to_azure_blob(console: Console, experiment_strategy: str):
    """Upload split files (train/val/test) to Azure Blob Storage under
    splits/<strategy>/"""
    # Get storage config and connect to Azure Blob storage in one step
    container_client, _, _, _ = get_azure_storage_client(console, require_key=True)

    # Format the split subdirectory name
    split_subfolder = experiment_strategy.lower().replace(".", "_").replace(" ", "_")
    split_dir = Path("data/model_ready/splits") / split_subfolder

    # Validate split files exist
    if not split_dir.exists():
        console.print(
            f"Error: Split directory '{split_dir}' not found. Generate splits first.",
            style="error",
        )
        sys.exit(1)

    for fname in ["train.json", "val.json", "test.json"]:
        if not (split_dir / fname).exists():
            console.print(
                f"Error: Missing split file: {split_dir / fname}", style="error"
            )
            sys.exit(1)

    try:
        blob_folder = f"splits/{split_subfolder}"
        upload_files(
            container_client=container_client,
            console=console,
            local_folder_path=split_dir,
            blob_folder=blob_folder,
        )
        console.print(
            f"Uploaded split files to Azure Blob Storage: {blob_folder}",
            style="success",
        )
    except Exception as e:
        console.print(f"Error uploading split files: {str(e)}", style="error")
        sys.exit(1)


def register_split_data_asset(
    ml_client: MLClient, console: Console, experiment_strategy: str
):
    """Register the split folder in Azure Blob Storage as a data asset in Azure ML."""
    split_subfolder = experiment_strategy.lower().replace(".", "_").replace(" ", "_")

    # Get storage configuration in one step
    _, container_name, storage_account, _ = get_azure_storage_client(console)

    version = datetime.now().strftime("%Y%m%d.%H%M")
    asset_name = f"split_{split_subfolder}"
    split_name = split_subfolder.replace("_", " ")
    description = f"Train/val/test split for {split_name}"

    # Get base dataset versions
    rock_images = get_data_asset(ml_client, console, "rock_images")
    rock_masks = get_data_asset(ml_client, console, "rock_masks")

    metadata = {
        "base_images_dataset": f"rock_images:{rock_images.version}",
        "base_masks_dataset": f"rock_masks:{rock_masks.version}",
        "split_strategy": experiment_strategy,
    }

    splits_blob_uri = (
        f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/"
        f"splits/{split_subfolder}"
    )
    asset = register_data_asset(
        ml_client=ml_client,
        console=console,
        name=asset_name,
        version=version,
        description=description,
        path=splits_blob_uri,
        asset_type=AssetTypes.URI_FOLDER,
        tags={"domain": "geology", "type": "data_split", "split": split_subfolder},
        metadata=metadata,
    )
    console.print(
        "\nSuccessfully registered split asset: "
        f"{asset.name} (version {asset.version})",
        style="success",
    )
    console.print(f"Asset path: {splits_blob_uri}", style="info")


def list_data_assets(ml_client: MLClient, console: Console, asset_name: str = None):
    """List all data assets or versions of a specific data asset.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
        asset_name: Optional name of a specific asset to list versions for
    """

    if asset_name:  # List versions of a specific asset
        console.print(f"Listing versions of data asset: {asset_name}", style="info")
        df = list_data_asset_versions(ml_client, console, asset_name)

        if df.empty:
            console.print(
                f"No versions found for asset '{asset_name}'", style="warning"
            )
        else:
            console.print(f"\nFound {len(df)} versions:", style="info")
            print(df.to_string(index=False))

    else:
        # List all data assets
        console.print("Listing all data assets in the workspace:", style="info")

        try:
            # Get all data assets
            assets = list(ml_client.data.list())

            if not assets:
                console.print("No data assets found in the workspace", style="warning")
                return

            # Group assets by name to show latest version
            assets_by_name = {}
            for asset in assets:
                if asset.name not in assets_by_name:
                    assets_by_name[asset.name] = []

                # Extract the relevant properties safely
                asset_info = {
                    "name": asset.name,
                    "version": asset.version,
                    "created_at": (
                        asset.creation_context.created_at
                        if hasattr(asset.creation_context, "created_at")
                        else "Unknown"
                    ),
                    "type": asset.type if hasattr(asset, "type") else "Unknown",
                }
                assets_by_name[asset.name].append(asset_info)

            # Print summary of latest versions
            console.print(
                f"\nFound {len(assets_by_name)} unique data assets:", style="info"
            )

            # Create a table of latest versions
            table_data = []
            for name, versions in assets_by_name.items():
                # Sort by version descending and get the latest
                latest = sorted(versions, key=lambda x: x["version"], reverse=True)[0]
                table_data.append(latest)

            # Convert to DataFrame for pretty printing
            if table_data:
                df = pd.DataFrame(table_data)
                df = df.sort_values("name").reset_index(drop=True)
                print(df.to_string(index=False))
            else:
                console.print("No data assets found to display", style="warning")

        except Exception as e:
            console.print(f"Error listing data assets: {str(e)}", style="error")
            console.print(
                "This could be due to permission issues or "
                "invalid Azure ML configuration.",
                style="warning",
            )


def compare_data_asset_versions(
    ml_client: MLClient, console: Console, name: str, version1: str, version2: str
) -> dict[str, str | list[str]]:
    """
    Compare two versions of a data asset.

    Args:
        ml_client: Azure ML client
        console: Console object for pretty printing
        name: Name of the data asset
        version1: First version to compare
        version2: Second version to compare

    Returns:
        Dictionary with comparison results
    """

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


def compare_assets(
    ml_client: MLClient, console: Console, asset_name: str, version1: str, version2: str
):
    """Compare two versions of a data asset.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
        asset_name: Name of the asset to compare
        version1: First version to compare
        version2: Second version to compare
    """

    if not all([asset_name, version1, version2]):
        err_msg = (
            "Error: Missing required parameters. Please provide asset_name, "
            "version1, and version2."
        )
        console.print(err_msg, style="error")
        return console.print(
            f"Comparing versions {version1} and {version2} of asset '{asset_name}'",
            style="info",
        )

    # Compare the versions
    comparison = compare_data_asset_versions(
        ml_client, console, asset_name, version1, version2
    )

    if "error" in comparison:
        console.print(f"Error comparing versions: {comparison['error']}", style="error")
        return

    # Print comparison results
    console.print("\nComparison results:", style="info")
    console.print(f"Asset name: {comparison['name']}", style="info")
    console.print(
        f"Comparing version {comparison['version1']} with {comparison['version2']}",
        style="info",
    )
    console.print(f"Path changed: {comparison['path_changed']}", style="info")

    if comparison["metadata_differences"]:
        console.print("\nMetadata differences:", style="warning")
        for diff in comparison["metadata_differences"]:
            console.print(f"- {diff}")
    else:
        console.print("\nNo metadata differences found", style="success")


def prepare_and_save_dataset_splits(
    console: Console,
    images_directory: str | Path,
    labels_directory: str | Path,
    experiment_strategy: str,
    dataset_strategies: dict[str, dict[str, list[str]]],
    dataset_prefixes: dict[str, list[str]],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
):
    """
    Prepare and save dataset splits for use in Azure ML.

    This function:
    1. Uses the dataset prefixes from the config to filter files.
    2. Gets the data files based on those prefixes.
    3. If the train and test sets are disjoint, no splitting is performed
       and all train and test files are used as provided, regardless of
       the fraction values.
    4. Otherwise, splits the data into train/val/test sets according to
       the provided fractions.
    5. Saves the resulting splits as JSON files in the correct subfolder
       under data/model_ready/splits/<strategy>/<subtype>/.

    Args:
        console: Console object for pretty printing.
        images_directory: Path to the images directory.
        labels_directory: Path to the labels directory.
        experiment_strategy: The experiment strategy (used for split folder).
        dataset_strategies: Dictionary of dataset strategies.
        dataset_prefixes: Dictionary of dataset prefixes.
        train_fraction: Fraction of data to use for training.
        val_fraction: Fraction of data to use for validation.
        test_fraction: Fraction of data to use for testing.
    """

    console.print("Preparing dataset splits for Azure ML", style="info")

    images_directory = Path(images_directory)
    labels_directory = Path(labels_directory)

    if not images_directory.exists():
        console.print(
            f"Error: Images directory '{images_directory}' not found", style="error"
        )
        sys.exit(1)

    if not labels_directory.exists():
        console.print(
            f"Error: Labels directory '{labels_directory}' not found", style="error"
        )
        sys.exit(1)

    # Determine split subfolder based on experiment_strategy
    # Example: verification_box, dfn_to_slope, etc.
    # User should set experiment_strategy to e.g. 'verification_box',
    # 'dfn_to_slope', etc.
    split_subfolder = experiment_strategy.lower().replace(".", "_").replace(" ", "_")
    split_dir = Path("data/model_ready/splits") / split_subfolder
    if not split_dir.exists():
        split_dir.mkdir(parents=True, exist_ok=True)

    # Break example command into multiple lines
    cmd = (
        "python scripts/azure_manage_data_assets.py "
        "azure_data_assets.command=register-splits"
    )
    console.print(cmd, style="info")

    console.print(f"Split files will be saved to: {split_dir}", style="info")

    # returns the prefixes for train and test datasets
    prefixes = get_datasets_prefixes(
        experiment_strategy=experiment_strategy,
        dataset_strategies=dataset_strategies,
        dataset_prefixes=dataset_prefixes,
    )

    # Example: train_prefixes: ['FracMan'], test_prefixes: ['Larvik', 'RV4]
    train_prefixes_list, test_prefixes_list = (
        prefixes["train_prefixes"],
        prefixes["test_prefixes"],
    )

    # Show which dataset prefixes are used for training and testing
    console.print(f"Train prefixes: {train_prefixes_list}", style="info")
    console.print(f"Test prefixes: {test_prefixes_list}", style="info")

    console.print("Getting data files for given prefixes...", style="info")
    train_files, test_files = get_data_files(
        images_directory, labels_directory, train_prefixes_list, test_prefixes_list
    )

    train_set = set(train_files)
    test_set = set(test_files)
    if train_set.isdisjoint(test_set):
        console.print(
            "Train and test sets are disjoint. No splitting will be performed; "
            "using all train and test files as provided.",
            style="info",
        )

        train_list = list(train_files)
        val_list = []
        test_list = list(test_files)
    else:
        console.print(
            f"Splitting data with train fraction: {train_fraction}, "
            f"val fraction: {val_fraction}, "
            f"test fraction: {test_fraction}...",
            style="info",
        )
        train_list, val_list, test_list = split_data(
            train_files,
            test_files,
            train_frac=train_fraction,
            val_frac=val_fraction,
            test_frac=test_fraction,
        )

    # Save splits in the correct subfolder
    with open(split_dir / "train.json", "w") as f:
        json.dump(train_list, f)
    with open(split_dir / "val.json", "w") as f:
        json.dump(val_list, f)
    with open(split_dir / "test.json", "w") as f:
        json.dump(test_list, f)

    console.print(f"Number of training samples: {len(train_list)}", style="success")
    console.print(f"Number of validation samples: {len(val_list)}", style="success")
    console.print(f"Number of test samples: {len(test_list)}", style="success")

    console.print("\nSplit files have been saved to:", style="success")
    console.print(f"  - {split_dir / 'train.json'}", style="info")
    console.print(f"  - {split_dir / 'val.json'}", style="info")
    console.print(f"  - {split_dir / 'test.json'}", style="info")
    console.print(
        "\nYou can now upload and register these splits in Azure ML using:",
        style="info",
    )
    console.print(
        "  python scripts/azure_manage_data_assets.py "
        "azure_data_assets.command=upload-splits",
        style="info",
    )
    console.print(
        "  python scripts/azure_manage_data_assets.py "
        "azure_data_assets.command=register-splits",
        style="info",
    )


def retrieve_and_validate_data_assets(
    ml_client: MLClient,
    console: Console,
    experiment_strategy: str,
    get_data_asset_func: Callable,
) -> tuple[Any, Any, Any]:
    """
    Retrieve and validate all required data assets for training.

    This function handles data asset retrieval with proper error handling,
    retry logic, and permission validation.

    Args:
        ml_client: Azure ML client
        console: Console for logging
        experiment_strategy: Strategy name for selecting split assets
        get_data_asset_func: Function to retrieve data assets

    Returns:
        tuple: (images_dataset, masks_dataset, splits_dataset)

    Raises:
        SystemExit: If data assets cannot be retrieved
    """
    try:
        console.print("Retrieving latest data assets from Azure ML...", style="info")

        # Apply retry logic to data asset retrieval
        get_data_asset_with_retry = retry_azure_operation(
            get_data_asset_func, operation_name="Data asset retrieval"
        )

        # Get base image and mask datasets
        images_dataset = get_data_asset_with_retry(ml_client, console, "rock_images")
        masks_dataset = get_data_asset_with_retry(ml_client, console, "rock_masks")

        # Select the correct split asset based on experiment strategy
        strategy = experiment_strategy.lower()
        split_asset_name = f"split_{strategy.replace('.', '_').replace(' ', '_')}"
        splits_dataset = get_data_asset_with_retry(ml_client, console, split_asset_name)

        # Log dataset information
        console.print(
            f"✓ Using split asset: {split_asset_name} (v{splits_dataset.version})",
            style="success",
        )
        console.print(
            f"✓ Using images dataset version: {images_dataset.version}", style="success"
        )
        console.print(
            f"✓ Using masks dataset version: {masks_dataset.version}", style="success"
        )

        # Verify data asset permissions
        _validate_data_asset_permissions(ml_client, console)

        return images_dataset, masks_dataset, splits_dataset

    except ResourceNotFoundError as e:
        console.print(f"Data asset not found: {str(e)}", style="error")
        console.print(
            "Solution: Register data assets using:\n"
            "python scripts/azure_manage_data_assets.py "
            "azure_data_assets.command=register-all",
            style="info",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"Error retrieving data assets: {str(e)}", style="error")
        console.print(
            "Solution: Ensure data assets are registered and you have "
            "access permissions",
            style="info",
        )
        sys.exit(1)


def _validate_data_asset_permissions(ml_client: MLClient, console: Console) -> None:
    """Validate permissions to access data assets."""
    try:
        console.print("Verifying data asset permissions...", style="info")
        assets = ml_client.data.list(max_results=5)
        asset_count = sum(1 for _ in assets)
        console.print(
            f"✓ Verified access to {asset_count} data assets", style="success"
        )
    except Exception as perm_e:
        console.print(
            f"Warning: Limited data asset permissions: {str(perm_e)}", style="warning"
        )
