"""
Manage Azure ML data assets for rock mass segmentation.

This script demonstrates how to use the Azure ML data asset management functionality
to handle your rock mass segmentation datasets properly.

Example usage:
    # Register base datasets
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=register-base-datasets

    # Register dataset splits
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=register-splits

    # List assets with specific name
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=list-assets \
        asset_name=rock_images

    # Compare two versions of an asset
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=compare-assets \
        asset_name=rock_images version1=1 version2=2

    # Upload new data
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=upload-data

    # Generate new dataset splits
    python scripts/manage_azure_data_assets.py \
        azure_data_assets.command=generate-splits
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.storage.blob import BlobServiceClient, ContentSettings
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from tqdm import tqdm

from ml_segmentation.azure_data_assets import (
    compare_data_asset_versions,
    connect_to_azure_ml,
    get_data_asset,
    list_data_asset_versions,
    register_data_asset,
    setup_azure_environment,
)
from ml_segmentation.data_loading import (
    get_data_files,
    get_datasets_prefixes,
    split_data,
)
from ml_segmentation.schema_config import AzureDataAssetsCommand, ConfigSchema
from ml_segmentation.utility import seed_everything


def get_command_description(command: AzureDataAssetsCommand) -> str:
    """Get a descriptive text for an Azure Data Assets command.

    This function provides human-readable descriptions of each command beyond
    the kebab-case identifiers in the enum. These descriptions are used for
    UI presentation and help text to improve usability, while maintaining
    separation between command identifiers and their presentation layer.

    Args:
        command: The AzureDataAssetsCommand enum value

    Returns:
        A description of what the command does
    """
    descriptions = {
        AzureDataAssetsCommand.REGISTER_BASE_DATASETS: (
            "Register base datasets in Azure ML"
        ),
        AzureDataAssetsCommand.REGISTER_SPLITS: "Register dataset splits in Azure ML",
        AzureDataAssetsCommand.UPLOAD_SPLITS: (
            "Upload dataset splits to Azure Blob Storage"
        ),
        AzureDataAssetsCommand.LIST_ASSETS: "List data assets in Azure ML",
        AzureDataAssetsCommand.COMPARE_ASSETS: "Compare two versions of a data asset",
        AzureDataAssetsCommand.UPLOAD_DATA: "Upload local data to Azure Blob storage",
        AzureDataAssetsCommand.GENERATE_SPLITS: "Generate dataset splits locally",
        AzureDataAssetsCommand.PROCESS_ALL_SPLITS: (
            "Process all experiment strategies: "
            "generate, upload, and register splits in one operation"
        ),
    }
    return descriptions.get(command, "Unknown command")


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


def get_latest_dataset_version(prefix: str, container_client: BlobServiceClient) -> str:
    """Get the latest dataset version directory for a given prefix."""
    blobs = container_client.list_blobs(name_starts_with=prefix)
    versions = set()
    for blob in blobs:
        # Extract version from path (e.g., 'rockmass/v20240522/file.jpg' -> 'v20240522')
        parts = blob.name.split("/")
        if len(parts) > 1 and parts[1].startswith("v"):
            versions.add(parts[1])
    if not versions:
        raise ValueError(f"No version directories found for {prefix}")
    return max(versions)  # Latest version based on string comparison


def register_base_datasets(ml_client: MLClient, console: Console):
    """Register base datasets in Azure ML.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
    """

    console.print("Registering base datasets in Azure ML", style="info")

    # Get storage config and connect to Azure Blob storage in one step
    container_client, container_name, storage_account, _ = get_azure_storage_client(
        console, require_key=False
    )

    # Define the dataset paths using proper Azure Blob storage URL format
    console.print("Reading dataset paths...", style="info")

    try:
        latest_images_version = get_latest_dataset_version("rockmass", container_client)
        latest_masks_version = get_latest_dataset_version(
            "label/binary", container_client
        )
        latest_raw_version = get_latest_dataset_version(
            "label/raw_data", container_client
        )

        # wasbs is a protocol identifier for Azure Blob Storage secure connection
        # (with SSL/TLS)
        dataset_paths = {
            "images": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/rockmass/{latest_images_version}",
            "masks": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/binary/{latest_masks_version}",
            "raw": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/raw_data/{latest_raw_version}",
        }

        console.print("Using latest versions:", style="info")
        console.print(f"  Images: {latest_images_version}", style="info")
        console.print(f"  Masks: {latest_masks_version}", style="info")
        console.print(f"  Raw: {latest_raw_version}", style="info")

    except Exception as e:
        console.print(f"Error finding latest versions: {str(e)}", style="error")
        sys.exit(1)

    # Generate a version based on current timestamp
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
    container_client, container_name, storage_account, _ = get_azure_storage_client(
        console, require_key=True
    )

    if not container_client:
        console.print("Failed to connect to Azure Blob storage", style="error")
        sys.exit(1)

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
    description = f"Train/val/test split for {split_subfolder.replace('_', ' ')}"

    # Get base dataset versions
    rock_images = get_data_asset(ml_client, "rock_images")
    rock_masks = get_data_asset(ml_client, "rock_masks")

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

    if asset_name:
        # List versions of a specific asset
        console.print(f"Listing versions of data asset: {asset_name}", style="info")
        df = list_data_asset_versions(ml_client, asset_name)

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
        return

    console.print(
        f"Comparing versions {version1} and {version2} of asset '{asset_name}'",
        style="info",
    )

    # Compare the versions
    comparison = compare_data_asset_versions(ml_client, asset_name, version1, version2)

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

    # Check if paths exist - paths have already been resolved by Hydra
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
    container_client, container_name, storage_account, _ = get_azure_storage_client(
        console, require_key=True
    )

    if not container_client:
        console.print("Failed to connect to Azure Blob storage", style="error")
        return

    # Generate version for this upload
    version = f"v{datetime.now().strftime('%Y%m%d')}"

    # Upload images with versioning
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["images"],
        blob_folder="rockmass",
        version=version,
        preserve_version=True,
    )

    # Upload processed masks with versioning
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["processed_masks"],
        blob_folder="label/binary",
        version=version,
        preserve_version=True,
    )

    # Upload raw masks with versioning
    upload_files(
        container_client=container_client,
        console=console,
        local_folder_path=paths_to_check["raw_masks"],
        blob_folder="label/raw_data",
        version=version,
        preserve_version=True,
    )

    console.print("\nUpload complete!", style="success")
    console.print("You can now register the data assets using:", style="info")
    console.print(
        "python manage_azure_data_assets.py "
        "azure_data_assets.command=register-base-datasets",
        style="info",
    )


def upload_files(
    container_client,
    console: Console,
    local_folder_path,
    blob_folder,
    version: str | None = None,
    preserve_version: bool = True,
):
    """Upload files from a local folder to Azure Blob storage.

    Args:
        container_client: Azure Blob container client
        console: Console object for pretty printing
        local_folder_path: Path to the local folder
        blob_folder: Base folder path in the blob container
        version: Optional version string. If None and preserve_version is True,
                generates version based on current date
        preserve_version: Whether to use versioned directories. If False, files are
                        uploaded directly to blob_folder without version subfolder
    """  # Get list of files to upload
    files = list(local_folder_path.glob("**/*"))
    files = [f for f in files if f.is_file()]

    if not files:
        console.print(f"No files found in {local_folder_path}", style="warning")
        return

    # If using versioned directories, update the blob folder path
    if preserve_version:
        blob_folder = get_dataset_version_path(blob_folder, version)

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
        "python scripts/manage_azure_data_assets.py "
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


def get_dataset_version_path(base_path: str, version: str | None = None) -> str:
    """Get the versioned path for a dataset in Azure Blob storage.

    Args:
        base_path: The base path for the dataset (e.g., 'rockmass' or 'label')
        version: Optional version string. If None, generates version based on
                current date

    Returns:
        str: The versioned path (e.g., 'rockmass/v20240522' or 'label/binary/v20240522')
    """
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d')}"

    return f"{base_path}/{version}"


def get_and_validate_azure_storage_config(
    console: Console, require_key: bool = False
) -> tuple[str, str, str, str | None]:
    """Get and validate Azure storage configuration from environment variables.

    Args:
        console: Console object for pretty printing
        require_key: Whether to require the storage key (default: False)

    Returns:
        tuple: (storage_account, container_name, connection_string, storage_key)
        where storage_key is None if require_key is False

    Raises:
        SystemExit: If required configuration is missing
    """
    _, container_name, storage_account, storage_key = get_azure_storage_client(
        console=console, require_key=require_key
    )

    # Build connection string for Azure storage account
    endpoint_protocol = "DefaultEndpointsProtocol=https"
    account_name = f"AccountName={storage_account}"
    account_key = f"AccountKey={storage_key}" if storage_key else ""
    endpoint_suffix = "EndpointSuffix=core.windows.net"

    connection_string = None
    if storage_key:
        connection_string = (
            f"{endpoint_protocol};{account_name};{account_key};{endpoint_suffix}"
        )

    return storage_account, container_name, connection_string, storage_key


def connect_to_azure_blob_storage(
    console: Console,
    require_key: bool = False,
    storage_account: str = None,
    container_name: str = None,
    connection_string: str = None,
) -> tuple[BlobServiceClient, str, str] | None:
    """Connect to Azure Blob storage and return a client.

    Args:
        console: Console object for pretty printing
        require_key: Whether to require the storage key (default: False)
        storage_account: Optional storage account name. If not provided, will be
        retrieved from environment.
        container_name: Optional container name. If not provided, will be retrieved
        from environment.
        connection_string: Optional connection string. If not provided, will be built
        from environment variables.

    Returns:
        tuple: (container_client, container_name, storage_account)
            or None if connection fails

    Raises:
        SystemExit: If required configuration is missing
    """
    container_client, container_name, storage_account, _ = get_azure_storage_client(
        console=console,
        require_key=require_key,
        storage_account=storage_account,
        container_name=container_name,
        connection_string=connection_string,
    )

    if container_client:
        return container_client, container_name, storage_account
    else:
        return None


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for the script using Hydra.

    Args:
        cfg: The Hydra configuration object
    """
    # Configure logging to reduce verbose Azure client output
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)

    # Initialize environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = setup_azure_environment()

    try:
        # Convert OmegaConf to a Python dictionary and validate with Pydantic
        cfg_dict = OmegaConf.to_object(cfg)
        pcfg = ConfigSchema(**cfg_dict)

        # Set random seed at the beginning to affect all operations
        seed_everything(pcfg.experiment.seed)

        console.print(f"Using configuration: {pcfg.azure_data_assets}", style="info")

        # Get command from validated config
        command = pcfg.azure_data_assets.command

        # Only connect to Azure ML if the command requires it and is valid
        ml_client = None
        try:
            valid_commands_requiring_connection = [
                AzureDataAssetsCommand.REGISTER_BASE_DATASETS,
                AzureDataAssetsCommand.REGISTER_SPLITS,
                AzureDataAssetsCommand.LIST_ASSETS,
                AzureDataAssetsCommand.COMPARE_ASSETS,
                AzureDataAssetsCommand.PROCESS_ALL_SPLITS,
            ]

            if command in valid_commands_requiring_connection:
                ml_client = connect_to_azure_ml(
                    subscription_id=subscription_id,
                    resource_group=resource_group,
                    workspace_name=workspace_name,
                )

            # Get asset parameters from validated config
            asset_name = pcfg.azure_data_assets.asset_name
            version1 = pcfg.azure_data_assets.version1
            version2 = pcfg.azure_data_assets.version2

            # Execute the appropriate command based on the enum value
            match command:
                case AzureDataAssetsCommand.UPLOAD_DATA:
                    upload_base_data_to_azure_blob(
                        console,
                        pcfg.dataset.path_images,
                        pcfg.dataset.path_raw_mask_labels,
                        pcfg.dataset.path_processed_mask_labels,
                    )
                case AzureDataAssetsCommand.REGISTER_BASE_DATASETS:
                    register_base_datasets(ml_client, console)
                case AzureDataAssetsCommand.GENERATE_SPLITS:
                    prepare_and_save_dataset_splits(
                        console,
                        images_directory=pcfg.dataset.path_images,
                        labels_directory=pcfg.dataset.path_processed_mask_labels,
                        experiment_strategy=pcfg.experiment.experiment_strategy,
                        dataset_strategies=pcfg.experiment.dataset_strategies,
                        dataset_prefixes=pcfg.dataset.prefixes,
                        train_fraction=pcfg.experiment.train_fraction,
                        val_fraction=pcfg.experiment.val_fraction,
                        test_fraction=pcfg.experiment.test_fraction,
                    )
                case AzureDataAssetsCommand.UPLOAD_SPLITS:
                    upload_split_data_to_azure_blob(
                        console,
                        pcfg.experiment.experiment_strategy,
                    )
                case AzureDataAssetsCommand.REGISTER_SPLITS:
                    register_split_data_asset(
                        ml_client,
                        console,
                        pcfg.experiment.experiment_strategy,
                    )
                case AzureDataAssetsCommand.LIST_ASSETS:
                    list_data_assets(ml_client, console, asset_name)
                case AzureDataAssetsCommand.COMPARE_ASSETS:
                    compare_assets(ml_client, console, asset_name, version1, version2)
                case AzureDataAssetsCommand.PROCESS_ALL_SPLITS:
                    # List of experiment strategies to process
                    strategies = [
                        "verification_box",
                        "verification_dfn",
                        "main_objective_dfn_rock_slope",
                        "main_objective_dfn_box",
                        "main_objective_box_rock_slope",
                        "main_objective_box_box",
                    ]
                    for strategy in strategies:
                        console.print(
                            "\n================ Processing strategy: "
                            f"{strategy} ================",
                            style="bold green",
                        )
                        try:
                            # Step 1: Generate splits
                            console.print(
                                f"Step 1: Generating splits for {strategy}...",
                                style="yellow",
                            )
                            prepare_and_save_dataset_splits(
                                console,
                                images_directory=pcfg.dataset.path_images,
                                labels_directory=pcfg.dataset.path_processed_mask_labels,
                                experiment_strategy=strategy,
                                dataset_strategies=pcfg.experiment.dataset_strategies,
                                dataset_prefixes=pcfg.dataset.prefixes,
                                train_fraction=pcfg.experiment.train_fraction,
                                val_fraction=pcfg.experiment.val_fraction,
                                test_fraction=pcfg.experiment.test_fraction,
                            )
                            # Step 2: Upload splits
                            console.print(
                                "Step 2: Uploading splits for "
                                + f"{strategy} to Azure Blob Storage...",
                                style="yellow",
                            )
                            upload_split_data_to_azure_blob(console, strategy)
                            # Step 3: Register splits
                            console.print(
                                "Step 3: Registering splits for "
                                + f"{strategy} in Azure ML...",
                                style="yellow",
                            )
                            register_split_data_asset(ml_client, console, strategy)
                            console.print(
                                f"Successfully processed {strategy}", style="green"
                            )
                        except Exception as e:
                            console.print(
                                f"Error processing {strategy}: {str(e)}", style="red"
                            )
                    console.print(
                        "\nAll strategies processed. Check above for any errors.",
                        style="bold green",
                    )
                case _:
                    show_command_help(console)
        except ValueError as e:
            error_str = str(e)
            # Check for specific types of errors and provide helpful messages
            if "azure_data_assets.command" in error_str:
                console.print(f"Error: {error_str}", style="error")
                show_command_help(console)
            else:
                # For other errors, show the error but still with command help
                console.print(f"Configuration error: {error_str}", style="error")
                console.print(
                    "Check your configuration and try again.", style="warning"
                )
                show_command_help(console)
    except Exception as e:
        # Catch any other exceptions
        console.print(f"Error: {str(e)}", style="error")
        show_command_help(console)


def show_command_help(console: Console):
    """Display available commands and usage examples.

    Args:
        console: Console object for pretty printing
    """
    # Print available commands
    console.print("Available commands:", style="info")
    for cmd in AzureDataAssetsCommand:
        console.print(f"  {cmd.value}: {get_command_description(cmd)}", style="info")

    console.print("\nUsage examples:", style="info")
    # Make long command examples more readable by splitting into multiple lines
    base_cmd = "python scripts/manage_azure_data_assets.py azure_data_assets.command="

    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.REGISTER_BASE_DATASETS.value}",
        style="info",
    )
    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.UPLOAD_SPLITS.value}",
        style="info",
    )
    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.REGISTER_SPLITS.value}",
        style="info",
    )


if __name__ == "__main__":
    main()
