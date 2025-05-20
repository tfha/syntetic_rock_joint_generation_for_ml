"""
Manage Azure ML data assets for rock mass segmentation.

This script demonstrates how to use the Azure ML data asset management functionality
to handle your rock mass segmentation datasets properly.

Usage:
    python scripts/manage_azure_data_assets.py azure_data_assets.command=register-base-datasets
    python scripts/manage_azure_data_assets.py azure_data_assets.command=register-splits
    python scripts/manage_azure_data_assets.py azure_data_assets.command=list-assets asset_name=rock_images
    python scripts/manage_azure_data_assets.py azure_data_assets.command=compare-assets asset_name=rock_images version1=1 version2=2
    python scripts/manage_azure_data_assets.py azure_data_assets.command=upload-data
    python scripts/manage_azure_data_assets.py azure_data_assets.command=generate-splits
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

    Args:
        command: The AzureDataAssetsCommand enum value

    Returns:
        A description of what the command does
    """
    descriptions = {
        AzureDataAssetsCommand.REGISTER_BASE_DATASETS: "Register base datasets in Azure ML",
        AzureDataAssetsCommand.REGISTER_SPLITS: "Register dataset splits in Azure ML",
        AzureDataAssetsCommand.LIST_ASSETS: "List data assets in Azure ML",
        AzureDataAssetsCommand.COMPARE_ASSETS: "Compare two versions of a data asset",
        AzureDataAssetsCommand.UPLOAD_DATA: "Upload local data to Azure Blob storage",
        AzureDataAssetsCommand.GENERATE_SPLITS: "Generate dataset splits locally",
        AzureDataAssetsCommand.UPLOAD_SPLITS: "Upload dataset split JSON files to Azure Blob storage",
    }
    return descriptions.get(command, "Unknown command")


def register_base_datasets(ml_client: MLClient, console: Console):
    """Register base datasets in Azure ML.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
    """

    console.print("Registering base datasets in Azure ML", style="info")

    # Get blob datastore name directly from environment variables
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    container_name = os.environ.get("AZURE_BLOB_DATASTORE")

    if not all([storage_account, container_name]):
        console.print(
            "Error: Missing Azure storage configuration in .env file.", style="error"
        )
        console.print(
            "Please set AZURE_STORAGE_ACCOUNT and AZURE_BLOB_DATASTORE environment variables.",
            style="error",
        )
        return

    # Define the dataset paths using proper Azure Blob storage URL format
    console.print("Reading dataset paths...", style="info")

    # Format paths using proper schema (abfss://) for Azure Data Lake Storage Gen2
    # Or wasbs:// for Azure Blob Storage
    dataset_paths = {
        "images": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/rockmass",
        "masks": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/binary",
        "raw": f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/label/raw_data",
    }

    # Generate a version based on current timestamp
    version = datetime.now().strftime("%Y%m%d.%H%M")

    # Register the datasets with proper metadata
    assets = {}

    # 1. Register the images dataset
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


def register_dataset_splits(ml_client: MLClient, console: Console):
    """Register train/val/test dataset splits in Azure ML.

    Args:
        ml_client: The Azure ML client
        console: Console object for pretty printing
    """

    console.print("Registering dataset splits in Azure ML", style="info")

    base_path = Path("data/model_ready")
    split_files = {
        "train": base_path / "train_files.json",
        "val": base_path / "val_files.json",
        "test": base_path / "test_files.json",
    }

    # Check if files exist
    for name, path in split_files.items():
        if not path.exists():
            console.print(f"Error: Split file '{path}' not found", style="error")
            sys.exit(1)

    # Get the latest version of the base datasets
    try:
        images_dataset = get_data_asset(ml_client, "rock_images")
        masks_dataset = get_data_asset(ml_client, "rock_masks")
        console.print(
            f"Using images dataset version: {images_dataset.version}", style="info"
        )
        console.print(
            f"Using masks dataset version: {masks_dataset.version}", style="info"
        )
    except Exception as e:
        console.print(f"Error retrieving base datasets: {e}", style="error")
        sys.exit(1)

    # Generate a version based on current timestamp
    version = datetime.now().strftime("%Y%m%d.%H%M")

    # Register individual split files
    registered_assets = {}

    for split_name, split_path in split_files.items():
        console.print(f"Registering {split_name} split...", style="info")

        # Load split file to get sample count
        with open(split_path) as f:
            split_data = json.load(f)

        # Create metadata with references to base datasets
        metadata = {
            "base_images_dataset": f"rock_images:{images_dataset.version}",
            "base_masks_dataset": f"rock_masks:{masks_dataset.version}",
            "num_samples": str(len(split_data)),
            "split_type": split_name,
        }

        # Register the split file as a data asset
        asset = register_data_asset(
            ml_client=ml_client,
            name=f"rock_segmentation_{split_name}_split",
            version=version,
            description=f"Rock segmentation {split_name} split",
            path=str(split_path.absolute()),
            asset_type=AssetTypes.URI_FILE,
            tags={"domain": "geology", "type": "data_split", "split": split_name},
            metadata=metadata,
        )

        registered_assets[split_name] = asset

    # Create a combined asset referencing all splits
    combined_metadata = {
        "train_split": f"rock_segmentation_train_split:{version}",
        "val_split": f"rock_segmentation_val_split:{version}",
        "test_split": f"rock_segmentation_test_split:{version}",
        "base_images_dataset": f"rock_images:{images_dataset.version}",
        "base_masks_dataset": f"rock_masks:{masks_dataset.version}",
    }

    # Register the combined split reference
    combined_asset = register_data_asset(
        ml_client=ml_client,
        name="rock_segmentation_splits",
        version=version,
        description="Combined rock segmentation dataset splits",
        path=str(
            base_path.absolute()
        ),  # Path to the directory containing the split files
        asset_type=AssetTypes.URI_FOLDER,
        tags={"domain": "geology", "type": "data_split", "split": "combined"},
        metadata=combined_metadata,
    )

    registered_assets["combined"] = combined_asset

    # Print summary
    console.print(
        "\nSuccessfully registered the following dataset splits:", style="success"
    )
    for name, asset in registered_assets.items():
        console.print(f"- {name}: {asset.name} (version {asset.version})", style="info")


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
                "This could be due to permission issues or invalid Azure ML configuration.",
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
        console.print(
            "Error: Missing required parameters. Please provide asset_name, version1, and version2.",
            style="error",
        )
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


def upload_data_to_azure_blob(
    console: Console, path_images: str, path_raw_masks: str, path_processed_masks: str
):
    """Upload local data to Azure Blob storage.

    Args:
        console: Console object for pretty printing
        path_images: Path to the images directory (may contain variables to replace)
        path_raw_masks: Path to the raw mask labels directory (may contain variables to replace)
        path_processed_masks: Path to the processed mask labels directory (may contain variables to replace)
    """

    console.print("Preparing to upload data to Azure Blob storage...", style="info")

    # Get storage account connection info from environment variables
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    storage_key = os.environ.get("AZURE_STORAGE_KEY")
    container_name = os.environ.get("AZURE_BLOB_DATASTORE")

    if not all([storage_account, storage_key, container_name]):
        console.print(
            "Error: Missing Azure storage configuration in .env file.", style="error"
        )
        console.print(
            "Please set AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, and AZURE_BLOB_DATASTORE.",
            style="error",
        )
        return

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
    confirmation = input("Do you want to continue? (y/n): ").strip().lower()

    if confirmation != "y":
        console.print("Upload canceled.", style="warning")
        return

    # Connect to Azure Blob storage
    console.print("Connecting to Azure Blob storage...", style="info")
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey={storage_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)
        console.print(
            f"Successfully connected to container: {container_name}", style="success"
        )
    except Exception as e:
        console.print(
            f"Error connecting to Azure Blob storage: {str(e)}", style="error"
        )
        return  # Upload images
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
        "python manage_azure_data_assets.py azure_data_assets.command=register-base-datasets",
        style="info",
    )


def upload_files(container_client, console: Console, local_folder_path, blob_folder):
    """Upload files from a local folder to Azure Blob storage.

    Args:
        container_client: Azure Blob container client
        console: Console object for pretty printing
        local_folder_path: Path to the local folder
        blob_folder: Folder path in the blob container
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


def upload_splits_to_azure_blob(console: Console):
    """Upload the dataset split JSON files to Azure Blob storage.

    This function uploads the train/val/test split JSON files to Azure Blob storage
    to make them available for Azure ML experiments.

    Args:
        console: Console object for pretty printing
    """

    console.print(
        "Preparing to upload dataset split files to Azure Blob storage...", style="info"
    )

    # Get storage account connection info from environment variables
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    storage_key = os.environ.get("AZURE_STORAGE_KEY")
    container_name = os.environ.get("AZURE_BLOB_DATASTORE")

    if not all([storage_account, storage_key, container_name]):
        console.print(
            "Error: Missing Azure storage configuration in .env file.", style="error"
        )
        console.print(
            "Please set AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, and AZURE_BLOB_DATASTORE.",
            style="error",
        )
        return

    # Check if split files exist
    split_files_dir = Path("data/model_ready")
    split_files = [
        split_files_dir / "train_files.json",
        split_files_dir / "val_files.json",
        split_files_dir / "test_files.json",
    ]

    for file_path in split_files:
        if not file_path.exists():
            console.print(f"Error: Split file '{file_path}' not found", style="error")
            console.print(
                "Please generate splits first using:\n  python scripts/manage_azure_data_assets.py azure_data_assets.command=generate-splits",
                style="info",
            )
            return

    # Connect to Azure Blob storage
    console.print("Connecting to Azure Blob storage...", style="info")
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey={storage_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)
        console.print(
            f"Successfully connected to container: {container_name}", style="success"
        )
    except Exception as e:
        console.print(
            f"Error connecting to Azure Blob storage: {str(e)}", style="error"
        )
        return

    # Upload files to Azure Blob Storage
    blob_folder = "dataset_splits"
    console.print(f"Uploading split files to {blob_folder}/...", style="info")

    for file_path in split_files:
        blob_path = f"{blob_folder}/{file_path.name}".replace("\\", "/")

        try:
            # Upload with proper content type
            blob_client = container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type="application/json")

            with open(file_path, "rb") as data:
                blob_client.upload_blob(
                    data, overwrite=True, content_settings=content_settings
                )

            console.print(f"Successfully uploaded: {file_path.name}", style="success")
        except Exception as e:
            console.print(f"Error uploading {file_path.name}: {str(e)}", style="error")

    console.print(
        "\nSplit files have been uploaded to Azure Blob Storage!", style="success"
    )
    console.print(f"Location: {blob_folder}/", style="info")
    console.print(
        "\nYou can now register these splits in Azure ML using:", style="info"
    )
    console.print(
        "  python scripts/manage_azure_data_assets.py azure_data_assets.command=register-splits",
        style="info",
    )


def generate_dataset_splits(
    console: Console,
    images_directory: str | Path,
    labels_directory: str | Path,
    experiment_strategy: str,
    dataset_strategies: dict[str, dict[str, list[str]]],
    dataset_prefixes: dict[str, list[str]],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    """Generate dataset splits for use in Azure ML.

    This function:
    1. Uses the dataset prefixes from the config
    2. Gets the data files based on those prefixes
    3. Splits the data into train/val/test sets
    4. Saves the splits as JSON files in the data/model_ready directory

    Args:
        console: Console object for pretty printing
        images_directory: Path to the images directory
        labels_directory: Path to the processed mask labels directory
        experiment_strategy: The experiment strategy to use
        dataset_strategies: Mapping of experiment strategies to their dataset configurations
        dataset_prefixes: Mapping of dataset names to lists of prefixes
        train_fraction: Fraction of data used for training
        val_fraction: Fraction of data used for validation
        test_fraction: Fraction of data used for testing
        seed: Random seed for reproducibility
    """

    console.print("Generating dataset splits for Azure ML", style="info")

    # Convert to Path objects if they're strings
    images_directory = Path(images_directory)
    labels_directory = Path(labels_directory)

    # Check if directories exist
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

    # Get prefixes for files in the dataset to use for training and testing
    console.print("Getting dataset prefixes...", style="info")
    prefixes = get_datasets_prefixes(
        experiment_strategy=experiment_strategy,
        dataset_strategies=dataset_strategies,
        dataset_prefixes=dataset_prefixes,
    )

    train_prefixes_list, test_prefixes_list = (
        prefixes["train_prefixes"],
        prefixes["test_prefixes"],
    )

    # Get data files
    console.print("Getting data files...", style="info")
    train_files, test_files = get_data_files(
        images_directory, labels_directory, train_prefixes_list, test_prefixes_list
    )

    # Split data - this also saves the splits to data/model_ready/*.json files
    console.print(
        f"Splitting data with train fraction: {train_fraction}, "
        f"val fraction: {val_fraction}, "
        f"test fraction: {test_fraction}...",
        style="info",
    )

    # Set a fixed random seed for reproducibility using the utility function
    seed_everything(seed)

    train_list, val_list, test_list = split_data(
        train_files,
        test_files,
        train_frac=train_fraction,
        val_frac=val_fraction,
        test_frac=test_fraction,
    )

    # Reset random seed (but retaining deterministic PyTorch behavior)
    seed_everything(None)

    # Print the number of samples in each split
    console.print(f"Number of training samples: {len(train_list)}", style="success")
    console.print(f"Number of validation samples: {len(val_list)}", style="success")
    console.print(f"Number of test samples: {len(test_list)}", style="success")

    console.print("\nSplit files have been saved to:", style="success")
    console.print("  - data/model_ready/train_files.json", style="info")
    console.print("  - data/model_ready/val_files.json", style="info")
    console.print("  - data/model_ready/test_files.json", style="info")
    console.print(
        "\nYou can now register these splits in Azure ML using:", style="info"
    )
    console.print(
        "  python scripts/manage_azure_data_assets.py azure_data_assets.command=register-splits",
        style="info",
    )


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for the script using Hydra.

    Args:
        cfg: The Hydra configuration object
    """
    # Convert OmegaConf to a Python dictionary and validate with Pydantic
    cfg_dict = OmegaConf.to_object(cfg)  # to dict
    pcfg = ConfigSchema(**cfg_dict)

    # Initialize environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = setup_azure_environment()
    console.print(f"Using configuration: {pcfg.azure_data_assets}", style="info")

    # Connect to Azure ML - do this once and pass the client to functions
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    # Configure logging to reduce verbose Azure client output
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)

    # Get command from validated config
    command = pcfg.azure_data_assets.command

    # Get asset parameters from validated config
    asset_name = pcfg.azure_data_assets.asset_name
    version1 = pcfg.azure_data_assets.version1
    version2 = pcfg.azure_data_assets.version2

    # Execute the appropriate command based on the enum value
    if command == AzureDataAssetsCommand.REGISTER_BASE_DATASETS:
        register_base_datasets(ml_client, console)
    elif command == AzureDataAssetsCommand.REGISTER_SPLITS:
        register_dataset_splits(ml_client, console)
    elif command == AzureDataAssetsCommand.LIST_ASSETS:
        list_data_assets(ml_client, console, asset_name)
    elif command == AzureDataAssetsCommand.COMPARE_ASSETS:
        compare_assets(ml_client, console, asset_name, version1, version2)
    elif command == AzureDataAssetsCommand.UPLOAD_DATA:
        upload_data_to_azure_blob(
            console,
            pcfg.dataset.path_images,
            pcfg.dataset.path_raw_mask_labels,
            pcfg.dataset.path_processed_mask_labels,
        )
    elif command == AzureDataAssetsCommand.GENERATE_SPLITS:
        generate_dataset_splits(
            console,
            images_directory=pcfg.dataset.path_images,
            labels_directory=pcfg.dataset.path_processed_mask_labels,
            experiment_strategy=pcfg.experiment.experiment_strategy,
            dataset_strategies=pcfg.experiment.dataset_strategies,
            dataset_prefixes=pcfg.dataset.prefixes,
            train_fraction=pcfg.experiment.train_fraction,
            val_fraction=pcfg.experiment.val_fraction,
            test_fraction=pcfg.experiment.test_fraction,
            seed=pcfg.experiment.seed,
        )
    elif command == AzureDataAssetsCommand.UPLOAD_SPLITS:
        upload_splits_to_azure_blob(console)
    else:
        console.print("Available commands:", style="info")
        for cmd in AzureDataAssetsCommand:
            console.print(
                f"  {cmd.value} - {get_command_description(cmd)}", style="info"
            )

        console.print("\nFor more details, run:", style="info")
        console.print(
            "  python manage_azure_data_assets.py azure_data_assets.command=help",
            style="info",
        )

        console.print("\nUsage examples:", style="info")
        console.print(
            f"  python manage_azure_data_assets.py azure_data_assets.command={AzureDataAssetsCommand.REGISTER_BASE_DATASETS.value}",
            style="info",
        )
        console.print(
            f"  python manage_azure_data_assets.py azure_data_assets.command={AzureDataAssetsCommand.UPLOAD_DATA.value}",
            style="info",
        )


if __name__ == "__main__":
    main()
