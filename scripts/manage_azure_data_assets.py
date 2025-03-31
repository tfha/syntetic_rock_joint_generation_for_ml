"""
Manage Azure ML data assets for rock mass segmentation.

This script demonstrates how to use the Azure ML data asset management functionality
to handle your rock mass segmentation datasets properly.

Usage:
    python manage_azure_data_assets.py register-base-datasets
    python manage_azure_data_assets.py register-splits
    python manage_azure_data_assets.py list-assets
    python manage_azure_data_assets.py compare-assets rock_images 1 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv

from ml_segmentation.azure_data_assets import (
    compare_data_asset_versions,
    connect_to_azure_ml,
    get_data_asset,
    list_data_asset_versions,
    register_data_asset,
)
from ml_segmentation.utility import get_custom_console


def setup_environment():
    """Set up the environment and return a console for pretty printing."""
    # Load environment variables from .env file
    load_dotenv()

    # Create a console for pretty printing
    console = get_custom_console()

    # Check if Azure environment variables are set
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        console.print(
            "Error: AZURE_SUBSCRIPTION_ID environment variable not set", style="error"
        )
        console.print(
            "Please set it with: export AZURE_SUBSCRIPTION_ID='your-subscription-id'",
            style="error",
        )
        sys.exit(1)

    return console


def register_base_datasets(args):
    """Register base datasets in Azure ML."""
    console = setup_environment()
    console.print("Registering base datasets in Azure ML", style="info")

    # Get Azure credentials
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "rg-rock-joint-detection")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "ws-rock-joint-det")

    # Connect to Azure ML
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    # Get Azure ML configuration from environment or defaults
    azure_blob_datastore = os.environ.get("AZURE_BLOB_DATASTORE", "rock_data")

    # Define the dataset paths
    console.print("Reading dataset paths...", style="info")
    blob_base_path = f"azureblob://{azure_blob_datastore}"

    # Base paths for the datasets
    dataset_paths = {
        "images": f"{blob_base_path}/rockmass",
        "masks": f"{blob_base_path}/label/binary",
        "raw": f"{blob_base_path}/label/raw_data",
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


def register_dataset_splits(args):
    """Register train/val/test dataset splits in Azure ML."""
    console = setup_environment()
    console.print("Registering dataset splits in Azure ML", style="info")

    # Connect to Azure ML
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_ML_WORKSPACE"),
    )

    # Paths to split files
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


def list_data_assets(args):
    """List all data assets or versions of a specific data asset."""
    console = setup_environment()

    # Connect to Azure ML
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_ML_WORKSPACE"),
    )

    if args.asset_name:
        # List versions of a specific asset
        console.print(
            f"Listing versions of data asset: {args.asset_name}", style="info"
        )
        df = list_data_asset_versions(ml_client, args.asset_name)

        if df.empty:
            console.print(
                f"No versions found for asset '{args.asset_name}'", style="warning"
            )
        else:
            console.print(f"\nFound {len(df)} versions:", style="info")
            print(df.to_string(index=False))

    else:
        # List all data assets
        console.print("Listing all data assets in the workspace:", style="info")
        assets = list(ml_client.data.list())

        if not assets:
            console.print("No data assets found in the workspace", style="warning")
            return

        # Group assets by name to show latest version
        assets_by_name = {}
        for asset in assets:
            if asset.name not in assets_by_name:
                assets_by_name[asset.name] = []
            assets_by_name[asset.name].append(
                {
                    "name": asset.name,
                    "version": asset.version,
                    "created_at": asset.creation_context.created_at,
                    "type": getattr(asset, "type", "unknown"),
                }
            )

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
        df = pd.DataFrame(table_data)
        df = df.sort_values("name").reset_index(drop=True)

        print(df.to_string(index=False))


def compare_assets(args):
    """Compare two versions of a data asset."""
    console = setup_environment()
    console.print(
        f"Comparing versions {args.version1} and {args.version2} of asset '{args.asset_name}'",
        style="info",
    )

    # Connect to Azure ML
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_ML_WORKSPACE"),
    )

    # Compare the versions
    comparison = compare_data_asset_versions(
        ml_client, args.asset_name, args.version1, args.version2
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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Manage Azure ML data assets for rock mass segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Register base datasets command
    _ = subparsers.add_parser(
        "register-base-datasets", help="Register base datasets in Azure ML"
    )

    # Register dataset splits command
    _ = subparsers.add_parser(
        "register-splits", help="Register dataset splits in Azure ML"
    )

    # List assets command
    list_parser = subparsers.add_parser(
        "list-assets", help="List data assets in Azure ML"
    )
    list_parser.add_argument(
        "--asset-name", "-n", help="Name of the data asset to list versions for"
    )

    # Compare assets command
    compare_parser = subparsers.add_parser(
        "compare-assets", help="Compare two versions of a data asset"
    )
    compare_parser.add_argument("asset_name", help="Name of the data asset to compare")
    compare_parser.add_argument("version1", help="First version to compare")
    compare_parser.add_argument("version2", help="Second version to compare")

    args = parser.parse_args()

    if args.command == "register-base-datasets":
        register_base_datasets(args)
    elif args.command == "register-splits":
        register_dataset_splits(args)
    elif args.command == "list-assets":
        list_data_assets(args)
    elif args.command == "compare-assets":
        compare_assets(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
