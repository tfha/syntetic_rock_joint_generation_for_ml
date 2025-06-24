"""
Azure ML specific dataset handling for rock mass segmentation project.
This module provides utilities for working with Azure ML datasets.
"""

import json
import os
from pathlib import Path

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from rich.console import Console
from torch.utils.data import DataLoader

from ml_segmentation.data_loading import (
    SegmentationDataset,
    get_transforms,
)


def create_azure_datasets(
    ml_client: MLClient,
    images_path: str,
    masks_path: str,
    images_dataset_name: str = "rockmass_images",
    masks_dataset_name: str = "rockmass_masks",
    description: str = "Rock mass segmentation dataset",
) -> tuple[Data, Data]:
    """
    Creates Azure ML datasets from blob storage paths.

    Args:
        ml_client: Azure ML client
        images_path: Path to the folder containing images
        masks_path: Path to the folder containing masks
        images_dataset_name: Name for the images dataset
        masks_dataset_name: Name for the masks dataset
        description: Description for the datasets

    Returns:
        Tuple of (images_dataset, masks_dataset)
    """
    # Create datasets if they don't exist
    try:
        images_dataset = ml_client.data.get(name=images_dataset_name, label="latest")
        print(f"Images dataset '{images_dataset_name}' already exists in workspace")
    except Exception:
        print(f"Creating images dataset '{images_dataset_name}'")
        images_dataset = Data(
            name=images_dataset_name,
            description=f"{description} - Images",
            path=images_path,
            type=AssetTypes.URI_FOLDER,
        )
        images_dataset = ml_client.data.create_or_update(images_dataset)

    try:
        masks_dataset = ml_client.data.get(name=masks_dataset_name, label="latest")
        print(f"Masks dataset '{masks_dataset_name}' already exists in workspace")
    except Exception:
        print(f"Creating masks dataset '{masks_dataset_name}'")
        masks_dataset = Data(
            name=masks_dataset_name,
            description=f"{description} - Masks",
            path=masks_path,
            type=AssetTypes.URI_FOLDER,
        )
        masks_dataset = ml_client.data.create_or_update(masks_dataset)

    return images_dataset, masks_dataset


def setup_azure_dataloader(
    console: Console,
    images_path: Path,
    labels_path: Path,
    batch_size: int,
    num_workers: int,
    optional_transforms: bool = False,
    splits_path: Path = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sets up data loaders for Azure ML training environment.
    Uses Azure ML mounted paths for dataset inputs and registered splits.

    Args:
        console: Rich console for pretty printing
        images_path: Path to the images directory
        labels_path: Path to the labels directory
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        optional_transforms: Whether to use optional data augmentation
        splits_path: Path to the directory containing registered splits

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Check if directories exist
    if not images_path.exists():
        raise ValueError(f"Images directory does not exist: {images_path}")
    if not labels_path.exists():
        raise ValueError(f"Labels directory does not exist: {labels_path}")

    # Log image and mask directory information
    img_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    mask_files = list(labels_path.glob("*.jpg")) + list(labels_path.glob("*.png"))

    console.print(
        f"Found {len(img_files)} images and {len(mask_files)} masks", style="info"
    )
    console.print(
        f"Sample image files: {[f.name for f in img_files[:5]]}", style="info"
    )
    console.print(
        f"Sample mask files: {[f.name for f in mask_files[:5]]}", style="info"
    )

    # Get transformations
    transforms_dict = get_transforms(optional_transforms=optional_transforms)
    # Use registered splits from Azure ML
    if splits_path is not None and splits_path.exists():
        console.print(f"Using registered splits from: {splits_path}", style="info")

        # Look for train/val/test split files
        train_file = splits_path / "train_files.json"
        val_file = splits_path / "val_files.json"
        test_file = splits_path / "test_files.json"  # Load splits from files
        if train_file.exists() and test_file.exists():
            with open(train_file, "r") as f:
                train_list = json.load(f)

            val_list = []
            if val_file.exists():
                with open(val_file, "r") as f:
                    val_list = json.load(f)

            with open(test_file, "r") as f:
                test_list = json.load(f)

            console.print(
                f"Loaded splits from registered files: "
                f"train={len(train_list)}, val={len(val_list)}, test={len(test_list)}",
                style="info",
            )

            # Log to MLflow if in Azure ML environment
            if os.environ.get("AZUREML_RUN_ID"):
                mlflow.log_param("registered_train_samples", len(train_list))
                mlflow.log_param("registered_val_samples", len(val_list))
                mlflow.log_param("registered_test_samples", len(test_list))
        else:
            raise ValueError(
                "Required split files not found in the splits directory. "
                "Please ensure train_files.json and test_files.json are available."
            )
    else:
        raise ValueError(
            "Splits path not found or is None. "
            "Please ensure the 'splits_data' input is correctly configured "
            "in your Azure ML job."
        )  # Log dataset splits
    console.print(
        f"Dataset splits: train={len(train_list)}, "
        f"val={len(val_list)}, test={len(test_list)}",
        style="info",
    )

    # Log to MLflow if in Azure ML environment
    if os.environ.get("AZUREML_RUN_ID"):
        mlflow.log_param("train_samples", len(train_list))
        mlflow.log_param("val_samples", len(val_list))
        mlflow.log_param("test_samples", len(test_list))

    # Create datasets
    train_dataset = SegmentationDataset(
        images_path, labels_path, train_list, transform=transforms_dict["train"]
    )
    val_dataset = SegmentationDataset(
        images_path, labels_path, val_list, transform=transforms_dict["val"]
    )
    test_dataset = SegmentationDataset(
        images_path, labels_path, test_list, transform=transforms_dict["test"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
