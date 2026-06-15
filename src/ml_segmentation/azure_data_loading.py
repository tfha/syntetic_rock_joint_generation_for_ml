"""
Azure ML specific dataset handling for rock mass segmentation project.
This module provides utilities for working with Azure ML datasets.
"""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from rich.console import Console
from torch.utils.data import DataLoader

from ml_segmentation.data_loading import (
    SegmentationDataset,
    get_transforms,
    validate_no_data_leakage,
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
        images_dataset = ml_client.data.get(
            name=images_dataset_name,
            label="latest",
        )
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
        masks_dataset = ml_client.data.get(
            name=masks_dataset_name,
            label="latest",
        )
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
    transforms_parameters: dict[str, Any] | None = None,
    splits_path: Path | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    generator: torch.Generator | None = None,
    experiment_strategy: str | None = None,
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
        transforms_parameters: Dictionary of transform parameters (e.g., crop_size)
        splits_path: Path to the directory containing registered splits
        device: Device to use for determining pin_memory setting
        pin_memory: Whether to use pinned memory (None for auto-detect)
        persistent_workers: Whether to use persistent workers (None for auto-detect)
        generator: Random generator for reproducible shuffling
        experiment_strategy: Experiment strategy name (e.g., 'simplemixed_box_50')

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
        f"Found {len(img_files)} images and {len(mask_files)} masks\n"
        f"Sample image files: {[f.name for f in img_files[:5]]}\n"
        f"Sample mask files: {[f.name for f in mask_files[:5]]}",
        style="blue",
    )

    # Get transformations
    # Training uses augmentation, validation/test use only basic transforms
    train_transforms = get_transforms(
        optional_transforms=optional_transforms,
        transforms_parameters=transforms_parameters,
    )
    # Val/test: disable all augmentations, keep only resize/crop and normalize
    val_test_transforms = get_transforms(
        optional_transforms=False,  # Disable all augmentations
        transforms_parameters=transforms_parameters,
    )
    # Use registered splits from Azure ML
    if splits_path is not None and splits_path.exists():
        console.print(
            f"Using registered splits from: {splits_path}",
            style="blue",
        )

        # Look for split files; support legacy *_files.json names as well.
        candidates = {
            "train": [
                splits_path / "train.json",
                splits_path / "train_files.json",
            ],
            "val": [
                splits_path / "val.json",
                splits_path / "val_files.json",
            ],
            "test": [
                splits_path / "test.json",
                splits_path / "test_files.json",
            ],
        }

        def first_existing(paths: list[Path]) -> Path | None:
            for p in paths:
                if p.exists():
                    return p
            return None

        train_file = first_existing(candidates["train"]) or splits_path / "train.json"
        val_file = first_existing(candidates["val"]) or splits_path / "val.json"
        test_file = first_existing(candidates["test"]) or splits_path / "test.json"

        if train_file.exists() and test_file.exists():
            with open(train_file) as f:
                train_list = json.load(f)

            val_list = []
            if val_file.exists():
                with open(val_file) as f:
                    val_list = json.load(f)

            with open(test_file) as f:
                test_list = json.load(f)

            console.print(
                "Loaded splits from registered files: "
                f"train={len(train_list)}, val={len(val_list)}, "
                f"test={len(test_list)}",
                style="blue",
            )

            # VALIDATE NO DATA LEAKAGE (CRITICAL)
            console.print(
                "Validating data integrity (checking for train/val/test leakage)...",
                style="blue",
            )
            experiment_name = os.environ.get(
                "AZUREML_RUN_DISPLAY_NAME", "azure_experiment"
            )
            # Check if SimpleMixed experiment (val=test by design)
            # Use experiment_strategy if provided, otherwise fall back to experiment_name
            strategy_check = (
                experiment_strategy if experiment_strategy else experiment_name
            )
            is_simplemixed = "simplemixed" in strategy_check.lower()
            validate_no_data_leakage(
                train_list=train_list,
                val_list=val_list,
                test_list=test_list,
                experiment_name=experiment_name,
                allow_val_test_overlap=is_simplemixed,
            )

            # Log to MLflow if in Azure ML environment
            if os.environ.get("AZUREML_RUN_ID"):
                mlflow.log_param("registered_train_samples", len(train_list))
                mlflow.log_param("registered_val_samples", len(val_list))
                mlflow.log_param("registered_test_samples", len(test_list))
        else:
            existing = [p.name for p in splits_path.glob("*.json")]
            raise ValueError(
                "Required split files not found. Expected train.json or "
                "train_files.json AND test.json or test_files.json. "
                f"Present JSON files: {existing}"
            )
    else:
        raise ValueError(
            "Splits path not found or is None. "
            "Please ensure the 'splits_data' input is correctly configured "
            "in your Azure ML job."
        )  # Log dataset splits
    console.print(
        "Dataset splits: "
        f"train={len(train_list)}, val={len(val_list)}, "
        f"test={len(test_list)}",
        style="blue",
    )

    # Log to MLflow if in Azure ML environment
    if os.environ.get("AZUREML_RUN_ID"):
        mlflow.log_param("train_samples", len(train_list))
        mlflow.log_param("val_samples", len(val_list))
        mlflow.log_param("test_samples", len(test_list))

    # Create datasets
    train_dataset = SegmentationDataset(
        images_path,
        labels_path,
        train_list,
        transform=train_transforms,
        return_original=True,
    )
    val_dataset = SegmentationDataset(
        images_path, labels_path, val_list, transform=val_test_transforms
    )
    test_dataset = SegmentationDataset(
        images_path, labels_path, test_list, transform=val_test_transforms
    )

    # Resolve DataLoader performance flags
    if pin_memory is None:
        if device is not None:
            pin_memory = device.type == "cuda"
        else:
            pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        # Disable persistent_workers by default to avoid shared memory issues in Azure ML
        # Can be overridden by explicitly passing persistent_workers=True
        persistent_workers = False

    console.print(
        "Dataloader settings -> "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers={persistent_workers}",
        style="blue",
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )

    return train_loader, val_loader, test_loader
