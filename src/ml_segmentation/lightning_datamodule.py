"""
PyTorch Lightning data module for rock mass segmentation.

This module provides a Lightning-compatible data interface that wraps
the existing Azure data loading functionality.

Classes:
    SegmentationDataModule: Lightning data module for segmentation
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from ml_segmentation.azure_data_loading import setup_azure_dataloader


class SegmentationDataModule(pl.LightningDataModule):
    """
    Lightning data module for segmentation.

    This module wraps the existing Azure data loading functionality
    to provide a Lightning-compatible interface.
    """

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        optional_transforms: bool = False,
        splits_path: Path | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initialize the data module.

        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            optional_transforms: Whether to use optional transforms
            splits_path: Path to dataset splits
            pin_memory: Whether to pin memory in data loaders
            persistent_workers: Whether to use persistent workers
        """
        super().__init__()
        self.save_hyperparameters()

        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optional_transforms = optional_transforms
        self.splits_path = splits_path
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Data loaders will be set in setup()
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Setup data loaders for the given stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        if stage == "fit" or stage is None:
            # Setup training and validation data loaders
            console = Console()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.train_loader, self.val_loader, self.test_loader = (
                setup_azure_dataloader(
                    console=console,
                    images_path=self.images_path,
                    labels_path=self.labels_path,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    optional_transforms=self.optional_transforms,
                    splits_path=self.splits_path,
                    device=device,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                )
            )

        if stage == "test":
            # Setup test data loader if not already done
            if self.test_loader is None:
                console = Console()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                _, _, self.test_loader = setup_azure_dataloader(
                    console=console,
                    images_path=self.images_path,
                    labels_path=self.labels_path,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    optional_transforms=self.optional_transforms,
                    splits_path=self.splits_path,
                    device=device,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                )

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_loader is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_loader is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_loader is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self.test_loader
