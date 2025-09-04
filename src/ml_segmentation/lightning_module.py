"""
PyTorch Lightning module for rock mass segmentation.

This module implements a Lightning wrapper for the segmentation models,
providing automatic mixed precision training, distributed training support,
and better memory management compared to manual training loops.

Classes:
    SegmentationLightningModule: Lightning module for segmentation training
"""

from typing import Any

import pytorch_lightning as pl
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import JaccardIndex, Precision, Recall
from torchmetrics.classification import BinaryF1Score

from ml_segmentation.define_model import choose_model


class SegmentationLightningModule(pl.LightningModule):
    """
    Lightning module for segmentation training.

    This module wraps the segmentation model and provides automatic:
    - Mixed precision training
    - Distributed training support
    - Learning rate scheduling
    - Metrics tracking
    - Model checkpointing
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any],
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        scheduler_gamma: float = 0.5,
        threshold: float = 0.5,
    ):
        """
        Initialize the Lightning module.

        Args:
            model_name: Name of the segmentation model
            model_params: Parameters for the model
            learning_rate: Learning rate for optimizer
            scheduler_patience: Patience for LR scheduler
            scheduler_gamma: Gamma for LR scheduler
            threshold: Threshold for binary predictions
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = choose_model(model_name, model_params)

        # Loss function
        self.criterion = DiceLoss(mode="binary", from_logits=True)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_gamma = scheduler_gamma
        self.threshold = threshold

        # Metrics for training
        self.train_iou = JaccardIndex(task="binary")
        self.train_iou_background = JaccardIndex(task="binary")
        self.train_iou_joints = JaccardIndex(task="binary")
        self.train_dice = BinaryF1Score()
        self.train_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")

        # Metrics for validation
        self.val_iou = JaccardIndex(task="binary")
        self.val_iou_background = JaccardIndex(task="binary")
        self.val_iou_joints = JaccardIndex(task="binary")
        self.val_dice = BinaryF1Score()
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")

        # Metrics for test
        self.test_iou = JaccardIndex(task="binary")
        self.test_iou_background = JaccardIndex(task="binary")
        self.test_iou_joints = JaccardIndex(task="binary")
        self.test_dice = BinaryF1Score()
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:
        """
        Shared step for training, validation, and test.

        Args:
            batch: Tuple of (images, masks)
            stage: One of 'train', 'val', 'test'

        Returns:
            Dictionary containing loss and predictions
        """
        images, masks = batch

        # Forward pass
        outputs = self.forward(images)
        loss = self.criterion(outputs, masks)

        # Get predictions
        preds = torch.sigmoid(outputs) > self.threshold

        # Calculate metrics based on stage
        if stage == "train":
            metrics = {
                "iou": self.train_iou,
                "iou_background": self.train_iou_background,
                "iou_joints": self.train_iou_joints,
                "dice": self.train_dice,
                "precision": self.train_precision,
                "recall": self.train_recall,
            }
        elif stage == "val":
            metrics = {
                "iou": self.val_iou,
                "iou_background": self.val_iou_background,
                "iou_joints": self.val_iou_joints,
                "dice": self.val_dice,
                "precision": self.val_precision,
                "recall": self.val_recall,
            }
        else:  # test
            metrics = {
                "iou": self.test_iou,
                "iou_background": self.test_iou_background,
                "iou_joints": self.test_iou_joints,
                "dice": self.test_dice,
                "precision": self.test_precision,
                "recall": self.test_recall,
            }

        # Update metrics
        metrics["iou"].update(preds, masks.int())

        # Background IoU (where mask == 1)
        metrics["iou_background"].update(preds, masks)

        # Joint IoU (where mask == 0, inverted)
        joint_preds = (~preds).int()
        joint_masks = (~masks.bool()).int()
        metrics["iou_joints"].update(joint_preds, joint_masks)

        # Other metrics
        metrics["dice"].update(preds, masks.int())
        metrics["precision"].update(preds, masks.int())
        metrics["recall"].update(preds, masks.int())

        return {"loss": loss, "preds": preds, "masks": masks}

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        result = self._shared_step(batch, "train")

        # Log loss
        self.log(
            "train_loss", result["loss"], on_step=True, on_epoch=True, prog_bar=True
        )

        return result["loss"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        result = self._shared_step(batch, "val")

        # Log loss
        self.log(
            "val_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True
        )

        return result["loss"]

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        result = self._shared_step(batch, "test")

        # Log loss
        self.log("test_loss", result["loss"], on_step=False, on_epoch=True)

        return result["loss"]

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Compute and log training metrics
        train_metrics = {
            "train_iou": self.train_iou.compute(),
            "train_iou_background": self.train_iou_background.compute(),
            "train_iou_joints": self.train_iou_joints.compute(),
            "train_dice": self.train_dice.compute(),
            "train_precision": self.train_precision.compute(),
            "train_recall": self.train_recall.compute(),
        }

        self.log_dict(train_metrics, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.train_iou.reset()
        self.train_iou_background.reset()
        self.train_iou_joints.reset()
        self.train_dice.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Compute and log validation metrics
        val_metrics = {
            "val_iou": self.val_iou.compute(),
            "val_iou_background": self.val_iou_background.compute(),
            "val_iou_joints": self.val_iou_joints.compute(),
            "val_dice": self.val_dice.compute(),
            "val_precision": self.val_precision.compute(),
            "val_recall": self.val_recall.compute(),
        }

        self.log_dict(val_metrics, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.val_iou.reset()
        self.val_iou_background.reset()
        self.val_iou_joints.reset()
        self.val_dice.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Compute and log test metrics
        test_metrics = {
            "test_iou": self.test_iou.compute(),
            "test_iou_background": self.test_iou_background.compute(),
            "test_iou_joints": self.test_iou_joints.compute(),
            "test_dice": self.test_dice.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
        }

        self.log_dict(test_metrics, on_epoch=True)

        # Reset metrics
        self.test_iou.reset()
        self.test_iou_background.reset()
        self.test_iou_joints.reset()
        self.test_dice.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_gamma,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
