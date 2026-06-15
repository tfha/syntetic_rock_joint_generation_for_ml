"""
Training module wrapper - provides training functions for two-stage finetuning.

This module re-exports and wraps functions from train_eval_funcs.py to provide
a consistent interface for training scripts. Specifically, it provides an alias
from loss_fn → criterion parameter names for finetuning-specific code.
"""

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn, optim

from .train_eval_funcs import (
    EarlyStopping,
    create_results_table,
    save_image_predictions,
)
from .train_eval_funcs import train_one_epoch as _train_one_epoch_base
from .train_eval_funcs import validate_one_epoch as _validate_one_epoch_base
from .utility import log_metrics_to_mlflow


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,  # Alias for criterion
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Train the model for one epoch. Wrapper for train_eval_funcs.train_one_epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for the training data.
        loss_fn: Loss function (aliased as criterion internally).
        optimizer: Optimizer.
        device: Device to train on.
        scaler: GradScaler for mixed precision training.
        threshold: Prediction threshold for binary classification.
        max_batches: Maximum number of batches to process (for testing).

    Returns:
        Dictionary of training metrics.
    """
    return _train_one_epoch_base(
        model=model,
        dataloader=dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        threshold=threshold,
        max_batches=max_batches,
    )


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,  # Alias for criterion
    device: torch.device,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Validate the model for one epoch. Wrapper for train_eval_funcs.validate_one_epoch.

    Args:
        model: The model to validate.
        dataloader: DataLoader for the validation data.
        loss_fn: Loss function (aliased as criterion internally).
        device: Device to validate on.
        threshold: Prediction threshold for binary classification.
        max_batches: Maximum number of batches to process (for testing).

    Returns:
        Dictionary of validation metrics.
    """
    return _validate_one_epoch_base(
        model=model,
        dataloader=dataloader,
        criterion=loss_fn,
        device=device,
        threshold=threshold,
        max_batches=max_batches,
    )


def get_loss_function(
    loss_name: str,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Get loss function based on configuration.

    Args:
        loss_name: Name of the loss function ('dice' or 'focal').
        focal_alpha: Alpha parameter for FocalLoss.
        focal_gamma: Gamma parameter for FocalLoss.

    Returns:
        Loss function module.
    """
    if loss_name == "focal":
        return FocalLoss(
            mode="binary",
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    else:  # Default to "dice"
        return DiceLoss(mode="binary", from_logits=True)


# Other exports needed by training scripts
__all__ = [
    "train_one_epoch",
    "validate_one_epoch",
    "save_image_predictions",
    "EarlyStopping",
    "create_results_table",
    "get_loss_function",
    "log_metrics_to_mlflow",
]
