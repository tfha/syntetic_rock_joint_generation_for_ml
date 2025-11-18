from collections.abc import Sized
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table

# from torch.amp import autocast
from torchmetrics import JaccardIndex, Precision, Recall
from torchmetrics.classification import BinaryF1Score


def _to_float(x: Any) -> float:
    """Coerce torchmetrics ``compute()`` results to ``float``.

    Condensed why:
    - Torchmetrics stubs can mark ``compute()`` as returning ``None``; using this helper avoids scattering casts/"type: ignore" at call sites.
    - Runtime return types vary (Python float, NumPy scalar, or 0-d ``torch.Tensor``), so we centralize the coercion here.
    - Keeps call sites clean and future-proofs minor return-type changes.

    Note: If a CUDA tensor is returned, this safely moves it to CPU before ``.item()``.
    """
    # Fast-path common cases
    if isinstance(x, int | float):
        return float(x)
    # Handle torch tensors (including CUDA 0-d tensors)
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().item()
    # Fallback for NumPy scalars and other number-like types
    return float(x)


def check_and_update_best_metrics(
    metrics: dict[str, float],
    best_metrics: dict[str, Any] | None,
    epoch: int,
    training_time: float,
    compare_metric: str = "loss",
) -> dict[str, Any]:
    """
    Check and update the best metrics if the current metrics are better.
    Args:
        metrics (dict[str, float]): A dictionary containing the current metrics with
        keys "loss", "iou", "dice", "precision", and "recall".
        best_metrics (dict[str, Any]): A dictionary containing the best metrics so far.
            If None, the current metrics will be considered the best.
        epoch (int): The current epoch number.
        training_time (float): The total training time up to the current epoch.
        compare_metric (str, optional): The metric to use for comparison. Default is
        "loss". For "loss", lower is better. For "iou", "dice", "precision", "recall"
        higher is better.
    Returns:
        dict[str, Any]: Updated best metrics dictionary if the current metrics are
        better,
        otherwise returns the original best metrics.
    """

    is_better = False
    if best_metrics is None:
        # initialize to force creation of a concrete dict below
        best_metrics = {}
        is_better = True
    elif compare_metric == "loss":
        is_better = metrics[compare_metric] < best_metrics[compare_metric]
    else:  # For metrics like "iou", "dice", "precision", "recall", higher is better
        is_better = metrics[compare_metric] > best_metrics[compare_metric]

    if is_better:
        best_metrics = {
            "epoch": epoch + 1,
            "loss": metrics["loss"],
            "iou": metrics["iou"],
            "iou_background": metrics["iou_background"],
            "iou_joints": metrics["iou_joints"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "training_time": training_time,
        }
        console = Console()
        console.print("[bold green]New best model found![/bold green]")
        console.print(
            create_results_table(epoch, best_metrics, session="Best Validation")
        )
    return best_metrics


def create_results_table(
    epoch: int, metrics: dict[str, float], session: str = "Training"
) -> Table:
    table = Table(title=f"Epoch {epoch + 1} {session} Results")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for metric, value in metrics.items():
        table.add_row(metric.capitalize(), f"{value:.4f}")
    return table


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on (CPU or GPU).
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision
        training.
        threshold (float, optional): Threshold for converting model outputs to
        binary predictions. Defaults to 0.5.
        max_batches (int | None, optional): Maximum number of batches to process.
        If None, process all batches. Defaults to None.

    Returns:
        dict[str, float]: Dictionary containing the training loss and metrics
        (IoU, Dice, Precision, Recall).
    """
    model.train()
    running_loss = 0.0

    # Initialize metrics
    # Overall metrics treat background (1) as positive class
    binary_iou_metric = JaccardIndex(task="binary").to(device)
    dice_metric = BinaryF1Score().to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)

    # Joint-specific metrics (inverted to treat joints as positive class)
    joint_iou_metric = JaccardIndex(task="binary").to(device)
    joint_dice_metric = BinaryF1Score().to(device)
    joint_precision_metric = Precision(task="binary").to(device)
    joint_recall_metric = Recall(task="binary").to(device)

    for batch_idx, (images, masks) in enumerate(
        track(dataloader, description="Training", total=len(dataloader))
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Match autocast to active device; disable on CPU to avoid crashes
        amp_ctx = (
            torch.amp.autocast(device_type=device.type)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            # Forward pass
            outputs = model(images)
            # Resize outputs to match mask dimensions (fixes DeepLabV3+ upsampling issue)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(outputs, masks)

        # Backward pass and optimization
        ######################################################################
        # The standard way to run backward propagation is to call loss.backward(). When
        # you call .backward() on scaler.scale(loss), you are still running backward
        # propagation on the loss object, but with the gradient values scaled up by a
        # dynamic factor managed by the GradScaler. This means that the optimizer will
        # apply the gradients scaled by the same factor. The net effect is that the
        # optimizer sees gradients that are of the right scale, and the optimizer’s
        # internal heuristics can be used as intended. This will typically improve the
        # numerical stability of training.
        if device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > threshold

            # Overall metrics (background as positive class since mask=1)
            binary_iou_metric.update(preds, masks.int())
            dice_metric.update(preds, masks.int())
            precision_metric.update(preds, masks.int())
            recall_metric.update(preds, masks.int())

            # Joint metrics: invert both predictions and masks to treat joints
            # (mask=0) as positive class
            joint_preds = ~preds
            joint_masks = ~masks.bool()
            joint_iou_metric.update(joint_preds, joint_masks.int())
            joint_dice_metric.update(joint_preds, joint_masks.int())
            joint_precision_metric.update(joint_preds, joint_masks.int())
            joint_recall_metric.update(joint_preds, joint_masks.int())

    ds_sized: Sized = dataloader.dataset  # type: ignore[assignment]
    epoch_loss = float(running_loss) / float(len(ds_sized))

    metrics = {
        "loss": epoch_loss,
        "iou": round(_to_float(binary_iou_metric.compute()), 4),  # type: ignore[func-returns-value]
        "iou_joints": round(_to_float(joint_iou_metric.compute()), 4),  # type: ignore[func-returns-value]
        "dice": round(_to_float(dice_metric.compute()), 4),  # type: ignore[func-returns-value]
        "dice_joints": round(_to_float(joint_dice_metric.compute()), 4),  # type: ignore[func-returns-value]
        "precision": round(_to_float(precision_metric.compute()), 4),  # type: ignore[func-returns-value]
        "precision_joints": round(_to_float(joint_precision_metric.compute()), 4),  # type: ignore[func-returns-value]
        "recall": round(_to_float(recall_metric.compute()), 4),  # type: ignore[func-returns-value]
        "recall_joints": round(_to_float(joint_recall_metric.compute()), 4),  # type: ignore[func-returns-value]
    }

    return metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0

    # Initialize metrics
    # Overall metrics treat background (1) as positive class
    binary_iou_metric = JaccardIndex(task="binary").to(device)
    dice_metric = BinaryF1Score().to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)

    # Joint-specific metrics (inverted to treat joints as positive class)
    joint_iou_metric = JaccardIndex(task="binary").to(device)
    joint_dice_metric = BinaryF1Score().to(device)
    joint_precision_metric = Precision(task="binary").to(device)
    joint_recall_metric = Recall(task="binary").to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
            track(dataloader, description="Validation", total=len(dataloader))
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, masks = images.to(device), masks.to(device)

            amp_ctx = (
                torch.amp.autocast(device_type=device.type)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                # Forward pass
                outputs = model(images)
                # Resize outputs to match mask dimensions (fixes DeepLabV3+ upsampling issue)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds = torch.sigmoid(outputs) > threshold

            # Overall metrics (background as positive class since mask=1)
            binary_iou_metric.update(preds, masks.int())
            dice_metric.update(preds, masks.int())
            precision_metric.update(preds, masks.int())
            recall_metric.update(preds, masks.int())

            # Joint metrics: invert both predictions and masks to treat joints
            # (mask=0) as positive class
            joint_preds = ~preds
            joint_masks = ~masks.bool()
            joint_iou_metric.update(joint_preds, joint_masks.int())
            joint_dice_metric.update(joint_preds, joint_masks.int())
            joint_precision_metric.update(joint_preds, joint_masks.int())
            joint_recall_metric.update(joint_preds, joint_masks.int())

    ds_sized: Sized = dataloader.dataset  # type: ignore[assignment]
    epoch_loss = float(running_loss) / float(len(ds_sized))

    metrics = {
        "loss": epoch_loss,
        "iou": round(_to_float(binary_iou_metric.compute()), 4),  # type: ignore[func-returns-value]
        "iou_joints": round(_to_float(joint_iou_metric.compute()), 4),  # type: ignore[func-returns-value]
        "dice": round(_to_float(dice_metric.compute()), 4),  # type: ignore[func-returns-value]
        "dice_joints": round(_to_float(joint_dice_metric.compute()), 4),  # type: ignore[func-returns-value]
        "precision": round(_to_float(precision_metric.compute()), 4),  # type: ignore[func-returns-value]
        "precision_joints": round(_to_float(joint_precision_metric.compute()), 4),  # type: ignore[func-returns-value]
        "recall": round(_to_float(recall_metric.compute()), 4),  # type: ignore[func-returns-value]
        "recall_joints": round(_to_float(joint_recall_metric.compute()), 4),  # type: ignore[func-returns-value]
    }

    return metrics


class EarlyStopping:
    """
    EarlyStopping is a class that implements early stopping functionality for model
    training.

    Args:
        patience (int): The number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints the early stopping counter.
        delta (float): The minimum change in the monitored metric to be considered as
        improvement.

    Attributes:
        patience (int): The number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints the early stopping counter.
        delta (float): The minimum change in the monitored metric to be considered as
        improvement.
        counter (int): The number of epochs since the last improvement.
        best_score (float or None): The best score achieved so far.
        early_stop (bool): Whether to stop the training early or not.
        val_loss_min (float): The minimum validation loss achieved so far.
        best_model (dict or None): The state dictionary of the best model.

    Methods:
        __call__(val_loss, model): Updates the early stopping criteria based on the
        validation loss.
        _save_best_model(model): Saves the state dictionary of the best model.

    """

    # Attribute type declarations (PEP 526)
    best_score: float | None
    early_stop: bool
    val_loss_min: float
    best_model: dict[str, Any] | None

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Updates the early stopping criteria based on the validation loss.

        Args:
            val_loss (float): The validation loss of the current epoch.
            model (nn.Module): The model being trained.

        Returns:
            None

        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_best_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_best_model(model)
            self.counter = 0

    def _save_best_model(self, model: nn.Module) -> None:
        """
        Saves the state dictionary of the best model.

        Args:
            model (nn.Module): The model to save.

        Returns:
            None

        """
        self.best_model = model.state_dict()
        # Guard against None; best_score is set before calling this method
        if self.best_score is None:
            self.val_loss_min = float("inf")
        else:
            self.val_loss_min = -self.best_score


def calculate_image_statistics(image_tensor: torch.Tensor) -> dict[str, float]:
    """
    Calculate brightness, contrast, saturation, and hue statistics.

    Args:
        image_tensor: Normalized tensor (C, H, W) with ImageNet normalization

    Returns:
        Dictionary with mean brightness, contrast, saturation, hue
    """
    # Denormalize from ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    # Convert to PIL for HSV conversion
    img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    img_hsv = np.array(img_pil.convert("HSV")).astype(np.float32) / 255.0

    # Calculate statistics
    brightness = float(img.mean())  # Mean RGB value
    contrast = float(img.std())  # Standard deviation as contrast
    saturation = float(img_hsv[:, :, 1].mean())  # Mean saturation
    hue = float(img_hsv[:, :, 0].mean())  # Mean hue (0-1 range)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "hue": hue,
    }


def save_image_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 3,
    threshold: float = 0.5,
    save_dir: Path = Path("plots/predictions"),
    epoch: int | None = None,
    exp_tag: str | None = None,
) -> None:
    """
    Save predictions from the model as images.

    Args:
        model (nn.Module): The model to use for predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader for the data.
        device (torch.device): Device to run the inference on (CPU or GPU).
        num_samples (int, optional): Number of samples to visualize. Defaults to 3.
        threshold (float, optional): Threshold for binary predictions. Defaults to 0.5.
        save_dir (Path, optional): Directory to save the images. Defaults to
        Path("plots/predictions").
        epoch (int, optional): Current epoch number for filename. Defaults to None.
        exp_tag (str, optional): Experiment tag/timestamp for filename. Defaults to
        None.
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images)) > threshold

            batch_size = images.size(0)
            for b in range(batch_size):
                if collected >= num_samples:
                    break

                # Move only the single sample to CPU/NumPy to reduce memory
                img_np: np.ndarray = images[b].detach().cpu().numpy()
                mask_np: np.ndarray = 1 - masks[b].detach().cpu().numpy()  # invert
                pred_np: np.ndarray = 1 - preds[b].detach().cpu().numpy()  # invert

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                img_to_display = img_np.transpose(1, 2, 0)
                img_to_display = (img_to_display - img_to_display.min()) / (
                    img_to_display.max() - img_to_display.min() + 1e-8
                )

                axes[0].imshow(img_to_display)
                axes[0].set_title("Original Image")
                axes[1].imshow(mask_np[0], cmap="gray")
                axes[1].set_title("True Mask")
                axes[2].imshow(pred_np[0], cmap="gray")
                axes[2].set_title("Predicted Mask")

                for ax in axes:
                    ax.axis("off")

                filename = f"sample_{collected}"
                if epoch is not None:
                    filename = f"{filename}_epoch_{epoch}"
                if exp_tag is not None:
                    filename = f"{filename}_{exp_tag}"

                plt.savefig(save_dir / f"{filename}.png")
                plt.close(fig)

                collected += 1

            if collected >= num_samples:
                break
