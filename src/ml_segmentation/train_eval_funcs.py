import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import track
from rich.table import Table

# from torch.amp import autocast
from torchmetrics import Dice, JaccardIndex, Precision, Recall


def check_and_update_best_metrics(
    metrics: dict[str, float],
    best_metrics: dict[str, Any],
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
    binary_iou_metric = JaccardIndex(task="binary").to(device)
    # For per-class IoU, use separate binary IoU metrics
    background_iou_metric = JaccardIndex(task="binary").to(device)
    joint_iou_metric = JaccardIndex(task="binary").to(device)
    dice_metric = Dice(num_classes=2).to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)

    for batch_idx, (images, masks) in enumerate(
        track(dataloader, description="Training", total=len(dataloader))
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        with torch.amp.autocast(device_type="cuda"):  # Mixed precision training
            # Forward pass
            outputs = model(images)
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
        scaler.scale(loss).backward()
        scaler.unscale_(
            optimizer
        )  # unscale the gradients of optimizer's assigned params in-place before the
        # optimizer's step
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Calculate metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > threshold
            binary_iou_metric.update(preds, masks.int())

            # For masks where joints=0 (black) and background=1 (white):
            # Calculate IoU for background (where mask == 1)
            # For background IoU, we're treating areas where mask=1 as the positive
            # class
            background_preds = preds
            background_masks = masks
            background_iou_metric.update(background_preds, background_masks)

            # Calculate IoU for joints (where mask == 0)
            # For joint IoU, we need to invert both predictions and masks to treat
            # joints as the positive class
            joint_preds = (
                ~preds
            )  # Invert to focus on areas where prediction is 0 (joints)
            joint_masks = (
                ~masks.bool()
            )  # Invert to focus on areas where mask is 0 (joints)
            joint_iou_metric.update(joint_preds, joint_masks)

            dice_metric.update(preds, masks.int())
            precision_metric.update(preds, masks.int())
            recall_metric.update(preds, masks.int())

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = {
        "loss": epoch_loss,
        "iou": round(binary_iou_metric.compute().item(), 4),
        "iou_background": round(background_iou_metric.compute().item(), 4),
        "iou_joints": round(joint_iou_metric.compute().item(), 4),
        "dice": round(dice_metric.compute().item(), 4),
        "precision": round(precision_metric.compute().item(), 4),
        "recall": round(recall_metric.compute().item(), 4),
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
    binary_iou_metric = JaccardIndex(task="binary").to(device)
    # For per-class IoU, use separate binary IoU metrics
    background_iou_metric = JaccardIndex(task="binary").to(device)
    joint_iou_metric = JaccardIndex(task="binary").to(device)
    dice_metric = Dice(num_classes=2).to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
            track(dataloader, description="Validation", total=len(dataloader))
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast(device_type="cuda"):
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds = torch.sigmoid(outputs) > threshold
            binary_iou_metric.update(preds, masks.int())

            # For masks where joints=0 (black) and background=1 (white):
            # Calculate IoU for background (where mask == 1)
            # For background IoU, we're treating areas where mask=1 as the positive
            # class
            background_preds = preds
            background_masks = masks
            background_iou_metric.update(background_preds, background_masks)

            # Calculate IoU for joints (where mask == 0)
            # For joint IoU, we need to invert both predictions and masks to treat
            # joints as the positive class
            joint_preds = (
                ~preds
            )  # Invert to focus on areas where prediction is 0 (joints)
            joint_masks = (
                ~masks.bool()
            )  # Invert to focus on areas where mask is 0 (joints)
            joint_iou_metric.update(joint_preds, joint_masks)

            dice_metric.update(preds, masks.int())
            precision_metric.update(preds, masks.int())
            recall_metric.update(preds, masks.int())

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = {
        "loss": epoch_loss,
        "iou": round(binary_iou_metric.compute().item(), 4),
        "iou_background": round(background_iou_metric.compute().item(), 4),
        "iou_joints": round(joint_iou_metric.compute().item(), 4),
        "dice": round(dice_metric.compute().item(), 4),
        "precision": round(precision_metric.compute().item(), 4),
        "recall": round(recall_metric.compute().item(), 4),
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

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: None | float = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.best_model: None | dict = None

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
        self.val_loss_min = -self.best_score


def save_image_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 3,
    threshold: float = 0.5,
    save_dir: Path = Path("plots/predictions"),
    epoch: int = None,
    exp_tag: str = None,
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
    samples = random.sample(list(dataloader), num_samples)

    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, (images, masks) in enumerate(samples):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(images)) > threshold

        # Convert tensors to CPU for plotting
        images = images.cpu().numpy()
        masks = (
            1 - masks.cpu().numpy()
        )  # Invert mask for black lines on white background
        preds = (
            1 - preds.cpu().numpy()
        )  # Invert prediction for black lines on white background

        # Plot original image, true mask, and predicted mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Normalize and transpose image correctly for display
        img_to_display = images[0].transpose(1, 2, 0)  # Move channels to the end
        img_to_display = (img_to_display - img_to_display.min()) / (
            img_to_display.max() - img_to_display.min() + 1e-8
        )

        axes[0].imshow(img_to_display)  # Already transposed above
        axes[0].set_title("Original Image")
        axes[1].imshow(masks[0][0], cmap="gray")
        axes[1].set_title("True Mask")
        axes[2].imshow(preds[0][0], cmap="gray")
        axes[2].set_title("Predicted Mask")

        # Remove axes
        for ax in axes:
            ax.axis("off")

        # Create a unique filename
        filename = f"sample_{idx}"
        if epoch is not None:
            filename = f"{filename}_epoch_{epoch}"
        if exp_tag is not None:
            filename = f"{filename}_{exp_tag}"

        # Save the figure
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{filename}.png"
        plt.savefig(save_path)
        plt.close(fig)
