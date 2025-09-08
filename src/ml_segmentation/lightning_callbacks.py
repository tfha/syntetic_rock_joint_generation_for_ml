"""
PyTorch Lightning callbacks for rock mass segmentation.

This module provides custom callbacks for Lightning training,
including MLflow integration and image prediction saving.

Classes:
    MLflowCallback: Callback for MLflow logging integration
    ImagePredictionCallback: Callback for saving image predictions
"""

import os
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from ml_segmentation.train_eval_funcs import save_image_predictions


class MLflowCallback(Callback):
    """
    Callback for MLflow logging integration.

    This callback ensures proper MLflow logging during Lightning training,
    including metrics, parameters, and artifacts.
    """

    def __init__(self, log_model: bool = True):
        """
        Initialize the MLflow callback.

        Args:
            log_model: Whether to log the model to MLflow
        """
        super().__init__()
        self.log_model = log_model

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when training starts."""
        # Log hyperparameters
        if trainer.logger is not None and hasattr(trainer.logger, "log_hyperparams"):
            # Convert to a plain dict for type compatibility with Lightning's logger API
            try:
                hparams_dict: dict[str, object] = dict(pl_module.hparams)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive fallback
                hparams_dict = {}
            trainer.logger.log_hyperparams(hparams_dict)  # type: ignore[call-arg]

        # Log hyperparameters to MLflow if available
        if os.environ.get("AZUREML_RUN_ID"):
            for key, value in pl_module.hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of each training epoch."""
        if os.environ.get("AZUREML_RUN_ID"):
            # Get current epoch metrics from trainer
            if trainer.logged_metrics:
                for metric_name, metric_value in trainer.logged_metrics.items():
                    if "train_" in metric_name:
                        mlflow.log_metric(
                            metric_name, float(metric_value), trainer.current_epoch
                        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of each validation epoch."""
        if os.environ.get("AZUREML_RUN_ID"):
            # Get current epoch metrics from trainer
            if trainer.logged_metrics:
                for metric_name, metric_value in trainer.logged_metrics.items():
                    if "val_" in metric_name:
                        mlflow.log_metric(
                            metric_name, float(metric_value), trainer.current_epoch
                        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        if os.environ.get("AZUREML_RUN_ID") and self.log_model:
            # Log the final model
            mlflow.pytorch.log_model(
                pl_module.model,
                "final_model",
                registered_model_name=f"{pl_module.hparams.get('model_name', 'segmentation')}-lightning-final",
            )


class ImagePredictionCallback(Callback):
    """
    Callback for saving image predictions during training.

    This callback saves example predictions at regular intervals
    to visualize training progress.
    """

    def __init__(
        self,
        output_dir: Path,
        save_every_n_epochs: int = 10,
        max_images: int = 5,
        threshold: float = 0.5,
    ):
        """
        Initialize the image prediction callback.

        Args:
            output_dir: Directory to save prediction images
            save_every_n_epochs: Save predictions every N epochs
            max_images: Maximum number of images to save
            threshold: Threshold for binary predictions
        """
        super().__init__()
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_images = max_images
        self.threshold = threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of each validation epoch."""
        # Only save predictions every N epochs
        if (trainer.current_epoch + 1) % self.save_every_n_epochs != 0:
            return

        # Obtain validation dataloader (Lightning may store a list)
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return
        if isinstance(val_dataloader, (list, tuple)):
            if not val_dataloader:
                return
            val_dataloader = val_dataloader[0]

        # Delegate to existing helper to avoid code duplication
        try:
            save_image_predictions(
                model=pl_module,
                dataloader=val_dataloader,
                device=pl_module.device,
                num_samples=self.max_images,
                threshold=self.threshold,
                save_dir=self.output_dir,
                epoch=trainer.current_epoch,
            )
        except Exception as e:  # pragma: no cover - non-critical logging
            print(f"Warning: Failed to save predictions: {e}")


class ModelCheckpointCallback(Callback):
    """
    Enhanced model checkpoint callback with Azure ML integration.

    This callback extends the standard Lightning checkpoint functionality
    with Azure ML artifact logging.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize the checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of each validation epoch."""
        # Get current validation loss
        if "val_loss" in trainer.logged_metrics:
            current_val_loss = float(trainer.logged_metrics["val_loss"])

            # Save best model if validation loss improved
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss

                # Save checkpoint
                checkpoint_path = (
                    self.checkpoint_dir
                    / f"best_model_epoch_{trainer.current_epoch}.pth"
                )
                torch.save(pl_module.state_dict(), checkpoint_path)

                # Log to MLflow if available
                if os.environ.get("AZUREML_RUN_ID"):
                    mlflow.log_artifact(str(checkpoint_path))
                    mlflow.log_metric(
                        "best_val_loss", current_val_loss, trainer.current_epoch
                    )
