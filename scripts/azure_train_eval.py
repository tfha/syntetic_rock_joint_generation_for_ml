"""
Azure ML native training script for rock mass segmentation.

This script is designed to work with Azure ML's native dataset handling,
using registered datasets passed as job inputs.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any  # Only import Any as it doesn't have a built-in equivalent

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from segmentation_models_pytorch.losses import DiceLoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from ml_segmentation.azure_data_loading import setup_azure_dataloader
from ml_segmentation.data_loading import get_datasets_prefixes
from ml_segmentation.debug_functionality import better_traceback
from ml_segmentation.define_model import choose_model
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.train_eval_funcs import (
    EarlyStopping,
    check_and_update_best_metrics,
    save_image_predictions,
    train_one_epoch,
    validate_one_epoch,
)
from ml_segmentation.utility import (
    create_results_table,
    get_custom_console,
    log_metrics_to_tensorboard,
    seed_everything,
)

# Configure logging to reduce verbose Azure client output
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 1. Initialize MLflow and configuration
    ########################################################################
    # Start MLflow tracking
    mlflow.start_run()

    # Setup configuration
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)
    console = get_custom_console()

    # Log key parameters
    mlflow.log_params(
        {
            "model_name": pcfg.model.name,
            "learning_rate": pcfg.model.learning_rate,
            "batch_size": pcfg.model.batch_size,
            "num_epochs": pcfg.model.num_epochs,
            "experiment_strategy": pcfg.experiment.experiment_strategy,
        }
    )

    # 2. Setup output directories
    ########################################################################
    console.print(
        "Starting Azure ML training run with strategy:"
        f"{pcfg.experiment.experiment_strategy}",
        style="info",
    )

    # Create standard output directories
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create TensorBoard log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir / "tensorboard_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Create directory for example images
    example_images_dir = output_dir / "example_images" / timestamp
    example_images_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed and device
    seed_everything(pcfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")

    # 3. Load data from Azure ML inputs
    ########################################################################
    console.print(
        "Loading training and testing data from Azure ML inputs...", style="info"
    )

    # In Azure ML, input datasets are mounted to paths defined in environment variables
    # AZUREML_RUN_ID environment variable is a standard environment variable set by
    # Azure ML when a job is running
    if "AZUREML_RUN_ID" in os.environ:
        # Get the paths from environment variables set by Azure ML
        images_path = Path(os.environ.get("AZUREML_DATAREFERENCE_images_data", ""))
        masks_path = Path(os.environ.get("AZUREML_DATAREFERENCE_masks_data", ""))
        console.print(f"Azure ML mounted images path: {images_path}", style="info")
        console.print(f"Azure ML mounted masks path: {masks_path}", style="info")

        # Check if splits dataset is mounted and should be used
        splits_path = None
        if pcfg.experiment.use_registered_splits:
            splits_path = Path(os.environ.get("AZUREML_DATAREFERENCE_splits_data", ""))
            if splits_path.exists():
                console.print(
                    f"Azure ML mounted splits path: {splits_path}", style="info"
                )
                mlflow.log_param("splits_path", str(splits_path))
            else:
                console.print(
                    "Splits path not found, using strategy-based filtering",
                    style="warning",
                )
                splits_path = None

        # Log dataset information in MLflow
        mlflow.log_param("images_path", str(images_path))
        mlflow.log_param("masks_path", str(masks_path))
    else:
        # Fallback to configured paths for local testing
        images_path = Path(pcfg.dataset.path_images)
        masks_path = Path(pcfg.dataset.path_processed_mask_labels)
        splits_path = None
        console.print("Running in local mode, using configured paths", style="warning")

    # 4. Prepare dataset prefixes and dataloaders
    ########################################################################
    # Get prefixes for dataset filtering based on experiment strategy
    prefixes = get_datasets_prefixes(
        experiment_strategy=pcfg.experiment.experiment_strategy,
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    train_prefixes_list, test_prefixes_list = (
        prefixes["train_prefixes"],
        prefixes["test_prefixes"],
    )

    # Log dataset prefixes to MLflow
    mlflow.log_param("train_prefixes", train_prefixes_list)
    mlflow.log_param("test_prefixes", test_prefixes_list)

    # Setup dataloaders
    train_loader, val_loader, test_loader = setup_azure_dataloader(
        images_path=images_path,
        labels_path=masks_path,
        train_prefixes_list=train_prefixes_list,
        test_prefixes_list=test_prefixes_list,
        batch_size=pcfg.model.batch_size,
        num_workers=pcfg.experiment.num_workers,
        train_fraction=pcfg.experiment.train_fraction,
        val_fraction=pcfg.experiment.val_fraction,
        test_fraction=pcfg.experiment.test_fraction,
        optional_transforms=pcfg.experiment.optional_transforms,
        splits_path=splits_path,
        use_registered_splits=pcfg.experiment.use_registered_splits,
    )

    # 5. Initialize model architecture
    ########################################################################
    console.print("Initializing model architecture...", style="info")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)

    # Log model architecture details
    mlflow.log_param("model_architecture", pcfg.model.name)

    # Log model info if possible
    try:
        from torchinfo import summary

        model_stats = summary(
            model, input_size=(pcfg.model.batch_size, 3, 224, 224), verbose=0
        )
        with open(output_dir / "model_summary.txt", "w") as f:
            f.write(str(model_stats))
        mlflow.log_artifact(output_dir / "model_summary.txt")
    except ImportError:
        console.print(
            "torchinfo not available, skipping model summary", style="warning"
        )

    # 6. Setup training components
    ########################################################################
    console.print("Setting up training components...", style="info")

    # Loss function
    criterion = DiceLoss(mode="binary", from_logits=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)

    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=pcfg.model.scheduler.gamma,
        patience=pcfg.model.scheduler.patience,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience,
        verbose=True,
        delta=pcfg.experiment.early_stopping_delta,
    )

    # 7. Execute training and validation loop
    ########################################################################
    console.print("Beginning training and validation...", style="info")
    start_time = time.time()
    best_metrics = None

    try:
        for epoch in range(pcfg.model.num_epochs):
            console.print(f"Epoch {epoch + 1}/{pcfg.model.num_epochs}")

            # Training
            metrics_training = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                threshold=0.5,
                max_batches=pcfg.experiment.sanity_check_num_batches,
            )
            console.print(
                create_results_table(epoch, metrics_training, session="Training")
            )

            # Log training metrics
            metrics_training["learning_rate"] = optimizer.param_groups[0]["lr"]
            log_metrics_to_tensorboard(
                writer=writer, metrics=metrics_training, prefix="Training", epoch=epoch
            )

            # Log to MLflow
            for name, value in metrics_training.items():
                mlflow.log_metric(f"train_{name}", value, epoch)

            # Validation
            metrics_validation = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                threshold=0.5,
                max_batches=pcfg.experiment.sanity_check_num_batches,
            )
            console.print(
                create_results_table(epoch, metrics_validation, session="Validation")
            )

            # Log validation metrics
            log_metrics_to_tensorboard(
                writer=writer,
                metrics=metrics_validation,
                prefix="Validation",
                epoch=epoch,
            )

            # Log to MLflow
            for name, value in metrics_validation.items():
                mlflow.log_metric(f"val_{name}", value, epoch)

            # Save example predictions periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                save_dir = example_images_dir / f"epoch_{epoch + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_image_predictions(
                    model, test_loader, device, num_samples=3, save_dir=save_dir
                )

                # Log example images to MLflow
                for img_path in save_dir.glob("*.png"):
                    mlflow.log_artifact(
                        str(img_path), f"example_predictions/epoch_{epoch + 1}"
                    )

            # Update learning rate
            scheduler.step(metrics_validation["loss"])

            # Update best metrics
            training_time = time.time() - start_time
            best_metrics = check_and_update_best_metrics(
                metrics_validation, best_metrics, epoch, training_time
            )

            # Check early stopping condition
            if not pcfg.experiment.sanity_check_num_batches:
                early_stopping(metrics_validation["loss"], model)
                if early_stopping.early_stop:
                    console.print("Early stopping triggered", style="warning")
                    break

    except Exception as e:
        console.print(f"Error during training: {str(e)}", style="danger")
        mlflow.log_param("error", str(e))
        mlflow.log_param("training_status", "failed")
        raise

    finally:
        # 8. Finalize and save model artifacts
        ########################################################################
        console.print("Finalizing training...", style="info")

        # Close the tensorboard writer
        writer.close()

        # Save final model
        final_model_path = models_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(final_model_path)

        # Save best model if available
        if early_stopping.best_model is not None:
            console.print("Saving best model from early stopping...", style="info")
            model.load_state_dict(early_stopping.best_model)
            best_model_path = models_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)

            # Register model in MLflow
            mlflow.pytorch.log_model(model, "best_model")

            # Save the run ID to a file for the registration step
            current_run_id = mlflow.active_run().info.run_id
            with open(models_dir / "best_run_id.txt", "w") as f:
                f.write(current_run_id)
            console.print(
                f"Saved run ID {current_run_id} for model registration", style="info"
            )

        # 9. Run final evaluation and generate results
        ########################################################################
        # Final test evaluation
        console.print("Running final evaluation on test set...", style="info")
        final_test_metrics = validate_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )
        console.print(create_results_table(0, final_test_metrics, session="Final Test"))

        # Log final test metrics
        for name, value in final_test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        # Generate and save final predictions
        final_predictions_dir = plots_dir / "predictions"
        final_predictions_dir.mkdir(parents=True, exist_ok=True)
        save_image_predictions(
            model,
            test_loader,
            device,
            num_samples=5,
            save_dir=final_predictions_dir,
        )

        # Log final predictions
        mlflow.log_artifacts(str(final_predictions_dir), "final_predictions")

        # Log best metrics
        if best_metrics:
            for name, value in best_metrics.items():
                if name != "epoch":
                    mlflow.log_metric(f"best_{name}", value)

            # Save best metrics to a file
            with open(output_dir / "best_metrics.txt", "w") as f:
                for name, value in best_metrics.items():
                    f.write(f"{name}: {value}\n")
            mlflow.log_artifact(output_dir / "best_metrics.txt")

        # Mark training as completed
        mlflow.log_param("training_status", "completed")
        mlflow.log_metric("total_training_time", time.time() - start_time)

        console.print("Training complete!", style="success")


if __name__ == "__main__":
    better_traceback()
    main()
