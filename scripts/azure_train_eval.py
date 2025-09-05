"""
Azure ML native training script for rock mass segmentation.

This script is designed to work with Azure ML's native dataset handling,
using registered datasets passed as job inputs. It implements the training
pipeline for segmentation models that detect rock joints and features.

Architecture:
- Leverages Azure ML mounted datasets for efficient data access
- Uses PyTorch and segmentation_models_pytorch for model training
- Implements MLflow for comprehensive experiment tracking
- TensorBoard integration for real-time visualization
- Robust error handling and metrics tracking
- Standardized output organization for models, visualizations, and logs

The script handles:
1. Configuration setup and validation using Hydra
2. Azure ML dataset mounting and loading
3. Model definition and initialization
4. Training/validation/testing loops with robust metrics tracking
5. Early stopping and best model checkpoint saving
6. Visualization of predictions at regular intervals
7. Comprehensive MLflow logging for experiment tracking
8. Clean output organization for artifacts and logs
"""

import os
import time
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from segmentation_models_pytorch.losses import DiceLoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from ml_segmentation.azure_core import configure_azure_logging
from ml_segmentation.azure_data_loading import setup_azure_dataloader
from ml_segmentation.cuda_memory_utils import (
    configure_dataloader_for_memory,
    handle_cuda_oom_error,
    monitor_gpu_memory,
    setup_mixed_precision_training,
)
from ml_segmentation.memory_management import setup_model_with_memory_optimization
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


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Configure logging to reduce verbose Azure client output
    configure_azure_logging()
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    warnings.filterwarnings("ignore", category=UserWarning, module="msrest")

    # Setup configuration
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_container, dict):
        raise TypeError("Expected Hydra cfg to convert to a dict")
    # Ensure type for pydantic parsing
    cfg_mapping = cast(Mapping[str, Any], cfg_container)
    cfg_dict_typed: dict[str, Any] = dict(cfg_mapping)
    pcfg = ConfigSchema(**cfg_dict_typed)
    console = get_custom_console()

    # Set the experiment name based on the experiment strategy
    experiment_name = f"rock-segmentation-{pcfg.experiment.experiment_strategy}"
    mlflow.set_experiment(experiment_name)

    # Start MLflow tracking - Azure ML automatically sets up the tracking URI
    mlflow.start_run()

    # 2. Setup output directories for storing results and logs
    ########################################################################
    console.print(
        "Starting Azure ML training run with strategy:"
        f"{pcfg.experiment.experiment_strategy}",
        style="info",
    )  # Create standard output directories
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)  # Create MLflow logs directory for Azure ML
    mlflow_logs_dir = output_dir / "mlruns"
    mlflow_logs_dir.mkdir(exist_ok=True)
    console.print(
        f"MLflow logs will be saved to: {mlflow_logs_dir}", style="info"
    )  # Create Hydra outputs directory for Azure ML
    hydra_outputs_dir = output_dir / "hydra_outputs"
    hydra_outputs_dir.mkdir(exist_ok=True)

    # Override Hydra's output directory configuration for Azure ML
    timestamp_date = datetime.now().strftime("%Y-%m-%d")
    timestamp_time = datetime.now().strftime("%H-%M-%S")
    hydra_run_dir = hydra_outputs_dir / timestamp_date / timestamp_time
    hydra_run_dir.mkdir(
        parents=True, exist_ok=True
    )  # Override Hydra's output directory
    hydra_config = {
        "hydra": {
            "run": {"dir": str(hydra_run_dir)},
            "sweep": {
                "dir": str(hydra_outputs_dir / "multirun"),
                "subdir": "${hydra.job.num}",
            },
        }
    }
    hydra.core.config_store.ConfigStore.instance().store(
        name="hydra_config", node=hydra_config
    )
    console.print(f"Hydra outputs will be saved to: {hydra_run_dir}", style="info")

    # Create TensorBoard log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir / "tensorboard_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Create directory for example images
    example_images_dir = output_dir / "example_images" / timestamp
    example_images_dir.mkdir(parents=True, exist_ok=True)

    # 3. Set random seed and device
    ########################################################################
    seed_everything(pcfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")

    # Monitor initial GPU memory if available
    if device.type == "cuda":
        initial_memory = monitor_gpu_memory(device, console)
        mlflow.log_params({
            f"gpu_{k}": v for k, v in initial_memory.items()
        })

    # 4. Load data from Azure ML inputs
    ########################################################################
    console.print(
        "Loading training and testing data from Azure ML inputs...", style="info"
    )  # In Azure ML, input datasets are mounted to paths defined in environment
    # variables    # Get the paths from environment variables set by Azure ML
    images_path = Path(os.environ.get("AZUREML_INPUT_images_data", ""))
    masks_path = Path(os.environ.get("AZUREML_INPUT_masks_data", ""))
    splits_path = Path(os.environ.get("AZUREML_INPUT_splits_data", ""))

    # Validate all required data inputs
    # Check images path
    if images_path.exists():
        console.print(f"Azure ML mounted images path: {images_path}", style="info")
        mlflow.log_param("images_path", str(images_path))
    else:
        console.print(
            "Images path not found. Please ensure the 'images_data' input is "
            "correctly configured in your Azure ML job.",
            style="danger",
        )
        raise ValueError(f"Images directory does not exist: {images_path}")

    # Check masks path
    if masks_path.exists():
        console.print(f"Azure ML mounted masks path: {masks_path}", style="info")
        mlflow.log_param("masks_path", str(masks_path))
    else:
        console.print(
            "Masks path not found. Please ensure the 'masks_data' input is "
            "correctly configured in your Azure ML job.",
            style="danger",
        )
        raise ValueError(f"Masks directory does not exist: {masks_path}")

    # Check splits path
    if splits_path.exists():
        console.print(f"Azure ML mounted splits path: {splits_path}", style="info")
        mlflow.log_param("splits_path", str(splits_path))
    else:
        console.print(
            "Splits path not found. Please ensure the 'splits_data' input is "
            "correctly configured in your Azure ML job.",
            style="warning",
        )
        # We'll keep splits_path as None if it doesn't exist

    # Log dataset information in MLflow
    mlflow.log_param("images_path", str(images_path))
    mlflow.log_param("masks_path", str(masks_path))

    # 5. Prepare dataset prefixes and dataloaders
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

    # Get optimal dataloader configuration for memory management
    num_workers, pin_memory, persistent_workers = configure_dataloader_for_memory(
        num_workers=pcfg.experiment.num_workers,
        device=device,
    )
    console.print(
        f"Optimized dataloader config: num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers={persistent_workers}",
        style="info"
    )

    # Setup dataloaders with memory-optimized settings
    train_loader, val_loader, test_loader = setup_azure_dataloader(
        console=console,
        images_path=images_path,
        labels_path=masks_path,
        batch_size=pcfg.model.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        optional_transforms=pcfg.experiment.optional_transforms,
        splits_path=splits_path,
        device=device,
    )

    # 6. Initialize model architecture
    ########################################################################
    console.print("Initializing model architecture...", style="info")

    model, optimal_batch_size = setup_model_with_memory_optimization(
        model=choose_model(pcfg.model.name, pcfg.model.params),
        pcfg=pcfg,
        device=device,
        console=console,
    )

    # If batch size was adjusted, recreate dataloaders
    if optimal_batch_size != pcfg.model.batch_size:
        train_loader, val_loader, test_loader = setup_azure_dataloader(
            console=console,
            images_path=images_path,
            labels_path=masks_path,
            batch_size=optimal_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            optional_transforms=pcfg.experiment.optional_transforms,
            splits_path=splits_path,
            device=device,
        )

    # Log model architecture details
    mlflow.log_param("model_architecture", pcfg.model.name)

    # Log model info if possible
    try:
        with torch.no_grad():
            # Use batch size = 1 to avoid large GPU allocations during summary
            model_stats = summary(model, input_size=(1, 3, 224, 224), verbose=0)
        with open(output_dir / "model_summary.txt", "w") as f:
            f.write(str(model_stats))
        mlflow.log_artifact(str(output_dir / "model_summary.txt"))
    except ImportError:
        console.print(
            "torchinfo not available, skipping model summary", style="warning"
        )

    # 7. Setup training components
    ########################################################################
    console.print("Setting up training components...", style="info")

    # Loss function
    criterion = DiceLoss(mode="binary", from_logits=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)

    # Mixed precision training with proper setup
    scaler, use_amp = setup_mixed_precision_training(device)
    console.print(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}", style="info")
    mlflow.log_param("mixed_precision", use_amp)

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

    # 8. Execute training and validation loop
    ########################################################################
    console.print("Beginning training and validation...", style="info")
    start_time = time.time()
    best_metrics: dict[str, Any] | None = None
    best_model_state: dict[str, Any] | None = None  # Store the state of the best model

    try:
        for epoch in range(pcfg.model.num_epochs):
            console.print(f"Epoch {epoch + 1}/{pcfg.model.num_epochs}")

            # Monitor GPU memory at start of epoch
            if device.type == "cuda":
                epoch_memory = monitor_gpu_memory(device, console)
                mlflow.log_metrics({
                    f"epoch_{epoch}_gpu_{k}": v for k, v in epoch_memory.items()
                }, step=epoch)

            try:
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

                # Log to MLflow - the last value of each metric will be shown on run page
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

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    console.print(f"CUDA OOM error in epoch {epoch + 1}", style="red")
                    new_batch_size = handle_cuda_oom_error(e, pcfg.model.batch_size, console)
                    console.print(
                        f"Consider reducing batch_size to {new_batch_size} and restarting training",
                        style="yellow"
                    )
                    mlflow.log_param("oom_error_epoch", epoch + 1)
                    mlflow.log_param("suggested_batch_size", new_batch_size)
                    raise RuntimeError(
                        f"CUDA out of memory in epoch {epoch + 1}. "
                        f"Reduce batch_size to {new_batch_size} or use a GPU with more memory."
                    ) from e
                else:
                    raise

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
            previous_best = best_metrics
            best_metrics = check_and_update_best_metrics(
                metrics_validation, best_metrics, epoch, training_time
            )

            # Save the best model state when best metrics are updated
            if previous_best != best_metrics:
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict().copy(),
                    "optimizer_state_dict": optimizer.state_dict().copy(),
                    "loss": metrics_validation["loss"],
                    "metrics": best_metrics,
                }
                console.print(
                    "Saved best model state from metrics update", style="info"
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
        # 9. Finalize and save model artifacts
        ########################################################################
        console.print("Finalizing training...", style="info")

        # Close the tensorboard writer
        writer.close()

        # Save final model
        final_model_path = models_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        # Log model in MLflow format for easier deployment
        mlflow.pytorch.log_model(
            model,
            "final_model",
            registered_model_name=f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-final",
        )

        # Save best model from best metrics tracking if available
        if best_model_state is not None:
            console.print("Saving best model from metrics tracking...", style="info")
            best_model_path = models_dir / "best_metrics_model.pth"
            torch.save(best_model_state["model_state_dict"], best_model_path)
            mlflow.log_artifact(str(best_model_path))

            # Save the model in the current state
            current_state = model.state_dict().copy()

            # Load the best model for MLflow registration
            model.load_state_dict(best_model_state["model_state_dict"])
            mlflow.pytorch.log_model(
                model,
                "best_metrics_model",
                registered_model_name=f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-best-metrics",
            )

            # Restore the model to its current state
            model.load_state_dict(current_state)

            console.print(
                f"Best model was from epoch {best_model_state['epoch'] + 1}",
                style="info",
            )

        # Save best model from early stopping if available
        if early_stopping.best_model is not None:
            console.print("Saving best model from early stopping...", style="info")
            model.load_state_dict(early_stopping.best_model)
            best_model_path = models_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(str(best_model_path))

            # Save the run ID to a file for the registration step
            active = mlflow.active_run()
            if active is not None:
                current_run_id = active.info.run_id
                with open(models_dir / "best_run_id.txt", "w") as f:
                    f.write(current_run_id)
                console.print(
                    f"Saved run ID {current_run_id} for model registration",
                    style="info",
                )
            else:
                console.print(
                    "No active MLflow run; skipping run ID file creation",
                    style="warning",
                )

        # 10. Run final evaluation and generate results
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
            # Log best metrics with explicit epoch information
            for name, value in best_metrics.items():
                if name != "epoch":
                    # Log as a metric with the epoch it was achieved in
                    mlflow.log_metric(f"best_{name}", value, step=best_metrics["epoch"])

            # Also log the epoch where best metrics were achieved
            mlflow.log_metric("best_metrics_epoch", best_metrics["epoch"])

            # Save best metrics to a file with more complete information
            with open(output_dir / "best_metrics.txt", "w") as f:
                f.write(f"Best metrics achieved at epoch {best_metrics['epoch']}:\n")
                f.write("-" * 50 + "\n")
                for name, value in best_metrics.items():
                    f.write(f"{name}: {value}\n")
            mlflow.log_artifact(str(output_dir / "best_metrics.txt"))

        # Mark training as completed
        mlflow.log_param("training_status", "completed")
        mlflow.log_metric("total_training_time", time.time() - start_time)

        console.print("Training complete!", style="success")


if __name__ == "__main__":
    better_traceback()
    main()
