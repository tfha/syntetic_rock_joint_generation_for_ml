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
- SDK v2 compatible with fallback mount path detection

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

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure our src package is importable
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# If the user configured a curated Azure ML environment, run the helper that
# installs any missing packages before importing heavy third‑party modules.
use_curated = False
try:
    import yaml

    cfg_path = Path("scripts/config/main.yaml")
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            use_curated = (
                yaml.safe_load(f).get("azure_ml", {}).get("use_curated_env", False)
            )
except Exception:
    use_curated = False

if use_curated:
    subprocess.run(
        [sys.executable, "scripts/install_missing_packages.py"],
        check=True,
    )

# Defer heavy imports until optional curated-env installer runs first
import hydra  # noqa: E402 (deferred until optional package install step)
import mlflow  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from torchinfo import summary  # noqa: E402

from ml_segmentation.azure_core import (  # noqa: E402
    configure_azure_logging_and_warning,
)
from ml_segmentation.azure_data_loading import (  # noqa: E402
    setup_azure_dataloader,
)
from ml_segmentation.data_loading import get_datasets_prefixes  # noqa: E402
from ml_segmentation.debug_functionality import better_traceback  # noqa: E402
from ml_segmentation.define_model import choose_model  # noqa: E402
from ml_segmentation.schema_config import ConfigSchema  # noqa: E402
from ml_segmentation.train_eval_funcs import (  # noqa: E402
    EarlyStopping,
    check_and_update_best_metrics,
    save_image_predictions,
    train_one_epoch,
    validate_one_epoch,
)
from ml_segmentation.utility import (  # noqa: E402
    create_results_table,
    get_custom_console,
    log_metrics_to_tensorboard,
    seed_everything,
)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Configure logging to reduce verbose Azure client output
    configure_azure_logging_and_warning()

    # 1. Initialize MLflow and configuration
    ########################################################################
    # Setup configuration
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)
    console = get_custom_console()

    # Set the experiment name based on the experiment strategy unless Azure ML
    # already supplied a run ID (in which case the experiment is pre-bound).
    experiment_name = f"rock-segmentation-{pcfg.experiment.experiment_strategy}"

    env_run_id = (
        os.getenv("MLFLOW_RUN_ID") or os.getenv("AZUREML_RUN_ID") or os.getenv("RUN_ID")
    )

    # Diagnostics: show MLflow context in Azure ML job environment
    console.print(
        f"MLflow tracking URI: {mlflow.get_tracking_uri()}",
        style="info",
    )
    console.print(
        (
            "Env run IDs -> MLFLOW_RUN_ID: "
            f"{os.getenv('MLFLOW_RUN_ID')}, AZUREML_RUN_ID: "
            f"{os.getenv('AZUREML_RUN_ID')}, RUN_ID: {os.getenv('RUN_ID')}"
        ),
        style="info",
    )

    if env_run_id:
        # In managed Azure ML a run may already exist; attempt to
        # attach to it without resetting the experiment to avoid mismatches.
        active = mlflow.active_run()
        if active is None:
            try:
                mlflow.start_run(run_id=env_run_id)
                active = mlflow.active_run()
                console.print(
                    f"Attached to existing MLflow run: {active.info.run_id}",
                    style="info",
                )
            except Exception as e:  # Fallback: nested start
                console.print(
                    f"Failed to attach to env run id {env_run_id}: {e}. "
                    "Starting nested run instead.",
                    style="warning",
                )
                mlflow.start_run(nested=True)
                active = mlflow.active_run()
                console.print(
                    f"Started nested MLflow run: {active.info.run_id}",
                    style="info",
                )
        else:
            console.print(
                f"Using already active MLflow run: {active.info.run_id}",
                style="info",
            )
    else:
        # No env-provided run; set experiment then start run.
        mlflow.set_experiment(experiment_name)
        active = mlflow.active_run()
        if active is None:
            try:
                mlflow.start_run()
                active = mlflow.active_run()
                console.print(
                    f"Started new MLflow run: {active.info.run_id}",
                    style="info",
                )
            except Exception as e:
                console.print(
                    f"Standard mlflow.start_run failed: {e}. Trying nested run.",
                    style="warning",
                )
                mlflow.start_run(nested=True)
                active = mlflow.active_run()
                console.print(
                    f"Started nested MLflow run: {active.info.run_id}",
                    style="info",
                )
        else:
            console.print(
                f"Using existing MLflow run: {active.info.run_id}",
                style="info",
            )

    # 2. Setup output directories for storing results and logs
    ########################################################################
    console.print(
        (
            "Starting Azure ML training run with strategy: "
            f"{pcfg.experiment.experiment_strategy}"
        ),
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
    console.print(
        f"Hydra outputs will be saved to: {hydra_run_dir}",
        style="info",
    )

    # Create TensorBoard log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir / "tensorboard_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Create directory for example images (unified structure with FineTune)
    example_images_dir = models_dir / "example_images" / timestamp
    example_images_dir.mkdir(parents=True, exist_ok=True)

    # 3. Set random seed and device
    ########################################################################
    seed_everything(pcfg.experiment.seed)

    # Create generator for reproducible DataLoader
    generator = torch.Generator()
    generator.manual_seed(pcfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")

    # 4. Load data from Azure ML inputs
    ########################################################################
    console.print(
        "Loading train/test data from Azure ML inputs...",
        style="info",
    )
    # Azure ML provides input paths via ${{inputs.name}} which get passed
    # as Hydra config overrides (dataset.azure_*_path)
    images_path = pcfg.dataset.azure_images_path
    masks_path = pcfg.dataset.azure_masks_path
    splits_path = pcfg.dataset.azure_splits_path

    console.print(f"Images path from config: {images_path}", style="info")
    console.print(f"Masks path from config: {masks_path}", style="info")
    console.print(f"Splits path from config: {splits_path}", style="info")

    def validate_nonempty_images(path: Path, label: str) -> None:
        # Treat '.' with zero files as missing mount rather than continuing
        img_like = list(path.glob("*.png")) + list(path.glob("*.jpg"))
        if len(img_like) == 0:
            console.print(
                (
                    f"{label} path '{path}' contains no image files; "
                    "check data asset or input name."
                ),
                style="warning",
            )
            raise ValueError(
                f"{label} directory appears empty at '{path}'. "
                "Verify Azure ML input binding and that it contains files."
            )

    # Validate all required data inputs
    # Check images path
    if images_path is None:
        console.print(
            "Missing Azure ML input 'images_data' - not provided via config.",
            style="danger",
        )
        raise ValueError("Azure ML input 'images_data' was not provided.")
    if images_path.exists():
        console.print(f"Mounted images path: {images_path}", style="info")
        mlflow.log_param("images_path", str(images_path))
        validate_nonempty_images(images_path, "Images")
    else:
        console.print(
            "Images path not found. Please ensure the 'images_data' input is "
            "correctly configured in your Azure ML job.",
            style="danger",
        )
        raise ValueError(f"Images directory does not exist: {images_path}")

    # Check masks path
    if masks_path is None:
        console.print(
            "Missing Azure ML input 'masks_data' (AZUREML_INPUT_MASKS_DATA).",
            style="danger",
        )
        raise ValueError("Azure ML input 'masks_data' is not mounted.")
    if masks_path.exists():
        console.print(f"Mounted masks path: {masks_path}", style="info")
        mlflow.log_param("masks_path", str(masks_path))
        validate_nonempty_images(masks_path, "Masks")
    else:
        console.print(
            "Masks path not found. Please ensure the 'masks_data' input is "
            "correctly configured in your Azure ML job.",
            style="danger",
        )
        raise ValueError(f"Masks directory does not exist: {masks_path}")

    # Check splits path
    if splits_path is not None and splits_path.exists():
        console.print(f"Mounted splits path: {splits_path}", style="info")
        mlflow.log_param("splits_path", str(splits_path))
    else:
        console.print(
            "Splits path not found. Please ensure the 'splits_data' input is "
            "correctly configured in your Azure ML job.",
            style="warning",
        )
        # We'll keep splits_path as None if it doesn't exist
        splits_path = None

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

    # Setup dataloaders
    train_loader, val_loader, test_loader = setup_azure_dataloader(
        console=console,
        images_path=images_path,
        labels_path=masks_path,
        batch_size=pcfg.model.batch_size,
        num_workers=pcfg.experiment.num_workers,
        optional_transforms=pcfg.experiment.optional_transforms,
        transforms_parameters={
            "crop_size": pcfg.dataset.crop_size,
        },
        splits_path=splits_path,
        device=device,  # let function decide pin_memory
        pin_memory=None,  # auto: True on CUDA, False on CPU
        persistent_workers=None,  # auto: True if num_workers > 0
        generator=generator,
        experiment_strategy=pcfg.experiment.experiment_strategy,
    )

    # 6. Initialize model architecture
    ########################################################################
    console.print("Initializing model architecture...", style="info")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)

    # Log model architecture details
    mlflow.log_param("model_architecture", pcfg.model.name)

    # Log model info if possible
    try:
        with torch.no_grad():
            # Use batch size = 1 to avoid large GPU allocations during summary
            model_stats = summary(
                model,
                input_size=(1, 3, 224, 224),
                verbose=0,
            )
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

    # Select loss function based on configuration
    criterion: nn.Module
    if pcfg.experiment.loss_function == "focal":
        # Focal Loss (based on Lin et al. 2017) from segmentation_models_pytorch
        criterion = FocalLoss(
            mode="binary",
            alpha=pcfg.experiment.focal_alpha,
            gamma=pcfg.experiment.focal_gamma,
        )
        console.print(
            f"Using Focal Loss (alpha={pcfg.experiment.focal_alpha}, "
            f"gamma={pcfg.experiment.focal_gamma})",
            style="info",
        )
        mlflow.log_param("loss_function", "focal")
        mlflow.log_param("focal_alpha", pcfg.experiment.focal_alpha)
        mlflow.log_param("focal_gamma", pcfg.experiment.focal_gamma)
    else:  # Default to "dice"
        # Dice Loss (based on V-Net, Milletari et al. 2016) from segmentation_models_pytorch
        criterion = DiceLoss(mode="binary", from_logits=True)
        console.print("Using Dice Loss", style="info")
        mlflow.log_param("loss_function", "dice")

    # Determine effective threshold (ablation_threshold overrides prediction_threshold)
    effective_threshold: float = (
        pcfg.experiment.ablation_threshold
        if pcfg.experiment.ablation_threshold is not None
        else pcfg.experiment.prediction_threshold
    )
    console.print(
        f"Using prediction threshold: {effective_threshold}",
        style="info",
    )
    mlflow.log_param("prediction_threshold", effective_threshold)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)

    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=pcfg.model.scheduler.mode,  # Use mode from config (max for dice_joints)
        factor=pcfg.model.scheduler.gamma,
        patience=pcfg.model.scheduler.patience,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience,
        verbose=True,
        delta=pcfg.experiment.early_stopping_delta,
        mode="max",  # Maximize dice_joints
    )

    # Initialize metrics tracking for CSV export
    metrics_history = []
    # Track if metrics CSV was already logged to avoid duplicate uploads
    metrics_csv_logged = False

    # 8. Execute training and validation loop
    ########################################################################
    console.print("Beginning training and validation...", style="info")
    start_time = time.time()
    best_metrics: dict[str, Any] | None = None
    # Store the state of the best model for later artifact saving
    best_model_state: dict[str, Any] | None = None

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
                threshold=effective_threshold,
                max_batches=pcfg.experiment.sanity_check_num_batches,
            )
            console.print(
                create_results_table(
                    epoch,
                    metrics_training,
                    session="Training",
                )
            )

            # Log training metrics
            metrics_training["learning_rate"] = optimizer.param_groups[0]["lr"]
            log_metrics_to_tensorboard(
                writer=writer,
                metrics=metrics_training,
                prefix="Training",
                epoch=epoch,
            )

            # Log training metrics (last value shown on run page)
            for name, value in metrics_training.items():
                mlflow.log_metric(f"train_{name}", value, epoch)

            # Store training metrics for CSV export
            epoch_metrics: dict[str, int | float] = {"epoch": epoch}
            for name, value in metrics_training.items():
                epoch_metrics[f"train_{name}"] = value

            # Validation (only if validation data exists)
            if len(val_loader.dataset) > 0:
                metrics_validation = validate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device,
                    threshold=effective_threshold,
                    max_batches=pcfg.experiment.sanity_check_num_batches,
                )
                console.print(
                    create_results_table(
                        epoch,
                        metrics_validation,
                        session="Validation",
                    )
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

                # Store validation metrics for CSV export
                for name, value in metrics_validation.items():
                    epoch_metrics[f"val_{name}"] = value

                # Update learning rate based on validation loss
                scheduler.step(metrics_validation["loss"])
            else:
                # No validation data - use training metrics instead
                console.print(
                    "No validation data available, using training loss "
                    "for scheduler and early stopping",
                    style="warning",
                )
                # Use training metrics as a proxy for validation
                metrics_validation = metrics_training.copy()
                scheduler.step(metrics_training["loss"])

            # Append metrics for this epoch to history
            metrics_history.append(epoch_metrics)

            # Save example predictions periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                # Save training set predictions with augmentation visualization
                train_pred_dir = example_images_dir / f"epoch_{epoch + 1}" / "training"
                train_pred_dir.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"Saving training set predictions with augmentation (epoch {epoch + 1})",
                    style="info",
                )
                save_image_predictions(
                    model,
                    train_loader,
                    device,
                    num_samples=10,
                    save_dir=train_pred_dir,
                    show_original=True,  # Show augmentation visualization
                )

                # Always save test set predictions
                test_pred_dir = example_images_dir / f"epoch_{epoch + 1}" / "test"
                test_pred_dir.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"Saving test set predictions (epoch {epoch + 1})",
                    style="info",
                )
                save_image_predictions(
                    model,
                    test_loader,
                    device,
                    num_samples=10,
                    save_dir=test_pred_dir,
                    show_original=False,
                )

                # Save validation predictions only for finetune experiments
                # (SimpleMixed uses test set as validation, so skip redundant folder)
                is_finetune = pcfg.experiment.experiment_strategy.startswith("finetune")
                if len(val_loader.dataset) > 0 and is_finetune:
                    val_pred_dir = (
                        example_images_dir / f"epoch_{epoch + 1}" / "validation"
                    )
                    val_pred_dir.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"Saving validation set predictions (epoch {epoch + 1})",
                        style="info",
                    )
                    save_image_predictions(
                        model,
                        val_loader,
                        device,
                        num_samples=10,
                        save_dir=val_pred_dir,
                        show_original=False,
                    )

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
                early_stopping(
                    metrics_validation[pcfg.experiment.compare_metric], model
                )  # Monitor dice_joints (max mode)
                if early_stopping.early_stop:
                    console.print("Early stopping triggered", style="warning")
                    break

    except Exception as e:
        console.print(f"Error during training: {str(e)}", style="danger")
        # Truncate error message to avoid MLflow 500-char parameter limit
        error_msg = str(e)
        if len(error_msg) > 490:
            error_msg = error_msg[:490] + "..."
        mlflow.log_param("error", error_msg)
        mlflow.log_param("training_status", "failed")
        raise

    finally:
        # 9. Finalize and save model artifacts
        # ------------------------------------------------------------------
        console.print("Finalizing training...", style="info")

        # Close the tensorboard writer
        writer.close()

        # Save metrics history to CSV
        if metrics_history:
            # Only log if not already logged during periodic saves
            if not metrics_csv_logged:
                # Get run name from MLflow active run
                active_run = mlflow.active_run()
                run_name = active_run.info.run_name if active_run else "unknown_run"
                metrics_csv_path = output_dir / f"{run_name}_metrics.csv"
                console.print(
                    f"Creating metrics CSV with {len(metrics_history)} rows",
                    style="info",
                )
                with open(metrics_csv_path, "w", newline="") as f:
                    writer_csv = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
                    writer_csv.writeheader()
                    writer_csv.writerows(metrics_history)
                console.print(
                    f"Saved metrics CSV to {metrics_csv_path} "
                    f"(size: {metrics_csv_path.stat().st_size} bytes)",
                    style="info",
                )
                # Log to metrics folder for better organization in Azure ML
                mlflow.log_artifact(str(metrics_csv_path), "metrics")
                console.print("Logged metrics CSV to MLflow artifacts", style="info")
            else:
                console.print(
                    "Metrics CSV already logged during periodic saves",
                    style="info",
                )
        else:
            console.print(
                "No metrics history to save (training may have failed early)",
                style="warning",
            )

        # Save end-of-training predictions on validation set (only for finetune)
        # SimpleMixed uses test set as validation, so skip redundant folder
        is_finetune = pcfg.experiment.experiment_strategy.startswith("finetune")
        if len(val_loader.dataset) > 0 and is_finetune:
            final_val_dir = example_images_dir / "final" / "validation"
            final_val_dir.mkdir(parents=True, exist_ok=True)
            save_image_predictions(
                model,
                val_loader,
                device,
                num_samples=10,
                save_dir=final_val_dir,
            )
            mlflow.log_artifacts(
                str(final_val_dir), f"example_images/{timestamp}/final/validation"
            )

        # Save final test predictions
        final_test_dir = example_images_dir / "final" / "test"
        final_test_dir.mkdir(parents=True, exist_ok=True)
        save_image_predictions(
            model,
            test_loader,
            device,
            num_samples=10,
            save_dir=final_test_dir,
        )
        mlflow.log_artifacts(
            str(final_test_dir), f"example_images/{timestamp}/final/test"
        )

        # Save final model
        final_model_path = models_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        # Log model in MLflow format for easier deployment
        final_registered_name = (
            f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-final"
        )
        mlflow.pytorch.log_model(
            model,
            "final_model",
            registered_model_name=final_registered_name,
        )

        # Save best model from best metrics tracking if available
        if best_model_state is not None:
            console.print(
                "Saving best model from metrics tracking...",
                style="info",
            )
            best_model_path = models_dir / "best_metrics_model.pth"
            torch.save(best_model_state["model_state_dict"], best_model_path)
            mlflow.log_artifact(str(best_model_path))

            # Save the model in the current state
            current_state = model.state_dict().copy()

            # Load the best model for MLflow registration
            model.load_state_dict(best_model_state["model_state_dict"])
            strategy_name = pcfg.experiment.experiment_strategy
            best_metrics_registered_name = (
                f"{pcfg.model.name}-{strategy_name}-best-metrics"
            )
            mlflow.pytorch.log_model(
                model,
                "best_metrics_model",
                registered_model_name=best_metrics_registered_name,
            )

            # Restore the model to its current state
            model.load_state_dict(current_state)

            console.print(
                f"Best model was from epoch {best_model_state['epoch'] + 1}",
                style="info",
            )

        # Save best model from early stopping if available
        if early_stopping.best_model is not None:
            console.print(
                "Saving best model from early stopping...",
                style="info",
            )
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
        # --------------------------------------------------------------
        # Final test evaluation
        console.print(
            "Running final evaluation on test set...",
            style="info",
        )
        final_test_metrics = validate_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            threshold=effective_threshold,
        )
        console.print(
            create_results_table(
                0,
                final_test_metrics,
                session="Final Test",
            )
        )

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
            num_samples=10,
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
                    mlflow.log_metric(
                        f"best_{name}",
                        value,
                        step=best_metrics["epoch"],
                    )

            # Also log the epoch where best metrics were achieved
            mlflow.log_metric("best_metrics_epoch", best_metrics["epoch"])

            # Save best metrics to a file with more complete information
            with open(output_dir / "best_metrics.txt", "w") as f:
                best_epoch = best_metrics["epoch"]
                epoch_line = f"Best metrics achieved at epoch {best_epoch}:\n"
                f.write(epoch_line)
                f.write("-" * 50 + "\n")
                for name, value in best_metrics.items():
                    f.write(f"{name}: {value}\n")
            mlflow.log_artifact(str(output_dir / "best_metrics.txt"))

        # Generate GradCAM visualizations if enabled
        # Mark training as completed
        mlflow.log_param("training_status", "completed")
        mlflow.log_metric("total_training_time", time.time() - start_time)

        console.print("Training complete!", style="success")


if __name__ == "__main__":
    better_traceback()
    main()
