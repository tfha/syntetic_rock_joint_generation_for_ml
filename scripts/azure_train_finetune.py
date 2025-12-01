"""
Azure ML two-stage Fine-Tuned (FT) training for Wachter et al. (2026) experiments.

This script implements the FT training strategy on Azure ML:
1. Stage 1 (Pretrain): Train on 100% synthetic data until real validation accuracy plateaus
2. Stage 2 (Finetune): Continue training on real data for remaining epochs

Architecture:
- Leverages Azure ML mounted datasets for efficient data access
- Uses PyTorch and segmentation_models_pytorch for model training
- Implements MLflow for comprehensive experiment tracking
- Two-stage training with separate dataloaders for synthetic and real data
- Early stopping for each stage with different patience values

Usage:
    Submitted via azure_submit_job.py with finetune_* experiment strategies
"""

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure our src package is importable
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Handle curated Azure ML environment package installation
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
import hydra  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from torchinfo import summary  # noqa: E402

from ml_segmentation.azure_core import (  # noqa: E402
    configure_azure_logging_and_warning,
)
from ml_segmentation.data_loading import (  # noqa: E402
    SegmentationDataset,
    get_transforms,
)
from ml_segmentation.debug_functionality import better_traceback  # noqa: E402
from ml_segmentation.define_model import choose_model  # noqa: E402
from ml_segmentation.schema_config import ConfigSchema  # noqa: E402
from ml_segmentation.train_eval_funcs import (  # noqa: E402
    EarlyStopping,
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

console = get_custom_console()


def load_finetune_splits(splits_path: Path) -> tuple[list, list, list, list]:
    """Load FT-specific splits from Azure mounted path."""
    train_synthetic_json = splits_path / "train_synthetic.json"
    train_real_json = splits_path / "train_real.json"
    val_json = splits_path / "val.json"
    test_json = splits_path / "test.json"

    # Validate files exist
    missing_files = []
    for fpath in [train_synthetic_json, train_real_json, val_json, test_json]:
        if not fpath.exists():
            missing_files.append(str(fpath))

    if missing_files:
        raise FileNotFoundError(
            f"Required FT split files not found: {missing_files}. "
            f"Available files: {[p.name for p in splits_path.glob('*.json')]}"
        )

    with open(train_synthetic_json) as f:
        train_synthetic = json.load(f)
    with open(train_real_json) as f:
        train_real = json.load(f)
    with open(val_json) as f:
        val_list = json.load(f)
    with open(test_json) as f:
        test_list = json.load(f)

    console.print(
        f"Loaded FT splits: {len(train_synthetic)} synthetic, "
        f"{len(train_real)} real, {len(val_list)} val, {len(test_list)} test",
        style="info",
    )

    return train_synthetic, train_real, val_list, test_list


def create_dataloader_from_files(
    file_list: list[str],
    images_dir: Path,
    labels_dir: Path,
    transform_dict,  # dict with "image" and "label" keys
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    return_original: bool = False,
) -> DataLoader:
    """Create a dataloader from a list of files using SegmentationDataset."""
    dataset = SegmentationDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        file_list=file_list,
        transform=transform_dict,
        return_original=return_original,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Execute two-stage FT training on Azure ML."""
    configure_azure_logging_and_warning()
    better_traceback()

    # Parse and validate configuration
    pcfg = ConfigSchema(**OmegaConf.to_container(cfg, resolve=True))

    # Verify this is a finetune experiment
    strategy_value = pcfg.experiment.experiment_strategy.value
    if not strategy_value.startswith("finetune_"):
        console.print(
            f"[red]Error: This script is only for finetune_* experiments. "
            f"Got: {strategy_value}[/red]"
        )
        console.print(
            "[cyan]For simplemixed_* experiments, use: azure_train_eval.py[/cyan]"
        )
        sys.exit(1)

    # Setup
    seed_everything(pcfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")

    # Detect if running in Azure ML
    is_azure = os.environ.get("AZUREML_RUN_ID") is not None
    run_id = os.environ.get("AZUREML_RUN_ID", "local_run")

    # Setup MLflow
    if is_azure:
        # Azure ML automatically sets up MLflow
        mlflow_uri = mlflow.get_tracking_uri()
        console.print(f"MLflow tracking URI: {mlflow_uri}", style="info")
        env_run_ids = (
            f"MLFLOW_RUN_ID: {os.environ.get('MLFLOW_RUN_ID')}, "
            f"AZUREML_RUN_ID: {os.environ.get('AZUREML_RUN_ID')}, "
            f"RUN_ID: {os.environ.get('RUN_ID')}"
        )
        console.print(f"Env run IDs -> {env_run_ids}", style="info")

        # Attach to existing run if available
        active_run = mlflow.active_run()
        if active_run:
            console.print(
                f"Attached to existing MLflow run: {active_run.info.run_id}",
                style="info",
            )
        else:
            mlflow.start_run()
            console.print("Started new MLflow run", style="info")
    else:
        # Local execution
        mlflow.set_tracking_uri(str(pcfg.mlflow.tracking_uri))
        experiment_name = f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    console.print(
        f"Starting Azure ML training run with strategy: "
        f"{pcfg.experiment.experiment_strategy}",
        style="bold green",
    )

    # MLflow output paths
    mlflow_output_dir = Path("outputs/mlruns")
    mlflow_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"MLflow logs will be saved to: {mlflow_output_dir}", style="info")

    # Hydra output paths
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    hydra_output_dir = Path(f"outputs/hydra_outputs/{timestamp}")
    hydra_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Hydra outputs will be saved to: {hydra_output_dir}", style="info")

    # Get Azure ML mounted paths from config
    # These are set via Hydra overrides in azure_submit_job.py:
    # +dataset.azure_images_path=${{inputs.images_data}}
    # +dataset.azure_masks_path=${{inputs.masks_data}}
    # +dataset.azure_splits_path=${{inputs.splits_data}}
    images_path = pcfg.dataset.azure_images_path
    masks_path = pcfg.dataset.azure_masks_path
    splits_path = pcfg.dataset.azure_splits_path

    console.print(f"Images path from config: {images_path}", style="info")
    console.print(f"Masks path from config: {masks_path}", style="info")
    console.print(f"Splits path from config: {splits_path}", style="info")

    # Validate mounted paths exist
    if images_path is None:
        console.print(
            "Missing Azure ML input 'images_data'. "
            "Ensure +dataset.azure_images_path=${{inputs.images_data}} is set.",
            style="danger",
        )
        raise ValueError("Azure ML input 'images_data' is not mounted.")

    if not images_path.exists():
        console.print(
            f"Images path not found: {images_path}. "
            f"Please ensure the 'images_data' input is correctly configured.",
            style="danger",
        )
        raise ValueError(f"Images directory does not exist: {images_path}")

    if masks_path is None:
        console.print(
            "Missing Azure ML input 'masks_data'. "
            "Ensure +dataset.azure_masks_path=${{inputs.masks_data}} is set.",
            style="danger",
        )
        raise ValueError("Azure ML input 'masks_data' is not mounted.")

    if not masks_path.exists():
        console.print(
            f"Masks path not found: {masks_path}. "
            f"Please ensure the 'masks_data' input is correctly configured.",
            style="danger",
        )
        raise ValueError(f"Masks directory does not exist: {masks_path}")

    if splits_path is None:
        console.print(
            "Missing Azure ML input 'splits_data'. "
            "Ensure +dataset.azure_splits_path=${{inputs.splits_data}} is set.",
            style="danger",
        )
        raise ValueError("Azure ML input 'splits_data' is not mounted.")

    if not splits_path.exists():
        console.print(
            f"Splits path not found: {splits_path}. "
            f"Please ensure the 'splits_data' input is correctly configured.",
            style="danger",
        )
        raise ValueError(f"Splits directory does not exist: {splits_path}")

    # Log configuration
    mlflow.log_params(
        {
            "model": pcfg.model.name,
            "experiment_strategy": str(pcfg.experiment.experiment_strategy),
            "training_mode": "finetune_two_stage",
            "max_epochs": pcfg.model.num_epochs,
            "batch_size": pcfg.model.batch_size,
            "learning_rate": pcfg.model.learning_rate,
            "loss_function": str(pcfg.experiment.loss_function),
            "seed": pcfg.experiment.seed,
            "device": str(device),
            "run_id": run_id,
        }
    )

    # Load FT-specific splits
    console.print(
        f"\n[bold cyan]Loading Fine-Tuned splits from {splits_path}[/bold cyan]"
    )
    console.print(f"Using registered splits from: {splits_path}", style="info")

    train_synthetic, train_real, val_list, test_list = load_finetune_splits(splits_path)

    # Log split sizes
    mlflow.log_param("train_synthetic_count", len(train_synthetic))
    mlflow.log_param("train_real_count", len(train_real))
    mlflow.log_param("val_count", len(val_list))
    mlflow.log_param("test_count", len(test_list))

    # Verify data files exist
    console.print("Verifying image and mask files...", style="info")
    image_files = sorted([f.name for f in images_path.glob("*.png")])
    mask_files = sorted([f.name for f in masks_path.glob("*.png")])

    console.print(
        f"Found {len(image_files)} images and {len(mask_files)} masks", style="info"
    )
    console.print(
        f"Sample image files: {image_files[:5] if image_files else 'None'}",
        style="info",
    )
    console.print(
        f"Sample mask files: {mask_files[:5] if mask_files else 'None'}", style="info"
    )

    # Get transforms
    train_transforms = get_transforms(
        optional_transforms=pcfg.experiment.optional_transforms,
        transforms_parameters={
            "crop_size": pcfg.dataset.crop_size,
        },
    )
    val_transforms = get_transforms(
        optional_transforms=False,  # No augmentation for validation/test
        transforms_parameters={
            "crop_size": pcfg.dataset.crop_size,
        },
    )

    # Determine DataLoader performance flags
    pin_memory_flag = device.type == "cuda"
    persistent_workers_flag = False  # Avoid shared memory issues in Azure ML

    console.print(
        f"DataLoader settings -> pin_memory={pin_memory_flag}, "
        f"persistent_workers={persistent_workers_flag}",
        style="info",
    )

    # Create dataloaders
    console.print("\n[bold]Creating dataloaders...[/bold]")

    # Stage 1: Synthetic data loader
    train_synthetic_loader = create_dataloader_from_files(
        train_synthetic,
        images_path,
        masks_path,
        train_transforms,
        pcfg.model.batch_size,
        pcfg.experiment.num_workers,
        shuffle=True,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers_flag,
        return_original=True,
    )

    # Stage 2: Real data loader
    train_real_loader = create_dataloader_from_files(
        train_real,
        images_path,
        masks_path,
        train_transforms,
        pcfg.model.batch_size,
        pcfg.experiment.num_workers,
        shuffle=True,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers_flag,
        return_original=True,
    )

    # Validation and test loaders (100% real data)
    val_loader = create_dataloader_from_files(
        val_list,
        images_path,
        masks_path,
        val_transforms,
        pcfg.model.batch_size,
        pcfg.experiment.num_workers,
        shuffle=False,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers_flag,
    )

    test_loader = create_dataloader_from_files(
        test_list,
        images_path,
        masks_path,
        val_transforms,
        pcfg.model.batch_size,
        pcfg.experiment.num_workers,
        shuffle=False,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers_flag,
    )

    console.print(
        f"  Stage 1 (synthetic): {len(train_synthetic_loader)} batches", style="info"
    )
    console.print(f"  Stage 2 (real): {len(train_real_loader)} batches", style="info")
    console.print(f"  Validation: {len(val_loader)} batches", style="info")
    console.print(f"  Test: {len(test_loader)} batches", style="info")

    # Initialize model
    console.print("\n[bold]Initializing model...[/bold]")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)

    # Model summary
    in_channels = pcfg.model.params.get("in_channels", 3)
    model_stats = summary(
        model,
        input_size=(
            1,
            in_channels,
            pcfg.dataset.crop_size,
            pcfg.dataset.crop_size,
        ),
        verbose=0,
    )
    console.print(f"Model parameters: {model_stats.total_params:,}", style="info")

    # Loss function
    criterion: nn.Module
    if str(pcfg.experiment.loss_function).lower() == "dice":
        criterion = DiceLoss(mode="binary", from_logits=True)
        console.print("Using Dice Loss", style="info")
        mlflow.log_param("loss_function", "dice")
    elif str(pcfg.experiment.loss_function).lower() == "focal":
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
    else:
        criterion = nn.BCEWithLogitsLoss()
        console.print("Using BCEWithLogits Loss", style="info")
        mlflow.log_param("loss_function", "bce")

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
    # Early stopping for stage 1 (pretraining on synthetic, validated on real)
    early_stopping_stage1 = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience,
        verbose=True,
        delta=0.0,
    )

    # Early stopping for stage 2 (finetuning on real)
    early_stopping_stage2 = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience,
        verbose=True,
        delta=pcfg.experiment.early_stopping_delta,
    )

    # TensorBoard writer
    tensorboard_dir = Path("outputs/tensorboard_logs")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tensorboard_dir))

    # Output directory for model checkpoints
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrics tracking for CSV export
    metrics_history = []

    # ====================================================================
    # STAGE 1: PRETRAIN ON SYNTHETIC DATA
    # ====================================================================
    console.print(
        "\n[bold yellow]═══ STAGE 1: Pretraining on Synthetic Data ═══[/bold yellow]"
    )
    console.print(
        f"Training until real validation accuracy plateaus "
        f"(patience={pcfg.experiment.early_stopping_patience} epochs)\n"
    )

    best_stage1_metric = 0.0
    best_stage1_epoch = 0
    stage1_model_path = output_dir / "stage1_best_model.pth"

    for epoch in range(1, pcfg.model.num_epochs + 1):
        console.print(
            f"\n[bold cyan]Stage 1 - Epoch {epoch}/{pcfg.model.num_epochs}[/bold cyan]"
        )

        # Train on synthetic data
        metrics_train = train_one_epoch(
            model=model,
            dataloader=train_synthetic_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        # Validate on REAL data (this is the key metric for stage 1)
        metrics_val = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Display results
        console.print(
            create_results_table(epoch, metrics_train, session="Stage1-Train")
        )
        console.print(create_results_table(epoch, metrics_val, session="Stage1-Val"))

        # Log to MLflow and TensorBoard
        metrics_train["learning_rate"] = optimizer.param_groups[0]["lr"]
        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_train, prefix="stage1_train", epoch=epoch
        )
        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_val, prefix="stage1_val", epoch=epoch
        )

        for key, value in metrics_train.items():
            mlflow.log_metric(f"stage1_train_{key}", value, step=epoch)
        for key, value in metrics_val.items():
            mlflow.log_metric(f"stage1_val_{key}", value, step=epoch)

        # Store metrics for CSV export
        epoch_metrics = {"epoch": epoch, "stage": 1}
        for name, value in metrics_train.items():
            epoch_metrics[f"train_{name}"] = value
        for name, value in metrics_val.items():
            epoch_metrics[f"val_{name}"] = value
        metrics_history.append(epoch_metrics)

        # Scheduler step
        current_metric = metrics_val[pcfg.experiment.compare_metric]
        scheduler.step(current_metric)

        # Track best model
        if current_metric > best_stage1_metric:
            best_stage1_metric = current_metric
            best_stage1_epoch = epoch

            # Save stage 1 best model
            torch.save(model.state_dict(), stage1_model_path)
            console.print(
                f"✓ Saved Stage 1 best model (epoch {epoch}, "
                f"{pcfg.experiment.compare_metric}={current_metric:.4f})",
                style="green",
            )

        # Save predictions periodically
        if epoch % 5 == 0 or epoch == 1:
            # Train predictions (synthetic) with original images
            train_viz_dir = output_dir / f"stage1_train_predictions_epoch{epoch}"
            train_viz_dir.mkdir(parents=True, exist_ok=True)
            save_image_predictions(
                model=model,
                dataloader=train_synthetic_loader,
                device=device,
                save_dir=train_viz_dir,
                num_samples=10,
                show_original=True,
            )

            # Validation predictions
            val_viz_dir = output_dir / f"stage1_val_predictions_epoch{epoch}"
            val_viz_dir.mkdir(parents=True, exist_ok=True)
            save_image_predictions(
                model=model,
                dataloader=val_loader,
                device=device,
                save_dir=val_viz_dir,
                num_samples=10,
            )

        # Early stopping check
        early_stopping_stage1(current_metric, model)
        if early_stopping_stage1.early_stop:
            console.print(
                f"\n[bold yellow]Stage 1 early stopping triggered at epoch {epoch}[/bold yellow]"
            )
            console.print(
                f"Best real validation {pcfg.experiment.compare_metric}: "
                f"{best_stage1_metric:.4f} (epoch {best_stage1_epoch})"
            )
            break

    # Log stage 1 summary
    mlflow.log_metrics(
        {
            "stage1_epochs": epoch,
            "stage1_best_epoch": best_stage1_epoch,
            f"stage1_best_{pcfg.experiment.compare_metric}": best_stage1_metric,
        }
    )

    # Load best stage 1 model for stage 2
    console.print("\n[bold]Loading best Stage 1 model for Stage 2...[/bold]")
    model.load_state_dict(torch.load(stage1_model_path))

    # ====================================================================
    # STAGE 2: FINETUNE ON REAL DATA
    # ====================================================================
    console.print(
        "\n[bold yellow]═══ STAGE 2: Finetuning on Real Data ═══[/bold yellow]"
    )
    remaining_epochs = pcfg.model.num_epochs - epoch
    console.print(f"Training for {remaining_epochs} more epochs on real data\n")

    best_stage2_metric = 0.0
    best_stage2_epoch = 0
    stage2_start_epoch = epoch + 1
    final_model_path = output_dir / "best_model.pth"

    for epoch in range(stage2_start_epoch, pcfg.model.num_epochs + 1):
        console.print(
            f"\n[bold cyan]Stage 2 - Epoch {epoch}/{pcfg.model.num_epochs}[/bold cyan]"
        )

        # Train on REAL data
        metrics_train = train_one_epoch(
            model=model,
            dataloader=train_real_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        # Validate on real data
        metrics_val = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Display results
        console.print(
            create_results_table(epoch, metrics_train, session="Stage2-Train")
        )
        console.print(create_results_table(epoch, metrics_val, session="Stage2-Val"))

        # Log to MLflow and TensorBoard
        metrics_train["learning_rate"] = optimizer.param_groups[0]["lr"]
        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_train, prefix="stage2_train", epoch=epoch
        )
        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_val, prefix="stage2_val", epoch=epoch
        )

        for key, value in metrics_train.items():
            mlflow.log_metric(f"stage2_train_{key}", value, step=epoch)
        for key, value in metrics_val.items():
            mlflow.log_metric(f"stage2_val_{key}", value, step=epoch)

        # Store metrics for CSV export
        epoch_metrics = {"epoch": epoch, "stage": 2}
        for name, value in metrics_train.items():
            epoch_metrics[f"train_{name}"] = value
        for name, value in metrics_val.items():
            epoch_metrics[f"val_{name}"] = value
        metrics_history.append(epoch_metrics)

        # Scheduler step
        current_metric = metrics_val[pcfg.experiment.compare_metric]
        scheduler.step(current_metric)

        # Track best model
        if current_metric > best_stage2_metric:
            best_stage2_metric = current_metric
            best_stage2_epoch = epoch

            # Save stage 2 best model (final model)
            torch.save(model.state_dict(), final_model_path)
            console.print(
                f"✓ Saved Stage 2 best model (epoch {epoch}, "
                f"{pcfg.experiment.compare_metric}={current_metric:.4f})",
                style="green",
            )

        # Save predictions periodically
        if epoch % 5 == 0:
            # Train predictions (real) with original images
            train_viz_dir = output_dir / f"stage2_train_predictions_epoch{epoch}"
            train_viz_dir.mkdir(parents=True, exist_ok=True)
            save_image_predictions(
                model=model,
                dataloader=train_real_loader,
                device=device,
                save_dir=train_viz_dir,
                num_samples=10,
                show_original=True,
            )

            # Validation predictions
            val_viz_dir = output_dir / f"stage2_val_predictions_epoch{epoch}"
            val_viz_dir.mkdir(parents=True, exist_ok=True)
            save_image_predictions(
                model=model,
                dataloader=val_loader,
                device=device,
                save_dir=val_viz_dir,
                num_samples=10,
            )

        # Early stopping check
        early_stopping_stage2(current_metric, model)
        if early_stopping_stage2.early_stop:
            console.print(
                f"\n[bold yellow]Stage 2 early stopping triggered at epoch {epoch}[/bold yellow]"
            )
            break

    # Log stage 2 summary
    mlflow.log_metrics(
        {
            "stage2_epochs": epoch - stage2_start_epoch + 1,
            "stage2_best_epoch": best_stage2_epoch,
            f"stage2_best_{pcfg.experiment.compare_metric}": best_stage2_metric,
            "total_epochs": epoch,
        }
    )

    # ====================================================================
    # FINAL EVALUATION ON TEST SET
    # ====================================================================
    console.print("\n[bold yellow]═══ Final Evaluation on Test Set ═══[/bold yellow]")
    console.print("Loading best Stage 2 model...\n")

    model.load_state_dict(torch.load(final_model_path))
    metrics_test = validate_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    console.print(create_results_table(0, metrics_test, session="Test"))

    # Log test metrics
    for key, value in metrics_test.items():
        mlflow.log_metric(f"test_{key}", value)

    # Save final predictions
    final_viz_dir = output_dir / "final_test_predictions"
    final_viz_dir.mkdir(parents=True, exist_ok=True)
    save_image_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=final_viz_dir,
        num_samples=10,
    )

    # Summary
    console.print("\n[bold green]═══ Training Complete ═══[/bold green]")
    console.print(f"Stage 1: {best_stage1_epoch} epochs (synthetic pretraining)")
    console.print(
        f"Stage 2: {best_stage2_epoch - stage2_start_epoch + 1} epochs (real finetuning)"
    )
    console.print(f"Total: {epoch} epochs")
    console.print(
        f"\nBest Stage 1 {pcfg.experiment.compare_metric}: {best_stage1_metric:.4f}"
    )
    console.print(
        f"Best Stage 2 {pcfg.experiment.compare_metric}: {best_stage2_metric:.4f}"
    )
    console.print(
        f"Final Test {pcfg.experiment.compare_metric}: "
        f"{metrics_test[pcfg.experiment.compare_metric]:.4f}"
    )

    # Save metrics history to CSV
    if metrics_history:
        run_name = f"{pcfg.model.name}_{pcfg.experiment.experiment_strategy}"
        metrics_csv_path = output_dir / f"{run_name}_metrics.csv"

        console.print(
            f"Creating metrics CSV with {len(metrics_history)} rows",
            style="info",
        )

        with open(metrics_csv_path, "w", newline="") as f:
            fieldnames = metrics_history[0].keys()
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(metrics_history)

        console.print(f"Metrics saved to {metrics_csv_path}", style="green")

        # Log CSV to MLflow
        mlflow.log_artifact(str(metrics_csv_path))

    # Log artifacts to MLflow
    mlflow.log_artifact(str(final_model_path))
    mlflow.log_artifact(str(stage1_model_path))

    # Close writer
    writer.close()

    console.print("\n[bold green]✓ All outputs saved to outputs/[/bold green]")


if __name__ == "__main__":
    main()
