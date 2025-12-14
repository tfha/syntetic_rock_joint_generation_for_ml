"""
Two-stage Fine-Tuned (FT) training for Wachter et al. (2025) experiments.

This script implements the FT training strategy:
1. Stage 1 (Pretrain): Train on 100% synthetic data until real validation accuracy plateaus
2. Stage 2 (Finetune): Continue training on real data for remaining epochs

Usage:
    poetry run python scripts/train_finetune.py \
        model=unet \
        experiment.experiment_strategy=finetune_box_10
"""

import json
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader

from ml_segmentation.data_loading import SegmentationDataset, get_transforms
from ml_segmentation.define_model import choose_model
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.training import (
    EarlyStopping,
    create_results_table,
    get_loss_function,
    log_metrics_to_mlflow,
    train_one_epoch,
    validate_one_epoch,
)
from ml_segmentation.utility import better_traceback, seed_everything, setup_mlflow

console = Console()


def load_finetune_splits(split_dir: Path) -> tuple[list, list, list, list]:
    """Load FT-specific splits: synthetic, real, val, test."""
    train_synthetic_json = split_dir / "train_synthetic.json"
    train_real_json = split_dir / "train_real.json"
    val_json = split_dir / "val.json"
    test_json = split_dir / "test.json"

    # Validate files exist
    for fpath in [train_synthetic_json, train_real_json, val_json, test_json]:
        if not fpath.exists():
            raise FileNotFoundError(f"Required FT split file not found: {fpath}")

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


def create_dataloader(
    file_list: list[str],
    images_dir: Path,
    labels_dir: Path,
    transforms,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a dataloader from a list of files."""
    dataset = SegmentationDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        file_list=file_list,
        transform=transforms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Execute two-stage FT training."""
    better_traceback()

    # Parse and validate configuration
    pcfg = ConfigSchema(**OmegaConf.to_container(cfg, resolve=True))

    # Verify this is a finetune experiment
    if not str(pcfg.experiment.experiment_strategy).startswith("finetune_"):
        console.print(
            f"Error: This script is only for finetune_* experiments. "
            f"Got: {pcfg.experiment.experiment_strategy}",
            style="error",
        )
        console.print(
            "For simplemixed_* experiments, use: python scripts/train_eval.py",
            style="info",
        )
        return

    # Setup
    seed_everything(pcfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")

    # Setup MLflow
    experiment_name = f"{pcfg.model.architecture}-{pcfg.experiment.experiment_strategy}"
    setup_mlflow(
        tracking_uri=str(pcfg.mlflow.tracking_uri),
        experiment_name=experiment_name,
    )

    with mlflow.start_run(run_name=experiment_name):
        # Log configuration
        mlflow.log_params(
            {
                "model": pcfg.model.architecture,
                "experiment_strategy": str(pcfg.experiment.experiment_strategy),
                "training_mode": "finetune_two_stage",
                "max_epochs": pcfg.model.num_epochs,
                "batch_size": pcfg.model.batch_size,
                "learning_rate": pcfg.model.learning_rate,
                "loss_function": str(pcfg.experiment.loss_function),
                "seed": pcfg.experiment.seed,
            }
        )

        # Load FT-specific splits
        dataset_strategy = pcfg.experiment.dataset_strategies[
            str(pcfg.experiment.experiment_strategy)
        ]
        split_dir = Path(dataset_strategy.custom_splits_dir)

        console.print(
            f"\n[bold cyan]Loading Fine-Tuned splits from {split_dir}[/bold cyan]"
        )
        train_synthetic, train_real, val_list, test_list = load_finetune_splits(
            split_dir
        )

        # Log split sizes
        mlflow.log_param("train_synthetic_count", len(train_synthetic))
        mlflow.log_param("train_real_count", len(train_real))
        mlflow.log_param("val_count", len(val_list))
        mlflow.log_param("test_count", len(test_list))

        # Setup paths
        images_dir = Path(pcfg.dataset.path_images)
        labels_dir = Path(pcfg.dataset.path_processed_mask_labels)

        # Get transforms
        train_transforms = get_transforms(
            crop_size=pcfg.dataset.crop_size,
            is_training=True,
        )
        val_transforms = get_transforms(
            crop_size=pcfg.dataset.crop_size,
            is_training=False,
        )

        # Create dataloaders
        console.print("\n[bold]Creating dataloaders...[/bold]")

        # Stage 1: Synthetic data loader
        train_synthetic_loader = create_dataloader(
            train_synthetic,
            images_dir,
            labels_dir,
            train_transforms,
            pcfg.model.batch_size,
            pcfg.experiment.num_workers,
            shuffle=True,
        )

        # Stage 2: Real data loader
        train_real_loader = create_dataloader(
            train_real,
            images_dir,
            labels_dir,
            train_transforms,
            pcfg.model.batch_size,
            pcfg.experiment.num_workers,
            shuffle=True,
        )

        # Validation and test loaders (100% real data)
        val_loader = create_dataloader(
            val_list,
            images_dir,
            labels_dir,
            val_transforms,
            pcfg.model.batch_size,
            pcfg.experiment.num_workers,
            shuffle=False,
        )

        test_loader = create_dataloader(
            test_list,
            images_dir,
            labels_dir,
            val_transforms,
            pcfg.model.batch_size,
            pcfg.experiment.num_workers,
            shuffle=False,
        )

        console.print(
            f"  Stage 1 (synthetic): {len(train_synthetic_loader)} batches",
            style="info",
        )
        console.print(
            f"  Stage 2 (real): {len(train_real_loader)} batches", style="info"
        )
        console.print(f"  Validation: {len(val_loader)} batches", style="info")
        console.print(f"  Test: {len(test_loader)} batches", style="info")

        # Initialize model
        console.print("\n[bold]Initializing model...[/bold]")
        model = choose_model(pcfg.model.name, pcfg.model.params).to(device)

        # Loss function and metrics
        loss_fn = get_loss_function(
            loss_name=str(pcfg.experiment.loss_function),
            focal_alpha=pcfg.experiment.focal_alpha,
            focal_gamma=pcfg.experiment.focal_gamma,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=pcfg.model.learning_rate,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=pcfg.model.scheduler.gamma,
            patience=pcfg.model.scheduler.patience,
        )

        # Early stopping for stage 1 (patience=10 as per Wachter et al.)
        early_stopping_stage1 = EarlyStopping(
            patience=10,
            mode="max",
            delta=0.0,
        )

        # Early stopping for stage 2 (use configured patience)
        early_stopping_stage2 = EarlyStopping(
            patience=pcfg.experiment.early_stopping_patience,
            mode="max",
            delta=0.0,
        )

        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()

        # ====================================================================
        # STAGE 1: PRETRAIN ON SYNTHETIC DATA
        # ====================================================================
        console.print(
            "\n[bold yellow]═══ STAGE 1: Pretraining on Synthetic Data ═══[/bold yellow]"
        )
        console.print(
            "Training until real validation accuracy plateaus (patience=10 epochs)\n"
        )

        best_stage1_metric = 0.0
        best_stage1_epoch = 0

        for epoch in range(1, pcfg.model.num_epochs + 1):
            console.print(
                f"\n[bold cyan]Stage 1 - Epoch {epoch}/{pcfg.model.num_epochs}[/bold cyan]"
            )

            # Train on synthetic data
            metrics_train = train_one_epoch(
                model=model,
                dataloader=train_synthetic_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
            )

            # Validate on REAL data (this is the key metric for stage 1)
            metrics_val = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            # Display results
            console.print(
                create_results_table(epoch, metrics_train, session="Stage1-Train")
            )
            console.print(
                create_results_table(epoch, metrics_val, session="Stage1-Val")
            )

            # Log to MLflow
            log_metrics_to_mlflow(
                metrics=metrics_train,
                prefix="stage1_train",
                epoch=epoch,
            )
            log_metrics_to_mlflow(
                metrics=metrics_val,
                prefix="stage1_val",
                epoch=epoch,
            )

            # Scheduler step
            current_metric = metrics_val[pcfg.experiment.compare_metric]
            scheduler.step(current_metric)

            # Track best model
            if current_metric > best_stage1_metric:
                best_stage1_metric = current_metric
                best_stage1_epoch = epoch

                # Save stage 1 best model
                stage1_model_path = Path("outputs") / "stage1_best_model.pth"
                stage1_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), stage1_model_path)
                console.print(
                    f"✓ Saved Stage 1 best model (epoch {epoch}, "
                    f"{pcfg.experiment.compare_metric}={current_metric:.4f})",
                    style="green",
                )

            # Early stopping check
            early_stopping_stage1(current_metric)
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

        for epoch in range(stage2_start_epoch, pcfg.model.num_epochs + 1):
            console.print(
                f"\n[bold cyan]Stage 2 - Epoch {epoch}/{pcfg.model.num_epochs}[/bold cyan]"
            )

            # Train on REAL data
            metrics_train = train_one_epoch(
                model=model,
                dataloader=train_real_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
            )

            # Validate on real data
            metrics_val = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            # Display results
            console.print(
                create_results_table(epoch, metrics_train, session="Stage2-Train")
            )
            console.print(
                create_results_table(epoch, metrics_val, session="Stage2-Val")
            )

            # Log to MLflow
            log_metrics_to_mlflow(
                metrics=metrics_train,
                prefix="stage2_train",
                epoch=epoch,
            )
            log_metrics_to_mlflow(
                metrics=metrics_val,
                prefix="stage2_val",
                epoch=epoch,
            )

            # Scheduler step
            current_metric = metrics_val[pcfg.experiment.compare_metric]
            scheduler.step(current_metric)

            # Track best model
            if current_metric > best_stage2_metric:
                best_stage2_metric = current_metric
                best_stage2_epoch = epoch

                # Save stage 2 best model (final model)
                final_model_path = Path(pcfg.model_output.directory) / "best_model.pth"
                final_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), final_model_path)
                console.print(
                    f"✓ Saved Stage 2 best model (epoch {epoch}, "
                    f"{pcfg.experiment.compare_metric}={current_metric:.4f})",
                    style="green",
                )

            # Early stopping check
            early_stopping_stage2(current_metric)
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
        console.print(
            "\n[bold yellow]═══ Final Evaluation on Test Set ═══[/bold yellow]"
        )
        console.print("Loading best Stage 2 model...\n")

        model.load_state_dict(torch.load(final_model_path))
        metrics_test = validate_one_epoch(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        console.print(create_results_table(0, metrics_test, session="Test"))

        # Log test metrics
        log_metrics_to_mlflow(
            metrics=metrics_test,
            prefix="test",
            epoch=0,
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
            f"Final Test {pcfg.experiment.compare_metric}: {metrics_test[pcfg.experiment.compare_metric]:.4f}"
        )

        mlflow.log_artifact(str(final_model_path))


if __name__ == "__main__":
    main()
