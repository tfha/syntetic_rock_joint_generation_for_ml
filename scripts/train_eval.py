"""
Script to run a training session on your local machine.
This script performs the following steps:
1. Setup: Initializes logging, sets random seeds, and configures the device.
2. Load Data: Loads and preprocesses training, validation, and test datasets.
3. Validate Data: Optionally validates data before and after transformations.
4. Define Model: Initializes the model based on the configuration.
5. Define Loss, Optimizer, Scheduler: Sets up the loss function, optimizer, learning
   rate scheduler, and early stopping.
6. Training and Validation: Trains the model for a specified number of epochs, performs
   validation, and logs metrics.
7. Logging: Logs results and configuration to MLflow. Results during training are
   logged to TensorBoard.

Functions:
    main(cfg: DictConfig) -> None
        Main function to run the training session.
Args:
    cfg (DictConfig): Configuration object containing all the parameters for the
    training session.
Usage:
    Run this script with the appropriate configuration file to start a training session.
    Example:
        python scripts/train_eval.py model=unet
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rich.progress import track
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from ml_segmentation.data_loading import (
    get_data_files,
    get_dataloaders,
    get_datasets,
    get_datasets_prefixes,
    get_transforms,
    split_data,
    validate_data_post_transform,
    validate_data_pre_transform,
    validate_no_data_leakage,
)
from ml_segmentation.debug_functionality import (
    better_traceback,
    visualize_sample,
)
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
    log_metrics_to_mlflow,
    log_metrics_to_tensorboard,
    seed_everything,
)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # to dict
    pcfg = ConfigSchema(**cfg_dict)
    console = get_custom_console()
    console.print(pcfg)
    input("Press any Enter to continue...")  # Pause here to inspect config

    # SETUP
    ########################################################################
    # Create a rich console with custom theme
    console.print(
        "Kicking off an experiment using the experiment strategy:"
        f" {pcfg.experiment.experiment_strategy}",
        style="info",
    )

    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Create a unique log directory for each run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(pcfg.tensorboard.path, timestamp)
    writer = SummaryWriter(log_dir=log_dir)

    # Create directory for example images and clear any existing images
    example_images_dir = Path(f"{pcfg.experiment.path_example_images}/{timestamp}")

    # Clear any existing example images from previous runs
    if Path(pcfg.experiment.path_example_images).exists():
        console.print("Cleaning up existing example images...", style="info")
        try:
            shutil.rmtree(pcfg.experiment.path_example_images)
            console.print("Removed existing example images directory", style="info")
        except Exception as e:
            console.print(f"Error removing example images: {e}", style="warning")

    # Create fresh directory for new example images
    example_images_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(pcfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")
    pcfg.mlflow.experiment_name = f"train_test_{pcfg.experiment.experiment_strategy}"

    # LOAD DATA
    ###############################################################
    console.print("Loading training and testing data..", style="info")
    images_directory = Path(pcfg.dataset.path_images)
    labels_directory = Path(pcfg.dataset.path_processed_mask_labels)

    # Get transformations (must be defined before splits for later use)
    transforms_dict = get_transforms(
        optional_transforms=pcfg.experiment.optional_transforms,
        transforms_parameters={
            "crop_size": pcfg.dataset.crop_size,
        },
    )

    # Check if this experiment strategy uses custom splits (Wachter et al. experiments)
    strategy_config = pcfg.experiment.dataset_strategies.get(
        pcfg.experiment.experiment_strategy
    )
    if strategy_config:
        use_custom_splits = strategy_config.use_custom_splits
        custom_splits_dir = strategy_config.custom_splits_dir
    else:
        use_custom_splits = False
        custom_splits_dir = None

    # Determine split directory
    if use_custom_splits and custom_splits_dir:
        split_dir = Path(str(custom_splits_dir))
        console.print(
            f"Using custom splits from {split_dir} (Wachter et al. experiment)",
            style="info",
        )
    else:
        split_subfolder = (
            pcfg.experiment.experiment_strategy.lower()
            .replace(".", "_")
            .replace(" ", "_")
        )
        split_dir = Path("data/model_ready/splits") / split_subfolder
        split_dir.mkdir(parents=True, exist_ok=True)

    # Detect training strategy from experiment name
    experiment_name = str(pcfg.experiment.experiment_strategy).lower()
    is_finetune = experiment_name.startswith("finetune_")
    is_simplemixed = experiment_name.startswith("simplemixed_")

    # Determine which split files to load based on experiment strategy
    if is_finetune:
        # FT strategy uses separate files for two stages
        train_synthetic_json = split_dir / "train_synthetic.json"
        train_real_json = split_dir / "train_real.json"
        val_json = split_dir / "val.json"
        test_json = split_dir / "test.json"
        splits_exist = all(
            [
                train_synthetic_json.exists(),
                train_real_json.exists(),
                val_json.exists(),
                test_json.exists(),
            ]
        )
    elif is_simplemixed:
        # SM strategy uses mixed file
        train_json = split_dir / "train_all.json"
        val_json = split_dir / "val.json"
        test_json = split_dir / "test.json"
        splits_exist = all([train_json.exists(), val_json.exists(), test_json.exists()])
    else:
        # Standard training (no Wachter strategy)
        train_json = split_dir / "train.json"
        val_json = split_dir / "val.json"
        test_json = split_dir / "test.json"
        splits_exist = all([train_json.exists(), val_json.exists(), test_json.exists()])

    if splits_exist:
        console.print(f"Loading dataset splits from {split_dir}", style="info")

        if is_finetune:
            # Load FT strategy splits (will use them in two-stage training)
            with open(train_synthetic_json) as f:
                train_synthetic_list = json.load(f)
            with open(train_real_json) as f:
                train_real_list = json.load(f)
            # For now, use combined for initial dataloader setup
            # (will be properly handled in FT training loop implementation)
            train_list = train_synthetic_list + train_real_list
        elif is_simplemixed:
            # Load SM strategy mixed split
            with open(train_json) as f:
                train_list = json.load(f)
        else:
            # Standard training
            with open(train_json) as f:
                train_list = json.load(f)

        with open(val_json) as f:
            val_list = json.load(f)
        with open(test_json) as f:
            test_list = json.load(f)
    else:
        console.print(
            f"Generating dataset splits for local training (not found in {split_dir})",
            style="info",
        )
        # Get prefixes for files in the dataset to use for training and testing
        prefixes = get_datasets_prefixes(
            experiment_strategy=pcfg.experiment.experiment_strategy,
            dataset_strategies=pcfg.experiment.dataset_strategies,
            dataset_prefixes=pcfg.dataset.prefixes,
        )
        train_prefixes_list, test_prefixes_list = (
            prefixes["train_prefixes"],
            prefixes["test_prefixes"],
        )
        # Get data files
        train_files, test_files = get_data_files(
            images_directory, labels_directory, train_prefixes_list, test_prefixes_list
        )
        train_set = set(train_files)
        test_set = set(test_files)
        if train_set.isdisjoint(test_set):
            console.print(
                "Train and test sets are disjoint. No splitting will be performed;"
                "using all train and test files as provided.",
                style="info",
            )
            train_list = list(train_files)
            val_list = []
            test_list = list(test_files)
        else:
            # Determine split mode for messaging
            has_fractions = any(
                [
                    pcfg.experiment.train_fraction,
                    pcfg.experiment.val_fraction,
                    pcfg.experiment.test_fraction,
                ]
            )
            has_counts = any(
                [
                    pcfg.experiment.train_count,
                    pcfg.experiment.val_count,
                    pcfg.experiment.test_count,
                ]
            )

            if has_fractions:
                console.print(
                    f"Splitting with train frac.: "
                    f"{pcfg.experiment.train_fraction}, "
                    f"val frac.: {pcfg.experiment.val_fraction}, "
                    f"test frac.: {pcfg.experiment.test_fraction}...",
                    style="info",
                )
            elif has_counts:
                console.print(
                    f"Splitting with train count: "
                    f"{pcfg.experiment.train_count}, "
                    f"val count: {pcfg.experiment.val_count}, "
                    f"test count: {pcfg.experiment.test_count}...",
                    style="info",
                )

            train_list, val_list, test_list = split_data(
                train_files,
                test_files,
                train_frac=pcfg.experiment.train_fraction,
                val_frac=pcfg.experiment.val_fraction,
                test_frac=pcfg.experiment.test_fraction,
                train_count=pcfg.experiment.train_count,
                val_count=pcfg.experiment.val_count,
                test_count=pcfg.experiment.test_count,
            )
        # Save splits
        with open(train_json, "w") as f:
            json.dump(train_list, f)
        with open(val_json, "w") as f:
            json.dump(val_list, f)
        with open(test_json, "w") as f:
            json.dump(test_list, f)
        console.print(f"Saved splits to {split_dir}", style="success")

    # Print the number of samples in train, validation, and test sets
    console.print(f"Number of training samples: {len(train_list)}")
    console.print(f"Number of validation samples: {len(val_list)}")
    console.print(f"Number of test samples: {len(test_list)}")

    # VALIDATE NO DATA LEAKAGE (CRITICAL)
    ########################################
    console.print(
        "Validating data integrity (checking for train/val/test leakage)...",
        style="info",
    )
    if is_finetune:
        # For FT experiments, validate stage-specific splits to ensure no leakage
        validate_no_data_leakage(
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
            experiment_name=pcfg.experiment.experiment_strategy,
            train_synthetic_list=train_synthetic_list,
            train_real_list=train_real_list,
        )
    elif is_simplemixed:
        # For SimpleMixed experiments, val=test by design (Wachter strategy)
        validate_no_data_leakage(
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
            experiment_name=pcfg.experiment.experiment_strategy,
            allow_val_test_overlap=True,
        )
    else:
        # For standard experiments, validate combined splits only
        validate_no_data_leakage(
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
            experiment_name=pcfg.experiment.experiment_strategy,
        )

    # VALIDATE DATA - OPTIONAL
    ############################
    if pcfg.experiment.quality_control_data:
        console.print("Validate data...", style="info")
        validate_data_pre_transform(
            images_directory, labels_directory, train_list + val_list + test_list
        )

        image_transform = transforms_dict["image"]
        label_transform = transforms_dict["label"]
        file_list = train_list + val_list + test_list

        for file_name in track(
            file_list, description="Validating data files post transform..."
        ):
            image_path = images_directory / file_name
            label_path = labels_directory / file_name

            image = Image.open(image_path)
            label = Image.open(label_path)

            # Apply transforms
            transformed_image = image_transform(image)
            transformed_label = label_transform(label)

            validate_data_post_transform(
                transformed_image, transformed_label, file_name
            )
        input("Press any key to continue...")  # Pause here

    ############################

    # Create pytorch datasets objects
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_directory,
        labels_directory,
        train_list,
        val_list,
        test_list,
        transform=transforms_dict,
    )

    # View sample for quality control
    if pcfg.experiment.quality_control_data:
        console.print("Visualize sample...", style="info")
        sample_idx = np.random.randint(0, len(train_dataset))
        image, label = train_dataset[sample_idx]  # type: ignore[misc]
        visualize_sample(image, label)
        input("Press any key to continue...")  # Pause here

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=pcfg.model.batch_size,
        num_workers=pcfg.experiment.num_workers,
    )

    if pcfg.experiment.quality_control_data:
        console.print("Shapes of data:", style="info")

        # Iterate through the Train DataLoader
        for images, labels in train_loader:
            print("Train Batch (image, label):", images.shape, labels.shape)
            break

        # Iterate through the Validation DataLoader
        for images, labels in val_loader:
            print("Validation Batch (image, label):", images.shape, labels.shape)
            break

        # Iterate through the Test DataLoader
        for images, labels in test_loader:
            print("Test Batch (image, label):", images.shape, labels.shape)
            break

        input("Press any key to continue...")  # Pause here

    # DEFINE MODEL
    ###############################################################
    console.print("Define model...", style="info")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)
    if pcfg.experiment.quality_control_data:
        # Print model summary
        summary(
            model, input_size=(1, 3, 768, 768), verbose=1
        )  # Adjust input size as per your data
        input("Press any key to continue...")  # Pause here

    # DEFINE LOSS FUNCTION, OPTIMIZER, SCHEDULER, EARLY STOPPING
    ########################################################################
    console.print(
        "Define loss function, optimizer, scheduler, and early stopping callback...",
        style="info",
    )

    # Select loss function based on configuration
    criterion: nn.Module
    if pcfg.experiment.loss_function == "focal":
        # Focal Loss (based on Lin et al. 2017) from smp
        criterion = FocalLoss(
            mode="binary",
            alpha=pcfg.experiment.focal_alpha,
            gamma=pcfg.experiment.focal_gamma,
        )
    else:  # Default to "dice"
        # Dice Loss (based on V-Net, Milletari et al. 2016) from smp
        criterion = DiceLoss(mode="binary", from_logits=True)

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

    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)
    scaler = torch.amp.GradScaler(
        device="cuda"
    )  # Automatic Mixed Precision makes training faster and more memory efficient
    scheduler = (
        ReduceLROnPlateau(  # Reduce learning rate when a metric has stopped improving
            optimizer,
            mode="min",
            factor=pcfg.model.scheduler.gamma,
            patience=pcfg.model.scheduler.patience,
        )
    )
    # Determine early stopping mode based on compare_metric
    es_mode = "min" if pcfg.experiment.compare_metric == "loss" else "max"
    early_stopping = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience,
        verbose=True,
        delta=pcfg.experiment.early_stopping_delta,
        mode=es_mode,
    )

    # ALTERNATIVE RUNS FOR DEBUG AND CHECKS
    ########################################################################
    num_epochs = pcfg.model.num_epochs

    if pcfg.experiment.overfit_check:  # Tested model should overfit
        console.print("Overfit check...", style="warning")
        test_loader = train_loader
        num_epochs = 10  # need several epochs to stabilize the loss

    # TRAINING, VALIDATION, AND LOGGING
    ########################################################################

    console.print("Training and validation...", style="info")

    start_time = time.time()
    best_metrics = None

    try:  # makes it possible to stop the training with Ctrl+C and log the results
        for epoch in range(num_epochs):
            console.print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
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
                create_results_table(epoch, metrics_training, session="Training")
            )

            metrics_training["learning_rate"] = optimizer.param_groups[0]["lr"]

            log_metrics_to_tensorboard(
                writer=writer, metrics=metrics_training, prefix="Training", epoch=epoch
            )

            # Validate for one epoch (only if validation data exists)
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
                        epoch, metrics_validation, session="Validation"
                    )
                )

                log_metrics_to_tensorboard(
                    writer=writer,
                    metrics=metrics_validation,
                    prefix="Validation",
                    epoch=epoch,
                )

                # Step the scheduler
                scheduler.step(metrics_validation["loss"])
            else:
                # No validation data - use training metrics
                console.print(
                    "No validation data available, using training loss "
                    "for scheduler and early stopping",
                    style="warning",
                )
                metrics_validation = metrics_training.copy()
                scheduler.step(metrics_training["loss"])

            # Save example predictions every 3rd epoch (use validation set)
            if (epoch + 1) % 3 == 0 and len(val_loader.dataset) > 0:
                save_image_predictions(
                    model,
                    val_loader,
                    device,
                    num_samples=3,
                    save_dir=example_images_dir / f"epoch_{epoch + 1}",
                )

            # Step the scheduler
            scheduler.step(metrics_validation["loss"])

            # Update best metrics if applicable
            training_time = time.time() - start_time
            best_metrics = check_and_update_best_metrics(
                metrics_validation,
                best_metrics,
                epoch,
                training_time,
                compare_metric=pcfg.experiment.compare_metric,
            )

            # Check early stopping condition
            if not pcfg.experiment.sanity_check_num_batches:
                # Use compare_metric for early stopping (e.g., 'iou_joints' or 'loss')
                metric_value = metrics_validation[pcfg.experiment.compare_metric]
                early_stopping(metric_value, model)
                if early_stopping.early_stop:
                    console.print(
                        "Early stopping triggered. Training stopped.", style="warning"
                    )
                    break

    except KeyboardInterrupt:
        console.print("Training interrupted by keyboard.", style="warning")

    finally:
        writer.close()
        console.print("Training complete.", style="info")

        # Load best model and evaluate on test set
        ###############################################################
        if early_stopping.best_model is not None:
            console.print("Loading best model for final evaluation...", style="info")
            model.load_state_dict(early_stopping.best_model)
            model_path = Path("models/best_model.pth")
            torch.save(model.state_dict(), model_path)

        # Final evaluation on test set (only after training is complete)
        if len(test_loader.dataset) > 0:
            console.print("Evaluating on test set...", style="info")
            metrics_test = validate_one_epoch(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                threshold=effective_threshold,
                max_batches=None,  # Evaluate on full test set
            )
            console.print(
                create_results_table(
                    epoch=-1, metrics=metrics_test, session="Test (Final)"
                )
            )
            # Add test metrics to best_metrics for logging
            if best_metrics is not None:
                best_metrics["test_loss"] = metrics_test["loss"]
                best_metrics["test_iou"] = metrics_test["iou"]
                best_metrics["test_iou_joints"] = metrics_test["iou_joints"]
                best_metrics["test_dice"] = metrics_test["dice"]
                best_metrics["test_dice_joints"] = metrics_test["dice_joints"]
                best_metrics["test_precision"] = metrics_test["precision"]
                best_metrics["test_precision_joints"] = metrics_test["precision_joints"]
                best_metrics["test_recall"] = metrics_test["recall"]
                best_metrics["test_recall_joints"] = metrics_test["recall_joints"]
        else:
            console.print(
                "No test data available for final evaluation.", style="warning"
            )

        # LOG RESULTS AND CONFIG TO MLFLOW
        ###############################################################
        console.print("Logging results to mlflow...", style="info")

        if pcfg.experiment.log_mlflow:
            save_image_predictions(
                model,
                test_loader,
                device,
                num_samples=3,
                save_dir=Path("plots/predictions"),
            )
            experiment_name = "train_test"
            log_metrics_to_mlflow(
                best_metrics if best_metrics is not None else {},
                pcfg.model.name,
                pcfg.model.params,
                pcfg.experiment.experiment_strategy,
                experiment_name,
                tracking_uri=str(pcfg.mlflow.path)
                if pcfg.mlflow.path is not None
                else None,
                hydra_cfg_dir=HydraConfig.get().run.dir,
                save_best_metrics=True,
                track_prediction_images=True,
                save_model=pcfg.mlflow.save_model,
            )


if __name__ == "__main__":
    better_traceback()
    main()
