import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from ml_segmentation.data_loading import (
    get_data_files,
    get_dataloaders,
    get_datasets,
    get_prefix_lists,
    get_transforms,
    split_data,
    validate_data,
)
from ml_segmentation.define_model import choose_model
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.train_eval_funcs import (
    EarlyStopping,
    save_image_predictions,
    train_one_epoch,
    validate_one_epoch,
)
from ml_segmentation.utility import (
    better_traceback,
    check_and_update_best_metrics,
    create_results_table,
    get_custom_console,
    log_metrics_to_mlflow,
    seed_everything,
)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # to dict
    pcfg = ConfigSchema(**cfg_dict)
    console = get_custom_console()
    console.print(pcfg)

    # SETUP
    ########################################################################
    writer = SummaryWriter(log_dir=pcfg.tensorboard.path)
    # Create a rich console with custom theme
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")
    pcfg.mlflow.experiment_name = "train_test"

    # LOAD DATA
    ###############################################################
    console.print("Loading training and testing data..", style="info")
    images_directory = pcfg.dataset.path_raw_rockmass
    labels_directory = pcfg.dataset.path_raw_labels

    # Get transformations
    transforms_dict = get_transforms(
        optional_transforms=pcfg.experiment.optional_transforms
    )

    # Get prefixes
    train_prefixes_list, test_prefixes_list = get_prefix_lists(
        pcfg.experiment.dataset_name_train,
        pcfg.experiment.dataset_name_test,
        pcfg.dataset.prefixes_synthetic_rock_slope,
        pcfg.dataset.prefixes_synthetic_fracman,
        pcfg.dataset.prefixes_synthetic_box,
        pcfg.dataset.prefixes_real_world_box,
    )

    # Get data files
    train_files, test_files = get_data_files(
        images_directory, labels_directory, train_prefixes_list, test_prefixes_list
    )

    # Split data and validate
    train_list, val_list, test_list = split_data(
        train_files,
        test_files,
        train_frac=pcfg.experiment.train_fraction,
        val_frac=pcfg.experiment.val_fraction,
        test_frac=pcfg.experiment.test_fraction,
    )
    validate_data(images_directory, labels_directory, train_list + val_list + test_list)

    # Create datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_directory,
        labels_directory,
        train_list,
        val_list,
        test_list,
        transform=transforms_dict,
    )

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=8
    )

    # Iterate through the Train DataLoader
    for images, labels in train_loader:
        print("Train Batch:", images.shape, labels.shape)
        break

    # Iterate through the Validation DataLoader
    for images, labels in val_loader:
        print("Validation Batch:", images.shape, labels.shape)
        break

    # Iterate through the Test DataLoader
    for images, labels in test_loader:
        print("Test Batch:", images.shape, labels.shape)
        break

    # DEFINE MODEL
    ###############################################################
    console.print("Define model...", style="info")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)
    # Print model summary
    summary(model, input_size=(1, 3, 224, 224))  # Adjust input size as per your data

    # DEFINE LOSS FUNCTION, OPTIMIZER, SCHEDULER, EARLY STOPPING
    ########################################################################
    console.print(
        "Define loss function, optimizer, scheduler, and early stopping...",
        style="info",
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=pcfg.model.scheduler.gamma,
        patience=pcfg.model.scheduler.patience,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        patience=pcfg.experiment.early_stopping_patience, verbose=True
    )

    # ALTERNATIVE RUNS FOR DEBUG AND CHECKS
    ########################################################################
    num_epochs = pcfg.model.num_epochs

    if pcfg.experiment.overfit_check:  # Tested model should overfit
        console.print("Overfit check...", style="warning")
        test_loader = train_loader
        num_epochs = 3

    # TRAINING, VALIDATION, AND LOGGING
    ########################################################################
    console.print("Training and validation...", style="info")

    start_time = time.time()
    best_metrics = None

    try:
        for epoch in range(num_epochs):
            console.print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            train_metrics = {"loss": train_loss}
            console.print(
                create_results_table(epoch, train_metrics, session="Training")
            )
            writer.add_scalar("Training/Loss", train_loss, epoch)
            writer.add_scalar(
                "Training/Learning Rate", optimizer.param_groups[0]["lr"], epoch
            )

            # Validate for one epoch
            metrics = validate_one_epoch(model, test_loader, criterion, device)
            console.print(create_results_table(epoch, metrics, session="Validation"))
            writer.add_scalar("Validation/Loss", metrics["loss"], epoch)
            writer.add_scalar("Validation/IoU", metrics["iou"], epoch)
            writer.add_scalar("Validation/Dice", metrics["dice"], epoch)
            writer.add_scalar("Validation/Precision", metrics["precision"], epoch)
            writer.add_scalar("Validation/Recall", metrics["recall"], epoch)

            # Step the scheduler
            scheduler.step(metrics["loss"])

            # Update best metrics if applicable
            training_time = time.time() - start_time
            best_metrics = check_and_update_best_metrics(
                metrics, best_metrics, epoch, training_time
            )

            # Check early stopping condition
            early_stopping(metrics["loss"], model)
            if early_stopping.early_stop:
                console.print(
                    "Early stopping triggered. Training stopped.", style="warning"
                )
                break

    except KeyboardInterrupt:
        console.print("Training interrupted by keyboard.", style="warning")

    finally:
        writer.close()
        console.print("Training complete.")

        # LOG RESULTS AND CONFIG TO MLFLOW
        ###############################################################
        console.print("Logging results to mlflow...", style="info")

        if early_stopping.best_model:
            model.load_state_dict(early_stopping.best_model)
            model_path = Path("models/best_model.pth")
            torch.save(model.state_dict(), model_path)

        if pcfg.experiment.log_mlflow:
            save_image_predictions(
                model,
                test_loader,
                device,
                num_samples=3,
                save_path=Path("plots/predictions"),
            )
            experiment_name = "train_test"
            log_metrics_to_mlflow(
                best_metrics,
                pcfg.model.name,
                pcfg.model.params,
                pcfg.experiment.dataset_name_train,
                pcfg.experiment.dataset_name_test,
                experiment_name,
                tracking_uri=pcfg.mlflow.path,
                hydra_cfg_dir=HydraConfig.get().run.dir,
                save_best_metrics=True,
                track_prediction_images=True,
                save_model=pcfg.mlflow.save_model,
            )


if __name__ == "__main__":
    better_traceback()
    main()
