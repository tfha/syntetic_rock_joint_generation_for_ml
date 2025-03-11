import os  # noqa
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np  # noqa
import segmentation_models_pytorch as smp  # noqa
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rich.progress import track
from segmentation_models_pytorch.losses import DiceLoss  # noqa
from torch import nn, optim  # noqa
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
)
from ml_segmentation.debug_functionality import (  # noqa
    better_traceback,
    visualize_sample,
)
from ml_segmentation.define_model import choose_model
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.train_eval_funcs import EarlyStopping  # noqa
from ml_segmentation.train_eval_funcs import (
    save_image_predictions,
    train_one_epoch,
    validate_one_epoch,
)
from ml_segmentation.utility import (
    check_and_update_best_metrics,
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

    # SETUP
    ########################################################################
    console.print(
        f"Kicking off an experiment using the experiment strategy: {pcfg.experiment.experiment_strategy}",
        style="info",
    )
    writer = SummaryWriter(log_dir=pcfg.tensorboard.path)
    # Create a rich console with custom theme
    seed_everything(pcfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="info")
    pcfg.mlflow.experiment_name = f"train_test_{pcfg.experiment.experiment_strategy}"

    # LOAD DATA
    ###############################################################
    console.print("Loading training and testing data..", style="info")
    images_directory = pcfg.dataset.path_images
    labels_directory = pcfg.dataset.path_processed_mask_labels

    # Get transformations
    transforms_dict = get_transforms(
        optional_transforms=pcfg.experiment.optional_transforms
    )

    # Get prefixes
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

    # Split data
    train_list, val_list, test_list = split_data(
        train_files,
        test_files,
        train_frac=pcfg.experiment.train_fraction,
        val_frac=pcfg.experiment.val_fraction,
        test_frac=pcfg.experiment.test_fraction,
    )

    # Print the number of samples in train, validation, and test sets
    console.print(f"Number of training samples: {len(train_list)}")
    console.print(f"Number of validation samples: {len(val_list)}")
    console.print(f"Number of test samples: {len(test_list)}")

    # VALIDATE DATA
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

    ############################

    # Create datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_directory,
        labels_directory,
        train_list,
        val_list,
        test_list,
        transform=transforms_dict,
    )

    # # View sample for quality control
    # sample_idx = np.random.randint(0, len(train_dataset))
    # image, label = train_dataset[sample_idx]
    # visualize_sample(image, label)

    # Get DataLoaders
    # num_cpu = os.cpu_count()
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=pcfg.model.batch_size,
        num_workers=2,
    )

    # console.print("Shapes of data:", style="info")

    # # Iterate through the Train DataLoader
    # for images, labels in train_loader:
    #     print("Train Batch (image, label):", images.shape, labels.shape)
    #     break

    # # Iterate through the Validation DataLoader
    # for images, labels in val_loader:
    #     print("Validation Batch (image, label):", images.shape, labels.shape)
    #     break

    # # Iterate through the Test DataLoader
    # for images, labels in test_loader:
    #     print("Test Batch (image, label):", images.shape, labels.shape)
    #     break

    # DEFINE MODEL
    ###############################################################
    console.print("Define model...", style="info")
    model = choose_model(pcfg.model.name, pcfg.model.params).to(device)
    # Print model summary
    summary(model, input_size=(1, 3, 768, 768))  # Adjust input size as per your data

    # DEFINE LOSS FUNCTION, OPTIMIZER, SCHEDULER, EARLY STOPPING
    ########################################################################
    console.print(
        "Define loss function, optimizer, scheduler, and early stopping...",
        style="info",
    )

    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceLoss(
        mode="binary", from_logits=True
    )  # works better for imbalanced datasets
    optimizer = optim.Adam(model.parameters(), lr=pcfg.model.learning_rate)
    scaler = torch.amp.GradScaler(device="cuda")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=pcfg.model.scheduler.gamma,
        patience=pcfg.model.scheduler.patience,
    )
    # early_stopping = EarlyStopping(
    #     patience=pcfg.experiment.early_stopping_patience, verbose=True
    # )

    # ALTERNATIVE RUNS FOR DEBUG AND CHECKS
    ########################################################################
    num_epochs = pcfg.model.num_epochs

    if pcfg.experiment.overfit_check:  # Tested model should overfit
        console.print("Overfit check...", style="warning")
        test_loader = train_loader
        num_epochs = 3

    # TRAINING, VALIDATION, AND LOGGING
    ########################################################################

    # Test eventuelt Diceloss, dvs som beskrevet i segmentation_models_pytorch dokumentasjonen.
    # Kan mask eller image eller true image ha feil verdier for svart og hvit. Foelg form og verdier i data gjennom alle trinn og sjekk at ting fungerer som forventet.
    # Test uten resnet for initalisation
    # Test ut weighting i loss function pga unbalanced dataset
    # Test med preprocessing function from segmentation_models_pytorch

    console.print("Training and validation...", style="info")

    start_time = time.time()
    best_metrics = None

    # try:
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
            threshold=0.5,
            max_batches=pcfg.experiment.sanity_check,
        )
        console.print(create_results_table(epoch, metrics_training, session="Training"))

        metrics_training["learning_rate"] = optimizer.param_groups[0]["lr"]

        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_training, prefix="Training", epoch=epoch
        )

        # Validate for one epoch
        metrics_validation = validate_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
            max_batches=pcfg.experiment.sanity_check,
        )

        console.print(
            create_results_table(epoch, metrics_validation, session="Validation")
        )

        log_metrics_to_tensorboard(
            writer=writer, metrics=metrics_validation, prefix="Validation", epoch=epoch
        )

        # writer.add_scalar("Validation/Loss", metrics_validation["loss"], epoch)
        # writer.add_scalar("Validation/IoU", metrics_validation["iou"], epoch)
        # writer.add_scalar("Validation/Dice", metrics_validation["dice"], epoch)
        # writer.add_scalar("Validation/Precision", metrics_validation["precision"], epoch)
        # writer.add_scalar("Validation/Recall", metrics_validation["recall"], epoch)

        # Step the scheduler
        scheduler.step(metrics_validation["loss"])

        # Update best metrics if applicable
        training_time = time.time() - start_time
        best_metrics = check_and_update_best_metrics(
            metrics_validation, best_metrics, epoch, training_time
        )

        # Check early stopping condition
        # if not pcfg.experiment.sanity_check:
        #     early_stopping(metrics_validation["loss"], model)
        #     if early_stopping.early_stop:
        #         console.print(
        #             "Early stopping triggered. Training stopped.", style="warning"
        #         )
        #         break

    # except KeyboardInterrupt:
    # console.print("Training interrupted by keyboard.", style="warning")

    # finally:
    writer.close()
    console.print("Training complete.", style="info")

    # LOG RESULTS AND CONFIG TO MLFLOW
    ###############################################################
    console.print("Logging results to mlflow...", style="info")

    # if early_stopping.best_model:
    #     model.load_state_dict(early_stopping.best_model)
    #     model_path = Path("models/best_model.pth")
    #     torch.save(model.state_dict(), model_path)

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
            best_metrics,
            pcfg.model.name,
            pcfg.model.params,
            pcfg.experiment.experiment_strategy,
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
