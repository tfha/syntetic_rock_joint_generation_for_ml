"""
Azure ML PyTorch Lightning training script for rock mass segmentation.

This script provides a PyTorch Lightning-based alternative to azure_train_eval.py,
offering improved stability, memory management, and built-in features like:
- Automatic mixed precision training
- Distributed training support
- Better error handling and debugging
- Built-in early stopping and checkpointing
- Automatic gradient clipping

Architecture:
- Uses PyTorch Lightning for training management
- Leverages existing Azure ML dataset handling
- Maintains compatibility with existing configuration and models
- Provides enhanced stability for CUDA and memory management

The script handles:
1. Configuration setup and validation using Hydra
2. Azure ML dataset mounting and loading via Lightning DataModule
3. Model definition and Lightning module setup
4. Training with Lightning Trainer (automatic mixed precision, etc.)
5. MLflow integration for experiment tracking
6. TensorBoard logging through Lightning
7. Model checkpointing and artifact management
8. Comprehensive error handling and logging
"""

import os
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLflowLogger, TensorBoardLogger

from ml_segmentation.azure_core import configure_azure_logging
from ml_segmentation.data_loading import get_datasets_prefixes
from ml_segmentation.debug_functionality import better_traceback
from ml_segmentation.lightning_callbacks import (
    ImagePredictionCallback,
    MLflowCallback,
    ModelCheckpointCallback,
)
from ml_segmentation.lightning_datamodule import SegmentationDataModule
from ml_segmentation.lightning_module import SegmentationLightningModule
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.utility import get_custom_console, seed_everything


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Configure logging to reduce verbose Azure client output
    configure_azure_logging()
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    warnings.filterwarnings("ignore", category=UserWarning, module="msrest")

    # 1. Initialize configuration and setup
    ########################################################################
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
    experiment_name = (
        f"rock-segmentation-lightning-{pcfg.experiment.experiment_strategy}"
    )
    mlflow.set_experiment(experiment_name)

    # Start MLflow tracking - Azure ML automatically sets up the tracking URI
    mlflow.start_run()

    # 2. Setup output directories for storing results and logs
    ########################################################################
    console.print(
        "Starting Azure ML Lightning training run with strategy: "
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

    # Create MLflow logs directory for Azure ML
    mlflow_logs_dir = output_dir / "mlruns"
    mlflow_logs_dir.mkdir(exist_ok=True)
    console.print(f"MLflow logs will be saved to: {mlflow_logs_dir}", style="info")

    # Create Hydra outputs directory for Azure ML
    hydra_outputs_dir = output_dir / "hydra_outputs"
    hydra_outputs_dir.mkdir(exist_ok=True)

    # Override Hydra's output directory configuration for Azure ML
    timestamp_date = datetime.now().strftime("%Y-%m-%d")
    timestamp_time = datetime.now().strftime("%H-%M-%S")
    hydra_run_dir = hydra_outputs_dir / timestamp_date / timestamp_time
    hydra_run_dir.mkdir(parents=True, exist_ok=True)

    # Override Hydra's output directory
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

    # Create directories for Lightning logs and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    lightning_log_dir = output_dir / "lightning_logs" / timestamp
    lightning_log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = output_dir / "tensorboard_logs" / timestamp
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # Create directory for example images
    example_images_dir = output_dir / "example_images" / timestamp
    example_images_dir.mkdir(parents=True, exist_ok=True)

    # 3. Set random seed and configure device
    ########################################################################
    seed_everything(pcfg.experiment.seed)

    # Let Lightning handle device selection automatically
    console.print("Lightning will automatically handle device selection", style="info")

    try:
        # 4. Setup data and model
        ########################################################################
        console.print("Setting up data loading...", style="info")

        # Get dataset prefixes and paths
        dataset_prefix_groups = get_datasets_prefixes(pcfg)
        train_prefixes = dataset_prefix_groups["train_prefixes"]
        test_prefixes = dataset_prefix_groups["test_prefixes"]

        console.print(f"Train prefixes: {train_prefixes}", style="info")
        console.print(f"Test prefixes: {test_prefixes}", style="info")

        # Setup data paths (assuming Azure ML mounted paths)
        images_path = Path(pcfg.dataset.path_images)
        labels_path = Path(pcfg.dataset.path_processed_mask_labels)

        # Create Lightning data module
        data_module = SegmentationDataModule(
            images_path=images_path,
            labels_path=labels_path,
            batch_size=pcfg.model.batch_size,
            num_workers=pcfg.experiment.num_workers,
            optional_transforms=pcfg.experiment.optional_transforms,
            splits_path=None,  # Will use default behavior
            pin_memory=True,
            persistent_workers=pcfg.experiment.num_workers > 0,
        )

        # 5. Setup model
        ########################################################################
        console.print("Setting up Lightning model...", style="info")

        # Create Lightning module
        lightning_module = SegmentationLightningModule(
            model_name=pcfg.model.name,
            model_params=pcfg.model.params,
            learning_rate=pcfg.model.learning_rate,
            scheduler_patience=pcfg.model.scheduler.patience,
            scheduler_gamma=pcfg.model.scheduler.gamma,
            threshold=0.5,
        )

        # 6. Setup Lightning trainer and callbacks
        ########################################################################
        console.print("Setting up Lightning trainer...", style="info")

        # Setup loggers
        loggers = []

        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(tensorboard_log_dir.parent),
            name="",
            version=timestamp,
        )
        loggers.append(tb_logger)

        # MLflow logger (if in Azure ML environment)
        if os.environ.get("AZUREML_RUN_ID"):
            mlflow_logger = MLflowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow.get_tracking_uri(),
            )
            loggers.append(mlflow_logger)

        # Setup callbacks
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=pcfg.experiment.early_stopping_patience,
            min_delta=pcfg.experiment.early_stopping_delta,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(models_dir),
            filename="best-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Custom callbacks
        mlflow_callback = MLflowCallback(log_model=True)
        callbacks.append(mlflow_callback)

        image_callback = ImagePredictionCallback(
            output_dir=example_images_dir,
            save_every_n_epochs=pcfg.lightning.save_predictions_every_n_epochs,
            max_images=pcfg.lightning.max_prediction_images,
        )
        callbacks.append(image_callback)

        model_checkpoint_callback = ModelCheckpointCallback(checkpoint_dir=models_dir)
        callbacks.append(model_checkpoint_callback)

        # Create Lightning trainer
        trainer = pl.Trainer(
            max_epochs=pcfg.model.num_epochs,
            accelerator="auto",  # Automatically select accelerator (GPU/CPU)
            devices="auto",  # Automatically select number of devices
            precision=pcfg.lightning.precision
            if torch.cuda.is_available()
            else "32",  # Use config precision for GPU
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=pcfg.lightning.gradient_clip_val,  # Gradient clipping for stability
            accumulate_grad_batches=pcfg.lightning.accumulate_grad_batches,  # Gradient accumulation
            deterministic=pcfg.lightning.deterministic,  # For reproducibility
            fast_dev_run=pcfg.experiment.sanity_check_num_batches
            if pcfg.experiment.sanity_check_num_batches
            else False,
        )

        # 7. Training
        ########################################################################
        console.print("Starting Lightning training...", style="info")

        # Log configuration to MLflow
        if os.environ.get("AZUREML_RUN_ID"):
            mlflow.log_param("model_name", pcfg.model.name)
            mlflow.log_param("batch_size", pcfg.model.batch_size)
            mlflow.log_param("learning_rate", pcfg.model.learning_rate)
            mlflow.log_param("num_epochs", pcfg.model.num_epochs)
            mlflow.log_param("experiment_strategy", pcfg.experiment.experiment_strategy)
            mlflow.log_param("framework", "pytorch_lightning")

        # Train the model
        trainer.fit(lightning_module, data_module)

        # 8. Testing
        ########################################################################
        console.print("Running final testing...", style="info")

        # Test the model
        test_results = trainer.test(lightning_module, data_module)

        if test_results and os.environ.get("AZUREML_RUN_ID"):
            # Log test results to MLflow
            for metric_name, metric_value in test_results[0].items():
                mlflow.log_metric(f"final_{metric_name}", float(metric_value))

        # 9. Save final artifacts
        ########################################################################
        console.print("Saving final artifacts...", style="info")

        # Save final model state
        final_model_path = models_dir / "final_model.pth"
        torch.save(lightning_module.state_dict(), final_model_path)

        if os.environ.get("AZUREML_RUN_ID"):
            mlflow.log_artifact(str(final_model_path))

        console.print("Lightning training completed successfully!", style="bold green")

    except Exception as e:
        console.print(f"Training failed with error: {str(e)}", style="bold red")
        better_traceback()

        # Log error to MLflow
        if os.environ.get("AZUREML_RUN_ID"):
            mlflow.log_param("error", str(e))
            mlflow.log_param("training_status", "failed")
        raise

    finally:
        # 10. Cleanup
        ########################################################################
        console.print("Finalizing training...", style="info")

        # MLflow will be automatically ended by the context
        if os.environ.get("AZUREML_RUN_ID"):
            mlflow.log_param("training_status", "completed")


if __name__ == "__main__":
    main()
