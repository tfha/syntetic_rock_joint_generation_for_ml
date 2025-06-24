"""
This module defines the configuration schema for a machine learning segmentation
project using Pydantic models. It includes configurations for the model,
experiment, dataset, logging, and hyperparameter optimization.

Classes:
    Scheduler: Configuration for the learning rate scheduler.
    ModelConfig: Configuration for the model, including parameters like name,
        number of epochs, batch size, learning rate, and scheduler.
    ExperimentStrategy: Enum class defining various experiment strategies.
    ExperimentConfig: Configuration for the experiment, including strategies,
        dataset configurations, and other training parameters.
    DatasetConfig: Configuration for the dataset paths and prefixes.
    MlflowConfig: Configuration for MLflow logging.
    TensorboardConfig: Configuration for Tensorboard logging.
    OptunaConfig: Configuration for Optuna hyperparameter optimization.
    AzureMLConfig: Configuration for Azure Machine Learning.
    AzureDataAssetsCommand: Enum class defining available commands for Azure
        data assets management.
    AzureDataAssetsConfig: Configuration for Azure ML data assets management.
    ConfigSchema: The main configuration schema that includes all other
        configurations.

Functions:
    testing_scheme_functionality(cfg: DictConfig): A Hydra main function that
        tests the schema functionality by converting the Hydra config to a
        Pydantic model and printing it using Rich console.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator
from rich.console import Console


class Scheduler(BaseModel):
    patience: int
    gamma: float


class ModelConfig(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Name of the model configuration to use, e.g., deeplabv3plus, "
            "unet, unetplusplus"
        ),
    )
    num_epochs: int = Field(..., description="Number of epochs for training.")
    batch_size: int = Field(..., description="Batch size for training.")
    learning_rate: float = Field(..., description="Learning rate for the optimizer.")
    scheduler: Scheduler = Field(..., description="Scheduler configuration.")
    params: dict[str, Any] = Field(..., description="Dictionary of model parameters.")


class ExperimentStrategy(str, Enum):
    VERIFICATION_BOX = "verification_box"
    VERIFICATION_DFN = "verification_dfn"
    MAIN_OBJECTIVE_DFN_ROCK_SLOPE = "main_objective_dfn_rock_slope"
    MAIN_OBJECTIVE_DFN_BOX = "main_objective_dfn_box"
    MAIN_OBJECITVE_BOX_ROCK_SLOPE = "main_objective_box_rock_slope"
    MAIN_OBJECTIVE_BOX_BOX = "main_objective_box_box"
    DATASET_SIZE_TEST = "dataset_size_test"
    SEMI_SUPERVISED_LEARNING = "semi_supervised_learning"
    ONE_SHOT_SEGMENTATION = "one_shot_segmentation"


class ExperimentConfig(BaseModel):
    experiment_strategy: ExperimentStrategy = Field(
        ..., description="The experiment strategy chosen for this run."
    )
    dataset_strategies: dict[str, dict[str, list[str]]] = Field(
        ...,
        description=(
            "Mapping of experiment strategies to their corresponding dataset "
            "configurations. Each strategy includes 'train_datasets' and "
            "'test_datasets', which are lists of dataset names used for "
            "training and testing respectively."
        ),
    )
    seed: int = Field(..., description="Random seed for reproducibility.")
    log_mlflow: bool = Field(..., description="Whether to log to mlflow or not.")
    compare_metric: str = Field(
        ..., description="Metric used for comparison in choosing new best metrics."
    )
    num_workers: int = Field(..., description="Number of workers for data loading.")
    train_fraction: float = Field(
        ..., description="Fraction of data used for training."
    )
    val_fraction: float = Field(
        ..., description="Fraction of data used for validation."
    )
    test_fraction: float = Field(..., description="Fraction of data used for testing.")
    early_stopping_patience: int = Field(
        ..., description="Patience for early stopping in training."
    )
    early_stopping_delta: float = Field(
        ...,
        description=(
            "Minimum change in the monitored metric to qualify as an "
            "improvement for early stopping."
        ),
    )
    optional_transforms: bool = Field(
        ..., description="Whether optional image transforms are used."
    )
    overfit_check: bool = Field(..., description="Whether overfit check is used.")
    sanity_check_num_batches: int | None = Field(
        ..., description="Whether to run a sanity check for a number of batches."
    )
    quality_control_data: bool = Field(
        ..., description="Whether quality control data is used."
    )
    crossvalidation: bool = Field(..., description="Whether cross-validation is used.")
    path_example_images: Path = Field(
        ..., description="Path where example images are saved during training."
    )
    download_outputs: bool = Field(
        False,
        description="Flag to control whether to download outputs after job completion.",
    )


class DatasetConfig(BaseModel):
    path_images: Path = Field(..., description="Path to raw rock mass data.")
    path_raw_mask_labels: Path = Field(..., description="Path to raw labels data.")
    path_processed_mask_labels: Path = Field(
        ..., description="Path to processed masks."
    )
    prefixes: dict[str, list[str]] = Field(
        ...,
        description=(
            "Mapping of dataset names to lists of prefixes used to filter "
            "files for that dataset."
        ),
    )


class MlflowConfig(BaseModel):
    path: Path = Field(..., description="Path where mlflow logs are saved.")
    experiment_name: str | None = Field(
        None, description="Experiment name for mlflow tracking."
    )
    save_model: bool = Field(..., description="Whether to save the model or not.")


class TensorboardConfig(BaseModel):
    path: Path = Field(..., description="Path where tensorboard logs are saved.")


class OptunaConfig(BaseModel):
    n_trials: int = Field(
        ..., description="Number of trials for hyperparameter optimization."
    )
    path_results: Path = Field(
        ..., description="Directory path for storing hyperparameter results."
    )


class AzureMLConfig(BaseModel):
    compute_name: str = Field(
        ..., description="Name of the compute cluster to use for Azure ML training."
    )
    experiment_name: str = Field(..., description="Name of the experiment in Azure ML.")
    environment_name: str = Field(
        "rock-segmentation-env",
        description="Name of the Azure ML environment to use or create.",
    )
    environment_version: str = Field(
        "latest", description="Version of the Azure ML environment to use."
    )
    use_new_version: bool = Field(
        False, description="Whether to create and use a new version of the environment."
    )


class AzureDataAssetsCommand(str, Enum):
    """Available commands for Azure data assets management."""

    REGISTER_BASE_DATASETS = "register-base-datasets"
    REGISTER_SPLITS = "register-splits"
    LIST_ASSETS = "list-assets"
    COMPARE_ASSETS = "compare-assets"
    UPLOAD_DATA = "upload-data"
    GENERATE_SPLITS = "generate-splits"
    UPLOAD_SPLITS = "upload-splits"
    PROCESS_ALL_SPLITS = "process-all-splits"


class AzureDataAssetsConfig(BaseModel):
    """Configuration for Azure ML data assets management."""

    command: AzureDataAssetsCommand | None = Field(
        None,
        description=(
            "Command to execute (register-base-datasets, register-splits, "
            "list-assets, compare-assets, upload-data, generate-splits)"
        ),
    )
    asset_name: str | None = Field(
        None, description="Name of the asset to list or compare"
    )
    version1: str | None = Field(None, description="First version for comparison")
    version2: str | None = Field(None, description="Second version for comparison")
    output_dir: str = Field(
        "outputs/azure_data_assets", description="Output directory for logs"
    )


class ConfigSchema(BaseModel):
    path_project: Path = Field(..., description="Path to the project directory.")
    matplotlib_config_path: Path = Field(
        ..., description="Path to the Matplotlib configuration stylesheet."
    )
    model: ModelConfig
    experiment: ExperimentConfig
    mlflow: MlflowConfig
    tensorboard: TensorboardConfig
    optuna: OptunaConfig
    dataset: DatasetConfig
    azure_ml: AzureMLConfig
    azure_data_assets: AzureDataAssetsConfig = Field(
        default_factory=AzureDataAssetsConfig,
        description="Configuration for Azure ML data assets management.",
    )

    @field_validator("azure_data_assets")
    @classmethod
    def validate_command(cls, v):
        # Check if command is valid
        if hasattr(v, "command"):
            try:
                AzureDataAssetsCommand(v.command)
            except ValueError:
                available_commands = [cmd.value for cmd in AzureDataAssetsCommand]
                raise ValueError(
                    f"'{v.command}' is not a valid command. "
                    f"Available commands: {', '.join(available_commands)}"
                )
        return v


@hydra.main(config_path="../../scripts/config", config_name="main", version_base="1.3")
def testing_scheme_functionality(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(
        cfg
    )  # Convert OmegaConf to a regular dictionary
    pcfg = ConfigSchema(
        **cfg_dict
    )  # Parsing hydra config with pydantic for quality control
    console = Console()
    console.print(pcfg)


if __name__ == "__main__":
    testing_scheme_functionality()
