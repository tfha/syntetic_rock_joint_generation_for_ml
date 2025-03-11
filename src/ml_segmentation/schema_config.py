from enum import Enum
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from rich.console import Console


class Scheduler(BaseModel):
    patience: int
    gamma: float


class ModelConfig(BaseModel):
    name: str = Field(
        ...,
        description="Name of the model configuration to use, e.g., deeplabv3plus, unet, unetplusplus",
    )
    num_epochs: int = Field(..., description="Number of epochs for training.")
    batch_size: int = Field(..., description="Batch size for training.")
    learning_rate: float = Field(..., description="Learning rate for the optimizer.")
    scheduler: Scheduler = Field(..., description="Scheduler configuration.")
    params: dict[str, Any] = Field(..., description="Dictionary of model parameters.")


class ExperimentStrategy(str, Enum):
    STANDARD_RUN = "standard_run"
    STUDY_VERIFICATION = "study_verification"
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
            "Mapping of experiment strategies to their corresponding dataset configurations. "
            "Each strategy includes 'train_datasets' and 'test_datasets', "
            "which are lists of dataset names used for training and testing respectively."
        ),
    )
    seed: int = Field(..., description="Random seed for reproducibility.")
    log_mlflow: bool = Field(..., description="Whether to log to mlflow or not.")
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
    optional_transforms: bool = Field(
        ..., description="Whether optional image transforms are used."
    )
    overfit_check: bool = Field(..., description="Whether overfit check is used.")
    sanity_check: int | None = Field(
        ..., description="Whether to run a sanity check for a number of batches."
    )
    quality_control_data: bool = Field(
        ..., description="Whether quality control data is used."
    )
    crossvalidation: bool = Field(..., description="Whether cross-validation is used.")


class DatasetConfig(BaseModel):
    path_images: Path = Field(..., description="Path to raw rock mass data.")
    path_raw_mask_labels: Path = Field(..., description="Path to raw labels data.")
    path_processed_mask_labels: Path = Field(
        ..., description="Path to processed masks."
    )
    prefixes: dict[str, list[str]] = Field(
        ...,
        description="Mapping of dataset names to lists of prefixes used to filter files for that dataset.",
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
