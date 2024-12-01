from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from rich.console import Console


class Scheduler(BaseModel):
    patience: int
    gamma: float


# Define the main configuration schema
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


class ExperimentConfig(BaseModel):
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
    dataset_name_train: str = Field(..., description="Dataset name for training.")
    dataset_name_test: str = Field(..., description="Dataset name for testing.")
    optional_transforms: bool = Field(
        ..., description="Whether optional image transforms are used."
    )
    overfit_check: bool = Field(..., description="Whether overfit check is used.")
    crossvalidation: bool = Field(..., description="Whether cross-validation is used.")


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


class DatasetConfig(BaseModel):
    path_raw_rockmass: Path = Field(..., description="Path to raw rock mass data.")
    path_raw_labels: Path = Field(..., description="Path to raw labels data.")
    prefixes_synthetic_rock_slope: list[str] = Field(
        ..., description="Prefixes for synthetic rock slope data."
    )
    prefixes_synthetic_fracman: list[str] = Field(
        ..., description="Prefixes for synthetic fracman data."
    )
    prefixes_synthetic_box: list[str] = Field(
        ..., description="Prefixes for synthetic box data."
    )
    prefixes_real_world_box: list[str] = Field(
        ..., description="Prefixes for real world box data."
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
