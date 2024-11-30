from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd
import torch.nn as nn
from rich.console import Console


def log_mlflow_metrics_and_model(
    mlflow_path: Path,
    experiment_name: str,
    metrics: dict,
    artifacts: dict,
    model_name: str,
    model_params: dict,
    undersample_level: int,
    oversample_level: int,
    hydra_cfg_dir: Path,
) -> None:
    """
    Logs:
    - metrics
    - model details
    - path to Hydra config files
    - 5 random images from the dataset in mode raw, mask, and predicted mask

    """
    # Setting MLflow experiment and tracking URI
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Logging to MLflow
    with mlflow.start_run():
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model details
        model_details = {
            "model_name": model_name,
            "scaler": "StandardScaler",
            "undersample_level": undersample_level,
            "oversample_level": oversample_level,
        }
        mlflow.log_params(model_details)
        mlflow.log_params(model_params)

        # Log confusion matrix, and eventual other figures as artifact
        for name, fig in artifacts.items():
            mlflow.log_figure(fig, f"{name}.png")

        # Log Hydra config files as artifacts
        hydra_configs = [f for f in hydra_cfg_dir.iterdir() if f.suffix == ".yaml"]

        # Log paths of Hydra config files as MLflow parameters
        hydra_cfg_paths = []
        for config_file in hydra_configs:
            mlflow.log_artifact(str(config_file), artifact_path="hydra_configs")
            hydra_cfg_paths.append(str(config_file))

        # Log paths as MLflow parameters
        hydra_cfg_path_str = ", ".join(hydra_cfg_paths)
        mlflow.log_param("hydra_config_paths", hydra_cfg_path_str)


class EarlyStopping:
    """
    EarlyStopping is a class that implements early stopping functionality for model training.

    Args:
        patience (int): The number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints the early stopping counter.
        delta (float): The minimum change in the monitored metric to be considered as improvement.

    Attributes:
        patience (int): The number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints the early stopping counter.
        delta (float): The minimum change in the monitored metric to be considered as improvement.
        counter (int): The number of epochs since the last improvement.
        best_score (float or None): The best score achieved so far.
        early_stop (bool): Whether to stop the training early or not.
        val_loss_min (float): The minimum validation loss achieved so far.
        best_model (dict or None): The state dictionary of the best model.

    Methods:
        __call__(val_loss, model): Updates the early stopping criteria based on the validation loss.
        _save_best_model(model): Saves the state dictionary of the best model.

    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: None | float = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.best_model: None | dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Updates the early stopping criteria based on the validation loss.

        Args:
            val_loss (float): The validation loss of the current epoch.
            model (nn.Module): The model being trained.

        Returns:
            None

        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_best_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_best_model(model)
            self.counter = 0

    def _save_best_model(self, model: nn.Module) -> None:
        """
        Saves the state dictionary of the best model.

        Args:
            model (nn.Module): The model to save.

        Returns:
            None

        """
        self.best_model = model.state_dict()
        self.val_loss_min = -self.best_score


def modify_filepath(original_path: Path, endsection: str) -> Path:
    """
    Modify the given filepath by appending an endsection before the file extension.

    For example:
    >>> original_path = Path('/ML-MWD-prediction-tabular/data/train.csv')
    >>> modify_filepath(original_path, '_modified')
    Path('/ML-MWD-prediction-tabular/data/train_modified.csv')

    Parameters:
        original_path (Path): Original file path.
        endsection (str): String to append before the file extension.

    Returns:
        Path: Modified file path.
    """
    parent = original_path.parent
    stem = original_path.stem
    suffix = original_path.suffix

    new_filename = f"{stem}{endsection}{suffix}"
    new_path = parent / new_filename

    return new_path


def track_sample_num(func: Callable) -> Callable:
    """Tracking number of samples of a dataframe before and after processing."""
    console = Console()

    def df_processing(*args: int, **kwargs: int) -> pd.DataFrame:
        res = func(*args, **kwargs)
        for data in args:
            if isinstance(data, pd.DataFrame):
                console.print("--------------------------------")
                console.print(
                    f"Number of samples before processing with {func.__name__} function"
                    f" (rows,cols): {data.shape}"
                )
                console.print(
                    f"Number of samples after processing with {func.__name__} function"
                    f" (rows,cols): {res.shape}"
                )
                console.print("--------------------------------")
        return res

    return df_processing
