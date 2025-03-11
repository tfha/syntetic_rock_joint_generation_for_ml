import random
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int = 42) -> None:
    """
    Function to set random seed for reproducibility, similar to PyTorch Lightning's seed_everything.

    Args:
        seed (int): The seed value to use for random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(
        seed
    )  # Sets the seed for all CUDA devices (if using GPU)
    torch.backends.cudnn.deterministic = (
        True  # Ensures reproducibility in CuDNN (some slight slowdown)
    )
    torch.backends.cudnn.benchmark = (
        False  # Disables benchmark mode for reproducibility
    )


def check_and_update_best_metrics(
    metrics: dict[str, float],
    best_metrics: dict[str, Any],
    epoch: int,
    training_time: float,
) -> dict[str, Any]:
    if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
        best_metrics = {
            "epoch": epoch + 1,
            "loss": metrics["loss"],
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "training_time": training_time,
        }
        console = Console()
        console.print("[bold green]New best model found![/bold green]")
        console.print(
            create_results_table(epoch, best_metrics, session="Best Validation")
        )
    return best_metrics


# Function to create results table
def create_results_table(
    epoch: int, metrics: dict[str, float], session: str = "Training"
) -> Table:
    table = Table(title=f"Epoch {epoch + 1} {session} Results")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for metric, value in metrics.items():
        table.add_row(metric.capitalize(), f"{value:.2f}")
    return table


def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    metrics: dict[str, float],
    prefix: str,
    epoch: int,
) -> None:
    """
    Logs metrics to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        metrics (dict[str, float]): Dictionary of metrics to log.
        prefix (str): Prefix for metric names (e.g., 'Validation' or 'Training').
        epoch (int): Current epoch number.
    """
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"{prefix}/{metric_name}", metric_value, epoch)


def log_metrics_to_mlflow(
    best_metrics: dict[str, Any],
    model_name: str,
    model_params: dict[str, Any],
    experiment_strategy: str,
    experiment_name: str,
    tracking_uri: str = None,
    hydra_cfg_dir: str = None,
    save_best_metrics: bool = True,
    track_prediction_images: bool = False,
    save_model: bool = False,
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log Hydra config files as artifacts if provided
        if hydra_cfg_dir:
            hydra_cfg_dir = Path(hydra_cfg_dir)
            hydra_configs = [f for f in hydra_cfg_dir.iterdir() if f.suffix == ".yaml"]
            hydra_cfg_paths = []
            for config_file in hydra_configs:
                mlflow.log_artifact(str(config_file), artifact_path="hydra_configs")
                hydra_cfg_paths.append(str(config_file))

        # Save best_metrics as YAML and log as an artifact
        if save_best_metrics:
            best_metrics_yaml = "best_metrics.yaml"
            with open(best_metrics_yaml, "w") as f:
                yaml.dump(best_metrics, f)
            mlflow.log_artifact(best_metrics_yaml)

        # Log Hydra config files as artifacts if provided
        if hydra_cfg_dir:
            hydra_cfg_dir = Path(hydra_cfg_dir)
            hydra_configs = [f for f in hydra_cfg_dir.iterdir() if f.suffix == ".yaml"]
            hydra_cfg_paths = []
            for config_file in hydra_configs:
                mlflow.log_artifact(str(config_file), artifact_path="hydra_configs")
                hydra_cfg_paths.append(str(config_file))

        # Log predictions as an artifact if provided
        if track_prediction_images:
            mlflow.log_artifact(
                local_path="plots/predictions", artifact_path="predictions"
            )

        if save_model:
            mlflow.log_artifact(
                local_path=str(Path("models/best_model.pth")), artifact_path="models"
            )

        # Log best metrics
        mlflow.log_metrics(best_metrics)

        # Log model details
        mlflow.log_param("Model Name", model_name)
        mlflow.log_params(model_params)
        mlflow.log_param("Experiment strategy", experiment_strategy)


def get_custom_console() -> Console:
    """
    Returns a custom console with a custom theme.

    Returns:
        Console: Rich Console object with custom theme.

    """
    custom_theme = Theme(
        {"info": "bold green", "warning": "yellow", "danger": "bold red"}
    )
    return Console(theme=custom_theme)


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
