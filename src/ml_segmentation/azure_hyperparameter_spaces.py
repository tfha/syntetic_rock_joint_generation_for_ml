"""
Azure ML hyperparameter search space definitions for various segmentation models.

This module contains definitions of hyperparameter search spaces for different
segmentation models, optimized for Azure ML's hyperparameter tuning capabilities.
"""

from typing import Any  # Only import Any as it doesn't have a built-in equivalent

# Search space definitions for different models
# Each model has a specific set of hyperparameters to tune


def get_unet_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the search space for UNet model hyperparameters.

    Returns:
        Dictionary with parameter names and their search spaces
    """
    return {
        # Architecture parameters
        "model.params.encoder_name": {
            "type": "choice",
            "values": ["resnet34", "resnet50", "efficientnet-b0", "efficientnet-b3"],
        },
        # encoder_weights is fixed to "imagenet" in base config, not tuned
        "model.params.decoder_use_batchnorm": {
            "type": "choice",
            "values": ["true", "false"],
        },
        # Training parameters
        "model.batch_size": {"type": "choice", "values": [4, 8, 12]},
        "model.learning_rate": {
            "type": "uniform",
            "min_value": 1e-5,
            "max_value": 1e-3,
        },
        "model.scheduler.gamma": {
            "type": "uniform",
            "min_value": 0.1,
            "max_value": 0.5,
        },
        "model.scheduler.patience": {"type": "choice", "values": [3, 5, 7]},
        # Data augmentation - independent control (16 combos: 2^4)
        "experiment.optional_transforms.random_crop": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.horizontal_flip": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.vertical_flip": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.color_jitter": {
            "type": "choice",
            "values": ["true", "false"],
        },
    }


def get_unetplusplus_search_space() -> dict[str, dict[str, Any]]:
    """UNet++ uses the same search space as UNet."""
    return get_unet_search_space()


def get_deeplabv3plus_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the search space for DeepLabV3+ model hyperparameters.

    Returns:
        Dictionary with parameter names and their search spaces
    """
    return {
        # Architecture parameters
        "model.params.encoder_name": {
            "type": "choice",
            "values": [
                "resnet50",
                "resnet101",
                "efficientnet-b4",
                "mobilenet_v2",
            ],
        },
        # encoder_weights is fixed to "imagenet" in base config, not tuned
        "model.params.decoder_channels": {
            "type": "choice",
            "values": [128, 256],
        },
        "model.params.upsampling": {"type": "choice", "values": [4, 8]},
        # Training parameters
        "model.batch_size": {"type": "choice", "values": [4, 8, 12]},
        "model.learning_rate": {
            "type": "uniform",
            "min_value": 5e-6,
            "max_value": 5e-4,
        },
        "model.scheduler.gamma": {
            "type": "uniform",
            "min_value": 0.1,
            "max_value": 0.5,
        },
        "model.scheduler.patience": {"type": "choice", "values": [3, 5, 7]},
        # Data augmentation - independent transform control (16 combos: 2^4)
        "experiment.optional_transforms.random_crop": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.horizontal_flip": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.vertical_flip": {
            "type": "choice",
            "values": ["true", "false"],
        },
        "experiment.optional_transforms.color_jitter": {
            "type": "choice",
            "values": ["true", "false"],
        },
    }


def get_model_search_space(model_name: str) -> dict[str, dict[str, Any]]:
    """
    Get the hyperparameter search space for a specific model.

    Args:
        model_name: Name of the model ("unet", "deeplabv3plus", etc.)

    Returns:
        Dictionary with parameter names and their search spaces

    Raises:
        ValueError: If the model name is not recognized
    """
    model_name = model_name.lower()

    if model_name == "unet" or model_name == "unetplusplus":
        return get_unet_search_space()
    elif model_name == "deeplabv3plus" or model_name == "deeplabv3":
        return get_deeplabv3plus_search_space()
    else:
        raise ValueError(f"Unsupported model for hyperparameter tuning: {model_name}")


def get_bayesian_sampling_params(
    max_total_trials: int = 200,
    max_concurrent_trials: int = 4,
    delay_evaluation: int = 15,
    slack_factor: float = 0.15,
    evaluation_interval: int = 1,
) -> dict[str, Any]:
    """
    Get the Bayesian sampling parameters for hyperparameter tuning.

    Uses Azure ML BanditPolicy for early termination with the following behavior:
    - Trials run for at least delay_evaluation epochs before termination is considered
    - After delay, checks every evaluation_interval epochs
    - Terminates if metric < best_metric × (1 - slack_factor) for maximization goals

    Args:
        max_total_trials: Maximum number of trials to run
        max_concurrent_trials: Maximum number of trials to run concurrently
        delay_evaluation: Number of intervals to delay first policy evaluation.
            Prevents premature termination during warm-up phase. Default=15 epochs.
        slack_factor: Ratio slack allowed vs best run. Trials with metric <
            best_metric × (1 - slack_factor) are terminated. Default=0.15 (15% slack).
        evaluation_interval: Frequency for applying policy (in epochs). Default=1.

    Returns:
        Dictionary with Bayesian sampling parameters and BanditPolicy configuration
    """
    return {
        "sampling_algorithm": "bayesian",
        "early_termination": {
            "type": "bandit",
            "evaluation_interval": evaluation_interval,
            "delay_evaluation": delay_evaluation,
            "slack_factor": slack_factor,
        },
        "primary_metric": "val_dice_score",
        "goal": "maximize",
        "max_total_trials": max_total_trials,
        "max_concurrent_trials": max_concurrent_trials,
    }
