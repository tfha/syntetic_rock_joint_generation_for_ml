"""
Memory management utilities for model training and memory optimization.

This module provides utilities for optimizing model memory usage,
handling CUDA OOM errors, and configuring optimal training parameters.
"""

import logging

import mlflow
import torch
from rich.console import Console

from ml_segmentation.cuda_memory_utils import (
    get_optimal_batch_size,
    handle_cuda_oom_error,
)
from ml_segmentation.schema_config import ConfigSchema

logger = logging.getLogger(__name__)


def setup_model_with_memory_optimization(
    model: torch.nn.Module,
    pcfg: ConfigSchema,
    device: torch.device,
    console: Console,
) -> tuple[torch.nn.Module, int]:
    """
    Set up model with memory optimization and batch size adjustment.

    Args:
        model: The model to optimize
        pcfg: Configuration schema with model and dataset parameters
        device: Target device for the model
        console: Rich console for logging

    Returns:
        Tuple of (optimized_model, optimal_batch_size)

    Raises:
        RuntimeError: If insufficient GPU memory even after optimization
    """
    try:
        model = model.to(device)

        # Check if batch size needs optimization based on available memory
        optimal_batch_size = pcfg.model.batch_size

        if device.type == "cuda":
            optimal_batch_size = get_optimal_batch_size(
                model=model,
                input_shape=(3, pcfg.dataset.crop_size, pcfg.dataset.crop_size),
                device=device,
                max_batch_size=pcfg.model.batch_size,
            )

            if optimal_batch_size < pcfg.model.batch_size:
                console.print(
                    f"Reducing batch size from {pcfg.model.batch_size} to {optimal_batch_size} "
                    "based on available GPU memory",
                    style="warning"
                )
                mlflow.log_param("adjusted_batch_size", optimal_batch_size)

        return model, optimal_batch_size

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print("Initial model allocation failed due to memory constraints", style="red")
            new_batch_size = handle_cuda_oom_error(e, pcfg.model.batch_size, console)
            raise RuntimeError(
                f"Insufficient GPU memory. Try reducing batch_size to {new_batch_size} "
                "or use a GPU with more memory."
            ) from e
        else:
            raise
