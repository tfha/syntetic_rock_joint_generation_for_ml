"""
CUDA memory management utilities for preventing OOM and SIGSEGV errors.

This module provides utilities for managing GPU memory efficiently, detecting
available memory, and handling CUDA errors gracefully in Azure ML environments.
"""

import logging
import os

import torch
from rich.console import Console

logger = logging.getLogger(__name__)


def setup_cuda_environment(console: Console | None = None) -> None:
    """
    Set up CUDA environment variables and memory management settings.

    This function configures various environment variables to prevent
    threading issues and optimize CUDA memory usage for Azure ML.

    Args:
        console: Optional console for logging messages
    """
    if console is None:
        console = Console()

    # Set threading limits to prevent oversubscription
    threading_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    for var, value in threading_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            console.print(f"Set {var}={value}", style="dim")

    # Configure CUDA memory management
    cuda_vars = {
        "CUDA_LAUNCH_BLOCKING": "0",  # Asynchronous for performance
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True",
    }

    for var, value in cuda_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            console.print(f"Set {var}={value}", style="dim")


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] = (3, 768, 768),
    device: torch.device | None = None,
    safety_factor: float = 0.8,
    max_batch_size: int = 32,
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        model: The model to estimate memory for
        input_shape: Shape of input tensors (C, H, W)
        device: CUDA device to test on
        safety_factor: Safety factor to apply (0.8 = use 80% of available memory)
        max_batch_size: Maximum batch size to try

    Returns:
        Optimal batch size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return min(16, max_batch_size)  # Default for CPU

    try:
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = total_memory * safety_factor

        # Estimate memory per sample by testing with batch size 1
        torch.cuda.empty_cache()
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape, device=device)
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated(device)

            _ = model(dummy_input)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated(device)

            memory_per_sample = peak_memory - initial_memory

        # Calculate optimal batch size
        if memory_per_sample > 0:
            optimal_batch = int(available_memory // memory_per_sample)
            optimal_batch = max(1, min(optimal_batch, max_batch_size))
        else:
            optimal_batch = min(8, max_batch_size)  # Conservative default

        torch.cuda.empty_cache()
        return optimal_batch

    except Exception as e:
        logger.warning(f"Failed to estimate optimal batch size: {e}")
        return min(4, max_batch_size)  # Very conservative default


def configure_dataloader_for_memory(
    num_workers: int,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    device: torch.device | None = None,
) -> tuple[int, bool, bool]:
    """
    Configure DataLoader parameters for optimal memory usage.

    Args:
        num_workers: Desired number of workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        device: Device being used

    Returns:
        Tuple of (num_workers, pin_memory, persistent_workers)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reduce workers if we're on a memory-constrained system
    if device.type == "cuda":
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory
            # If less than 24GB, reduce workers to prevent OOM
            if total_memory < 24 * 1024**3:  # 24GB in bytes
                num_workers = min(num_workers, 2)
        except Exception:
            num_workers = min(num_workers, 2)  # Conservative default

    # Configure pin_memory based on device
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    # Configure persistent_workers
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # For single worker or CPU, disable persistent workers to save memory
    if num_workers <= 1 or device.type == "cpu":
        persistent_workers = False

    return num_workers, pin_memory, persistent_workers


def handle_cuda_oom_error(
    error: Exception,
    current_batch_size: int,
    console: Console | None = None,
) -> int:
    """
    Handle CUDA out-of-memory errors by suggesting reduced batch size.

    Args:
        error: The CUDA OOM error
        current_batch_size: Current batch size when error occurred
        console: Optional console for logging

    Returns:
        Suggested new batch size
    """
    if console is None:
        console = Console()

    console.print(f"CUDA OOM Error: {error}", style="red")

    # Suggest reducing batch size
    new_batch_size = max(1, current_batch_size // 2)

    console.print(
        f"Suggested solution: Reduce batch_size from {current_batch_size} to {new_batch_size}",
        style="yellow"
    )

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        console.print("Cleared CUDA cache", style="info")

    return new_batch_size


def setup_mixed_precision_training(device: torch.device) -> tuple[torch.cuda.amp.GradScaler | None, bool]:
    """
    Set up mixed precision training configuration.

    Args:
        device: Device being used for training

    Returns:
        Tuple of (GradScaler, enabled_flag)
    """
    use_amp = device.type == "cuda"

    if use_amp:
        try:
            # Test if mixed precision is supported
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            return scaler, True
        except Exception as e:
            logger.warning(f"Mixed precision not available: {e}")
            return None, False

    return None, False


def monitor_gpu_memory(device: torch.device, console: Console | None = None) -> dict[str, float]:
    """
    Monitor and report GPU memory usage.

    Args:
        device: CUDA device to monitor
        console: Optional console for logging

    Returns:
        Dictionary with memory statistics in GB
    """
    if device.type != "cuda":
        return {}

    try:
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB

        memory_stats = {
            "allocated_gb": round(allocated, 2),
            "cached_gb": round(cached, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "total_gb": round(total, 2),
            "utilization_percent": round((allocated / total) * 100, 1),
        }

        if console:
            console.print(
                f"GPU Memory: {memory_stats['allocated_gb']:.1f}GB/{memory_stats['total_gb']:.1f}GB "
                f"({memory_stats['utilization_percent']:.1f}%)",
                style="info"
            )

        return memory_stats

    except Exception as e:
        logger.warning(f"Failed to get GPU memory stats: {e}")
        return {}


def optimize_for_inference(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Optimize model for inference to reduce memory usage.

    Args:
        model: Model to optimize
        device: Device the model is on

    Returns:
        Optimized model
    """
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Try to compile model if supported (PyTorch 2.0+)
    try:
        if hasattr(torch, "compile") and device.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        logger.info(f"Model compilation not available: {e}")

    return model
