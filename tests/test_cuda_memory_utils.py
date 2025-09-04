"""
Tests for CUDA memory management utilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from ml_segmentation.cuda_memory_utils import (
    configure_dataloader_for_memory,
    get_optimal_batch_size,
    handle_cuda_oom_error,
    setup_cuda_environment,
    setup_mixed_precision_training,
    monitor_gpu_memory,
)


def test_setup_cuda_environment():
    """Test CUDA environment setup function."""
    import os

    # Mock console
    console = Mock()

    # Test environment variable setup
    setup_cuda_environment(console)

    # Check that console.print was called
    assert console.print.called

    # Check that some environment variables are set
    expected_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_LAUNCH_BLOCKING"]
    for var in expected_vars:
        assert var in os.environ


def test_configure_dataloader_for_memory():
    """Test dataloader configuration for memory optimization."""

    # Test with CUDA device (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = configure_dataloader_for_memory(
        num_workers=4,
        device=device,
    )

    # Check that configuration is returned
    assert isinstance(config, dict)
    assert "num_workers" in config
    assert "pin_memory" in config
    assert "persistent_workers" in config

    # Check that num_workers is adjusted for memory constraints
    assert config["num_workers"] <= 4

    # Check pin_memory is set correctly based on device
    if device.type == "cuda":
        assert config["pin_memory"] is True
    else:
        assert config["pin_memory"] is False


def test_get_optimal_batch_size_cpu():
    """Test optimal batch size estimation for CPU."""

    # Create a simple model
    model = torch.nn.Linear(10, 1)
    device = torch.device("cpu")

    batch_size = get_optimal_batch_size(
        model=model,
        input_shape=(10,),
        device=device,
        max_batch_size=32,
    )

    # For CPU, should return a reasonable default
    assert isinstance(batch_size, int)
    assert 1 <= batch_size <= 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_optimal_batch_size_cuda():
    """Test optimal batch size estimation for CUDA."""

    # Create a simple model
    model = torch.nn.Linear(10, 1)
    device = torch.device("cuda")

    batch_size = get_optimal_batch_size(
        model=model,
        input_shape=(10,),
        device=device,
        max_batch_size=32,
    )

    # Should return a valid batch size
    assert isinstance(batch_size, int)
    assert 1 <= batch_size <= 32


def test_handle_cuda_oom_error():
    """Test CUDA OOM error handling."""

    # Mock console
    console = Mock()

    # Create a mock CUDA OOM error
    error = RuntimeError("CUDA out of memory")

    new_batch_size = handle_cuda_oom_error(
        error=error,
        current_batch_size=16,
        console=console,
    )

    # Should suggest reduced batch size
    assert new_batch_size < 16
    assert new_batch_size >= 1

    # Should have printed messages
    assert console.print.called


def test_setup_mixed_precision_training():
    """Test mixed precision training setup."""

    # Test with CPU device
    device = torch.device("cpu")
    scaler, enabled = setup_mixed_precision_training(device)

    # Should be disabled for CPU
    assert enabled is False
    assert scaler is None

    # Test with CUDA device (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        scaler, enabled = setup_mixed_precision_training(device)

        # Should be enabled for CUDA
        assert enabled is True
        assert scaler is not None
        assert isinstance(scaler, torch.cuda.amp.GradScaler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_monitor_gpu_memory():
    """Test GPU memory monitoring."""

    device = torch.device("cuda")

    # Test without console
    memory_stats = monitor_gpu_memory(device)

    # Should return memory statistics
    assert isinstance(memory_stats, dict)
    expected_keys = ["allocated_gb", "cached_gb", "max_allocated_gb", "total_gb", "utilization_percent"]
    for key in expected_keys:
        assert key in memory_stats
        assert isinstance(memory_stats[key], (int, float))

    # Test with console
    console = Mock()
    memory_stats_with_console = monitor_gpu_memory(device, console)

    # Should print memory information
    assert console.print.called

    # Should return same type of data
    assert isinstance(memory_stats_with_console, dict)


def test_monitor_gpu_memory_cpu():
    """Test GPU memory monitoring with CPU device."""

    device = torch.device("cpu")
    memory_stats = monitor_gpu_memory(device)

    # Should return empty dict for CPU
    assert memory_stats == {}


@patch("torch.cuda.is_available", return_value=False)
def test_functions_with_cuda_unavailable(mock_cuda_available):
    """Test functions behave correctly when CUDA is unavailable."""

    device = torch.device("cpu")

    # Test mixed precision setup
    scaler, enabled = setup_mixed_precision_training(device)
    assert enabled is False
    assert scaler is None

    # Test memory monitoring
    memory_stats = monitor_gpu_memory(device)
    assert memory_stats == {}

    # Test batch size estimation
    model = torch.nn.Linear(10, 1)
    batch_size = get_optimal_batch_size(model, (10,), device)
    assert isinstance(batch_size, int)
    assert batch_size > 0
