# CUDA Memory Management and OOM Error Prevention

This document describes the memory management improvements implemented to prevent SIGSEGV and OOM errors when training on Azure ML with T4 GPUs (Standard_NC64as_T4_v3).

## Root Causes Identified

### 1. Memory Configuration Issues
- **Batch size too large**: Default `batch_size=16` with `crop_size=768` exceeded T4's 16GB VRAM
- **Inadequate worker management**: `num_workers=4` without proper thread limiting caused multiprocessing contention
- **No CUDA memory management**: Missing environment variables and memory optimization settings

### 2. Docker Environment Issues
- **Development image overhead**: Using `nvidia/cuda:12.1.1-cudnn8-devel` instead of runtime image
- **Missing environment variables**: No thread limiting for OpenMP/MKL operations
- **Suboptimal memory allocation**: No PyTorch CUDA memory management configuration

### 3. Threading and Synchronization Issues
- **Thread oversubscription**: Multiple libraries competing for CPU threads
- **CUDA context conflicts**: Missing proper CUDA environment setup

## Implemented Solutions

### 1. Optimized Docker Environment (`Dockerfile`)

**Changes made:**
- Switched from `cudnn8-devel` to `cudnn8-runtime` for smaller image size
- Added comprehensive environment variables for thread management:
  ```dockerfile
  ENV OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      NUMBA_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      CUDA_LAUNCH_BLOCKING=0 \
      PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"
  ```

**Benefits:**
- Prevents thread oversubscription
- Optimizes CUDA memory allocation
- Reduces image size by ~2GB

### 2. CUDA Memory Management Utilities (`src/ml_segmentation/cuda_memory_utils.py`)

**New functions:**
- `setup_cuda_environment()`: Configures optimal CUDA environment variables
- `get_optimal_batch_size()`: Automatically determines safe batch size based on GPU memory
- `configure_dataloader_for_memory()`: Optimizes DataLoader settings for memory efficiency
- `handle_cuda_oom_error()`: Provides actionable guidance when OOM errors occur
- `monitor_gpu_memory()`: Real-time GPU memory monitoring and logging

**Key features:**
- Automatic batch size adjustment based on available GPU memory
- Mixed precision training setup with error handling
- Memory-aware DataLoader configuration
- Comprehensive error handling and recovery suggestions

### 3. Memory-Optimized Configuration Defaults

**Model configuration changes:**
- Reduced default `batch_size` from 16 to 8
- Reduced `crop_size` from 768 to 512 pixels
- Reduced `num_workers` from 4 to 2
- Added memory-optimized model variants

**DataLoader optimizations:**
- Added `prefetch_factor=2` for efficient memory usage
- Enabled `drop_last=True` for training to maintain consistent memory usage
- Automatic `pin_memory` and `persistent_workers` configuration based on device

### 4. Enhanced Training Loop (`scripts/azure_train_eval.py`)

**Memory management features:**
- Automatic optimal batch size detection and adjustment
- Per-epoch GPU memory monitoring and logging
- Graceful CUDA OOM error handling with actionable suggestions
- Mixed precision training with proper error handling

**Error handling improvements:**
- Specific CUDA OOM error detection and handling
- Memory statistics logging to MLflow
- Suggested configuration changes when errors occur

## Usage Guidelines

### For T4 GPUs (16GB VRAM)
**Recommended settings:**
```yaml
model:
  batch_size: 8  # or lower if still getting OOM
dataset:
  crop_size: 512  # reduced from 768
experiment:
  num_workers: 2  # reduced from 4
```

### For V100 GPUs (32GB VRAM)
**Recommended settings:**
```yaml
model:
  batch_size: 16  # can use higher batch sizes
dataset:
  crop_size: 768  # can use larger images
experiment:
  num_workers: 4  # can use more workers
```

### For A100 GPUs (40GB/80GB VRAM)
**Recommended settings:**
```yaml
model:
  batch_size: 32  # can use much higher batch sizes
dataset:
  crop_size: 1024  # can use very large images
experiment:
  num_workers: 8  # can use many workers
```

## Azure ML Compute Resources

The following Azure ML compute nodes are available in the workspace:

### Available Compute Nodes

1. **Standard-NC6s-v3**: NVIDIA Tesla V100 — 16 GB per GPU — 1 GPU/node — total 16 GB
2. **NC64as-T4-v3**: NVIDIA T4 — 16 GB per GPU — 4 GPUs/node — total 64 GB  
3. **NC80adis-H100-v5**: NVIDIA H100 NVL — 94 GB per GPU — 2 GPUs/node — total 188 GB
4. **cpu-standard (Standard_DS11_v2)**: no GPU

### Recommended Configuration Mapping

**Standard-NC6s-v3 (V100 - 16GB)**: Use **T4 GPU settings** due to limited 16GB VRAM
```yaml
model:
  batch_size: 8
dataset:
  crop_size: 512
experiment:
  num_workers: 2
```

**NC64as-T4-v3 (T4 - 4 GPUs)**: Use **T4 GPU settings** per GPU, can leverage multiple GPUs
```yaml
model:
  batch_size: 8  # per GPU, total effective batch_size: 32
dataset:
  crop_size: 512
experiment:
  num_workers: 2  # per GPU
```

**NC80adis-H100-v5 (H100 - 94GB)**: Use **enhanced A100+ settings** for maximum performance
```yaml
model:
  batch_size: 64  # can use very large batch sizes
dataset:
  crop_size: 1024  # can use very large images
experiment:
  num_workers: 8  # can use many workers
```

**cpu-standard**: CPU-only training (not recommended for large models)
```yaml
model:
  batch_size: 4  # small batch size for CPU
dataset:
  crop_size: 256  # smaller images for CPU
experiment:
  num_workers: 1  # single worker for CPU
```

## Monitoring and Debugging

### GPU Memory Monitoring
The training script now automatically logs GPU memory usage:
- Initial GPU memory state
- Per-epoch memory usage
- Peak memory allocation
- Memory utilization percentage

### Error Handling
When CUDA OOM errors occur, the system will:
1. Detect the OOM error automatically
2. Clear CUDA cache
3. Log the error details to MLflow
4. Suggest specific configuration changes
5. Provide the exact batch size to try

### MLflow Integration
New memory-related metrics logged:
- `gpu_allocated_gb`: Current GPU memory usage
- `gpu_total_gb`: Total GPU memory available
- `gpu_utilization_percent`: Memory utilization percentage
- `adjusted_batch_size`: Auto-adjusted batch size (if changed)
- `mixed_precision`: Whether mixed precision is enabled

## Best Practices

### 1. Start with Conservative Settings
Begin with the T4-optimized settings even on larger GPUs, then scale up gradually.

### 2. Monitor Memory Usage
Check MLflow logs for memory utilization patterns to optimize settings.

### 3. Use Mixed Precision Training
Enable mixed precision to reduce memory usage by 30-50%:
```python
# Automatically enabled in the updated training script
use_amp = device.type == "cuda"
```

### 4. Batch Size Optimization
Let the system automatically determine optimal batch size:
```python
# Automatically handled in azure_train_eval.py
optimal_batch_size = get_optimal_batch_size(model, input_shape, device)
```

### 5. Error Recovery
When encountering OOM errors:
1. Check the suggested batch size in the error message
2. Update your configuration with the suggested value
3. Restart training with the new settings

## Configuration Examples

### Memory-Optimized for T4
```yaml
defaults:
  - model: unet_memory_optimized  # Use the memory-optimized variant

azure_ml:
  compute_name: "NC64as-T4-v3"

model:
  batch_size: 8

dataset:
  crop_size: 512

experiment:
  num_workers: 2
```

### High-Performance for V100/A100
```yaml
defaults:
  - model: unet

azure_ml:
  compute_name: "your-v100-cluster"

model:
  batch_size: 16  # or higher for A100

dataset:
  crop_size: 768  # or 1024 for A100

experiment:
  num_workers: 4  # or 8 for A100
```

## Troubleshooting

### Still Getting OOM Errors?
1. Further reduce batch size (try 4 or 2)
2. Reduce crop size (try 256 or 384)
3. Set num_workers to 0 or 1
4. Check GPU memory usage patterns in MLflow

### SIGSEGV Errors?
1. Ensure all environment variables are set correctly
2. Reduce num_workers to 0 (disable multiprocessing)
3. Check for CUDA driver compatibility issues
4. Verify PyTorch and CUDA versions match

### Performance Issues?
1. Gradually increase batch size while monitoring memory
2. Enable mixed precision training
3. Use persistent workers if memory allows
4. Monitor GPU utilization in addition to memory

## Testing the Changes

Run the memory utilities tests to verify functionality:
```bash
poetry run pytest tests/test_cuda_memory_utils.py -v
```

Test with a smoke test job:
```bash
python scripts/azure_submit_job.py experiment.smoke_test=true
```