# PyTorch Lightning Training for Azure ML

This document describes the PyTorch Lightning implementation for rock mass segmentation training in Azure ML, providing a more stable and user-friendly alternative to the traditional training approach.

## Overview

The Lightning implementation (`azure_train_eval_lightning.py`) addresses common issues with CUDA memory management and provides enhanced stability through:

- **Automatic mixed precision training** - Better memory efficiency and performance
- **Built-in distributed training support** - Seamless scaling across multiple GPUs
- **Enhanced memory management** - Reduces SIGSEGV and out-of-memory errors
- **Better error handling and debugging** - More informative error messages and stack traces
- **Automatic gradient clipping** - Prevents gradient explosion and training instability
- **Simplified configuration** - Consistent interface with existing config system

## Files Overview

### Core Lightning Modules

- `src/ml_segmentation/lightning_module.py` - Main Lightning module wrapping segmentation models
- `src/ml_segmentation/lightning_datamodule.py` - Data module for Lightning-compatible data loading
- `src/ml_segmentation/lightning_callbacks.py` - Custom callbacks for MLflow, checkpointing, and predictions

### Training Scripts

- `scripts/azure_train_eval_lightning.py` - Main Lightning training script for Azure ML
- `scripts/azure_train_eval.py` - Original manual training script (still available)

## Configuration

### Lightning-Specific Settings

Add the following to your `main.yaml` configuration:

```yaml
lightning:
  use_lightning: true  # Enable Lightning training
  precision: "16-mixed"  # Mixed precision: "16-mixed", "32", "bf16-mixed"
  gradient_clip_val: 1.0  # Gradient clipping value
  accumulate_grad_batches: 1  # Gradient accumulation
  save_predictions_every_n_epochs: 10  # Save predictions every N epochs
  max_prediction_images: 5  # Max prediction images to save
  deterministic: true  # Deterministic training for reproducibility
```

### Azure ML Environment

The Lightning implementation is designed to work with Azure ML curated environments:

```yaml
azure_ml:
  use_curated_env: true
  curated_env_name: "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"
```

This curated environment includes:
- PyTorch 2.2 with CUDA 12.1 support
- Compatible with PyTorch Lightning
- Pre-installed common ML packages
- Optimized for Azure ML GPU instances

## Usage

### Option 1: Direct Script Execution

```bash
# Run Lightning training with default config
python scripts/azure_train_eval_lightning.py

# Run with specific config overrides
python scripts/azure_train_eval_lightning.py \
    lightning.precision=32 \
    model.batch_size=16 \
    model.num_epochs=50
```

### Option 2: Azure ML Job Submission

Update `azure_submit_job.py` to use the Lightning script:

```python
# In azure_submit_job.py, change the command to:
command = "python azure_train_eval_lightning.py"
```

### Option 3: Hybrid Approach

You can switch between implementations by modifying the command in job submission:

```python
# Use Lightning for stable training
command = "python azure_train_eval_lightning.py"

# Or use traditional approach
command = "python azure_train_eval.py"
```

## Key Benefits

### Memory Management
- Automatic memory optimization through Lightning's built-in features
- Mixed precision training reduces memory usage by ~50%
- Better handling of large batch sizes and high-resolution images

### Training Stability
- Automatic gradient clipping prevents training explosions
- Better error recovery and logging
- Deterministic training for reproducible results

### Monitoring and Logging
- Seamless MLflow integration for experiment tracking
- TensorBoard logging with Lightning's enhanced metrics
- Automatic model checkpointing with best model selection

### Scalability
- Built-in support for multi-GPU training
- Easy transition to distributed training across multiple nodes
- Automatic device selection and optimization

## Troubleshooting

### CUDA Memory Issues
If you still encounter CUDA memory issues:

1. Reduce batch size in configuration
2. Use gradient accumulation: `lightning.accumulate_grad_batches: 4`
3. Switch to 32-bit precision: `lightning.precision: "32"`

### Performance Optimization
For better performance:

1. Use mixed precision: `lightning.precision: "16-mixed"`
2. Increase batch size if memory allows
3. Use persistent workers: `experiment.num_workers: 4`

### Debugging
Enable verbose logging:

```yaml
lightning:
  deterministic: true  # For reproducible debugging
```

## Comparison with Traditional Training

| Feature | Traditional (`azure_train_eval.py`) | Lightning (`azure_train_eval_lightning.py`) |
|---------|-------------------------------------|---------------------------------------------|
| Memory Management | Manual GradScaler | Automatic mixed precision |
| Multi-GPU Support | Manual implementation | Built-in distributed training |
| Checkpointing | Custom logic | Automatic with best model selection |
| Early Stopping | Custom callback | Built-in with monitoring |
| Logging | Manual MLflow calls | Integrated logging system |
| Error Handling | Basic try/catch | Enhanced error recovery |
| Configuration | Static settings | Dynamic trainer configuration |
| Debugging | Limited tools | Rich debugging features |

## Migration Guide

To migrate from traditional to Lightning training:

1. **Update configuration** - Add Lightning section to `main.yaml`
2. **Test locally** - Run with small dataset first
3. **Update job submission** - Change script name in Azure ML submission
4. **Monitor first runs** - Check logs and metrics for consistency
5. **Optimize settings** - Adjust Lightning-specific parameters based on performance

## Advanced Features

### Custom Callbacks
The Lightning implementation includes custom callbacks for:
- MLflow experiment tracking
- Image prediction saving
- Model checkpointing with Azure ML integration

### Multi-GPU Training
Lightning automatically handles multi-GPU training:

```yaml
azure_ml:
  compute_name: "Standard_NC24s_v3"  # Multi-GPU instance
```

Lightning will automatically distribute training across available GPUs.

### Hyperparameter Optimization
Lightning integrates well with Optuna for hyperparameter tuning:

```python
# In your optimization script
def objective(trial):
    lightning_config = {
        "precision": trial.suggest_categorical("precision", ["16-mixed", "32"]),
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.5, 2.0),
    }
    # Train with Lightning using these parameters
```

## Best Practices

1. **Start with curated environments** - Use Azure ML curated environments for stability
2. **Use mixed precision** - Enable `16-mixed` precision for better performance
3. **Monitor memory usage** - Start with smaller batch sizes and increase gradually
4. **Enable checkpointing** - Use automatic model checkpointing for long training runs
5. **Test configurations** - Validate new configurations with short test runs first

## Support and Issues

For issues specific to the Lightning implementation:

1. Check Lightning documentation: https://lightning.ai/docs/pytorch/stable/
2. Verify Azure ML environment compatibility
3. Review Lightning-specific logs in Azure ML outputs
4. Compare with traditional training results for validation

The Lightning implementation is designed to be a drop-in replacement for the traditional training approach while providing enhanced stability and features for Azure ML deployment.