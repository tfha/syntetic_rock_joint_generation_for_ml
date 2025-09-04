# Azure ML GPU Testing with MNIST Tutorial

This document provides a comprehensive guide for testing Azure ML GPU compute nodes using a simple MNIST classification tutorial. This script serves as both a validation tool for avoiding SIGSEGV errors and a learning resource for computer vision training on Azure ML.

## Overview

The MNIST tutorial script (`azure_mnist_tutorial.py`) is designed to:

- **Validate GPU environment** compatibility with Azure ML compute nodes
- **Test CUDA and PyTorch** functionality on Standard_NC64as_T4_v3 (Tesla T4)
- **Demonstrate best practices** for avoiding memory-related crashes (SIGSEGV)
- **Provide a working example** of computer vision training on Azure ML
- **Use minimal resources** to ensure stability on GPU compute

## Key Features

### 🎯 **Targeted Environment**
- **Compute Node**: Standard_NC64as_T4_v3 (64 cores, 440 GB RAM, Tesla T4 GPUs)
- **Azure ML Environment**: `acpt-pytorch-2.2-cuda12.1:40`
- **PyTorch Version**: Compatible with 2.2.x and CUDA 12.1
- **Python Version**: 3.11

### 🛡️ **Stability Features**
- Conservative memory usage (32 batch size, 0 workers)
- Proper CUDA memory management with periodic cleanup
- Error handling for common GPU issues
- Gradual resource scaling to prevent SIGSEGV
- Fallback to CPU if GPU issues are detected

### 📊 **Model Architecture**
- Simple CNN with ~421K parameters
- 2 Convolutional layers + 2 Fully connected layers
- Dropout regularization for stability
- He weight initialization for better convergence

## Usage

### 🚀 **Quick Start**

#### 1. Local Testing (for development)
```bash
# Test the script locally (will use synthetic data if MNIST download fails)
python scripts/azure_mnist_tutorial.py experiment.mnist_tutorial=true
```

#### 2. Azure ML Submission (recommended)
```bash
# Submit to Azure ML GPU compute using the curated environment
python scripts/azure_submit_job.py experiment.mnist_tutorial=true
```

### ⚙️ **Configuration**

The tutorial uses these conservative settings for maximum stability:

```yaml
# In scripts/config/main.yaml
experiment:
  mnist_tutorial: true          # Enable MNIST tutorial mode
  
azure_ml:
  compute_name: "NC64as-T4-v3" # Tesla T4 compute node
  curated_env_name: "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/version/40"
```

## Architecture Details

### 🏗️ **CNN Model Structure**

```python
SimpleMNISTCNN(
    conv1: Conv2d(1, 32, kernel_size=3)      # 32 filters, 3x3 kernel
    conv2: Conv2d(32, 64, kernel_size=3)     # 64 filters, 3x3 kernel  
    pool: MaxPool2d(kernel_size=2)           # 2x2 max pooling
    fc1: Linear(3136, 128)                   # Fully connected layer
    fc2: Linear(128, 10)                     # Output layer (10 classes)
    dropout: Dropout(0.25)                   # Regularization
)
```

**Total Parameters**: ~421,642 (lightweight for GPU testing)

### 🔄 **Training Pipeline**

1. **Environment Validation**
   - GPU availability check
   - CUDA driver compatibility
   - Memory allocation test
   - Fallback to CPU if needed

2. **Data Loading**
   - MNIST dataset (28x28 grayscale images)
   - Conservative DataLoader settings
   - Synthetic fallback for testing

3. **Training Loop**
   - 3 epochs (short for testing)
   - Adam optimizer (lr=0.001)
   - Cross-entropy loss
   - Progress logging every 100 batches

4. **Evaluation & Artifacts**
   - Test set evaluation after each epoch
   - Model checkpoints saved to `./outputs/`
   - MLflow experiment tracking
   - Comprehensive metrics logging

## Best Practices for Avoiding SIGSEGV

### 🛡️ **Memory Management**

```python
# Conservative DataLoader settings
DataLoader(
    dataset,
    batch_size=32,           # Small batch size
    num_workers=0,           # No multiprocessing
    pin_memory=False,        # Disable pinned memory
    persistent_workers=False # Disable worker persistence
)
```

### 🧹 **GPU Memory Cleanup**

```python
# Periodic cleanup during training
if device.type == "cuda":
    torch.cuda.empty_cache()

# Garbage collection after epochs
gc.collect()
```

### ⚠️ **Error Handling**

```python
try:
    # GPU operations
    output = model(data)
except Exception as e:
    console.print(f"GPU error: {e}", style="error")
    # Continue with next batch
    torch.cuda.empty_cache()
```

## Expected Results

### ✅ **Successful Run Indicators**

- **Environment validation passes** with GPU detected
- **MNIST dataset loads** successfully (or fallback works)
- **Training completes** all 3 epochs without crashes
- **Test accuracy** reaches 85-95% (with real MNIST data)
- **Artifacts saved** to `./outputs/` directory
- **MLflow tracking** records metrics correctly

### 📈 **Typical Performance**

With real MNIST data on GPU:
- **Training Time**: ~2-3 minutes for 3 epochs
- **Final Accuracy**: 85-95% (baseline CNN performance)
- **Memory Usage**: <2GB GPU memory
- **Stability**: Should complete without SIGSEGV errors

## Troubleshooting

### 🔧 **Common Issues**

#### 1. SIGSEGV Errors
```bash
# Symptoms: Segmentation fault, sudden job termination
# Solutions:
- Reduce batch_size from 32 to 16 or 8
- Set num_workers=0 (disable multiprocessing)
- Add more frequent torch.cuda.empty_cache() calls
```

#### 2. CUDA Out of Memory
```bash
# Symptoms: "CUDA out of memory" errors
# Solutions:
- Reduce batch_size
- Add more aggressive memory cleanup
- Check for memory leaks in model forward pass
```

#### 3. Environment Issues
```bash
# Symptoms: Import errors, CUDA not available
# Solutions:
- Verify curated environment version
- Check compute node has GPU capability
- Ensure environment includes required packages
```

#### 4. Data Loading Issues
```bash
# Symptoms: MNIST download fails, data corruption
# Solutions:
- Check network connectivity in Azure ML
- Use synthetic data fallback (automatically handled)
- Pre-download data to Azure storage if needed
```

### 🔍 **Debugging Steps**

1. **Check Logs**: Look for GPU validation output
2. **Monitor Memory**: Watch GPU memory usage patterns
3. **Test Locally**: Run with synthetic data first
4. **Gradual Scaling**: Start with smaller batch sizes
5. **Environment Test**: Try minimal smoke test first

## Extending the Tutorial

### 🔧 **Customization Options**

```python
# Modify training parameters
tutorial_config = {
    "batch_size": 16,        # Reduce for more stability
    "num_epochs": 5,         # Increase for better accuracy
    "learning_rate": 0.0005, # Fine-tune learning rate
    "dropout_rate": 0.3      # Adjust regularization
}
```

### 📊 **Adding More Metrics**

```python
# Add additional tracking
mlflow.log_metric("gpu_memory_used", torch.cuda.memory_allocated())
mlflow.log_metric("training_time", epoch_duration)
```

### 🔄 **Model Variations**

```python
# Try different architectures
class DeepMNISTCNN(nn.Module):
    # Add more layers, batch normalization, etc.
```

## Integration with Main Project

### 🔗 **Using as Reference**

This tutorial demonstrates patterns used throughout the main project:

- **Hydra configuration management**
- **Rich console output and progress tracking**
- **MLflow experiment tracking**
- **Azure ML job submission**
- **Proper error handling and logging**

### 📋 **Checklist for New Models**

Before implementing complex models, verify:

- [ ] GPU environment validates correctly
- [ ] Basic training loop completes without errors
- [ ] Memory usage stays within bounds
- [ ] MLflow logging works as expected
- [ ] Azure ML job submission succeeds

## Conclusion

This MNIST tutorial provides a solid foundation for validating Azure ML GPU environments and serves as a reference implementation for computer vision training. It demonstrates best practices for stability and provides a working example that can be extended for more complex use cases.

**Key Takeaways:**
- Start simple and scale gradually
- Use conservative memory settings
- Implement proper error handling
- Monitor resource usage
- Test thoroughly before complex deployments

For questions or issues, refer to the main project documentation or create an issue in the repository.