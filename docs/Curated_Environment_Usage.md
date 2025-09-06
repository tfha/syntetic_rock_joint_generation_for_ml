# Using Azure ML Curated Environment with Additional Packages

This document explains how to use Azure ML curated environments while adding the additional packages required by this project.

## Overview

Azure ML curated environments provide pre-built, optimized environments with common packages already installed. The `acpt-pytorch-2.2-cuda12.1` curated environment includes:

- PyTorch 2.2 with CUDA 12.1 support
- Common scientific computing packages: numpy, scipy, pandas, matplotlib
- Standard machine learning libraries: scikit-learn
- Jupyter/IPython for interactive development

However, this project requires additional specialized packages that are not included in the curated environment.

## Solution: Runtime Package Installation

We've implemented a solution that:

1. **Uses the curated environment** as the base for better stability and compatibility
2. **Installs missing packages at runtime** before running the actual training

This approach provides the benefits of curated environments while ensuring all required dependencies are available.

## How It Works

### 1. Curated Environment Configuration

In `scripts/config/main.yaml`:

```yaml
azure_ml:
  use_curated_env: true  # Enable curated environment usage
  curated_env_name: "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"
```

### 2. Additional Package Requirements

The file `scripts/curated_env_requirements.txt` contains packages not included in the curated environment:

```
# Configuration management
hydra-core>=1.3.2
omegaconf>=2.3.0
pydantic>=2.10.2

# PyTorch Lightning and related
pytorch-lightning>=2.4.0
torchmetrics>=1.6.0
torchinfo>=1.8.0

# Segmentation models
segmentation-models-pytorch>=0.3.4

# Experiment tracking and optimization
mlflow>=2.18.0
optuna>=4.1.0

# Azure ML SDK
azure-ai-ml>=1.26.0
azure-identity>=1.21.0

# Utilities
rich>=13.9.4
python-dotenv>=1.1.0
toml>=0.10.2
absl-py==1.4.0
```

### 3. Installation Script

The `scripts/install_missing_packages.py` script:

- Installs packages from `curated_env_requirements.txt`
- Handles installation errors gracefully
- Provides clear feedback on installation progress
- Falls back to individual package installation if batch installation fails

### 4. Automatic Integration

When `use_curated_env: true` is set, the Azure ML job submission automatically:

1. Uses the specified curated environment
2. Prepends the package installation command to the training command
3. Runs package installation before executing the actual training script

## Usage Examples

### Standard Training with Curated Environment

```bash
# The job submission will automatically run:
# python scripts/install_missing_packages.py && python scripts/azure_train_eval.py
python scripts/azure_submit_job.py
```

### Lightning Training with Curated Environment

Uncomment the Lightning command in `azure_submit_job.py`:

```python
train_command = (
    f"{base_command}python scripts/azure_train_eval_lightning.py "
    f"model={pcfg.model.name} "
    f"experiment.experiment_strategy={pcfg.experiment.experiment_strategy} "
    f"lightning.use_lightning=true "
)
```

### Smoke Testing with Curated Environment

```bash
# Set in config or override:
python scripts/azure_submit_job.py experiment.smoke_test=true
```

## Benefits

### Stability
- Uses Microsoft-maintained curated environment as base
- Reduces environment-related issues and CUDA problems
- Provides consistent PyTorch and CUDA versions

### Flexibility
- Allows adding project-specific packages as needed
- Easy to maintain and update package requirements
- Supports both traditional and Lightning training approaches

### Maintenance
- Clear separation between base environment and project requirements
- Easy to see which packages are project-specific
- Simple to update package versions

## Troubleshooting

### Package Installation Failures

If package installation fails:

1. Check the Azure ML job logs for specific error messages
2. Verify package compatibility with PyTorch 2.2 and CUDA 12.1
3. Update version constraints in `curated_env_requirements.txt` if needed

### Environment Issues

If the curated environment doesn't work:

1. Set `use_curated_env: false` in the configuration
2. Fall back to custom environment creation
3. Check Azure ML for available curated environment versions

### Package Conflicts

If packages conflict with curated environment:

1. Pin specific versions in `curated_env_requirements.txt`
2. Remove conflicting packages if they're already in the curated environment
3. Test locally with the same Python/PyTorch versions

## Best Practices

1. **Keep requirements minimal** - Only include packages not in the curated environment
2. **Use version constraints** - Specify minimum versions to ensure compatibility
3. **Test locally first** - Verify package compatibility before submitting to Azure ML
4. **Monitor installation time** - Large packages may increase job startup time
5. **Document changes** - Update this file when adding new requirements

## Migration from Custom Environments

To migrate from custom environments to curated environments:

1. Set `use_curated_env: true` in configuration
2. Move project-specific packages to `curated_env_requirements.txt`
3. Remove packages already included in curated environment
4. Test with a smoke test first
5. Monitor for any compatibility issues

This approach provides the best of both worlds: the stability of curated environments with the flexibility to add project-specific requirements.