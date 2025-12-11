# Results Analysis Scripts

This directory contains scripts for downloading, analyzing, and visualizing experimental results from Azure ML training jobs.

## Overview

After training models in Azure ML, use these tools to:
1. Download training metrics (CSV files with epoch-by-epoch scores)
2. Download example prediction images
3. Generate publication-quality comparison plots
4. Create training progression visualizations

## Scripts

### Data Download

- **`download_metrics.py`** - Download metrics CSV files from completed Azure ML jobs
- **`download_images.py`** - Download example prediction images from Azure blob storage

### Visualization

- **`plot_journal_figures.py`** - Create publication plots comparing experiments (5×2 grids)
- **`plot_progression.py`** - Generate training progression visualizations showing prediction improvement

### Legacy/Experimental

- `aggregate_results.py` - Aggregate metrics from multiple experiments
- `compare_models.py` - Compare performance across models and strategies
- `visualize_results.py` - Create plots and visualizations from results

## Usage

### 1. Download Training Metrics

Download metrics (training/validation scores per epoch) from completed jobs:

```bash
# Download metrics for all jobs in a text file
poetry run python scripts/results_analysis/download_metrics.py --jobs-file experiments/results/job_names_mode=max.txt --output-folder-name "mode=max"
```

**What gets downloaded:**
- Epoch-by-epoch training and validation metrics
- Dice scores, IoU, precision, recall (overall and joints-specific)
- Learning rates and loss values
- Saved as CSV files in `experiments/results/metrics/{output-folder-name}/`

**Input format:** Text file with one job name per line:
```
deeplabv3plus-simplemixed_box_10-20251209-1543
deeplabv3plus-finetune_box_10-20251210-0956
unet-simplemixed_slope_30-20251209-1523
```

### 2. Download Example Images

Download prediction images (composites of original, ground truth, prediction):

```bash
# Download images for jobs in a text file (default: 4 parallel workers)
poetry run python scripts/results_analysis/download_images.py --jobs-file scripts/results_analysis/test_image_jobs.txt --output-dir experiments/results/images --use-display-names

# Faster download with more parallel workers (recommended for good internet)
poetry run python scripts/results_analysis/download_images.py --jobs-file scripts/results_analysis/test_image_jobs.txt --output-dir experiments/results/images --use-display-names --max-workers 8
```

**Parallel Downloads:**
- `--max-workers` controls concurrent downloads (default: 4)
- Recommended values:
  - **4 workers** (default): Safe for all connections
  - **8 workers**: Good balance for fast internet (50+ Mbps)
  - **12 workers**: Aggressive for very fast connections (100+ Mbps)
- Network bandwidth is usually the bottleneck, not CPU

**Strategy-specific epochs:**

- **SimpleMixed**: Downloads epochs 5, 10, 15, 20, and final
- **Finetune**: Downloads epochs 5, 10, stage 2 start, stage 2 middle, and final

**Output structure:**
```
experiments/results/images/
├── deeplabv3plus-simplemixed_generalisation_cardboard_box_30/
│   ├── epoch_5/
│   │   ├── sample_0.png
│   │   ├── sample_1.png
│   │   └── ...
│   ├── epoch_10/
│   ├── epoch_15/
│   ├── epoch_20/
│   └── final/
└── deeplabv3plus-finetune_generalisation_pattern_box_10/
    ├── epoch_5/
    ├── epoch_10/
    ├── epoch_13_stage2_first_epoch/
    ├── epoch_17_stage2_fifth_epoch/
    └── final/
```

**Image format:** Each image is a 1500×500 pixel composite:
- **Left (188-530px)**: Original 3D rock box image
- **Middle (598-940px)**: Ground truth mask (line drawing)
- **Right (1008-1350px)**: Model prediction mask

### 3. Create Publication Plots

Generate publication-quality comparison plots (5×2 grids):

```bash
poetry run python scripts/results_analysis/plot_journal_figures.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots
```

**Generated plots compare:**
- Training strategies (simplemixed vs finetune)
- Synthetic/real data ratios (0%, 10%, 30%, 50%, 70%, 90%, 100%)
- Model architectures (UNet vs DeepLabV3+)
- Test datasets (box, pattern_box, cardboard_box, slope, larvik, rv4)

**Output:** `experiments/results/plots/validation_dice_joints_{comparison_type}.png`

### 4. Create Progression Plots

Generate training progression visualizations showing prediction improvement:

```bash
# SimpleMixed experiment
poetry run python scripts/results_analysis/plot_progression.py --image-dir experiments/results/images --job-name "deeplabv3plus-simplemixed_generalisation_cardboard_box_30-20251209-2337" --strategy simplemixed --num-samples 10

# Finetune experiment
poetry run python scripts/results_analysis/plot_progression.py --image-dir experiments/results/images --job-name "deeplabv3plus-finetune_generalisation_pattern_box_10-20251210-0957" --strategy finetune --num-samples 7
```

**Progression plot layout:**
- **Rows**: Test samples (10 by default)
- **Columns**:
  1. Original image
  2. Ground truth mask
  3-7. Predictions at different epochs
- **Headers**: Epoch labels with validation Dice scores (`val_dice_joints`)
- **Left margin**: Sample numbers (aligned with rows)

**Features:**
- Exact crop coordinates to extract clean images (no white padding)
- Dice scores shown for all epochs including final
- Automatic metrics file matching (handles different timestamps)
- Consistent 7-column layout for both strategies
- Clean titles without date/time stamps

**Output:** `experiments/results/plots/progression/{job_name}_progression.png`

## Metrics Explained

**Dice Score (`val_dice_joints`)**: The primary metric shown in plots.

- **Formula**: Dice = 2 × |Predicted ∩ Ground Truth| / (|Predicted| + |Ground Truth|)
- **Range**: 0.0 (no overlap) to 1.0 (perfect prediction)
- **Meaning**: Measures overlap between predicted and ground truth joint masks
- **Context**: Validation Dice coefficient specifically for the joints class

This metric evaluates how accurately the model segments rock joints (fractures) at each training epoch.

## Workflow Summary

1. **Train models** in Azure ML (simplemixed and finetune strategies)
2. **Download metrics** using `download_metrics.py` to get epoch-wise performance data
3. **Download images** using `download_images.py` for visual inspection
4. **Create comparisons** using `plot_journal_figures.py` for publication plots
5. **Visualize progression** using `plot_progression.py` to show training improvement
6. **Analyze results** to determine best models and strategies

## Configuration Files

- `test_image_jobs.txt` - List of representative jobs for image download
- `job_names_mode=max.txt` - Complete list of jobs for metrics download

## Output Organization

```
experiments/results/
├── metrics/
│   └── mode=max/
│       ├── deeplabv3plus_finetune_box_10_metrics.csv
│       ├── deeplabv3plus-simplemixed_box_10-20251209-1543_metrics.csv
│       └── ...
├── images/
│   ├── {job-display-name}/
│   │   ├── epoch_5/
│   │   ├── epoch_10/
│   │   └── ...
│   └── ...
└── plots/
    ├── progression/
    │   └── {job-name}_progression.png
    └── validation_dice_joints_{comparison}.png
```
