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

- **`plot_metrics_vs_proportion_real.py`** - Create plots showing metric vs synthetic/real data ratio
- **`plot_epoch_progression.py`** - Create 4-page plots showing training curves over epochs
- **`plot_progression.py`** - Generate training progression visualizations showing prediction improvement
- **`plot_progression_batch.py`** - Batch generate progression plots for all downloaded jobs automatically

### Qualitative Evaluation

- **`qualitative_evaluation_app.py`** - Streamlit app for rating epoch progression images interactively

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
helpful_rice_czsjvjzqz8
dreamy_needle_qr9t9jd920
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

#### Metrics vs Proportion of Real Data

Generate publication-quality plots showing how validation Dice score varies with synthetic/real data ratio:

```bash
poetry run python scripts/results_analysis/plot_metrics_vs_proportion_real.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric dice_joints
```

**Plot features:**
- Compares training strategies (simplemixed vs finetune)
- Shows effect of synthetic/real data ratios (0%, 10%, 30%, 50%, 70%, 90%, 100%)
- Compares model architectures (UNet vs DeepLabV3+)
- Separate plots for different test dataset groups (box-based, slope/rock)
- Colorblind-friendly with hollow/filled markers for strategies
- X-axis reversed to show increasing synthetic data left-to-right

**Output:** `experiments/results/plots/{metric}_vs_proportion_*.png`

#### Epoch Progression Plots

Generate training progression plots over epochs for Box and Slope experiments.

**Option 1: Single grid plot (5×8 layout, all experiments on one figure):**

```bash
poetry run python scripts/results_analysis/plot_epoch_progression.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric dice_joints
```

**Output:** `experiments/results/plots/figure_epoch_progression_{metric}_grid.png`

**Option 2: Separate pages (4-page format for publication):**

```bash
poetry run python scripts/results_analysis/plot_epoch_progression.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric dice_joints --separate-pages
```

**Output:** 4 separate files:
- `epoch_progression_{metric}_box_unet_page*.png`
- `epoch_progression_{metric}_box_deeplabv3plus_page*.png`
- `epoch_progression_{metric}_slope_unet_page*.png`
- `epoch_progression_{metric}_slope_deeplabv3plus_page*.png`

**Note:** Use metric names without `val_` or `train_` prefix (e.g., `dice_joints` not `val_dice_joints`). The script automatically plots both training (dashed) and validation (solid) curves.

**Plot organization (4 separate pages when using --separate-pages):**
- Page 1: Box experiments - UNet (finetune left, simplemixed right)
- Page 2: Box experiments - DeepLabV3+ (finetune left, simplemixed right)
- Page 3: Slope experiments - UNet (finetune left, simplemixed right)
- Page 4: Slope experiments - DeepLabV3+ (finetune left, simplemixed right)

**Plot features:**
- 5 rows of subplots per page (different data proportions: 10%, 30%, 50%, 70%, 90%)
- Training curves show progression from epoch 0 to 100
- Compares train_dice_joints (dashed) and val_dice_joints (solid)
- Colorblind-friendly colors
- Optimized layout for publication (14×15 inch pages)

**Batch generation for multiple metrics:**

Generate plots for all key metrics in one go:

```bash
# Generate 4-page format for dice_joints, dice, and loss
poetry run python scripts/results_analysis/plot_epoch_progression.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric dice_joints --separate-pages
poetry run python scripts/results_analysis/plot_epoch_progression.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric dice --separate-pages
poetry run python scripts/results_analysis/plot_epoch_progression.py --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots --metric loss --separate-pages
```

Available metrics: `dice_joints`, `dice`, `iou_joints`, `iou`, `loss`, `precision_joints`, `recall_joints`

### 4. Create Training Progression Visualizations

Generate visualizations showing how predictions improve during training:

```bash
# SimpleMixed experiment (5 epochs: 5, 10, 15, 20, final)
poetry run python scripts/results_analysis/plot_progression.py --image-dir experiments/results/images --job-name "deeplabv3plus-simplemixed_generalisation_cardboard_box_30-20251211-1639" --strategy simplemixed --num-samples 10

# Finetune experiment (5 epochs: 5, 10, stage2_start, stage2_mid, final)
poetry run python scripts/results_analysis/plot_progression.py --image-dir experiments/results/images --job-name "deeplabv3plus-finetune_generalisation_pattern_box_10-20251210-0956" --strategy finetune --num-samples 7
```

**Progression plot layout:**
- **Rows**: Test samples (configurable with `--num-samples`, default 10)
- **Columns** (7 total):
  1. Original image
  2. Ground truth mask
  3-7. Predictions at 5 key training epochs
- **Headers**: Epoch labels with validation Dice scores (`val_dice_joints`)
- **Left margin**: Sample numbers (aligned with rows)

**Features:**
- Exact crop coordinates to extract clean images (no white padding)
- Dice scores shown for all epochs including final
- Automatic metrics file matching (handles different timestamps)
- Consistent 7-column layout for both strategies
- Clean titles without date/time stamps
- Strategy-specific epoch selection (early, mid, late training)

**Output:** `experiments/results/plots/progression/{job_name}_progression.png`

**Batch generation for all jobs:**

Automatically generate progression plots for all downloaded job folders:

```bash
poetry run python scripts/results_analysis/plot_progression_batch.py --image-dir experiments/results/images --metrics-dir experiments/results/metrics/mode=max --output-dir experiments/results/plots/progression --num-samples 10
```

**How it works:**
- Automatically detects all job folders in the image directory
- Identifies training strategy (finetune/simplemixed) from folder names
- Matches each job with its corresponding metrics CSV file
- Handles timestamp differences between image folders and metrics files
- Generates progression plots for all jobs with available images and metrics

**Features:**
- **Auto-detection**: Determines finetune vs simplemixed from folder names
- **Smart matching**: Finds metrics files despite different naming conventions
- **Error handling**: Skips jobs with missing data and reports summary
- **Consistent output**: All plots use the same number of samples for fair comparison

**Summary output example:**
```
Found 50 job folders
Processing: unet-simplemixed_box_10-20251211-1603
  Detected strategy: simplemixed
  Using metrics: unet-simplemixed_box_10-20251211-1603_metrics.csv
  ✓ Saved: unet-simplemixed_box_10-20251211-1603_progression.png
...
Summary:
  Processed: 45 plots
  Skipped: 5 jobs
```

## Metrics Explained

**Dice Score (`val_dice_joints`)**: The primary metric shown in plots.

- **Formula**: Dice = 2 × |Predicted ∩ Ground Truth| / (|Predicted| + |Ground Truth|)
- **Range**: 0.0 (no overlap) to 1.0 (perfect prediction)
- **Meaning**: Measures overlap between predicted and ground truth joint masks
- **Context**: Validation Dice coefficient specifically for the joints class

This metric evaluates how accurately the model segments rock joints (fractures) at each training epoch.

### 5. Qualitative Evaluation (Interactive)

Launch the Streamlit app to rate epoch progression images interactively:

```bash
streamlit run scripts/results_analysis/qualitative_evaluation_app.py

# If port 8501 is blocked or in use, specify an alternative port:
streamlit run scripts/results_analysis/qualitative_evaluation_app.py --server.port 8502
```

**App features:**
- Navigate through images one-by-one with Prev/Next buttons
- Progress tracker showing which images have been rated
- Rate final epoch on 4 criteria using 1-5 scale:
  - **Geological recognisability**: How realistic do the predicted joints look?
  - **Joint persistence**: Are continuous joints properly connected?
  - **Boundary localisation & thickness**: Are joint boundaries precise?
  - **False positives / noise**: How much spurious segmentation exists?
- Optional field to note best epoch(s) that outperform final in general
- Add optional text notes
- Ratings saved automatically to CSV file
- Supports multiple raters with unique IDs
- Clickable sidebar list showing rated (✓) vs unrated (○) images

**Configuration (sidebar):**
- **Image folder**: Path to folder containing images (e.g., `experiments/results/images/job_name/`)
- **Rater ID**: Identifier for the person rating (e.g., `rater_1`, `geologist_A`)
- **Output CSV**: Path to save ratings (default: `qualitative_ratings.csv`)

**Output format:** CSV with columns:
- `timestamp`: When rating was saved
- `rater`: Who provided the rating
- `image`: Image filename
- `stage`: Training stage (always "final")
- `geological_recognisability`: Score 1-5
- `joint_persistence`: Score 1-5
- `boundary_localisation`: Score 1-5
- `false_positives`: Score 1-5
- `better_epoch`: Optional - best epoch(s) that outperform final in general
- `notes`: Optional text comments

**Workflow:**
1. Download images using `download_images.py`
2. Select a job folder in the app (contains epoch subfolders)
3. Navigate through sample images with Prev/Next or click from sidebar list
4. Rate final epoch on 4 criteria (1-5 scale)
5. Optionally note best epoch(s) that outperform final in general
6. Add any notes
7. Click Save to record ratings
8. Progress automatically tracked in sidebar

**Use case:** Complement quantitative metrics (Dice, IoU) with expert assessment of prediction quality for publication and model selection.

## Workflow Summary

1. **Train models** in Azure ML (simplemixed and finetune strategies)
2. **Download metrics** using `download_metrics.py` to get epoch-wise performance data
3. **Download images** using `download_images.py` for visual inspection
4. **Create comparison plots** using `plot_metrics_vs_proportion_real.py` and `plot_epoch_progression.py`
5. **Visualize progression** using `plot_progression.py` to show training improvement
6. **Qualitative evaluation** using `qualitative_evaluation_app.py` for expert ratings
7. **Analyze results** to determine best models and strategies

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
