# Results Analysis Scripts

This directory contains scripts for analyzing experimental results from Azure ML jobs.

## Scripts

- `download_metrics.py` - Download metrics CSV files from completed jobs
- `aggregate_results.py` - Aggregate metrics from multiple experiments
- `compare_models.py` - Compare performance across models and strategies
- `visualize_results.py` - Create plots and visualizations from results

## Usage

See individual script files for detailed usage instructions.

## Workflow

1. Download metrics from Azure ML jobs using `download_metrics.py`
2. Aggregate results using `aggregate_results.py`
3. Generate comparisons and visualizations as needed

## Data Format

Expected metrics CSV format:
- Columns: metric_name, value, epoch/step (optional)
- Rows: individual metric values

Output aggregated data will be saved to `experiments/results/`
