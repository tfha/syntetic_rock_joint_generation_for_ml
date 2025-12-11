"""Plot training and test Dice score progression over epochs.

Creates a 5×2 grid showing train_dice_joints and test_dice_joints over epochs
for different experiments, comparing models and training strategies.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Publication-quality plot settings
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }
)

# Colorblind-friendly colors
COLORS = {
    "unet": "#0173B2",  # Blue
    "deeplabv3plus": "#DE8F05",  # Orange
}

# Line styles for strategies
STRATEGY_LINESTYLE = {
    "finetune": "-",  # Solid
    "simplemixed": "--",  # Dashed
}


def parse_filename(filename: str) -> dict[str, str] | dict[str, str | float] | None:
    """Parse metrics filename to extract metadata.

    Examples:
        unet_finetune_box_10_metrics.csv
        deeplabv3plus-simplemixed_box_0-20251209-1543_metrics.csv
        unet_simplemixed_generalisation_larvik_100_metrics.csv
    """
    # Remove _metrics.csv suffix
    name = filename.replace("_metrics.csv", "")

    # Pattern 1: simplemixed with hyphens and timestamp
    # {model}-simplemixed_{experiment}_{proportion}-{timestamp}
    match = re.match(r"(unet|deeplabv3plus)-simplemixed_(.+)_(\d+)-\d{8}-\d{4}$", name)
    if match:
        model, experiment, proportion = match.groups()
        return {
            "model": model,
            "strategy": "simplemixed",
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
            "filename": filename,
        }

    # Pattern 2: finetune with underscores
    # {model}_{strategy}_{experiment}_{proportion}
    match = re.match(r"(unet|deeplabv3plus)_(finetune|simplemixed)_(.+)_(\d+)$", name)
    if match:
        model, strategy, experiment, proportion = match.groups()
        return {
            "model": model,
            "strategy": strategy,
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
            "filename": filename,
        }

    return None


def normalize_experiment_name(exp: str) -> str:
    """Normalize experiment names for grouping."""
    # Map variations to canonical names
    mappings = {
        "box": "Box",
        "pattern_box": "Pattern Box",
        "cardboard_box": "Cardboard Box",
        "generalisation_pattern_box": "Generalisation Pattern Box",
        "generalization_pattern_box": "Generalisation Pattern Box",
        "generalisation_cardboard_box": "Generalisation Cardboard Box",
        "generalization_cardboard_box": "Generalisation Cardboard Box",
        "slope": "Slope",
        "larvik": "Larvik",
        "rv4": "Rv 4",
        "generalisation_larvik": "Generalisation Larvik",
        "generalization_larvik": "Generalisation Larvik",
        "generalisation_rv4": "Generalisation Rv 4",
        "generalization_rv4": "Generalisation Rv 4",
    }
    return mappings.get(exp, exp)


def load_epoch_metrics(
    metrics_dir: Path, experiment: str, model: str, strategy: str, proportion: float
) -> pd.DataFrame | None:
    """Load epoch-by-epoch metrics for a specific experiment configuration."""
    # Find matching CSV file
    for csv_file in metrics_dir.glob("*.csv"):
        metadata = parse_filename(csv_file.name)
        if metadata is None:
            continue

        experiment_name = metadata.get("experiment", "")
        if (
            metadata["model"] == model
            and metadata["strategy"] == strategy
            and isinstance(experiment_name, str)
            and normalize_experiment_name(experiment_name) == experiment
            and metadata["proportion"] == proportion
        ):
            try:
                df = pd.read_csv(csv_file)
                # Ensure we have required columns
                if "epoch" in df.columns:
                    return df
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
                return None

    return None


def plot_model_strategy_experiment(
    ax,
    metrics_dir: Path,
    experiment: str,
    model: str,
    strategy: str,
    proportions: list[float],
    metric: str = "dice_joints",
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_title: bool = False,
) -> None:
    """Plot epoch progression for a specific model/strategy/experiment with all proportions.

    Args:
        metric: Name of metric to plot (e.g., 'dice_joints', 'dice', 'loss')
    """
    # Colors: Light to dark gradient - darker colors indicate more real data
    proportion_colors = {
        0.0: "#F5F5F5",  # Very light gray (0% real = 100% synthetic)
        0.1: "#E0E0E0",  # Light gray (10% real)
        0.3: "#9E9E9E",  # Medium gray (30% real)
        0.5: "#90CAF9",  # Light blue (50% real)
        0.7: "#42A5F5",  # Medium blue (70% real)
        0.9: "#1976D2",  # Dark blue (90% real)
        1.0: "#0D47A1",  # Very dark blue (100% real)
    }

    has_data = False
    all_values = []  # Collect all values for y-axis range

    # Plot each proportion
    for proportion in proportions:
        df = load_epoch_metrics(metrics_dir, experiment, model, strategy, proportion)

        if df is None or df.empty:
            continue

        has_data = True

        # Get color for this proportion
        color = proportion_colors.get(proportion, "#000000")

        # Convert to percentage for label (real data proportion)
        real_pct = round(proportion * 100)

        # Plot training metric (dashed line)
        train_col = f"train_{metric}"
        if train_col in df.columns:
            ax.plot(
                df["epoch"],
                df[train_col],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                zorder=3,
            )
            all_values.extend(df[train_col].dropna().tolist())

        # Plot validation metric (solid line) - this is the test/validation set
        val_col = f"val_{metric}"
        if val_col in df.columns:
            ax.plot(
                df["epoch"],
                df[val_col],
                color=color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.9,
                label=f"{real_pct}%",
                zorder=3,
            )
            all_values.extend(df[val_col].dropna().tolist())

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Styling
    if show_title:
        title = f"{model.upper()}\n{strategy.capitalize()}"
        ax.set_title(title, fontsize=9, pad=5)

    ax.set_xlim(0, 100)

    # Set y-axis limits based on data range
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        # Add 10% padding above and below
        padding = y_range * 0.1 if y_range > 0 else 0.1
        ax.set_ylim(max(0, y_min - padding), y_max + padding)
    else:
        ax.set_ylim(0, 1.0)  # Default range if no values
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_zorder(2)
    ax.spines["left"].set_zorder(2)

    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=8)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ylabel = metric.replace("_", " ").title()
        ax.set_ylabel(ylabel, fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=7)

    # Add legend for proportions in all subplots if has data
    if has_data:
        # Use top right for loss metrics to avoid blocking curves
        legend_loc = "upper right" if metric == "loss" else "lower right"
        ax.legend(
            fontsize=6,
            loc=legend_loc,
            framealpha=0.8,
            title="% real data",
            title_fontsize=6,
        )


def plot_experiment_grid(
    metrics_dir: Path,
    box_experiments: list[str],
    slope_experiments: list[str],
    proportions: list[float],
    output_path: Path,
    metric: str = "dice_joints",
) -> None:
    """Create a 5×8 grid of subplots for all model/strategy/experiment combinations.

    Args:
        metric: Name of metric to plot (e.g., 'dice_joints', 'dice', 'loss')
    """
    fig, axes = plt.subplots(5, 8, figsize=(40, 15))

    # Define the combinations for each column
    # Columns: 4 for box experiments, 4 for slope experiments
    # Order from left to right: UNet-Finetune, DeepLabV3+-Finetune, UNet-SimpleMixed, DeepLabV3+-SimpleMixed

    models = ["unet", "deeplabv3plus", "unet", "deeplabv3plus"]
    strategies = ["finetune", "finetune", "simplemixed", "simplemixed"]

    # Plot box experiments (left 4 columns)
    for col_idx in range(4):
        for row_idx, experiment in enumerate(box_experiments):
            show_xlabel = row_idx == 4  # Only bottom row
            show_ylabel = col_idx == 0  # Only leftmost column
            show_title = row_idx == 0  # Only top row

            plot_model_strategy_experiment(
                axes[row_idx, col_idx],
                metrics_dir,
                experiment,
                models[col_idx],
                strategies[col_idx],
                proportions,
                metric,
                show_ylabel,
                show_xlabel,
                show_title,
            )

            # Add experiment name as row label on the leftmost column
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    -0.25,
                    0.5,
                    experiment,
                    transform=axes[row_idx, col_idx].transAxes,
                    fontsize=9,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=0,
                )

    # Plot slope experiments (right 4 columns)
    for col_idx in range(4):
        for row_idx, experiment in enumerate(slope_experiments):
            show_xlabel = row_idx == 4  # Only bottom row
            show_ylabel = False  # No ylabel for right columns
            show_title = row_idx == 0  # Only top row

            plot_model_strategy_experiment(
                axes[row_idx, col_idx + 4],
                metrics_dir,
                experiment,
                models[col_idx],
                strategies[col_idx],
                proportions,
                metric,
                show_ylabel,
                show_xlabel,
                show_title,
            )

            # Add experiment name as row label on the first slope column
            if col_idx == 0:
                axes[row_idx, col_idx + 4].text(
                    -0.10,
                    0.5,
                    experiment,
                    transform=axes[row_idx, col_idx + 4].transAxes,
                    fontsize=9,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=0,
                )

    # Add super titles for box and slope sections
    # Calculate positions based on subplot locations (4 columns each out of 8)
    box_center = 0.02 + (0.48 * 0.5)  # Center of first 4 columns
    slope_center = 0.52 + (0.46 * 0.5)  # Center of last 4 columns

    fig.text(
        box_center, 0.98, "Box Experiments", ha="center", fontsize=12, fontweight="bold"
    )
    fig.text(
        slope_center,
        0.98,
        "Slope Experiments",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Create custom legend at the bottom
    from matplotlib.lines import Line2D

    # Format metric name for legend labels
    metric_label = metric.replace("_", " ").title()
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Train {metric_label}",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            label=f"Val {metric_label}",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_combined_page(
    metrics_dir: Path,
    experiments: list[str],
    experiment_type: str,
    model: str,
    proportions: list[float],
    output_path: Path,
    metric: str = "dice_joints",
) -> None:
    """Create a page with Finetune on left, SimpleMixed on right (same size as grid plot).

    Args:
        experiments: List of experiment names
        experiment_type: "Box" or "Slope"
        model: "unet" or "deeplabv3plus"
    """
    # Wider figure for better readability
    fig, axes = plt.subplots(5, 2, figsize=(14, 15))

    # Left column: Finetune strategy
    for row_idx, experiment in enumerate(experiments):
        show_xlabel = row_idx == 4
        show_ylabel = True
        show_title = False

        plot_model_strategy_experiment(
            axes[row_idx, 0],
            metrics_dir,
            experiment,
            model,
            "finetune",
            proportions,
            metric,
            show_ylabel,
            show_xlabel,
            show_title,
        )

        # Add experiment name as row label (only on left column)
        axes[row_idx, 0].text(
            -0.18,
            0.5,
            experiment,
            transform=axes[row_idx, 0].transAxes,
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=0,
        )

    # Right column: SimpleMixed strategy
    for row_idx, experiment in enumerate(experiments):
        show_xlabel = row_idx == 4
        show_ylabel = False
        show_title = False

        plot_model_strategy_experiment(
            axes[row_idx, 1],
            metrics_dir,
            experiment,
            model,
            "simplemixed",
            proportions,
            metric,
            show_ylabel,
            show_xlabel,
            show_title,
        )

    # Add column titles
    axes[0, 0].text(
        0.5,
        1.15,
        "Finetune",
        ha="center",
        transform=axes[0, 0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].text(
        0.5,
        1.15,
        "SimpleMixed",
        ha="center",
        transform=axes[0, 1].transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # Add page title at top (centered)
    model_name = "U-Net" if model == "unet" else "DeepLabV3+"
    fig.suptitle(
        f"{experiment_type} Experiments - {model_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        x=0.5,
    )

    # Create custom legend at the bottom (centered)
    from matplotlib.lines import Line2D

    metric_label = metric.replace("_", " ").title()
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Train {metric_label}",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            label=f"Val {metric_label}",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.96])
    plt.subplots_adjust(wspace=0.1, left=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_two_column_page(
    metrics_dir: Path,
    experiments: list[str],
    experiment_type: str,
    model: str,
    strategy: str,
    proportions: list[float],
    output_path: Path,
    metric: str = "dice_joints",
) -> None:
    """Create a single A4-sized page with 2 columns for one model/strategy combination.

    Args:
        experiment_type: "Box Experiments" or "Slope Experiments"
        model: "unet" or "deeplabv3plus"
        strategy: "finetune" or "simplemixed"
    """
    # A4 size: 8.27 x 11.69 inches
    fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69))

    # Split experiments into two groups for the two columns
    # Left column: first half, Right column: second half
    experiments_col1: list[str | None] = list(
        experiments[:3]
        if len(experiments) == 5
        else experiments[: len(experiments) // 2]
    )
    experiments_col2: list[str | None] = list(
        experiments[3:]
        if len(experiments) == 5
        else experiments[len(experiments) // 2 :]
    )

    # Pad to ensure we have 5 rows
    while len(experiments_col1) < 5:
        experiments_col1.append(None)
    while len(experiments_col2) < 5:
        experiments_col2.append(None)

    # Plot left column
    for row_idx in range(5):
        experiment = experiments_col1[row_idx]
        if experiment:
            show_xlabel = row_idx == 4
            show_ylabel = True
            show_title = False

            plot_model_strategy_experiment(
                axes[row_idx, 0],
                metrics_dir,
                experiment,
                model,
                strategy,
                proportions,
                metric,
                show_ylabel,
                show_xlabel,
                show_title,
            )

            # Add experiment name as row label
            axes[row_idx, 0].text(
                -0.35,
                0.5,
                experiment,
                transform=axes[row_idx, 0].transAxes,
                fontsize=10,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=0,
            )
        else:
            axes[row_idx, 0].axis("off")

    # Plot right column
    for row_idx in range(5):
        experiment = experiments_col2[row_idx]
        if experiment:
            show_xlabel = row_idx == 4
            show_ylabel = False
            show_title = False

            plot_model_strategy_experiment(
                axes[row_idx, 1],
                metrics_dir,
                experiment,
                model,
                strategy,
                proportions,
                metric,
                show_ylabel,
                show_xlabel,
                show_title,
            )

            # Add experiment name as row label
            axes[row_idx, 1].text(
                -0.15,
                0.5,
                experiment,
                transform=axes[row_idx, 1].transAxes,
                fontsize=10,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=0,
            )
        else:
            axes[row_idx, 1].axis("off")

    # Add page title at top
    model_name = "U-Net" if model == "unet" else "DeepLabV3+"
    strategy_name = strategy.capitalize()
    fig.suptitle(
        f"{experiment_type} - {model_name} ({strategy_name})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Create custom legend at the bottom
    from matplotlib.lines import Line2D

    metric_label = metric.replace("_", " ").title()
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Train {metric_label}",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            label=f"Val {metric_label}",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot epoch progression for experiments"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="dice_joints",
        help="Metric to plot (e.g., 'dice_joints', 'dice', 'loss')",
    )
    parser.add_argument(
        "--separate-pages",
        action="store_true",
        help="Generate separate A4 pages (4 pages total) instead of single grid",
    )

    args = parser.parse_args()

    if not args.metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {args.metrics_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiment groups
    box_experiments = [
        "Box",
        "Pattern Box",
        "Cardboard Box",
        "Generalisation Pattern Box",
        "Generalisation Cardboard Box",
    ]

    slope_experiments = [
        "Slope",
        "Larvik",
        "Rv 4",
        "Generalisation Larvik",
        "Generalisation Rv 4",
    ]

    # Define all proportions to plot
    proportions = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    if args.separate_pages:
        # Generate 4 separate A4 pages
        print(f"\nGenerating 4 separate A4 pages for metric: {args.metric}...")

        # Page 1: Box - UNet Finetune (left) | UNet SimpleMixed (right)
        output_file = f"figure_epoch_progression_{args.metric}_page1_box_unet.png"
        plot_combined_page(
            args.metrics_dir,
            box_experiments,
            "Box",
            "unet",
            proportions,
            args.output_dir / output_file,
            args.metric,
        )

        # Page 2: Box - DeepLabV3+ Finetune (left) | DeepLabV3+ SimpleMixed (right)
        output_file = (
            f"figure_epoch_progression_{args.metric}_page2_box_deeplabv3plus.png"
        )
        plot_combined_page(
            args.metrics_dir,
            box_experiments,
            "Box",
            "deeplabv3plus",
            proportions,
            args.output_dir / output_file,
            args.metric,
        )

        # Page 3: Slope - UNet Finetune (left) | UNet SimpleMixed (right)
        output_file = f"figure_epoch_progression_{args.metric}_page3_slope_unet.png"
        plot_combined_page(
            args.metrics_dir,
            slope_experiments,
            "Slope",
            "unet",
            proportions,
            args.output_dir / output_file,
            args.metric,
        )

        # Page 4: Slope - DeepLabV3+ Finetune (left) | DeepLabV3+ SimpleMixed (right)
        output_file = (
            f"figure_epoch_progression_{args.metric}_page4_slope_deeplabv3plus.png"
        )
        plot_combined_page(
            args.metrics_dir,
            slope_experiments,
            "Slope",
            "deeplabv3plus",
            proportions,
            args.output_dir / output_file,
            args.metric,
        )

        print(f"\nDone! 4 pages saved to: {args.output_dir}")
    else:
        # Generate single 5×8 grid
        print(
            f"\nGenerating 5×8 grid epoch progression plot for metric: {args.metric}..."
        )
        output_file = f"figure_epoch_progression_{args.metric}_grid.png"
        plot_experiment_grid(
            args.metrics_dir,
            box_experiments,
            slope_experiments,
            proportions,
            args.output_dir / output_file,
            args.metric,
        )

        print(f"\nDone! Plot saved to: {args.output_dir / output_file}")


if __name__ == "__main__":
    main()
