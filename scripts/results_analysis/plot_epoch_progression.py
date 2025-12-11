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
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_title: bool = False,
) -> None:
    """Plot epoch progression for a specific model/strategy/experiment with all proportions."""
    # Colors: Blue tones for 0-30%, Green tones for 50-100%
    proportion_colors = {
        0.0: "#E3F2FD",  # Very light blue (0%)
        0.1: "#90CAF9",  # Light blue (10%)
        0.3: "#42A5F5",  # Medium blue (30%)
        0.5: "#C8E6C9",  # Light green (50%)
        0.7: "#66BB6A",  # Medium green (70%)
        0.9: "#388E3C",  # Medium-dark green (90%)
        1.0: "#004D40",  # Very dark teal-green (100%)
    }

    has_data = False

    # Plot each proportion
    for proportion in proportions:
        df = load_epoch_metrics(metrics_dir, experiment, model, strategy, proportion)

        if df is None or df.empty:
            continue

        has_data = True

        # Get color for this proportion
        color = proportion_colors.get(proportion, "#000000")

        # Convert to synthetic proportion for label (1 - real proportion)
        synthetic_pct = round((1.0 - proportion) * 100)

        # Plot training dice (dashed line)
        if "train_dice_joints" in df.columns:
            ax.plot(
                df["epoch"],
                df["train_dice_joints"],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

        # Plot validation dice (solid line) - this is the test/validation set
        if "val_dice_joints" in df.columns:
            ax.plot(
                df["epoch"],
                df["val_dice_joints"],
                color=color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.9,
                label=f"{synthetic_pct}%",
            )

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Styling
    if show_title:
        title = f"{model.upper()}\n{strategy.capitalize()}"
        ax.set_title(title, fontsize=9, pad=5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=8)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("Dice score (joints)", fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=7)

    # Add legend for proportions in all subplots if has data
    if has_data:
        ax.legend(
            fontsize=6,
            loc="lower right",
            framealpha=0.8,
            title="% synthetic data",
            title_fontsize=6,
        )


def plot_experiment_grid(
    metrics_dir: Path,
    box_experiments: list[str],
    slope_experiments: list[str],
    proportions: list[float],
    output_path: Path,
) -> None:
    """Create a 5×8 grid of subplots for all model/strategy/experiment combinations."""
    fig, axes = plt.subplots(5, 8, figsize=(40, 15))

    # Define the combinations for each column
    # Columns: 4 for box experiments, 4 for slope experiments
    # For each experiment type: UNet-Finetune, UNet-SimpleMixed, DeepLabV3+-Finetune, DeepLabV3+-SimpleMixed

    models = ["unet", "unet", "deeplabv3plus", "deeplabv3plus"]
    strategies = ["finetune", "simplemixed", "finetune", "simplemixed"]

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

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label="Train_dice_joints",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            label="Val_dice_joints",
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

    print("\nGenerating 5×8 grid epoch progression plot...")
    plot_experiment_grid(
        args.metrics_dir,
        box_experiments,
        slope_experiments,
        proportions,
        args.output_dir / "figure_epoch_progression_grid.png",
    )

    print("\nDone! Plot saved to:", args.output_dir)


if __name__ == "__main__":
    main()
