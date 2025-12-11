"""Generate publication-quality plots for journal article.

Creates two plots showing best validation Dice score for joints across different
proportions of synthetic data, comparing models and training strategies.
Note: X-axis is reversed to show synthetic proportion increasing left to right.
"""

from __future__ import annotations

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

# Colorblind-friendly colors that work in grayscale
# Using different markers for experiments and model/strategy combinations
COLORS = {
    "unet": "#0173B2",  # Blue
    "deeplabv3plus": "#DE8F05",  # Orange
}

# Markers for different experiment strategies
EXPERIMENT_MARKERS = {
    "Box": "o",  # Circle
    "Pattern Box": "s",  # Square
    "Cardboard Box": "^",  # Triangle up
    "Generalisation Pattern Box": "X",  # Cross (X)
    "Generalisation Cardboard Box": "P",  # Plus (+)
    "Slope": "o",  # Circle
    "Larvik": "s",  # Square
    "Rv 4": "^",  # Triangle up
    "Generalisation Larvik": "X",  # Cross (X)
    "Generalisation Rv 4": "P",  # Plus (+)
}

# Marker face colors for strategy (filled vs hollow)
STRATEGY_FILL = {
    "finetune": "full",  # Filled markers
    "simplemixed": "none",  # Hollow markers (only edge)
}


def parse_filename(filename: str) -> dict[str, str | float] | None:
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


def get_best_val_dice_joints(csv_path: Path) -> float | None:
    """Extract best validation Dice score for joints from metrics CSV."""
    try:
        df = pd.read_csv(csv_path)

        # Check which column name is used (finetune vs simplemixed may differ)
        if "val_dice_joints" in df.columns:
            return df["val_dice_joints"].max()
        elif "val_dice_joint" in df.columns:
            return df["val_dice_joint"].max()
        else:
            print(f"Warning: No val_dice_joints column in {csv_path.name}")
            return None
    except Exception as e:
        print(f"Error reading {csv_path.name}: {e}")
        return None


def load_all_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load all metrics from CSV files into a structured DataFrame."""
    data = []

    for csv_file in metrics_dir.glob("*.csv"):
        metadata = parse_filename(csv_file.name)
        if metadata is None:
            continue

        best_dice = get_best_val_dice_joints(csv_file)
        if best_dice is None:
            continue

        data.append(
            {
                "model": metadata["model"],
                "strategy": metadata["strategy"],
                "experiment": metadata["experiment"],
                "proportion": metadata["proportion"],
                "val_dice_joints": best_dice,
            }
        )

    return pd.DataFrame(data)


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


def plot_individual_experiment(
    ax,
    df: pd.DataFrame,
    experiment: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    """Plot a single experiment on the given axis."""
    df_exp = df[df["experiment_normalized"] == experiment].copy()

    if df_exp.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Plot each combination of model and strategy
    for strategy in ["finetune", "simplemixed"]:
        for model in ["unet", "deeplabv3plus"]:
            df_subset = df_exp[
                (df_exp["model"] == model) & (df_exp["strategy"] == strategy)
            ]

            if df_subset.empty:
                continue

            # Sort by proportion
            df_subset = df_subset.sort_values("proportion")

            # Use real data proportion directly
            real_proportion = df_subset["proportion"]

            # Get marker and fill style
            marker = EXPERIMENT_MARKERS.get(experiment, "o")
            fillstyle = STRATEGY_FILL[strategy]
            linestyle = "-" if strategy == "finetune" else "--"

            # Plot line
            ax.plot(
                real_proportion,
                df_subset["val_dice_joints"],
                color=COLORS[model],
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.7,
            )

            # Plot points on top
            ax.scatter(
                real_proportion,
                df_subset["val_dice_joints"],
                color=COLORS[model],
                marker=marker,
                s=60,
                edgecolors=COLORS[model],
                linewidths=1.5,
                facecolors=COLORS[model] if fillstyle == "full" else "none",
                alpha=0.8,
                zorder=3,
            )

    # Styling
    ax.set_title(experiment, fontsize=10, pad=5)
    ax.set_xlim(-0.05, 1.05)  # 0.0 to 1.0 for real data proportion
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Proportion of real data", fontsize=9)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("val_dice_joints", fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=8)


def plot_experiment_grid(
    df: pd.DataFrame,
    box_experiments: list[str],
    slope_experiments: list[str],
    output_path: Path,
) -> None:
    """Create a 5x2 grid of subplots for box and slope experiments."""
    fig, axes = plt.subplots(5, 2, figsize=(10, 12))

    # Left column: Box experiments
    for i, experiment in enumerate(box_experiments):
        show_xlabel = i == 4  # Only bottom row
        show_ylabel = True  # All left column
        plot_individual_experiment(axes[i, 0], df, experiment, show_ylabel, show_xlabel)

    # Right column: Slope experiments
    for i, experiment in enumerate(slope_experiments):
        show_xlabel = i == 4  # Only bottom row
        show_ylabel = False  # No ylabel for right column
        plot_individual_experiment(axes[i, 1], df, experiment, show_ylabel, show_xlabel)

    # Add column titles
    axes[0, 0].text(
        0.5,
        1.15,
        "Box Experiments",
        ha="center",
        transform=axes[0, 0].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].text(
        0.5,
        1.15,
        "Slope Experiments",
        ha="center",
        transform=axes[0, 1].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    # Create custom legend at the bottom
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["unet"],
            markeredgecolor=COLORS["unet"],
            markersize=8,
            label="U-Net (Finetune)",
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=COLORS["unet"],
            markersize=8,
            label="U-Net (SimpleMixed)",
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["deeplabv3plus"],
            markeredgecolor=COLORS["deeplabv3plus"],
            markersize=8,
            label="DeepLabV3+ (Finetune)",
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=COLORS["deeplabv3plus"],
            markersize=8,
            label="DeepLabV3+ (SimpleMixed)",
            markeredgewidth=1.5,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def main() -> None:
    """Generate journal figures."""
    # Load all metrics
    metrics_dir = Path("experiments/results/metrics/mode=max")
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found: {metrics_dir}")
        return

    print("Loading metrics from CSV files...")
    df = load_all_metrics(metrics_dir)

    if df.empty:
        print("Error: No data loaded from CSV files")
        return

    # Normalize experiment names
    df["experiment_normalized"] = df["experiment"].apply(normalize_experiment_name)

    print(f"Loaded {len(df)} data points")
    print(f"Models: {df['model'].unique()}")
    print(f"Strategies: {df['strategy'].unique()}")
    print(f"Experiments: {df['experiment_normalized'].unique()}")

    # Create output directory
    output_dir = Path("experiments/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("\nGenerating 5x2 grid plot...")
    plot_experiment_grid(
        df,
        box_experiments,
        slope_experiments,
        output_dir / "figure_all_experiments_grid.png",
    )

    print("\nDone! Plot saved to:", output_dir)


if __name__ == "__main__":
    main()
