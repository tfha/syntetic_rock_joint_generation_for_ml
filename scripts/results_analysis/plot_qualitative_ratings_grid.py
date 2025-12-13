"""Generate grid plot for qualitative ratings (mean quality scores).

Creates a 5x2 grid plot showing mean quality score across different
proportions of real data, comparing models and training strategies.
Similar format to figure_all_experiments_grid but using qualitative ratings.
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

# Colorblind-friendly colors
COLORS = {
    "unet": "#0173B2",  # Blue
    "deeplabv3plus": "#DE8F05",  # Orange
}

# Markers for all experiments (using circles to match legend)
EXPERIMENT_MARKERS = {
    "Box": "o",  # Circle
    "Pattern Box": "o",  # Circle
    "Cardboard Box": "o",  # Circle
    "Generalisation Pattern Box": "o",  # Circle
    "Generalisation Cardboard Box": "o",  # Circle
    "Slope": "o",  # Circle
    "Larvik": "o",  # Circle
    "Rv 4": "o",  # Circle
    "Generalisation Larvik": "o",  # Circle
    "Generalisation Rv 4": "o",  # Circle
}

# Marker face colors for strategy (filled vs hollow)
STRATEGY_FILL = {
    "finetune": "full",  # Filled markers
    "simplemixed": "none",  # Hollow markers (only edge)
}


def parse_image_filename(filename: str) -> dict[str, str | float] | None:
    """Parse progression image filename to extract metadata.

    Examples:
        unet-finetune_box_10-20251211-1702_progression.png
        deeplabv3plus-simplemixed_generalisation_cardboard_box_30-20251211-1639_progression.png
    """
    # Remove _progression.png suffix
    name = filename.replace("_progression.png", "")

    # Pattern: {model}-{strategy}_{experiment}_{proportion}-{timestamp}
    match = re.match(
        r"(unet|deeplabv3plus)-(finetune|simplemixed)_(.+)_(\d+)-\d{8}-\d{4}$", name
    )
    if match:
        model, strategy, experiment, proportion = match.groups()
        return {
            "model": model,
            "strategy": strategy,
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
        }

    return None


def normalize_experiment_name(experiment: str) -> str:
    """Normalize experiment names to match standard format."""
    # Convert underscores to spaces and capitalize words
    normalized = experiment.replace("_", " ").title()

    # Handle special cases for Rv4 -> Rv 4
    if "Rv4" in normalized:
        normalized = normalized.replace("Rv4", "Rv 4")
    if "rv4" in experiment.lower():
        normalized = normalized.replace("Rv4", "Rv 4")

    return normalized


def load_qualitative_ratings(csv_path: Path) -> pd.DataFrame:
    """Load qualitative ratings and calculate mean quality scores."""
    df = pd.read_csv(csv_path)

    # Calculate mean quality score from the four criteria
    criteria_columns = [
        "geological_recognisability",
        "joint_persistence",
        "boundary_localisation",
        "false_positives",
    ]

    df["mean_quality_score"] = df[criteria_columns].mean(axis=1)

    # Parse image filenames to extract metadata
    metadata_list = []
    for _, row in df.iterrows():
        metadata = parse_image_filename(row["image"])
        if metadata:
            metadata["mean_quality_score"] = row["mean_quality_score"]
            # Include better_epoch information
            metadata["has_better_epoch"] = (
                pd.notna(row.get("better_epoch")) and row.get("better_epoch") != ""
            )
            metadata_list.append(metadata)

    if not metadata_list:
        return pd.DataFrame()

    result_df = pd.DataFrame(metadata_list)

    # Normalize experiment names
    result_df["experiment_normalized"] = result_df["experiment"].apply(
        normalize_experiment_name
    )

    return result_df


def plot_individual_experiment(
    ax: plt.Axes,
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
                df_subset["mean_quality_score"],
                color=COLORS[model],
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.7,
            )

            # Plot points on top
            ax.scatter(
                real_proportion,
                df_subset["mean_quality_score"],
                color=COLORS[model],
                marker=marker,
                s=60,
                edgecolors=COLORS[model],
                linewidths=1.5,
                facecolors=COLORS[model] if fillstyle == "full" else "none",
                alpha=0.8,
                zorder=3,
            )

            # Mark points with better epochs: + for finetune, x for simplemixed
            df_better = df_subset[df_subset["has_better_epoch"]]
            if not df_better.empty:
                marker_better = "+" if strategy == "finetune" else "x"
                ax.scatter(
                    df_better["proportion"],
                    df_better["mean_quality_score"],
                    marker=marker_better,
                    s=200,
                    color="black",
                    linewidths=2,
                    alpha=0.9,
                    zorder=4,
                )

    # Styling
    ax.set_title(experiment, fontsize=10, pad=5)
    ax.set_xlim(-0.05, 1.05)  # 0.0 to 1.0 for real data proportion
    ax.set_ylim(0, 5.5)  # Quality scores range from 1-5
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Proportion of real data", fontsize=9)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("Mean quality score", fontsize=9)
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
        Line2D(
            [0],
            [0],
            marker="+",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label='Better epoch than "best" (Finetune)',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="Better epoch than final (SimpleMixed)",
            markeredgewidth=2,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def main() -> None:
    """Generate qualitative ratings grid plot."""
    # Load qualitative ratings
    csv_path = Path("experiments/results/qualitative_ratings.csv")
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print("Loading qualitative ratings...")
    df = load_qualitative_ratings(csv_path)

    if df.empty:
        print("Error: No data loaded from CSV file")
        return

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
        output_dir / "figure_qualitative_ratings_grid.png",
    )

    print("\nDone! Plot saved to:", output_dir / "figure_qualitative_ratings_grid.png")


if __name__ == "__main__":
    main()
