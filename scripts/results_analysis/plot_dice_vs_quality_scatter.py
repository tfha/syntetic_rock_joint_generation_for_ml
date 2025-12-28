"""Scatter plot comparing Dice Joints vs Mean Quality Score across all experiments.

Groups experiments by test object (Box, Pattern Box, Cardboard Box, Slope, Larvik, Rv 4)
and uses different symbols for each group to visualize correlation.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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
        result: dict[str, str | float] = {
            "model": model,
            "strategy": strategy,
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
            "full_name": f"{model}-{strategy}_{experiment}_{proportion}",
        }
        return result

    return None


def normalize_experiment_name(experiment: str) -> str:
    """Normalize experiment names to standard format.

    Keep generalisation experiments separate from base experiments (10 groups total).
    """
    exp_lower = experiment.lower()

    # Check if generalisation experiment
    is_generalisation = exp_lower.startswith("generalisation_")
    if is_generalisation:
        exp_lower = exp_lower.replace("generalisation_", "")

    # Map to standardized names
    if "pattern" in exp_lower and "box" in exp_lower:
        base_name = "Pattern Box"
    elif "cardboard" in exp_lower and "box" in exp_lower:
        base_name = "Cardboard Box"
    elif "box" in exp_lower:
        base_name = "Box"
    elif "rv4" in exp_lower or "rv_4" in exp_lower:
        base_name = "Rv 4"
    elif "larvik" in exp_lower:
        base_name = "Larvik"
    elif "slope" in exp_lower:
        base_name = "Slope"
    else:
        # Fallback: convert underscores to spaces and capitalize
        base_name = experiment.replace("_", " ").title()
        if "Rv4" in base_name:
            base_name = base_name.replace("Rv4", "Rv 4")

    # Add "Generalisation" prefix if it was a generalisation experiment
    if is_generalisation:
        return f"Generalisation {base_name}"
    else:
        return base_name


def load_metrics_data(metrics_dir: Path) -> pd.DataFrame:
    """Load metrics CSVs and extract dice joints scores.

    For finetune: use best val_dice_joints
    For simplemixed: use final epoch val_dice_joints
    """
    data = []

    for csv_file in metrics_dir.glob("*.csv"):
        name = csv_file.stem.replace("_metrics", "")

        # Pattern 1: simplemixed with hyphens and timestamp
        match = re.match(
            r"(unet|deeplabv3plus)-simplemixed_(.+)_(\d+)-\d{8}-\d{4}$", name
        )
        if match:
            model, experiment, proportion = match.groups()
            strategy = "simplemixed"
            metadata = {
                "model": model,
                "strategy": strategy,
                "experiment": experiment,
                "proportion": float(proportion) / 100.0,
                "full_name": f"{model}-{strategy}_{experiment}_{proportion}",
            }
        else:
            # Pattern 2: finetune with underscores
            match = re.match(
                r"(unet|deeplabv3plus)_(finetune|simplemixed)_(.+)_(\d+)$", name
            )
            if match:
                model, strategy, experiment, proportion = match.groups()
                metadata = {
                    "model": model,
                    "strategy": strategy,
                    "experiment": experiment,
                    "proportion": float(proportion) / 100.0,
                    "full_name": f"{model}-{strategy}_{experiment}_{proportion}",
                }
            else:
                continue

        # Read metrics
        try:
            df = pd.read_csv(csv_file)
            if "val_dice_joints" not in df.columns:
                continue

            # For finetune: use best (max) val_dice_joints
            # For simplemixed: use final epoch val_dice_joints
            if metadata["strategy"] == "finetune":
                dice_score = df["val_dice_joints"].max()
            else:  # simplemixed
                dice_score = df["val_dice_joints"].iloc[-1]  # Final epoch

            metadata["dice_joints"] = dice_score
            if isinstance(metadata["experiment"], str):
                metadata["experiment_normalized"] = normalize_experiment_name(
                    metadata["experiment"]
                )
                data.append(metadata)

        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")

    return pd.DataFrame(data)


def load_quality_ratings(ratings_csv: Path) -> pd.DataFrame:
    """Load qualitative ratings and compute mean quality scores."""
    df = pd.read_csv(ratings_csv)

    # Compute mean quality score from criteria
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
        if metadata and isinstance(metadata.get("experiment"), str):
            metadata["mean_quality_score"] = row["mean_quality_score"]
            experiment: str = metadata["experiment"]  # type: ignore[assignment]
            metadata["experiment_normalized"] = normalize_experiment_name(experiment)
            metadata_list.append(metadata)

    return pd.DataFrame(metadata_list)


def create_scatter_plot_all(
    metrics_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create scatter plot of dice joints vs mean quality score for all experiments."""

    # Merge datasets on full_name
    merged = pd.merge(
        metrics_df,
        ratings_df[["full_name", "mean_quality_score"]],
        on="full_name",
        how="inner",
    )

    if merged.empty:
        print("No matching data found!")
        return

    print(f"Plotting {len(merged)} experiments...")

    # Define object groups (10 groups, matching figure_all_experiments_grid order)
    # Blue markers for box types, red markers for slope types
    # Shapes: circle (base), square (pattern/larvik), triangle (cardboard/rv4), diamond (gen pattern/gen larvik), inverted triangle (gen cardboard/gen rv4)
    object_groups = {
        "Box": {"marker": "o", "color": "blue", "label": "Box"},
        "Pattern Box": {"marker": "s", "color": "blue", "label": "Pattern Box"},
        "Cardboard Box": {"marker": "^", "color": "blue", "label": "Cardboard Box"},
        "Generalisation Pattern Box": {
            "marker": "d",
            "color": "blue",
            "label": "Generalisation Pattern Box",
        },
        "Generalisation Cardboard Box": {
            "marker": "v",
            "color": "blue",
            "label": "Generalisation Cardboard Box",
        },
        "Slope": {"marker": "o", "color": "red", "label": "Slope"},
        "Larvik": {"marker": "s", "color": "red", "label": "Larvik"},
        "Rv 4": {"marker": "^", "color": "red", "label": "Rv 4"},
        "Generalisation Larvik": {
            "marker": "d",
            "color": "red",
            "label": "Generalisation Larvik",
        },
        "Generalisation Rv 4": {
            "marker": "v",
            "color": "red",
            "label": "Generalisation Rv 4",
        },
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each group with different markers
    for group_name, style in object_groups.items():
        group_data = merged[merged["experiment_normalized"] == group_name]

        if not group_data.empty:
            ax.scatter(
                group_data["dice_joints"],
                group_data["mean_quality_score"],
                marker=style["marker"],
                color=style["color"],
                s=80,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
                label=f"{style['label']} (n={len(group_data)})",
            )
            print(f"  {group_name}: {len(group_data)} experiments")

    # Add labels and title
    ax.set_xlabel("Validation Dice Joints", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Quality Score", fontsize=12, fontweight="bold")

    # Set consistent axis limits (include low Rv 4 scores)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.8, 5.2)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Calculate statistics
    x = merged["dice_joints"].values
    y = merged["mean_quality_score"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate R² for linear model
    y_pred_linear = slope * x + intercept
    ss_res_linear = np.sum((y - y_pred_linear) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_linear = 1 - (ss_res_linear / ss_tot)

    # Calculate R² for polynomial model
    poly_coeffs = np.polyfit(x, y, 2)
    y_pred_poly = np.polyval(poly_coeffs, x)
    ss_res_poly = np.sum((y - y_pred_poly) ** 2)
    r2_poly = 1 - (ss_res_poly / ss_tot)
    delta_r2 = r2_poly - r2_linear

    # Add title with r value
    ax.set_title(
        f"All 240 Experiments\n(n={len(merged)}, r={r_value:.3f})",
        fontsize=13,
        fontweight="bold",
    )

    # Add text box with R² statistics (matching grid format)
    text_str = f"R²ₗ={r2_linear:.3f}\nR²ₚ={r2_poly:.3f}\nΔR²={delta_r2:.3f}"
    ax.text(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "gray",
        },
    )

    # Add trend lines (without R² in labels)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    poly_y = np.polyval(poly_coeffs, line_x)

    ax.plot(line_x, line_y, "k--", linewidth=2, alpha=0.7, label="Linear")
    ax.plot(line_x, poly_y, "r:", linewidth=2, alpha=0.7, label="Polynomial")

    # Add legend with two columns
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")
    print(f"All 240 experiments: r={r_value:.3f}, ΔR²={delta_r2:.3f}")


def create_scatter_plot_grid(
    metrics_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create 5×2 grid of scatter plots (Box types on left, Slope types on right)."""

    # Merge datasets on full_name
    merged = pd.merge(
        metrics_df,
        ratings_df[["full_name", "mean_quality_score"]],
        on="full_name",
        how="inner",
    )

    if merged.empty:
        print("No matching data found!")
        return

    print("\nCreating 5×2 subplot grid...")

    # Define the grid layout matching figure_all_experiments_grid order:
    # (row, col, experiment_name, color, marker, title)
    subplot_config = [
        (0, 0, "Box", "blue", "o", "Box"),
        (0, 1, "Slope", "red", "o", "Slope"),
        (1, 0, "Pattern Box", "blue", "s", "Pattern Box"),
        (1, 1, "Larvik", "red", "s", "Larvik"),
        (2, 0, "Cardboard Box", "blue", "^", "Cardboard Box"),
        (2, 1, "Rv 4", "red", "^", "Rv 4"),
        (3, 0, "Generalisation Pattern Box", "blue", "d", "Generalisation Pattern Box"),
        (3, 1, "Generalisation Larvik", "red", "d", "Generalisation Larvik"),
        (
            4,
            0,
            "Generalisation Cardboard Box",
            "blue",
            "v",
            "Generalisation Cardboard Box",
        ),
        (4, 1, "Generalisation Rv 4", "red", "v", "Generalisation Rv 4"),
    ]

    # Create figure with 5 rows and 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(12, 22))

    # Plot each subplot
    for row, col, exp_name, color, marker, title in subplot_config:
        ax = axes[row, col]
        group_data = merged[merged["experiment_normalized"] == exp_name]

        if not group_data.empty:
            ax.scatter(
                group_data["dice_joints"],
                group_data["mean_quality_score"],
                marker=marker,
                color=color,
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.8,
            )

            # Compute correlation for this subset
            correlation = group_data["dice_joints"].corr(
                group_data["mean_quality_score"]
            )

            # Calculate trend lines and R² for this subset
            x = group_data["dice_joints"].values
            y = group_data["mean_quality_score"].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept

            # Calculate R² for linear model
            y_pred_linear = slope * x + intercept
            ss_res_linear = np.sum((y - y_pred_linear) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_linear = 1 - (ss_res_linear / ss_tot)

            ax.plot(line_x, line_y, "k--", linewidth=1.5, alpha=0.6, label="Linear")

            # Add polynomial (2nd degree) trend line
            poly_coeffs = np.polyfit(x, y, 2)
            poly_y = np.polyval(poly_coeffs, line_x)

            # Calculate R² for polynomial model
            y_pred_poly = np.polyval(poly_coeffs, x)
            ss_res_poly = np.sum((y - y_pred_poly) ** 2)
            r2_poly = 1 - (ss_res_poly / ss_tot)
            delta_r2 = r2_poly - r2_linear

            ax.plot(line_x, poly_y, "r:", linewidth=1.5, alpha=0.6, label="Polynomial")

            # Add legend for trend lines
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

            # Add title with sample count and correlation
            ax.set_title(
                f"{title}\n(n={len(group_data)}, r={correlation:.3f})",
                fontsize=11,
                fontweight="bold",
            )

            # Add statistics text box with R² values
            ax.text(
                0.05,
                0.95,
                f"R²ₗ={r2_linear:.3f}\nR²ₚ={r2_poly:.3f}\nΔR²={delta_r2:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            print(
                f"  {exp_name}: {len(group_data)} experiments, r={correlation:.3f}, ΔR²={delta_r2:.3f}"
            )

        # Labels
        if row == 4:  # Bottom row
            ax.set_xlabel("Validation Dice Joints", fontsize=10, fontweight="bold")
        if col == 0:  # Left column
            ax.set_ylabel("Mean Quality Score", fontsize=10, fontweight="bold")

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set consistent axis limits with padding to avoid cutting off symbols
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.8, 5.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved grid plot: {output_path}")


def main():
    """Main function."""
    base_dir = Path("experiments/results")
    metrics_dir = base_dir / "metrics" / "mode=max"
    ratings_csv = base_dir / "qualitative_ratings.csv"
    output_all = base_dir / "plots" / "figure_dice_vs_quality_scatter_all.png"
    output_grid = base_dir / "plots" / "figure_dice_vs_quality_scatter_grid.png"

    print("Loading metrics data...")
    metrics_df = load_metrics_data(metrics_dir)
    print(f"Loaded {len(metrics_df)} metric entries")

    print("\nLoading quality ratings...")
    ratings_df = load_quality_ratings(ratings_csv)
    print(f"Loaded {len(ratings_df)} rating entries")

    print("\n" + "=" * 70)
    print("FIGURE 1: All 240 experiments in one plot")
    print("=" * 70)
    create_scatter_plot_all(metrics_df, ratings_df, output_all)

    print("\n" + "=" * 70)
    print("FIGURE 2: 3×2 grid by test object")
    print("=" * 70)
    create_scatter_plot_grid(metrics_df, ratings_df, output_grid)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
