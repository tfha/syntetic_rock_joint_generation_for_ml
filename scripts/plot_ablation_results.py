"""
Plot ablation study results showing Val_dice_joints vs prediction threshold
for different datasets and training strategies.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Load ablation metrics and create visualization."""
    # Define paths
    metrics_dir = Path("experiments/results/ablation_studies/metrics")

    # Initialize data containers
    data: dict[str, dict[str, list[float]]] = {
        "Finetune_box": {"thresholds": [], "dice_joints": []},
        "Simplemixed_box": {"thresholds": [], "dice_joints": []},
        "Finetune_slope": {"thresholds": [], "dice_joints": []},
        "Simplemixed_slope": {"thresholds": [], "dice_joints": []},
    }

    # Process box metrics
    for csv_file in sorted((metrics_dir / "box").glob("*.csv")):
        # Parse filename: metrics_box_threshold_0.2_finetune.csv
        parts = csv_file.stem.split("_")
        threshold = float(parts[3])
        strategy = parts[4]  # finetune or simplemixed

        # Read the last row (final epoch metrics)
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            final_metrics = rows[-1]  # Last row
            val_dice_joints = float(final_metrics["val_dice_joints"])

        key = f"{strategy.capitalize()}_box"
        data[key]["thresholds"].append(threshold)
        data[key]["dice_joints"].append(val_dice_joints)

    # Process slope metrics
    for csv_file in sorted((metrics_dir / "slope").glob("*.csv")):
        # Parse filename: metrics_slope_threshold_0.2_finetune.csv
        parts = csv_file.stem.split("_")
        threshold = float(parts[3])
        strategy = parts[4]  # finetune or simplemixed

        # Read the last row (final epoch metrics)
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            final_metrics = rows[-1]  # Last row
            val_dice_joints = float(final_metrics["val_dice_joints"])

        key = f"{strategy.capitalize()}_slope"
        data[key]["thresholds"].append(threshold)
        data[key]["dice_joints"].append(val_dice_joints)

    # Sort by threshold for each dataset
    for key in data:
        sorted_pairs = sorted(zip(data[key]["thresholds"], data[key]["dice_joints"]))
        if sorted_pairs:
            thresholds_sorted, dice_sorted = zip(*sorted_pairs)
            data[key]["thresholds"] = list(thresholds_sorted)
            data[key]["dice_joints"] = list(dice_sorted)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors and markers for each dataset
    colors = {
        "Finetune_box": "blue",  # blue
        "Simplemixed_box": "blue",  # blue hollow circle
        "Finetune_slope": "red",  # red rectangle
        "Simplemixed_slope": "red",  # red hollow rectangle
    }

    markers = {
        "Finetune_box": "o",
        "Simplemixed_box": "o",  # hollow circle
        "Finetune_slope": "s",  # rectangle
        "Simplemixed_slope": "s",  # hollow rectangle
    }

    linestyles = {
        "Finetune_box": "-",  # solid line (no change)
        "Simplemixed_box": "--",  # dashed line
        "Finetune_slope": "-",  # solid line
        "Simplemixed_slope": "--",  # dashed line
    }

    fillstyles = {
        "Finetune_box": "full",  # filled (no change)
        "Simplemixed_box": "none",  # hollow circle
        "Finetune_slope": "full",  # filled rectangle
        "Simplemixed_slope": "none",  # hollow rectangle
    }

    # Plot each dataset
    for label in data:
        if data[label]["thresholds"]:  # Only plot if data exists
            ax.plot(
                data[label]["thresholds"],
                data[label]["dice_joints"],
                marker=markers[label],
                color=colors[label],
                label=label,
                linewidth=2,
                markersize=8,
                linestyle=linestyles[label],
                fillstyle=fillstyles[label],
                alpha=0.8,
            )

    # Customize plot
    ax.set_xlabel("Prediction Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Dice Joints", fontsize=12, fontweight="bold")
    ax.set_title(
        "Ablation Study: Validation Dice Joints vs Prediction Threshold",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc=(0.8, 0.2), fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set x-axis ticks to thresholds
    all_thresholds = set()
    for key in data:
        all_thresholds.update(data[key]["thresholds"])
    if all_thresholds:
        ax.set_xticks(sorted(all_thresholds))

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_dir = Path("experiments/results/ablation_studies")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_dice_joints_by_threshold.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Plot saved to: {output_path}")

    # Print summary statistics
    print("\nAblation Study Summary:")
    print("-" * 80)
    for label in sorted(data.keys()):
        if data[label]["dice_joints"]:
            mean_dice = np.mean(data[label]["dice_joints"])
            max_dice = np.max(data[label]["dice_joints"])
            max_threshold = data[label]["thresholds"][
                data[label]["dice_joints"].index(max_dice)
            ]
            print(
                f"{label:20} | Mean: {mean_dice:.4f} | Max: {max_dice:.4f} (@ {max_threshold})"
            )


if __name__ == "__main__":
    main()
