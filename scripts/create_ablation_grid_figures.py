"""
Create grid figures showing predictions across prediction thresholds.
Each figure: rows = images 1-10, columns = [raw | gt | predictions for each threshold]
One figure per strategy (finetune_box, simplemixed_box, finetune_slope, simplemixed_slope).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def add_red_border(ax, linewidth=2):
    """Add red border to an axes."""
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(linewidth)
        spine.set_visible(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def crop_image_section(img, section: str) -> np.ndarray:
    """Crop specific section from composite image.

    Uses exact crop coordinates from plot_best_worst_comparison.py:
    - Original: (188, 82) to (530, 423)
    - Ground Truth: (598, 82) to (940, 423)
    - Prediction: (1008, 82) to (1350, 423)
    """
    crop_coords = {
        "original": (188, 82, 530, 423),
        "ground_truth": (598, 82, 940, 423),
        "prediction": (1008, 82, 1350, 423),
    }

    if section not in crop_coords:
        raise ValueError(f"Invalid section: {section}")

    cropped = img.crop(crop_coords[section])
    return np.array(cropped)


def extract_sections(img_path):
    """
    Load image and extract sections using proper cropping coordinates.
    Returns: raw_img, gt_img, pred_img (each as numpy array)
    """
    img = Image.open(img_path)

    raw = crop_image_section(img, "original")
    gt = crop_image_section(img, "ground_truth")
    pred = crop_image_section(img, "prediction")

    return raw, gt, pred


def create_grid_figures():
    """Create grid figures for each strategy."""
    metrics_dir = Path("experiments/results/ablation_studies/final_test_images")

    # Define strategies and thresholds
    geometries = ["box", "slope"]
    strategies = ["finetune", "simplemixed"]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    image_nums = list(range(1, 11))  # 01 to 10

    # Create figure for each strategy
    for geom in geometries:
        for strat in strategies:
            strategy_name = f"{strat.capitalize()}_{geom}"
            print(f"\nProcessing {strategy_name}...")

            # Create figure: 10 rows x 9 columns (raw, gt, 7 thresholds)
            fig, axes = plt.subplots(10, 9, figsize=(27, 30))
            fig.suptitle(
                f"Ablation Study: {strategy_name}\nRows=Images 1-10, "
                f"Columns=[Original | Ground truth | Prediction threshold 0.2-0.8]",
                fontsize=24,
                fontweight="bold",
                y=0.995,
            )

            # Add column headers
            col_labels = ["Original", "Ground truth"] + [f"{t}" for t in thresholds]
            for col_idx, label in enumerate(col_labels):
                axes[0, col_idx].text(
                    0.5,
                    1.15,
                    label,
                    ha="center",
                    va="bottom",
                    transform=axes[0, col_idx].transAxes,
                    fontsize=24,
                    fontweight="bold",
                )
                # Add red border to header cells
                add_red_border(axes[0, col_idx])

            # Populate grid
            for row_idx, img_num in enumerate(image_nums):
                img_id = f"{img_num:02d}"

                # Load first threshold image to get raw and GT
                first_threshold_file = (
                    metrics_dir / geom / f"{geom}_threshold_0.2_{strat}_{img_id}.png"
                )

                if not first_threshold_file.exists():
                    print(f"[WARNING] Not found: {first_threshold_file}")
                    continue

                raw, gt, _ = extract_sections(first_threshold_file)

                # Column 0: Raw
                axes[row_idx, 0].imshow(raw)
                add_red_border(axes[row_idx, 0])

                # Column 1: Ground Truth
                axes[row_idx, 1].imshow(gt)
                add_red_border(axes[row_idx, 1])

                # Columns 2-8: Predictions for each threshold
                for col_idx, threshold in enumerate(thresholds, start=2):
                    pred_file = (
                        metrics_dir
                        / geom
                        / f"{geom}_threshold_{threshold:.1f}_{strat}_{img_id}.png"
                    )

                    if pred_file.exists():
                        _, _, pred = extract_sections(pred_file)
                        axes[row_idx, col_idx].imshow(pred)
                    else:
                        print(f"[WARNING] Not found: {pred_file}")

                    add_red_border(axes[row_idx, col_idx])

                # Row label (Sample number, two lines)
                axes[row_idx, 0].text(
                    -0.25,
                    0.5,
                    f"Sample\n{img_num}",
                    ha="center",
                    va="center",
                    transform=axes[row_idx, 0].transAxes,
                    fontsize=20,
                    fontweight="bold",
                )

            # Save figure
            output_dir = Path("experiments/results/ablation_studies/grid_figures")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"grid_{strategy_name}.png"

            plt.tight_layout(rect=[0, 0, 1, 0.99])
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            print(f"[OK] Saved: {output_path}")

            plt.close(fig)

    print("\n[OK] All grid figures created!")


if __name__ == "__main__":
    create_grid_figures()
