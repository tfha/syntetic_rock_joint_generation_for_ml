"""Generate 10×6 grid comparing best/worst predictions across experiments.

For each test dataset, shows:
- Original image and ground truth
- Best val_dice_joints prediction (with full experiment name)
- Best, average, and worst mean quality score predictions

Uses one randomly selected sample (seed=42) per test dataset.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Publication-quality plot settings
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "font.family": "sans-serif",
    }
)


def parse_image_filename(filename: str) -> dict[str, str | float] | None:
    """Parse progression image filename to extract metadata.

    Examples:
        unet-finetune_box_10-20251211-1702_progression.png
        deeplabv3plus-simplemixed_generalisation_cardboard_box_30-20251211-1639_progression.png

    Returns:
        Dict with keys: model, strategy, experiment (all str), proportion (float), full_name (str)
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
    """Normalize experiment names to match standard format."""
    # Convert underscores to spaces and capitalize words
    normalized = experiment.replace("_", " ").title()

    # Handle special cases for Rv4 -> Rv 4
    if "Rv4" in normalized:
        normalized = normalized.replace("Rv4", "Rv 4")
    if "rv4" in experiment.lower():
        normalized = normalized.replace("Rv4", "Rv 4")

    return normalized


def load_metrics_data(metrics_dir: Path) -> pd.DataFrame:
    """Load metrics CSVs and get best val_dice_joints for each experiment."""
    data = []

    for csv_file in metrics_dir.glob("*.csv"):
        # Parse filename to extract metadata
        name = csv_file.stem.replace("_metrics", "")

        # Pattern 1: simplemixed with hyphens and timestamp
        match = re.match(
            r"(unet|deeplabv3plus)-simplemixed_(.+)_(\d+)-\d{8}-\d{4}$", name
        )
        if match:
            model, experiment, proportion = match.groups()
            metadata = {
                "model": model,
                "strategy": "simplemixed",
                "experiment": experiment,
                "proportion": float(proportion) / 100.0,
                "full_name": f"{model}-simplemixed_{experiment}_{proportion}",
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

        # Read metrics and get best val_dice_joints
        try:
            df = pd.read_csv(csv_file)
            if "val_dice_joints" in df.columns:
                best_dice = df["val_dice_joints"].max()
                metadata["best_val_dice_joints"] = best_dice
                if isinstance(metadata["experiment"], str):
                    metadata["experiment_normalized"] = normalize_experiment_name(
                        metadata["experiment"]
                    )
                data.append(metadata)
        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")

    return pd.DataFrame(data)


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
        if metadata and isinstance(metadata.get("experiment"), str):
            metadata["mean_quality_score"] = row["mean_quality_score"]
            # Type assertion for mypy
            experiment: str = metadata["experiment"]  # type: ignore[assignment]
            metadata["experiment_normalized"] = normalize_experiment_name(experiment)
            metadata_list.append(metadata)

    return pd.DataFrame(metadata_list)


def crop_image_section(img: Image.Image, section: str) -> np.ndarray:
    """Crop specific section from composite image.

    Uses exact crop coordinates matching plot_progression.py:
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


def find_best_experiments(
    test_dataset: str,
    metrics_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> dict[str, tuple[str, float]]:
    """Find experiments with best/worst metrics for a test dataset.

    Returns dict with keys: 'best_dice', 'best_quality', 'avg_quality', 'worst_quality'
    Each value is a tuple of (full_experiment_name, score).
    """
    # Filter for this test dataset
    metrics_subset = metrics_df[
        metrics_df["experiment_normalized"] == test_dataset
    ].copy()
    ratings_subset = ratings_df[
        ratings_df["experiment_normalized"] == test_dataset
    ].copy()

    results = {}

    # Best val_dice_joints
    if not metrics_subset.empty:
        best_idx = metrics_subset["best_val_dice_joints"].idxmax()
        full_name: str = metrics_subset.loc[best_idx, "full_name"]  # type: ignore[assignment]
        dice_score: float = metrics_subset.loc[best_idx, "best_val_dice_joints"]  # type: ignore[assignment]
        results["best_dice"] = (full_name, dice_score)

    # Quality scores
    if not ratings_subset.empty:
        # Best quality
        best_idx = ratings_subset["mean_quality_score"].idxmax()
        full_name_best: str = ratings_subset.loc[best_idx, "full_name"]  # type: ignore[assignment]
        quality_best: float = ratings_subset.loc[best_idx, "mean_quality_score"]  # type: ignore[assignment]
        results["best_quality"] = (full_name_best, quality_best)

        # Worst quality
        worst_idx = ratings_subset["mean_quality_score"].idxmin()
        full_name_worst: str = ratings_subset.loc[worst_idx, "full_name"]  # type: ignore[assignment]
        quality_worst: float = ratings_subset.loc[worst_idx, "mean_quality_score"]  # type: ignore[assignment]
        results["worst_quality"] = (full_name_worst, quality_worst)

        # Average quality (closest to median)
        median_score = ratings_subset["mean_quality_score"].median()
        ratings_subset["dist_to_median"] = abs(
            ratings_subset["mean_quality_score"] - median_score
        )
        avg_idx = ratings_subset["dist_to_median"].idxmin()
        full_name_avg: str = ratings_subset.loc[avg_idx, "full_name"]  # type: ignore[assignment]
        quality_avg: float = ratings_subset.loc[avg_idx, "mean_quality_score"]  # type: ignore[assignment]
        results["avg_quality"] = (full_name_avg, quality_avg)

    return results


def load_experiment_image(
    image_dir: Path, experiment_name: str, sample_num: int, section: str
) -> np.ndarray | None:
    """Load a specific section from an experiment's final epoch image.

    Args:
        image_dir: Base directory containing experiment folders
        experiment_name: Full experiment name (e.g., 'unet-finetune_box_10')
        sample_num: Sample number (0-9)
        section: 'original', 'ground_truth', or 'prediction'
    """
    # Find matching folder (name includes timestamp)
    matching_folders = list(image_dir.glob(f"{experiment_name}-*"))

    if not matching_folders:
        print(f"Warning: No folder found for {experiment_name}")
        return None

    # Use first match (should only be one)
    exp_folder = matching_folders[0]

    # Look for final epoch folder
    final_folder = exp_folder / "final"
    if not final_folder.exists():
        print(f"Warning: No final folder in {exp_folder.name}")
        return None

    # Load sample image
    sample_path = final_folder / f"sample_{sample_num}.png"
    if not sample_path.exists():
        print(f"Warning: Sample {sample_num} not found in {exp_folder.name}/final")
        return None

    try:
        img = Image.open(sample_path)
        return crop_image_section(img, section)
    except Exception as e:
        print(f"Error loading {sample_path}: {e}")
        return None


def create_comparison_grid(
    test_datasets: list[str],
    metrics_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    image_dir: Path,
    output_path: Path,
    figure_title: str = "Comparison Grid",
) -> None:
    """Create grid comparing best/worst predictions."""
    num_rows = len(test_datasets)
    num_cols = 6

    # Calculate figure size with larger cells and space for text below
    cell_height = 2.8  # Larger to accommodate text below
    cell_width = 2.2  # Slightly wider
    figsize = (cell_width * num_cols + 2.0, cell_height * num_rows + 1)

    # Create figure with appropriate size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    # Adjust subplot positioning to make room for labels and text
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.95, bottom=0.02, hspace=0.35, wspace=0.15
    )

    # Column titles
    col_titles = [
        "Original",
        "Ground truth",
        "Best/final epoch",
        "Best quality",
        "Average quality",
        "Worst quality",
    ]

    # Define specific sample numbers for each test dataset
    sample_mapping = {
        "Box": 1,
        "Pattern Box": 0,
        "Cardboard Box": 0,
        "Generalisation Pattern Box": 0,
        "Generalisation Cardboard Box": 0,
        "Slope": 3,
        "Larvik": 5,
        "Rv 4": 8,
        "Generalisation Larvik": 5,
        "Generalisation Rv 4": 8,
    }

    # Process each test dataset
    for row, test_dataset in enumerate(test_datasets):
        # Get the specific sample number for this dataset
        sample_num = sample_mapping.get(test_dataset, 0)

        # Find best experiments for this dataset
        best_experiments = find_best_experiments(test_dataset, metrics_df, ratings_df)

        if not best_experiments:
            # No data for this test dataset
            for col in range(num_cols):
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].text(
                        0.5, 0.5, "No data", ha="center", va="center", fontsize=9
                    )
            continue

        # Get a reference experiment for original and ground truth
        ref_tuple = (
            best_experiments.get("best_dice") or list(best_experiments.values())[0]
        )
        ref_experiment = ref_tuple[0] if isinstance(ref_tuple, tuple) else ref_tuple

        # Column 0: Original image
        original = load_experiment_image(
            image_dir, ref_experiment, sample_num, "original"
        )
        if original is not None:
            axes[row, 0].imshow(original)
        axes[row, 0].set_aspect("equal")
        # Add thin red border
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=10, fontweight="bold")

        # Column 1: Ground truth
        gt = load_experiment_image(
            image_dir, ref_experiment, sample_num, "ground_truth"
        )
        if gt is not None:
            axes[row, 1].imshow(gt)
        axes[row, 1].set_aspect("equal")
        # Add thin red border
        for spine in axes[row, 1].spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        if row == 0:
            axes[row, 1].set_title(col_titles[1], fontsize=10, fontweight="bold")

        # Columns 2-5: Predictions
        pred_cols = [
            ("best_dice", 2, "Dice joints: {:.2f}"),
            ("best_quality", 3, "Mean quality score: {:.1f}"),
            ("avg_quality", 4, "Mean quality score: {:.1f}"),
            ("worst_quality", 5, "Mean quality score: {:.1f}"),
        ]

        for exp_key, col, score_fmt in pred_cols:
            if exp_key in best_experiments:
                exp_name, score = best_experiments[exp_key]
                pred = load_experiment_image(
                    image_dir, exp_name, sample_num, "prediction"
                )
                if pred is not None:
                    axes[row, col].imshow(pred)

                # Add column title for first row
                if row == 0:
                    axes[row, col].set_title(
                        col_titles[col], fontsize=10, fontweight="bold"
                    )

                # Extract short name: model-strategy_proportion (remove test dataset)
                # e.g., "unet-finetune_box_10" -> "unet-finetune_10"
                parts = exp_name.split("_")
                if len(parts) >= 2:
                    model_strategy = parts[0]  # e.g., "unet-finetune"
                    proportion = parts[-1]  # e.g., "10"
                    short_name = f"{model_strategy}_{proportion}"
                else:
                    short_name = exp_name

                # For best_dice column, also add mean quality score
                if exp_key == "best_dice":
                    # Look up mean quality score for this experiment
                    quality_match = ratings_df[ratings_df["full_name"] == exp_name]
                    if not quality_match.empty:
                        quality_score = quality_match["mean_quality_score"].iloc[0]
                        score_text = f"Mean quality score: {quality_score:.1f}\n{score_fmt.format(score)}"
                    else:
                        score_text = score_fmt.format(score)
                else:
                    # For quality columns, also add dice joints score
                    dice_match = metrics_df[metrics_df["full_name"] == exp_name]
                    if not dice_match.empty:
                        dice_score = dice_match["best_val_dice_joints"].iloc[0]
                        score_text = (
                            f"{score_fmt.format(score)}\nDice Joints: {dice_score:.3f}"
                        )
                    else:
                        score_text = score_fmt.format(score)

                # Add experiment name and score below image
                axes[row, col].text(
                    0.5,
                    -0.04,
                    f"{short_name}\n{score_text}",
                    fontsize=7,
                    ha="center",
                    va="top",
                    transform=axes[row, col].transAxes,
                )

            axes[row, col].set_aspect("equal")
            # Add thin red border
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(1.5)
                spine.set_visible(True)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Adjust layout FIRST - leave more space on the left for sample labels
    plt.tight_layout(rect=(0.08, 0, 1, 0.99))

    # Add sample labels AFTER layout is finalized
    for row, test_dataset in enumerate(test_datasets):
        # Get sample number for this row
        sample_num = sample_mapping.get(test_dataset, 0)

        # Get the bbox of the subplot in figure coordinates
        bbox = axes[row, 0].get_position()
        # Calculate vertical center of this subplot
        row_center_y = (bbox.y0 + bbox.y1) / 2

        # Position labels to the left of the leftmost subplot
        label_x = bbox.x0 - 0.02  # 2% to the left of subplot

        fig.text(
            label_x,
            row_center_y,
            f"{test_dataset}\n(Sample {sample_num})",
            fontsize=9,
            fontweight="bold",
            ha="right",  # Right-align so text extends leftward
            va="center",
            transform=fig.transFigure,
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def main() -> None:
    """Generate best/worst comparison grid plot."""
    # Load data
    metrics_dir = Path("experiments/results/metrics/mode=max")
    ratings_csv = Path("experiments/results/qualitative_ratings.csv")
    image_dir = Path("experiments/results/images")

    print("Loading metrics and ratings...")
    metrics_df = load_metrics_data(metrics_dir)
    ratings_df = load_qualitative_ratings(ratings_csv)

    print(f"Loaded {len(metrics_df)} metric entries")
    print(f"Loaded {len(ratings_df)} rating entries")

    # Define test datasets in specified order
    box_datasets = [
        "Box",
        "Cardboard Box",
        "Generalisation Cardboard Box",
        "Pattern Box",
        "Generalisation Pattern Box",
    ]

    slope_datasets = [
        "Slope",
        "Larvik",
        "Generalisation Larvik",
        "Rv 4",
        "Generalisation Rv 4",
    ]

    # Create output directory
    output_dir = Path("experiments/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Box experiments comparison grid (5×6)...")
    create_comparison_grid(
        box_datasets,
        metrics_df,
        ratings_df,
        image_dir,
        output_dir / "figure_best_worst_comparison_box.png",
        figure_title="Box Experiments",
    )

    print("\nGenerating Slope experiments comparison grid (5×6)...")
    create_comparison_grid(
        slope_datasets,
        metrics_df,
        ratings_df,
        image_dir,
        output_dir / "figure_best_worst_comparison_slope.png",
        figure_title="Slope Experiments",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
