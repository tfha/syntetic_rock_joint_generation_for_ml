"""Generate comparison plot showing experiments with better epochs than final/best.

Creates a 22×N grid showing original, ground truth, and predictions at various epochs,
with red outlines highlighting the epochs identified as better than final/best.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Sample mapping (same as figure_best_worst_comparison)
SAMPLE_MAPPING = {
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


def parse_image_filename(filename: str) -> dict[str, str | float] | None:
    """Parse progression image filename to extract metadata."""
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
            "full_name": f"{model}-{strategy}_{experiment}_{proportion}",
        }

    return None


def normalize_experiment_name(experiment: str) -> str:
    """Normalize experiment names to match standard format."""
    normalized = experiment.replace("_", " ").title()
    if "Rv4" in normalized:
        normalized = normalized.replace("Rv4", "Rv 4")
    if "rv4" in experiment.lower():
        normalized = normalized.replace("Rv4", "Rv 4")
    return normalized


def load_experiments_with_better_epochs(csv_path: Path) -> pd.DataFrame:
    """Load experiments that have better epochs than final/best."""
    df = pd.read_csv(csv_path)

    # Filter for rows with better_epoch
    better_epochs = df[df["better_epoch"].notna() & (df["better_epoch"] != "")].copy()

    # Parse image filenames
    metadata_list = []
    for _, row in better_epochs.iterrows():
        metadata = parse_image_filename(row["image"])
        if metadata:
            metadata["better_epoch"] = row["better_epoch"]
            experiment_str = str(metadata["experiment"])
            metadata["experiment_normalized"] = normalize_experiment_name(
                experiment_str
            )
            metadata_list.append(metadata)

    return pd.DataFrame(metadata_list)


def find_experiment_directory(image_dir: Path, experiment_name: str) -> Path | None:
    """Find the experiment directory."""
    matching_folders = list(image_dir.glob(f"{experiment_name}-*"))

    if not matching_folders:
        print(f"Warning: No folder found for {experiment_name}")
        return None

    return matching_folders[0]


def crop_section(img: Image.Image, section: str) -> Image.Image:
    """Crop specific section from composite sample image.

    Crop coordinates from plot_best_worst_comparison.py:
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

    return img.crop(crop_coords[section])


def load_epoch_images(
    exp_dir: Path, sample_num: int, better_epochs: list[int], strategy: str
) -> list[tuple[str, Image.Image, bool]]:
    """Load prediction images for all available epochs for a specific sample.

    Returns list of (epoch_label, image, is_better) tuples.
    Also includes original and ground truth from one epoch.

    For finetune: marks stage2 epochs as better.
    For simplemixed: uses better_epochs list from CSV.
    """
    image_list = []

    # Get all epoch directories
    epoch_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])

    # Load original and ground truth from any epoch (they're the same)
    for epoch_dir in epoch_dirs:
        sample_file = epoch_dir / f"sample_{sample_num}.png"
        if sample_file.exists():
            try:
                img = Image.open(sample_file)
                original = crop_section(img, "original")
                gt = crop_section(img, "ground_truth")
                image_list.append(("Original", original, False))
                image_list.append(("Ground Truth", gt, False))
                break
            except Exception as e:
                print(f"Error loading original/GT from {sample_file}: {e}")
                continue

    # Load predictions from each epoch
    for epoch_dir in epoch_dirs:
        sample_file = epoch_dir / f"sample_{sample_num}.png"
        if not sample_file.exists():
            continue

        try:
            img = Image.open(sample_file)
            prediction = crop_section(img, "prediction")

            # Determine if this is a better epoch
            is_better = False
            epoch_label = epoch_dir.name

            if epoch_label == "final":
                epoch_label = "Final"
            elif epoch_label.startswith("epoch_"):
                # Parse full directory name for stage info
                dir_name = epoch_dir.name

                # Extract epoch number
                match = re.search(r"epoch_(\d+)", dir_name)
                if match:
                    epoch_num = int(match.group(1))

                    # For finetune: CSV uses epoch_13 = stage2_first, epoch_17 = stage2_fifth
                    # For simplemixed: use actual epoch numbers
                    if strategy == "finetune":
                        # Map CSV epoch codes to stage directories
                        if "stage2_first_epoch" in dir_name:
                            is_better = 13 in better_epochs
                        elif "stage2_fifth_epoch" in dir_name:
                            is_better = 17 in better_epochs
                        else:
                            # Regular epoch directories (e.g., epoch_5, epoch_10)
                            is_better = epoch_num in better_epochs
                    else:
                        # simplemixed: use actual epoch numbers from CSV
                        is_better = epoch_num in better_epochs

                    # Check for stage 2 information
                    if "stage2_first_epoch" in dir_name:
                        epoch_label = f"Epoch {epoch_num}\n(Stage 2 Start)"
                    elif "stage2_fifth_epoch" in dir_name:
                        epoch_label = f"Epoch {epoch_num}\n(Stage 2 5th)"
                    else:
                        epoch_label = f"Epoch {epoch_num}"

            image_list.append((epoch_label, prediction, is_better))

        except Exception as e:
            print(f"Error loading {sample_file}: {e}")
            continue

    return image_list


def add_red_border(img: Image.Image, border_width: int = 8) -> Image.Image:
    """Add a red border to an image."""
    # Create new image with border
    new_width = img.width + 2 * border_width
    new_height = img.height + 2 * border_width

    bordered_img = Image.new("RGB", (new_width, new_height), (255, 0, 0))
    bordered_img.paste(img, (border_width, border_width))

    return bordered_img


def get_actual_better_epochs(
    exp_dir: Path, better_epochs: list[int], strategy: str
) -> list[int]:
    """Get actual epoch numbers for plotting markers.

    For finetune: epoch_13 in CSV maps to stage2_first_epoch directory,
                  epoch_17 in CSV maps to stage2_fifth_epoch directory.
    For simplemixed: use epoch numbers directly.
    """
    if strategy != "finetune":
        return better_epochs

    actual_epochs = []

    # Map CSV epoch codes to actual epoch numbers from directories
    for epoch_dir in exp_dir.iterdir():
        if not epoch_dir.is_dir():
            continue

        dir_name = epoch_dir.name
        if "stage2_first_epoch" in dir_name and 13 in better_epochs:
            # Extract actual epoch number
            match = re.search(r"epoch_(\d+)", dir_name)
            if match:
                actual_epochs.append(int(match.group(1)))
        elif "stage2_fifth_epoch" in dir_name and 17 in better_epochs:
            match = re.search(r"epoch_(\d+)", dir_name)
            if match:
                actual_epochs.append(int(match.group(1)))
        elif dir_name.startswith("epoch_") and "stage2" not in dir_name:
            # Regular epoch directory (e.g., epoch_5, epoch_10)
            match = re.search(r"epoch_(\d+)", dir_name)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num in better_epochs:
                    actual_epochs.append(epoch_num)

    return actual_epochs


def load_experiment_metrics(
    metrics_dir: Path, model: str, strategy: str, experiment: str, proportion: str
) -> pd.DataFrame | None:
    """Load epoch metrics CSV for a specific experiment."""
    # Metrics are in mode=max subdirectory
    metrics_dir = metrics_dir / "mode=max"

    # Try different naming patterns:
    # 1. Finetune: model_strategy_experiment_proportion_metrics.csv (no timestamp, underscores)
    # 2. Simplemixed: model-strategy_experiment_proportion-timestamp_metrics.csv

    patterns = [
        f"{model}_{strategy}_{experiment}_{proportion}_metrics.csv",  # finetune
        f"{model}-{strategy}_{experiment}_{proportion}-*_metrics.csv",  # simplemixed
    ]

    for pattern in patterns:
        matching_files = list(metrics_dir.glob(pattern))
        if matching_files:
            try:
                df = pd.read_csv(matching_files[0])
                if "epoch" in df.columns:
                    return df
            except Exception as e:
                print(f"Error loading {matching_files[0].name}: {e}")
                continue

    return None


def load_all_proportions_metrics(
    metrics_dir: Path, model: str, strategy: str, experiment: str
) -> dict[str, pd.DataFrame]:
    """Load metrics for all proportions of a given experiment.

    Returns dict mapping proportion string to DataFrame.
    """
    metrics_dir = metrics_dir / "mode=max"
    all_metrics = {}

    # Define all possible proportions
    if strategy == "finetune":
        proportions = ["10", "30", "50", "70", "90"]
    else:  # simplemixed
        proportions = ["0", "10", "30", "50", "70", "90", "100"]

    for prop in proportions:
        patterns = [
            f"{model}_{strategy}_{experiment}_{prop}_metrics.csv",  # finetune
            f"{model}-{strategy}_{experiment}_{prop}-*_metrics.csv",  # simplemixed
        ]

        for pattern in patterns:
            matching_files = list(metrics_dir.glob(pattern))
            if matching_files:
                try:
                    df = pd.read_csv(matching_files[0])
                    if "epoch" in df.columns:
                        all_metrics[prop] = df
                        break
                except Exception:
                    continue

    return all_metrics


def parse_better_epochs(better_epoch_str: str) -> list[int]:
    """Parse better_epoch string to extract epoch numbers.

    Examples:
        "epoch_13" -> [13]
        "epoch_13, epoch_17" -> [13, 17]
    """
    epochs = []
    parts = better_epoch_str.split(",")
    for part in parts:
        part = part.strip()
        match = re.search(r"epoch_(\d+)", part)
        if match:
            epochs.append(int(match.group(1)))
    return epochs


def create_comparison_figure(
    experiments_df: pd.DataFrame,
    image_dir: Path,
    metrics_dir: Path,
    output_path: Path,
) -> None:
    """Create 22×N comparison figure showing better epoch predictions."""

    # Sort by test dataset for logical grouping
    dataset_order = [
        "Box",
        "Cardboard Box",
        "Generalisation Cardboard Box",
        "Pattern Box",
        "Generalisation Pattern Box",
        "Slope",
        "Larvik",
        "Generalisation Larvik",
        "Rv 4",
        "Generalisation Rv 4",
    ]

    experiments_df["dataset_rank"] = experiments_df["experiment_normalized"].map(
        {ds: i for i, ds in enumerate(dataset_order)}
    )
    experiments_df = experiments_df.sort_values("dataset_rank")

    print(f"\nProcessing {len(experiments_df)} experiments with better epochs...")

    # Load all epoch images for each experiment
    row_data_list = []
    for _, row in experiments_df.iterrows():
        full_name = row["full_name"]
        experiment_norm = row["experiment_normalized"]
        sample_num = SAMPLE_MAPPING.get(experiment_norm, 0)
        better_epochs = parse_better_epochs(row["better_epoch"])

        # Find experiment directory
        exp_dir = find_experiment_directory(image_dir, full_name)
        if exp_dir is None:
            continue

        # Extract strategy from full_name
        strategy = (
            full_name.split("-")[1].split("_")[0] if "-" in full_name else "unknown"
        )

        # Load all epoch images
        image_list = load_epoch_images(exp_dir, sample_num, better_epochs, strategy)
        if not image_list:
            print(f"Warning: No images found for {full_name}")
            continue

        # Extract model, strategy, experiment, proportion from full_name
        # Format: model-strategy_experiment_proportion
        # Example: unet-finetune_box_10 or unet-finetune_generalisation_box_10
        parts = full_name.split("-")
        model = ""
        strategy_parsed = ""
        experiment = ""
        proportion = ""

        if len(parts) == 2:
            model = parts[0]
            rest = parts[1]  # strategy_experiment_proportion
            rest_parts = rest.split("_")
            if len(rest_parts) >= 2:
                strategy_parsed = rest_parts[0]
                proportion = rest_parts[-1]
                # Everything between strategy and proportion is the experiment name
                experiment = "_".join(rest_parts[1:-1])
                experiment_label = (
                    f"{model}_{strategy_parsed}_{experiment}_{proportion}"
                )
            else:
                experiment_label = full_name
        else:
            experiment_label = full_name

        row_data_list.append(
            {
                "experiment": full_name,
                "dataset": experiment_norm,
                "label": experiment_label,
                "images": image_list,
                "model": model,
                "strategy": strategy_parsed,
                "experiment_name": experiment,
                "proportion": proportion,
                "better_epochs": better_epochs,
            }
        )

    if not row_data_list:
        print("No images loaded!")
        return

    # Determine column count from first experiment + 2 for metrics plots
    max_img_cols = max(len(rd["images"]) for rd in row_data_list)
    n_rows = len(row_data_list)
    total_cols = max_img_cols + 2  # Add 2 columns for dice_joints and loss

    print(f"Creating {n_rows}×{total_cols} comparison grid...")

    # Create figure with extra width for metric plots
    fig_width = max_img_cols * 2.0 + 2 * 3.0  # Wider for metrics
    fig_height = n_rows * 2.0

    fig, axes = plt.subplots(
        n_rows, total_cols, figsize=(fig_width, fig_height), squeeze=False
    )

    # Populate grid
    for row_idx, row_data in enumerate(row_data_list):
        images = row_data["images"]

        # Sort images: Original, Ground Truth, then epochs (5, 10, 13, 17), then Final
        def sort_key(item):
            label = item[0]
            if label == "Original":
                return 0
            elif label == "Ground Truth":
                return 1
            elif label == "Final":
                return 999
            elif label.startswith("Epoch"):
                return 2 + int(label.split()[1])
            else:
                return 500

        sorted_images = sorted(images, key=sort_key)

        # Plot images in first columns
        for col_idx in range(max_img_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sorted_images):
                label, img, is_better = sorted_images[col_idx]

                # Add red border if this is a better epoch
                if is_better:
                    img = add_red_border(img, border_width=8)

                ax.imshow(img)
                ax.set_title(label, fontsize=9)

            ax.axis("off")

        # Load metrics for current proportion and all other proportions
        metrics_df = load_experiment_metrics(
            metrics_dir,
            row_data["model"],
            row_data["strategy"],
            row_data["experiment_name"],
            row_data["proportion"],
        )

        all_proportions_metrics = load_all_proportions_metrics(
            metrics_dir,
            row_data["model"],
            row_data["strategy"],
            row_data["experiment_name"],
        )

        # Get actual epoch numbers for markers
        exp_dir = find_experiment_directory(image_dir, row_data["experiment"])
        actual_better_epochs = []
        if exp_dir is not None:
            actual_better_epochs = get_actual_better_epochs(
                exp_dir, row_data["better_epochs"], row_data["strategy"]
            )

        if metrics_df is not None:
            # Plot dice_joints in second-to-last column
            ax_dice = axes[row_idx, max_img_cols]
            if "val_dice_joints" in metrics_df.columns:
                # Plot all other proportions in grey (background)
                for prop, prop_df in all_proportions_metrics.items():
                    if (
                        prop != row_data["proportion"]
                        and "val_dice_joints" in prop_df.columns
                    ):
                        ax_dice.plot(
                            prop_df["epoch"],
                            prop_df["val_dice_joints"],
                            color="lightgrey",
                            linewidth=1.0,
                            alpha=0.5,
                            zorder=1,
                        )

                # Plot current proportion in blue (foreground)
                ax_dice.plot(
                    metrics_df["epoch"],
                    metrics_df["val_dice_joints"],
                    "b-",
                    linewidth=2.0,
                    zorder=3,
                )

                # Add markers at better epochs
                for better_epoch in actual_better_epochs:
                    epoch_data = metrics_df[metrics_df["epoch"] == better_epoch]
                    if not epoch_data.empty:
                        ax_dice.scatter(
                            epoch_data["epoch"],
                            epoch_data["val_dice_joints"],
                            color="red",
                            s=80,
                            zorder=5,
                            marker="o",
                            edgecolors="black",
                            linewidths=1.5,
                        )

                ax_dice.set_xlabel("Epoch", fontsize=8)
                ax_dice.set_ylabel("Val Dice Joints", fontsize=8)
                ax_dice.tick_params(labelsize=7)
                ax_dice.grid(True, alpha=0.3)
            else:
                ax_dice.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax_dice.axis("off")

            # Plot loss in last column
            ax_loss = axes[row_idx, max_img_cols + 1]
            if "val_loss" in metrics_df.columns:
                # Plot all other proportions in grey (background)
                for prop, prop_df in all_proportions_metrics.items():
                    if prop != row_data["proportion"] and "val_loss" in prop_df.columns:
                        ax_loss.plot(
                            prop_df["epoch"],
                            prop_df["val_loss"],
                            color="lightgrey",
                            linewidth=1.0,
                            alpha=0.5,
                            zorder=1,
                        )

                # Plot current proportion (foreground)
                ax_loss.plot(
                    metrics_df["epoch"],
                    metrics_df["val_loss"],
                    "r-",
                    linewidth=2.0,
                    zorder=3,
                )

                # Add markers at better epochs
                for better_epoch in actual_better_epochs:
                    epoch_data = metrics_df[metrics_df["epoch"] == better_epoch]
                    if not epoch_data.empty:
                        ax_loss.scatter(
                            epoch_data["epoch"],
                            epoch_data["val_loss"],
                            color="red",
                            s=80,
                            zorder=5,
                            marker="o",
                            edgecolors="black",
                            linewidths=1.5,
                        )

                ax_loss.set_xlabel("Epoch", fontsize=8)
                ax_loss.set_ylabel("Val Loss", fontsize=8)
                ax_loss.tick_params(labelsize=7)
                ax_loss.grid(True, alpha=0.3)
            else:
                ax_loss.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax_loss.axis("off")
        else:
            # No metrics available
            for col_offset in [0, 1]:
                ax = axes[row_idx, max_img_cols + col_offset]
                ax.text(0.5, 0.5, "No metrics", ha="center", va="center", fontsize=8)
                ax.axis("off")

        # Add experiment label (model_strategy_proportion) to left
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            row_data["label"],
            transform=axes[row_idx, 0].transAxes,
            fontsize=8,
            va="center",
            ha="right",
            fontweight="bold",
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def main() -> None:
    """Generate better epoch comparison plot."""
    csv_path = Path("experiments/results/qualitative_ratings.csv")
    image_dir = Path("experiments/results/images")
    metrics_dir = Path("experiments/results/metrics")
    output_dir = Path("experiments/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiments with better epochs...")
    experiments_df = load_experiments_with_better_epochs(csv_path)
    print(f"Found {len(experiments_df)} experiments with better epochs")

    create_comparison_figure(
        experiments_df,
        image_dir,
        metrics_dir,
        output_dir / "figure_better_epoch_comparisons.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
