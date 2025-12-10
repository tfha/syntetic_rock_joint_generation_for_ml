"""Plot training progression showing model predictions at different checkpoints.

This script creates publication-quality figures showing how model predictions
improve during training. Each figure shows multiple test samples (rows) across
different training epochs (columns), with Dice scores annotated.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def read_metrics_for_epochs(
    metrics_path: Path, epoch_numbers: list[int]
) -> dict[int, float]:
    """Read Dice scores for specific epochs from metrics CSV.

    Args:
        metrics_path: Path to the metrics CSV file
        epoch_numbers: List of epoch numbers to extract scores for

    Returns:
        Dictionary mapping epoch number to Dice score
    """
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found: {metrics_path}")
        return {}

    try:
        df = pd.read_csv(metrics_path)
        # Check for required columns
        if "epoch" not in df.columns:
            print(f"Warning: epoch column not found in {metrics_path}")
            return {}

        # Try to find the appropriate Dice score column
        dice_col = None
        for col_name in ["val_dice_joints", "best_val_dice_joint", "val_dice"]:
            if col_name in df.columns:
                dice_col = col_name
                break

        if dice_col is None:
            print(f"Warning: No Dice score column found in {metrics_path}")
            return {}

        epoch_scores = {}
        for epoch_num in epoch_numbers:
            epoch_data = df[df["epoch"] == epoch_num]
            if not epoch_data.empty:
                # Get the validation Dice score for this epoch
                score = epoch_data[dice_col].iloc[0]
                epoch_scores[epoch_num] = score

        return epoch_scores
    except Exception as e:
        print(f"Error reading metrics from {metrics_path}: {e}")
        return {}


def plot_progression_grid(
    image_dir: Path,
    metrics_path: Path,
    output_path: Path,
    strategy: str,
    num_samples: int = 5,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Create a progression grid showing model predictions over training.

    Args:
        image_dir: Directory containing epoch folders with images
        metrics_path: Path to metrics CSV file
        output_path: Where to save the output figure
        strategy: Training strategy ('simplemixed' or 'finetune')
        num_samples: Number of test samples to show (rows)
        figsize: Figure size (width, height) in inches. If None, auto-calculated
    """
    # Determine epoch folders based on strategy
    if strategy == "finetune":
        epoch_folders = [
            "epoch_5",
            "epoch_10",
            "epoch_13_stage2_first_epoch",
            "epoch_17_stage2_fifth_epoch",
            "final",
        ]
        epoch_numbers = [5, 10, 13, 17, None]  # None for final
    else:  # simplemixed
        epoch_folders = ["epoch_5", "epoch_10", "final"]
        epoch_numbers = [5, 10, None]

    # Check which epoch folders exist
    available_folders = []
    available_epochs = []
    for folder, epoch_num in zip(epoch_folders, epoch_numbers, strict=False):
        folder_path = image_dir / folder
        if folder_path.exists():
            available_folders.append(folder)
            available_epochs.append(epoch_num)
        else:
            print(f"Warning: Epoch folder not found: {folder_path}")

    if not available_folders:
        print(f"Error: No epoch folders found in {image_dir}")
        return

    num_epochs = len(available_folders)

    # Read Dice scores from metrics
    epoch_scores = {}
    if metrics_path.exists():
        valid_epoch_nums = [e for e in available_epochs if e is not None]
        epoch_scores = read_metrics_for_epochs(metrics_path, valid_epoch_nums)

    # Get list of available samples from first epoch folder
    first_epoch_dir = image_dir / available_folders[0]
    sample_files = sorted(first_epoch_dir.glob("sample_*.png"))[:num_samples]

    if not sample_files:
        print(f"Error: No sample images found in {first_epoch_dir}")
        return

    actual_num_samples = len(sample_files)
    print(
        f"Creating progression plot with {actual_num_samples} samples × {num_epochs} epochs"
    )

    # Calculate figure size if not provided
    if figsize is None:
        # Each cell should be reasonably sized for the composite images
        cell_height = 2.5
        cell_width = 3.5
        figsize = (cell_width * num_epochs + 1, cell_height * actual_num_samples + 1)

    # Create figure and axes
    fig, axes = plt.subplots(
        actual_num_samples,
        num_epochs,
        figsize=figsize,
        squeeze=False,
    )

    # Plot images
    for row, sample_file in enumerate(sample_files):
        sample_name = sample_file.name

        for col, (epoch_folder, epoch_num) in enumerate(
            zip(available_folders, available_epochs, strict=False)
        ):
            ax = axes[row, col]

            # Load and display image
            image_path = image_dir / epoch_folder / sample_name
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    ax.imshow(img)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        "Image\nError",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.axis("off")

            # Add column titles (epoch + score) on first row
            if row == 0:
                if epoch_num is None:
                    title = "Final"
                else:
                    title = f"Epoch {epoch_num}"

                # Add Dice score if available
                if epoch_num in epoch_scores:
                    dice = epoch_scores[epoch_num]
                    title += f"\nDice: {dice:.3f}"

                ax.set_title(title, fontsize=10, fontweight="bold")

            # Add row labels (sample number) on first column
            if col == 0:
                ax.set_ylabel(
                    f"Sample {row}",
                    fontsize=9,
                    rotation=90,
                    labelpad=10,
                )

    # Add overall title with job information
    job_display_name = image_dir.name
    fig.suptitle(
        f"Training Progression: {job_display_name}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved progression plot to {output_path}")
    plt.close()


def main() -> None:
    """Generate progression plots from downloaded images."""
    parser = argparse.ArgumentParser(
        description="Create training progression plots from downloaded images"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing job folders with epoch images",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="Job display name (folder name under image-dir)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["simplemixed", "finetune"],
        help="Training strategy",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("experiments/results/metrics/mode=max"),
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/plots/progression"),
        help="Output directory for progression plots",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples to include in the plot",
    )

    args = parser.parse_args()

    # Construct paths
    job_image_dir = args.image_dir / args.job_name
    if not job_image_dir.exists():
        print(f"Error: Job image directory not found: {job_image_dir}")
        return

    # Find corresponding metrics file
    # Try with full display name first, then try without timestamp
    metrics_file = args.metrics_dir / f"{args.job_name}_metrics.csv"
    if not metrics_file.exists():
        # Try alternate naming (without timestamp for finetune jobs)
        # e.g., "deeplabv3plus-finetune_generalisation_pattern_box_10-20251210-0957"
        # becomes "deeplabv3plus_finetune_generalisation_pattern_box_10"
        name_parts = args.job_name.rsplit("-", 2)  # Split off last 2 parts (date-time)
        if len(name_parts) == 3:
            alt_name = name_parts[0].replace("-", "_")
            metrics_file = args.metrics_dir / f"{alt_name}_metrics.csv"
            if metrics_file.exists():
                print(f"Found metrics file: {metrics_file.name}")

    # Output path
    output_file = args.output_dir / f"{args.job_name}_progression.png"

    # Create progression plot
    plot_progression_grid(
        image_dir=job_image_dir,
        metrics_path=metrics_file,
        output_path=output_file,
        strategy=args.strategy,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
