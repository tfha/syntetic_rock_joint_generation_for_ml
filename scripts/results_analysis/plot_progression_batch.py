"""Batch create progression plots for all job directories.

This script processes a directory containing multiple job result folders and
automatically generates progression plots for each job. It detects the training
strategy (finetune or simplemixed) from the folder name and applies the
appropriate epoch selection.
"""

import argparse
from pathlib import Path

from plot_progression import plot_progression_grid


def detect_strategy(folder_name: str) -> str | None:
    """Detect training strategy from folder name.

    Args:
        folder_name: Name of the job folder

    Returns:
        "finetune" or "simplemixed" if detected, None otherwise
    """
    folder_lower = folder_name.lower()
    if "finetune" in folder_lower:
        return "finetune"
    elif "simplemixed" in folder_lower:
        return "simplemixed"
    return None


def find_metrics_file(job_folder: Path, metrics_dir: Path) -> Path | None:
    """Find the metrics CSV file for a job.

    Args:
        job_folder: Path to the job folder in image directory
        metrics_dir: Path to the directory containing metrics CSV files

    Returns:
        Path to metrics CSV file if found, None otherwise
    """
    job_name = job_folder.name

    # Remove timestamp from job name if present (e.g., -20251210-0957)
    # This is in the image folder names but not in metrics file names
    base_name = job_name
    if "-20" in job_name:  # Has timestamp like -20251209-1543
        # Remove the timestamp portion
        parts = job_name.rsplit("-", 2)
        if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 4:
            base_name = parts[0]

    # For finetune: convert hyphens to underscores
    # Example: deeplabv3plus-finetune_cardboard_box_10 -> deeplabv3plus_finetune_cardboard_box_10
    if "finetune" in base_name.lower():
        # Replace hyphens before strategy name with underscores
        base_name = base_name.replace("-finetune", "_finetune")
        base_name = base_name.replace("deeplabv3plus-", "deeplabv3plus_")
        base_name = base_name.replace("unet-", "unet_")

    # Try exact match without timestamp
    exact_match = metrics_dir / f"{base_name}_metrics.csv"
    if exact_match.exists():
        return exact_match

    # For simplemixed: metrics files may have different timestamps than image folders
    # Try glob pattern matching
    if "simplemixed" in job_name.lower():
        # Try to match the base pattern ignoring timestamps
        pattern = f"{base_name}*_metrics.csv"
        matches = list(metrics_dir.glob(pattern))
        if matches:
            return matches[0]

    print(f"Warning: No metrics file found for {job_name}")
    print(f"  Tried: {base_name}_metrics.csv")
    return None


def process_all_jobs(
    image_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
    num_samples: int,
) -> None:
    """Process all job folders and create progression plots.

    Args:
        image_dir: Directory containing job folders with images
        metrics_dir: Directory containing metrics CSV files
        output_dir: Directory to save output plots
        num_samples: Number of samples to plot per job
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all job folders (subdirectories in image_dir)
    job_folders = [d for d in image_dir.iterdir() if d.is_dir()]

    if not job_folders:
        print(f"No job folders found in {image_dir}")
        return

    print(f"Found {len(job_folders)} job folders")

    processed_count = 0
    skipped_count = 0

    for job_folder in sorted(job_folders):
        job_name = job_folder.name
        print(f"\nProcessing: {job_name}")

        # Detect strategy from folder name
        strategy = detect_strategy(job_name)
        if strategy is None:
            print("  Skipping: Cannot detect strategy (finetune/simplemixed)")
            skipped_count += 1
            continue

        print(f"  Detected strategy: {strategy}")

        # Find metrics file
        metrics_path = find_metrics_file(job_folder, metrics_dir)
        if metrics_path is None:
            print("  Skipping: No metrics file found")
            skipped_count += 1
            continue

        print(f"  Using metrics: {metrics_path.name}")

        # Set output path
        output_path = output_dir / f"{job_name}_progression.png"

        # Check if images exist
        epoch_folders = list(job_folder.glob("epoch_*"))
        if not epoch_folders:
            print("  Skipping: No epoch folders found")
            skipped_count += 1
            continue

        # Create progression plot
        try:
            plot_progression_grid(
                image_dir=job_folder,
                metrics_path=metrics_path,
                output_path=output_path,
                strategy=strategy,
                num_samples=num_samples,
            )
            print(f"  ✓ Saved: {output_path.name}")
            processed_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            skipped_count += 1
            continue

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Processed: {processed_count} plots")
    print(f"  Skipped: {skipped_count} jobs")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch create progression plots for all job folders"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing job folders with images",
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
        default=Path("experiments/results/plots/progression"),
        help="Output directory for plots (default: experiments/results/plots/progression)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to plot per job (default: 10)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if not args.metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {args.metrics_dir}")

    # Process all jobs
    process_all_jobs(
        image_dir=args.image_dir,
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
