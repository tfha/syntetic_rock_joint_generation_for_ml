"""
Extract and visualize Slope experiments with boundary_localisation score of 2.

Creates a figure showing all experiments that scored 2 for boundary localisation,
with all 10 sample images for each experiment.
"""

import csv
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image


def load_ratings(csv_path: Path) -> list[dict]:
    """Load qualitative ratings from CSV."""
    ratings = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append(row)
    return ratings


def extract_slope_boundary_score_2(
    ratings: list[dict], image_dir: Path
) -> list[tuple[str, Path]]:
    """
    Extract Slope experiments with boundary_localisation score of 2.

    Returns list of (experiment_name, final_dir_path) tuples.
    """
    results = []

    for rating in ratings:
        # Check if it's a Slope experiment
        image_filename = rating["image"].lower()
        if "slope" not in image_filename:
            continue

        # Check if boundary_localisation score is 2
        if int(rating["boundary_localisation"]) != 2:
            continue

        # Parse directory path
        dir_name = rating["image"].replace("_progression.png", "")
        final_dir = image_dir / dir_name / "final"

        if final_dir.exists():
            experiment_name = dir_name
            results.append((experiment_name, final_dir))

    return results


def plot_slope_boundary_score_2(
    experiments: list[tuple[str, Path]], output_dir: Path
) -> None:
    """Create a figure showing all Slope experiments with boundary_localisation score 2."""

    if not experiments:
        print("No Slope experiments found with boundary_localisation score of 2.")
        return

    num_experiments = len(experiments)

    # Create figure with gridspec: 10 rows (for samples) × num_experiments columns
    fig = plt.figure(figsize=(4 * num_experiments, 16))
    gs = gridspec.GridSpec(10, num_experiments, figure=fig, hspace=0.25, wspace=0.3)

    fig.suptitle(
        "Slope Experiments with Boundary Localisation Score = 2\n(All 10 Samples per Experiment)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Color for score 2
    color = "#ff7f0e"

    for exp_idx, (experiment_name, final_dir) in enumerate(experiments):
        # Add experiment header
        fig.text(
            0.05 + (exp_idx + 0.5) * (0.9 / num_experiments),
            0.985,
            f"{experiment_name}\n(Score 2)",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "alpha": 0.7},
        )

        # Add all 10 samples
        for sample_idx in range(10):
            ax = fig.add_subplot(gs[sample_idx, exp_idx])

            sample_path = final_dir / f"sample_{sample_idx}.png"

            if sample_path.exists():
                try:
                    img = Image.open(sample_path)
                    ax.imshow(img)
                    ax.set_title(f"Sample {sample_idx}", fontsize=8, pad=2)
                except Exception:
                    ax.text(
                        0.5,
                        0.5,
                        "Error\nloading",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"sample_{sample_idx}.png\nnot found",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )

            ax.axis("off")

    output_path = output_dir / "slope_boundary_localisation_score_2.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Total experiments shown: {num_experiments}")
    for exp_name, _ in experiments:
        print(f"  - {exp_name}")
    plt.close()


def main():
    """Main execution."""
    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / "experiments" / "results" / "qualitative_ratings.csv"
    image_dir = repo_root / "experiments" / "results" / "images"
    output_dir = repo_root / "experiments" / "results" / "rubrics_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ratings from: {csv_path}")
    ratings = load_ratings(csv_path)

    print("\nExtracting Slope experiments with boundary_localisation score = 2...")
    experiments = extract_slope_boundary_score_2(ratings, image_dir)

    if experiments:
        print(f"Found {len(experiments)} experiment(s)")
        plot_slope_boundary_score_2(experiments, output_dir)
    else:
        print("No matching experiments found.")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
