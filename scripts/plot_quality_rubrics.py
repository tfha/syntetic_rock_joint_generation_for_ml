"""
Plot quality score rubrics with illustrative examples from qualitative ratings.

Organizes final-stage images from Larvik and Generalisation Larvik datasets
by quality criteria (geological_recognisability, joint_persistence,
boundary_localisation, false_positives) and score level (1-5).

Creates visualization plots to support rubric development.
"""

import csv
import textwrap
from collections import defaultdict
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


def filter_larvik_final_stage(ratings: list[dict]) -> list[dict]:
    """Filter for final stage Larvik or Generalisation Larvik images."""
    criteria = ["larvik", "generalisation_larvik"]
    filtered = []

    for rating in ratings:
        if rating["stage"] != "final":
            continue

        image_name = rating["image"].lower()
        if any(criterion in image_name for criterion in criteria):
            filtered.append(rating)

    return filtered


def organize_by_criterion_and_score(
    ratings: list[dict], image_dir: Path
) -> dict[str, dict[int, list]]:
    """
    Organize ratings by criterion and score level.

    Returns:
        dict mapping criterion -> {score: [ratings_list, final_dir_path]}
    """
    criteria = [
        "geological_recognisability",
        "joint_persistence",
        "boundary_localisation",
        "false_positives",
    ]

    organized: dict[str, dict[int, list]] = {}
    for criterion in criteria:
        organized[criterion] = defaultdict(list)

    for rating in ratings:
        # Parse image directory from filename
        # CSV has 'unet-finetune_box_10-20251211-1702_progression.png'
        # Directory is 'unet-finetune_box_10-20251211-1702/'
        image_filename = rating["image"]
        dir_name = image_filename.replace("_progression.png", "")

        model_dir = image_dir / dir_name
        final_dir = model_dir / "final"

        if not final_dir.exists():
            continue

        for criterion in criteria:
            score = int(rating[criterion])
            organized[criterion][score].append((rating, final_dir))

    return organized


def select_representative_images(
    organized: dict[str, dict[int, list]],
) -> dict[str, dict[int, tuple[str, str]]]:
    """
    Select one representative experiment for each criterion-score combination.

    Returns:
        dict mapping criterion -> {score: (final_dir_path, experiment_name)}
    """
    representative: dict[str, dict[int, tuple[str, str]]] = {}

    for criterion, scores_dict in organized.items():
        representative[criterion] = {}

        for score in range(1, 6):
            if score in scores_dict and scores_dict[score]:
                # Pick first available experiment for this score
                rating, final_dir = scores_dict[score][0]
                # Extract experiment name from image filename
                # e.g., 'unet-finetune_larvik_90-20251211-1626_progression.png' -> 'unet-finetune_larvik_90'
                image_filename = rating["image"]
                exp_full_name = image_filename.replace("_progression.png", "")

                # Remove timestamp (last 14 chars: -YYYYMMDD-HHMM)
                if len(exp_full_name) > 14 and exp_full_name[-14] == "-":
                    experiment_name = exp_full_name[:-14]
                else:
                    experiment_name = exp_full_name

                # Return the final_dir path so we can access all samples
                representative[criterion][score] = (str(final_dir), experiment_name)

    return representative


def plot_criterion_rubric(
    criterion: str, image_data: dict[int, tuple[str, str]], output_dir: Path
) -> None:
    """
    Plot all scores for a single criterion, showing all 10 samples for each score.

    image_data: dict mapping score -> (image_path, experiment_name)
                Note: image_path is actually final_dir from organize_by_criterion_and_score
    """
    fig = plt.figure(figsize=(26, 16))
    gs = gridspec.GridSpec(5, 10, figure=fig, hspace=0.20, wspace=0.20)

    fig.suptitle(
        f"Quality Rubric: {criterion.replace('_', ' ').title()} - All 10 Samples per Score",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    score_labels = [
        "Score 1\n(Poor)",
        "Score 2\n(Fair)",
        "Score 3\n(Good)",
        "Score 4\n(Very Good)",
        "Score 5\n(Excellent)",
    ]

    for score in range(1, 6):
        if score in image_data:
            final_dir_or_path, experiment_name = image_data[score]
            final_dir = Path(final_dir_or_path)

            # Load and display all 10 samples
            for sample_idx in range(10):
                row = sample_idx // 2  # rows 0-4 (5 rows for 10 samples)
                col = (score - 1) * 2 + (sample_idx % 2)  # 2 columns per score

                ax = fig.add_subplot(gs[row, col])

                sample_path = final_dir / f"sample_{sample_idx}.png"

                if sample_path.exists():
                    try:
                        img = Image.open(sample_path)
                        ax.imshow(img)
                        ax.set_title(f"Sample {sample_idx}", fontsize=10, pad=1)
                    except Exception:
                        ax.text(
                            0.5,
                            0.5,
                            "Error\nloading",
                            ha="center",
                            va="center",
                            fontsize=10,
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
        else:
            # No example for this score - fill with empty cells
            for sample_idx in range(10):
                row = sample_idx // 2
                col = (score - 1) * 2 + (sample_idx % 2)

                ax = fig.add_subplot(gs[row, col])
                ax.text(
                    0.5,
                    0.5,
                    "No example",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="lightgray",
                )
                ax.axis("off")

        # Add score header above each pair of columns
        score_col_center = 0.08 + (score - 1) * 0.18
        fig.text(
            score_col_center,
            0.985,
            score_labels[score - 1],
            ha="center",
            va="top",
            fontsize=16,
            color=colors[score - 1],
        )

        # Add experiment name below score header with word-based wrapping (no box)
        if score in image_data:
            _, experiment_name = image_data[score]
            # Wrap long experiment names breaking between words
            if len(experiment_name) > 30:
                wrapped_lines = textwrap.wrap(experiment_name, width=30)
                wrapped_name = "\n".join(wrapped_lines)
            else:
                wrapped_name = experiment_name
            fig.text(
                score_col_center,
                0.945,
                wrapped_name,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
                color="gray",
            )

    plt.savefig(output_dir / f"rubric_{criterion}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / f'rubric_{criterion}.png'}")
    plt.close()


def create_combined_rubric(
    organized: dict[str, dict[int, list]], image_dir: Path, output_dir: Path
) -> None:
    """
    Create a combined rubric with 4 subplots, one per criterion, each showing 5 scores.

    For boundary_localisation score 2 (not available in Larvik data),
    use sample_3 from unet-finetune_slope_10.
    """
    criteria = list(organized.keys())

    # Criterion descriptions
    criterion_descriptions = {
        "geological_recognisability": "How realistic and geologically plausible do the predicted joints appear?",
        "joint_persistence": "Are continuous joints properly connected without gaps or interruptions?",
        "boundary_localisation": "Are joint boundaries precise and is the thickness appropriate?",
        "false_positives": "How much spurious segmentation or noise is present? (1=very noisy, 5=very clean)",
    }

    # Score descriptions for each criterion
    score_descriptions = {
        "geological_recognisability": {
            1: "Unrealistic, implausible joint patterns.",
            2: "Partially realistic but with notable geological inconsistencies.",
            3: "Reasonably realistic with minor geological concerns.",
            4: "Realistic and mostly geologically plausible.",
            5: "Highly realistic and fully geologically plausible patterns.",
        },
        "joint_persistence": {
            1: "Joints frequently interrupted with major gaps and discontinuities.",
            2: "Joints partially continuous with several interruptions.",
            3: "Mostly continuous joints with minor interruptions.",
            4: "Continuous joints with minimal to no gaps.",
            5: "Perfectly continuous joints throughout.",
        },
        "boundary_localisation": {
            1: "Poorly defined boundaries, highly inappropriate thickness.",
            2: "Vague boundaries with thickness issues.",
            3: "Adequately defined boundaries with acceptable thickness.",
            4: "Well-defined boundaries with appropriate thickness.",
            5: "Precisely defined boundaries with perfect thickness.",
        },
        "false_positives": {
            1: "Extremely noisy with extensive spurious segmentation.",
            2: "Noisy with considerable spurious artifacts.",
            3: "Moderate noise with some spurious segmentation.",
            4: "Minimal noise with few spurious artifacts.",
            5: "Clean segmentation with negligible noise.",
        },
    }

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    score_labels = ["Score 1", "Score 2", "Score 3", "Score 4", "Score 5"]

    # Create figure with 4 subplots stacked vertically (4x1 layout)
    fig = plt.figure(figsize=(28, 28))
    fig.suptitle(
        "Quality Score Rubric - Four Criteria", fontsize=24, fontweight="bold", y=0.995
    )

    # Create 4 subplots with GridSpec (4 rows, 1 column)
    gs_main = gridspec.GridSpec(4, 1, figure=fig, hspace=0.4, wspace=0.1)

    for criterion_idx, criterion in enumerate(criteria):
        # Create subplot for this criterion
        row = criterion_idx
        col = 0
        ax_main = fig.add_subplot(gs_main[row, col])
        ax_main.axis("off")

        # Create nested GridSpec for this criterion's 5 score columns
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_main[row, col], hspace=0.05, wspace=0.15
        )

        # Define fixed positions for this row (working backwards from row spacing)
        row_top = 0.96 - (row * 0.24)
        row_bottom = row_top - 0.24

        # Fixed positions for score headers (consistent across all rows)
        score_header_y = row_top - 0.065

        # Add criterion title above score header with consistent gap
        description = criterion_descriptions.get(criterion, "")
        criterion_title = criterion.replace("_", " ").title()
        criterion_text = f"{criterion_title}\n{description}"

        # Position criterion text above score headers (with fixed gap of 0.025)
        criterion_y = score_header_y + 0.025
        fig.text(
            0.5,
            criterion_y,
            criterion_text,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            wrap=True,
        )

        # Create axes for each score in this criterion
        for score in range(1, 6):
            ax = fig.add_subplot(gs_inner[0, score - 1])

            # Add score label at fixed position
            score_desc = score_descriptions.get(criterion, {}).get(score, "")
            # Wrap description breaking between words
            if len(score_desc) > 20:
                wrapped_lines = textwrap.wrap(score_desc, width=20)
                wrapped_score_desc = "\n".join(wrapped_lines)
            else:
                wrapped_score_desc = score_desc

            score_header = f"{score_labels[score - 1]}\n{wrapped_score_desc}"

            # Calculate x position for this score column (consistent across all rows)
            col_width = 0.95 / 5
            score_x = 0.05 + (score - 0.5) * col_width

            # Position score header at fixed y position (same for all rows)
            fig.text(
                score_x,
                score_header_y,
                score_header,
                ha="center",
                va="top",
                fontsize=10,
                color=colors[score - 1],
                fontweight="bold",
            )

            # Load and display image
            image_loaded = False
            experiment_name = ""

            # Special handling for boundary_localisation score 2
            if criterion == "boundary_localisation" and score == 2:
                slope_dir_name = "unet-finetune_slope_10-20251211-1618"
                slope_final_dir = image_dir / slope_dir_name / "final"
                sample_path = slope_final_dir / "sample_3.png"

                if sample_path.exists():
                    try:
                        img = Image.open(sample_path)
                        ax.imshow(img)
                        experiment_name = "unet-finetune_slope_10 (sample_3)"
                        image_loaded = True
                    except Exception:
                        ax.text(
                            0.5,
                            0.5,
                            "Error loading",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )

            # Standard handling if not special case or special case failed
            if not image_loaded:
                if score in organized[criterion] and organized[criterion][score]:
                    rating, final_dir = organized[criterion][score][0]
                    sample_path = final_dir / "sample_0.png"

                    # Extract experiment name and remove timestamp
                    image_filename = rating["image"]
                    exp_full_name = image_filename.replace("_progression.png", "")
                    # Remove timestamp (last 14 chars: -YYYYMMDD-HHMM)
                    if len(exp_full_name) > 14 and exp_full_name[-14] == "-":
                        experiment_name = exp_full_name[:-14] + " (sample_0)"
                    else:
                        experiment_name = exp_full_name + " (sample_0)"

                    if sample_path.exists():
                        try:
                            img = Image.open(sample_path)
                            ax.imshow(img)
                            image_loaded = True
                        except Exception:
                            ax.text(
                                0.5,
                                0.5,
                                "Error loading",
                                ha="center",
                                va="center",
                                fontsize=9,
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Image not found",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="gray",
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No example",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="lightgray",
                    )

            # Add experiment name at bottom of row
            if image_loaded and experiment_name:
                if len(experiment_name) > 25:
                    wrapped_lines = textwrap.wrap(experiment_name, width=25)
                    wrapped_name = "\n".join(wrapped_lines)
                else:
                    wrapped_name = experiment_name

                exp_name_y = row_bottom + 0.015
                fig.text(
                    score_x,
                    exp_name_y,
                    wrapped_name,
                    ha="center",
                    va="top",
                    fontsize=8,
                    style="italic",
                    color="gray",
                )

            ax.set_aspect("equal")
            ax.axis("off")

    output_path = output_dir / "quality_rubric_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_single_criterion_detailed(
    criterion: str,
    organized: dict[str, dict[int, list]],
    image_dir: Path,
    output_dir: Path,
) -> None:
    """
    Plot a single criterion with 1 representative sample for each score (1-5).
    Layout: 5 columns (scores), each showing score header, image, experiment name, and sample number.
    """
    criterion_descriptions = {
        "geological_recognisability": "How realistic and geologically plausible do the predicted joints appear?",
        "joint_persistence": "Are continuous joints properly connected without gaps or interruptions?",
        "boundary_localisation": "Are joint boundaries precise and is the thickness appropriate?",
        "false_positives": "How much spurious segmentation or noise is present? (1=very noisy, 5=very clean)",
    }

    score_descriptions = {
        "geological_recognisability": {
            1: "Unrealistic, implausible joint patterns.",
            2: "Partially realistic but with notable geological inconsistencies.",
            3: "Reasonably realistic with minor geological concerns.",
            4: "Realistic and mostly geologically plausible.",
            5: "Highly realistic and fully geologically plausible patterns.",
        },
        "joint_persistence": {
            1: "Joints frequently interrupted with major gaps and discontinuities.",
            2: "Joints partially continuous with several interruptions.",
            3: "Mostly continuous joints with minor interruptions.",
            4: "Continuous joints with minimal to no gaps.",
            5: "Perfectly continuous joints throughout.",
        },
        "boundary_localisation": {
            1: "Poorly defined boundaries, highly inappropriate thickness.",
            2: "Vague boundaries with thickness issues.",
            3: "Adequately defined boundaries with acceptable thickness.",
            4: "Well-defined boundaries with appropriate thickness.",
            5: "Precisely defined boundaries with perfect thickness.",
        },
        "false_positives": {
            1: "Extremely noisy with extensive spurious segmentation.",
            2: "Noisy with considerable spurious artifacts.",
            3: "Moderate noise with some spurious segmentation.",
            4: "Minimal noise with few spurious artifacts.",
            5: "Clean segmentation with negligible noise.",
        },
    }

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    # Create figure: 5 columns (scores) x 1 row (samples) + headers/labels
    fig = plt.figure(figsize=(24, 5.3))

    # Main title
    criterion_title = criterion.replace("_", " ").title()
    description = criterion_descriptions.get(criterion, "")
    fig.suptitle(
        f"{criterion_title}\n{description}", fontsize=11, fontweight="bold", y=0.99
    )

    # Create 5 score columns with GridSpec - reduced spacing
    gs_main = gridspec.GridSpec(
        1,
        5,
        figure=fig,
        hspace=0.05,
        wspace=0.15,
        left=0.05,
        right=0.95,
        top=0.765,
        bottom=0.13,
    )

    for score in range(1, 6):
        ax = fig.add_subplot(gs_main[0, score - 1])

        # Score header with description - centered above image
        score_desc = score_descriptions.get(criterion, {}).get(score, "")
        if len(score_desc) > 25:
            wrapped_lines = textwrap.wrap(score_desc, width=25)
            wrapped_desc = "\n".join(wrapped_lines)
        else:
            wrapped_desc = score_desc

        score_header = f"Score {score}\n{wrapped_desc}"

        # Position score header - centered with image column
        # GridSpec: left=0.05, right=0.95, 5 columns
        # Each column width = 0.90 / 5 = 0.18, center offset = 0.05 + 0.09 = 0.14
        score_x = 0.14 + (score - 1) * 0.18
        fig.text(
            score_x,
            0.81,
            score_header,
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[score - 1],
            fontweight="bold",
        )

        # Load and display sample
        if score in organized[criterion] and organized[criterion][score]:
            rating, final_dir = organized[criterion][score][0]

            # Get experiment name
            image_filename = rating["image"]
            exp_full_name = image_filename.replace("_progression.png", "")
            if len(exp_full_name) > 14 and exp_full_name[-14] == "-":
                experiment_name = exp_full_name[:-14]
            else:
                experiment_name = exp_full_name

            # Load sample_0
            sample_path = final_dir / "sample_0.png"
            if sample_path.exists():
                try:
                    img = Image.open(sample_path)
                    ax.imshow(img)
                except OSError:
                    ax.text(
                        0.5, 0.5, "Error loading", ha="center", va="center", fontsize=9
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Not found",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=8,
                )

            # Experiment name and sample number below image
            info_text = f"{experiment_name}\n(sample_0)"
            if len(info_text) > 30:
                wrapped_lines = textwrap.wrap(info_text, width=30)
                info_text = "\n".join(wrapped_lines)

            fig.text(
                score_x,
                0.035,
                info_text,
                ha="center",
                va="top",
                fontsize=6.5,
                style="italic",
                color="gray",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No example",
                ha="center",
                va="center",
                color="lightgray",
                fontsize=9,
            )

        ax.axis("off")

    output_path = output_dir / f"single_criterion_{criterion}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution."""
    # Paths
    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / "experiments" / "results" / "qualitative_ratings.csv"
    image_dir = repo_root / "experiments" / "results" / "images"
    output_dir = repo_root / "experiments" / "results" / "rubrics_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ratings from: {csv_path}")
    ratings = load_ratings(csv_path)
    print(f"Total ratings: {len(ratings)}")

    print("\nFiltering for final stage Larvik/Generalisation Larvik...")
    filtered = filter_larvik_final_stage(ratings)
    print(f"Filtered ratings: {len(filtered)}")

    print("\nOrganizing by criterion and score...")
    organized = organize_by_criterion_and_score(filtered, image_dir)

    # Print coverage
    print("\nCoverage by criterion and score:")
    for criterion in organized:
        scores_with_examples = [s for s in range(1, 6) if s in organized[criterion]]
        print(f"  {criterion}: scores {scores_with_examples}")

    print("\nCreating criterion visualizations (1 sample per score)...")

    # Generate for all 4 criteria
    for criterion in organized:
        plot_single_criterion_detailed(criterion, organized, image_dir, output_dir)

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
