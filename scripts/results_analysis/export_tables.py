"""Export numerical values from figure_all_experiments_grid and figure_qualitative_ratings_grid to CSV.

Exports per-experiment averages across proportions:
- Metrics: val_dice_joints (best epoch for finetune, final for simplemixed)
- Qualitative: four criteria + mean quality score
- Grouping: 5×2 layout (Box types | Slope types)
- Models: unet, deeplabv3plus
- Strategies: simplemixed, finetune
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def parse_filename(filename: str) -> dict[str, str | float] | None:
    """Parse metrics filename to extract metadata."""
    name = filename.replace("_metrics.csv", "")

    # Pattern 1: simplemixed with hyphens and timestamp
    match = re.match(r"(unet|deeplabv3plus)-simplemixed_(.+)_(\d+)-\d{8}-\d{4}$", name)
    if match:
        model, experiment, proportion = match.groups()
        return {
            "model": model,
            "strategy": "simplemixed",
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
        }

    # Pattern 2: finetune with underscores
    match = re.match(r"(unet|deeplabv3plus)_(finetune|simplemixed)_(.+)_(\d+)$", name)
    if match:
        model, strategy, experiment, proportion = match.groups()
        return {
            "model": model,
            "strategy": strategy,
            "experiment": experiment,
            "proportion": float(proportion) / 100.0,
        }

    return None


def get_best_val_dice_joints(csv_path: Path) -> float | None:
    """Get best val_dice_joints from metrics CSV (max for all strategies)."""
    try:
        df = pd.read_csv(csv_path)
        if "val_dice_joints" in df.columns:
            return df["val_dice_joints"].max()
    except Exception:
        pass
    return None


def load_metrics_data(metrics_dir: Path) -> pd.DataFrame:
    """Load all metrics from CSV files."""
    data = []

    for csv_file in (metrics_dir / "mode=max").glob("*.csv"):
        metadata = parse_filename(csv_file.name)
        if metadata is None:
            continue

        best_dice = get_best_val_dice_joints(csv_file)
        if best_dice is None:
            continue

        data.append(
            {
                "model": metadata["model"],
                "strategy": metadata["strategy"],
                "experiment": metadata["experiment"],
                "proportion": metadata["proportion"],
                "val_dice_joints": best_dice,
            }
        )

    return pd.DataFrame(data)


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
        }

    return None


def normalize_experiment_name(exp: str) -> str:
    """Normalize experiment names for grouping."""
    mappings = {
        "box": "Box",
        "pattern_box": "Pattern Box",
        "cardboard_box": "Cardboard Box",
        "generalisation_pattern_box": "Gen. Pattern Box",
        "generalization_pattern_box": "Gen. Pattern Box",
        "generalisation_cardboard_box": "Gen. Cardboard Box",
        "generalization_cardboard_box": "Gen. Cardboard Box",
        "slope": "Slope",
        "larvik": "Larvik",
        "rv4": "Rv 4",
        "generalisation_larvik": "Gen. Larvik",
        "generalization_larvik": "Gen. Larvik",
        "generalisation_rv4": "Gen. Rv 4",
        "generalization_rv4": "Generalisation Rv 4",
    }
    return mappings.get(exp, exp)


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

    # Parse image filenames to extract metadata
    metadata_list = []
    for _, row in df.iterrows():
        metadata = parse_image_filename(row["image"])
        if metadata:
            metadata["geological_recognisability"] = row["geological_recognisability"]
            metadata["joint_persistence"] = row["joint_persistence"]
            metadata["boundary_localisation"] = row["boundary_localisation"]
            metadata["false_positives"] = row["false_positives"]
            metadata["mean_quality_score"] = row[criteria_columns].mean()
            metadata_list.append(metadata)

    if not metadata_list:
        return pd.DataFrame()

    result_df = pd.DataFrame(metadata_list)

    # Normalize experiment names
    result_df["experiment_normalized"] = result_df["experiment"].apply(
        normalize_experiment_name
    )

    return result_df


def compute_averages(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    """Compute averages across proportions for each experiment."""
    # Group by model, strategy, experiment and average across proportions
    grouped = df.groupby(group_cols)[value_cols].mean().reset_index()
    return grouped


def export_to_csv(
    metrics_df: pd.DataFrame,
    qualitative_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export averaged data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiment order (matching 5×2 grid)
    experiment_order = [
        "Box",
        "Pattern Box",
        "Cardboard Box",
        "Gen. Pattern Box",
        "Gen. Cardboard Box",
        "Slope",
        "Larvik",
        "Rv 4",
        "Gen. Larvik",
        "Gen. Rv 4",
    ]

    # Compute averages for metrics (across proportions)
    if not metrics_df.empty:
        metrics_avg = compute_averages(
            metrics_df,
            ["model", "strategy", "experiment_normalized"],
            ["val_dice_joints"],
        )

        # Sort by experiment order, then strategy (finetune first), then model (unet first)
        metrics_avg["exp_order"] = metrics_avg["experiment_normalized"].map(
            {exp: i for i, exp in enumerate(experiment_order)}
        )
        metrics_avg["strat_order"] = metrics_avg["strategy"].map(
            {"finetune": 0, "simplemixed": 1}
        )
        metrics_avg["model_order"] = metrics_avg["model"].map(
            {"unet": 0, "deeplabv3plus": 1}
        )
        metrics_avg = metrics_avg.sort_values(
            ["exp_order", "strat_order", "model_order"]
        ).drop(columns=["exp_order", "strat_order", "model_order"])

        # Format dice to 3 decimals
        metrics_avg["val_dice_joints"] = metrics_avg["val_dice_joints"].round(3)

        # Save metrics
        metrics_path = output_dir / "table_metrics_averaged.csv"
        metrics_avg.to_csv(metrics_path, index=False)
        print(f"Saved metrics table: {metrics_path}")
        print(f"\nMetrics preview:\n{metrics_avg.head(15)}")

    # Compute averages for qualitative (across proportions)
    if not qualitative_df.empty:
        qual_avg = compute_averages(
            qualitative_df,
            ["model", "strategy", "experiment_normalized"],
            [
                "geological_recognisability",
                "joint_persistence",
                "boundary_localisation",
                "false_positives",
                "mean_quality_score",
            ],
        )

        # Sort by experiment order, then strategy (finetune first), then model (unet first)
        qual_avg["exp_order"] = qual_avg["experiment_normalized"].map(
            {exp: i for i, exp in enumerate(experiment_order)}
        )
        qual_avg["strat_order"] = qual_avg["strategy"].map(
            {"finetune": 0, "simplemixed": 1}
        )
        qual_avg["model_order"] = qual_avg["model"].map({"unet": 0, "deeplabv3plus": 1})
        qual_avg = qual_avg.sort_values(
            ["exp_order", "strat_order", "model_order"]
        ).drop(columns=["exp_order", "strat_order", "model_order"])

        # Format quality scores to 1 decimal
        for col in [
            "geological_recognisability",
            "joint_persistence",
            "boundary_localisation",
            "false_positives",
            "mean_quality_score",
        ]:
            qual_avg[col] = qual_avg[col].round(1)

        # Save qualitative
        qual_path = output_dir / "table_qualitative_averaged.csv"
        qual_avg.to_csv(qual_path, index=False)
        print(f"\nSaved qualitative table: {qual_path}")
        print(f"\nQualitative preview:\n{qual_avg.head(15)}")

    # Merge metrics and qualitative into simplified combined table
    # Only include dice and mean quality score
    if not metrics_df.empty and not qualitative_df.empty:
        combined_simple = pd.merge(
            metrics_avg[
                ["model", "strategy", "experiment_normalized", "val_dice_joints"]
            ],
            qual_avg[
                ["model", "strategy", "experiment_normalized", "mean_quality_score"]
            ],
            on=["model", "strategy", "experiment_normalized"],
            how="outer",
        )

        # Rename columns to match LaTeX headers
        combined_simple = combined_simple.rename(
            columns={
                "model": "Model",
                "strategy": "Strategy",
                "experiment_normalized": "Experiment",
                "val_dice_joints": "Mean val dice joints",
                "mean_quality_score": "Mean quality score",
            }
        )

        # Sort by experiment order, then strategy (finetune first), then model (unet first)
        combined_simple["exp_order"] = combined_simple["Experiment"].map(
            {exp: i for i, exp in enumerate(experiment_order)}
        )
        combined_simple["strat_order"] = combined_simple["Strategy"].map(
            {"finetune": 0, "simplemixed": 1}
        )
        combined_simple["model_order"] = combined_simple["Model"].map(
            {"unet": 0, "deeplabv3plus": 1}
        )
        combined_simple = combined_simple.sort_values(
            ["exp_order", "strat_order", "model_order"]
        ).drop(columns=["exp_order", "strat_order", "model_order"])

        # Save simplified combined table
        combined_path = output_dir / "table_combined_averaged.csv"
        combined_simple.to_csv(combined_path, index=False)
        print(f"\nSaved combined table: {combined_path}")
        print(f"\nCombined preview:\n{combined_simple.head(15)}")


def main() -> None:
    """Export averaged tables to CSV."""
    # Paths
    metrics_dir = Path("experiments/results/metrics")
    qualitative_csv = Path("experiments/results/qualitative_ratings.csv")
    output_dir = Path("experiments/results/tables")

    print("=" * 70)
    print("Exporting averaged tables to CSV")
    print("=" * 70)

    # Load metrics data
    print("\nLoading metrics data...")
    metrics_df = load_metrics_data(metrics_dir)
    if not metrics_df.empty:
        # Normalize experiment names
        metrics_df["experiment_normalized"] = metrics_df["experiment"].apply(
            normalize_experiment_name
        )
        print(f"Loaded {len(metrics_df)} metric entries")
        print(f"Models: {sorted(metrics_df['model'].unique())}")
        print(f"Strategies: {sorted(metrics_df['strategy'].unique())}")
        print(f"Experiments: {sorted(metrics_df['experiment_normalized'].unique())}")

    # Load qualitative data
    print("\nLoading qualitative ratings...")
    qualitative_df = load_qualitative_ratings(qualitative_csv)
    if not qualitative_df.empty:
        print(f"Loaded {len(qualitative_df)} qualitative entries")
        print(f"Models: {sorted(qualitative_df['model'].unique())}")
        print(f"Strategies: {sorted(qualitative_df['strategy'].unique())}")
        print(
            f"Experiments: {sorted(qualitative_df['experiment_normalized'].unique())}"
        )

    # Export to CSV
    print("\n" + "=" * 70)
    print("Exporting averaged data to CSV...")
    print("=" * 70)
    export_to_csv(metrics_df, qualitative_df, output_dir)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
