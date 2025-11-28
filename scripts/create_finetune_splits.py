"""
Create dataset splits for Wachter et al. finetuning experiments.

This script generates train/test splits following the experimental design:
- BOX experiments: varying ratios of synthetic/real for training
- SLOPE experiments: varying ratios of synthetic/real for training
- Test set: always 100% real data (fixed size)

Usage:
    python scripts/create_finetune_splits.py
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from ml_segmentation.data_loading import get_data_files, get_datasets_prefixes

console = Console()


# Wachter et al. (2026) Experimental design from Table 1
# Simple Mixed (SM) strategy - train on shuffled synthetic+real
SIMPLEMIXED_EXPERIMENTS_BOX = [
    {
        "name": "simplemixed_box_0",
        "percent_real": 0,
        "train_synthetic": 180,
        "train_real": 0,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_10",
        "percent_real": 10,
        "train_synthetic": 162,
        "train_real": 18,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_30",
        "percent_real": 30,
        "train_synthetic": 126,
        "train_real": 54,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_50",
        "percent_real": 50,
        "train_synthetic": 90,
        "train_real": 90,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_70",
        "percent_real": 70,
        "train_synthetic": 54,
        "train_real": 126,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_90",
        "percent_real": 90,
        "train_synthetic": 18,
        "train_real": 162,
        "test_real": 20,
    },
    {
        "name": "simplemixed_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 180,
        "test_real": 20,
    },
]

SIMPLEMIXED_EXPERIMENTS_SLOPE = [
    {
        "name": "simplemixed_slope_0",
        "percent_real": 0,
        "train_synthetic": 2700,
        "train_real": 0,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_10",
        "percent_real": 10,
        "train_synthetic": 2430,
        "train_real": 270,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_30",
        "percent_real": 30,
        "train_synthetic": 1890,
        "train_real": 810,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_50",
        "percent_real": 50,
        "train_synthetic": 1350,
        "train_real": 1350,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_70",
        "percent_real": 70,
        "train_synthetic": 810,
        "train_real": 1890,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_90",
        "percent_real": 90,
        "train_synthetic": 270,
        "train_real": 2430,
        "test_real": 300,
    },
    {
        "name": "simplemixed_slope_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 2700,
        "test_real": 300,
    },
]

# Fine-Tuned (FT) strategy - pretrain on synthetic, finetune on real
FINETUNE_EXPERIMENTS_BOX = [
    {
        "name": "finetune_box_0",
        "percent_real": 0,
        "train_synthetic": 180,
        "train_real": 0,
        "test_real": 20,
    },
    {
        "name": "finetune_box_10",
        "percent_real": 10,
        "train_synthetic": 162,
        "train_real": 18,
        "test_real": 20,
    },
    {
        "name": "finetune_box_30",
        "percent_real": 30,
        "train_synthetic": 126,
        "train_real": 54,
        "test_real": 20,
    },
    {
        "name": "finetune_box_50",
        "percent_real": 50,
        "train_synthetic": 90,
        "train_real": 90,
        "test_real": 20,
    },
    {
        "name": "finetune_box_70",
        "percent_real": 70,
        "train_synthetic": 54,
        "train_real": 126,
        "test_real": 20,
    },
    {
        "name": "finetune_box_90",
        "percent_real": 90,
        "train_synthetic": 18,
        "train_real": 162,
        "test_real": 20,
    },
    {
        "name": "finetune_box_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 180,
        "test_real": 20,
    },
]

FINETUNE_EXPERIMENTS_SLOPE = [
    {
        "name": "finetune_slope_0",
        "percent_real": 0,
        "train_synthetic": 2700,
        "train_real": 0,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_10",
        "percent_real": 10,
        "train_synthetic": 2430,
        "train_real": 270,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_30",
        "percent_real": 30,
        "train_synthetic": 1890,
        "train_real": 810,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_50",
        "percent_real": 50,
        "train_synthetic": 1350,
        "train_real": 1350,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_70",
        "percent_real": 70,
        "train_synthetic": 810,
        "train_real": 1890,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_90",
        "percent_real": 90,
        "train_synthetic": 270,
        "train_real": 2430,
        "test_real": 300,
    },
    {
        "name": "finetune_slope_100",
        "percent_real": 100,
        "train_synthetic": 0,
        "train_real": 2700,
        "test_real": 300,
    },
]


def create_splits_for_experiment(
    experiment: dict,
    synthetic_files: list[str],
    real_files: list[str],
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Create train/test splits for a single experiment."""
    import random

    random.seed(seed)

    # Shuffle files for randomness
    synthetic_shuffled = synthetic_files.copy()
    real_shuffled = real_files.copy()
    random.shuffle(synthetic_shuffled)
    random.shuffle(real_shuffled)

    # Extract counts
    n_train_synth = experiment["train_synthetic"]
    n_train_real = experiment["train_real"]
    n_test_real = experiment["test_real"]

    # Validate we have enough data
    if n_train_synth > len(synthetic_shuffled):
        raise ValueError(
            f"Not enough synthetic data: need {n_train_synth}, "
            f"have {len(synthetic_shuffled)}"
        )
    if (n_train_real + n_test_real) > len(real_shuffled):
        raise ValueError(
            f"Not enough real data: need {n_train_real + n_test_real}, "
            f"have {len(real_shuffled)}"
        )

    # Split synthetic data
    train_synthetic = synthetic_shuffled[:n_train_synth]

    # Split real data (test first to ensure it's reserved)
    test_real = real_shuffled[:n_test_real]
    train_real = real_shuffled[n_test_real : n_test_real + n_train_real]

    # Combine training data
    train_all = train_synthetic + train_real
    random.shuffle(train_all)  # Shuffle combined training set

    # For FT strategy: also save separate synthetic and real training lists
    train_synthetic_only = train_synthetic.copy()
    train_real_only = train_real.copy()

    # Create output directory
    exp_dir = output_dir / experiment["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Determine strategy type from experiment name
    is_simplemixed = experiment["name"].startswith("simplemixed_")

    # Save splits
    splits = {
        "train.json": train_all,  # Azure expects train.json
        "train_all.json": train_all,  # For SM strategy (kept for reference)
        "train_synthetic.json": train_synthetic_only,  # For FT stage 1
        "train_real.json": train_real_only,  # For FT stage 2
        "test.json": test_real,  # Always real data
        "val.json": test_real,  # Use same as test (following Wachter)
    }

    for filename, data in splits.items():
        with open(exp_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    # Save metadata
    metadata = {
        "experiment_name": experiment["name"],
        "strategy_type": "simplemixed" if is_simplemixed else "finetune",
        "percent_real": experiment["percent_real"],
        "train_synthetic_count": len(train_synthetic),
        "train_real_count": len(train_real),
        "train_total_count": len(train_all),
        "test_count": len(test_real),
        "val_count": len(test_real),
        "synth_real_ratio": (
            f"{100 - experiment['percent_real']}/{experiment['percent_real']}"
        ),
        "note": (
            "train.json contains shuffled synthetic+real for Azure compatibility. "
            "SM uses train.json directly. FT will use train_synthetic.json "
            "then train_real.json in two-stage training (not yet implemented)."
        ),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(
        f"✓ Created splits for [cyan]{experiment['name']}[/cyan]: "
        f"{len(train_synthetic)} synth + {len(train_real)} real "
        f"→ {len(train_all)} train, {len(test_real)} test"
    )


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Generate all Simple Mixed and Fine-Tuned experiment splits."""
    console.print(
        "[bold green]Creating Wachter et al. (2026) Experiment Splits[/bold green]"
    )
    console.print("Generating both SM and FT strategy splits\n")

    # Load configuration
    from ml_segmentation.schema_config import ConfigSchema

    pcfg = ConfigSchema(**OmegaConf.to_container(cfg, resolve=True))

    # Output directory
    output_base = Path("data/model_ready/wachter_splits")
    output_base.mkdir(parents=True, exist_ok=True)

    console.print(f"Output directory: {output_base}\n")

    # === LOAD DATA FILES ===
    # Get synthetic box files (Benchmark prefix)
    synthetic_box_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_box",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    synthetic_box_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=synthetic_box_prefixes["train_prefixes"],
        test_prefixes=[],  # We want all files
    )

    # Get real box files (Box prefix)
    real_box_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_real_box",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    real_box_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=real_box_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # Get synthetic slope files (FracMan prefix)
    synthetic_slope_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_dfn",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    synthetic_slope_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=synthetic_slope_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # Get real slope files (Larvik, Rv4 prefixes)
    real_slope_prefixes = get_datasets_prefixes(
        experiment_strategy="verification_real_rock_slope",
        dataset_strategies=pcfg.experiment.dataset_strategies,
        dataset_prefixes=pcfg.dataset.prefixes,
    )
    real_slope_files, _ = get_data_files(
        images_dir=Path(pcfg.dataset.path_images),
        labels_dir=Path(pcfg.dataset.path_processed_mask_labels),
        train_prefixes=real_slope_prefixes["train_prefixes"],
        test_prefixes=[],
    )

    # === SIMPLE MIXED (SM) EXPERIMENTS ===
    console.print("[bold yellow]Simple Mixed (SM) Strategy - BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth, {len(real_box_files)} real"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=real_box_files,
            output_dir=output_base / "sm_box",
            seed=pcfg.experiment.seed,
        )
    console.print()

    console.print("[bold yellow]Simple Mixed (SM) Strategy - SLOPE[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_slope_files)} synth, {len(real_slope_files)} real"
    )
    for exp in SIMPLEMIXED_EXPERIMENTS_SLOPE:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_slope_files,
            real_files=real_slope_files,
            output_dir=output_base / "sm_slope",
            seed=pcfg.experiment.seed,
        )
    console.print()

    # === FINE-TUNED (FT) EXPERIMENTS ===
    console.print("[bold yellow]Fine-Tuned (FT) Strategy - BOX[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_box_files)} synth, {len(real_box_files)} real"
    )
    for exp in FINETUNE_EXPERIMENTS_BOX:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_box_files,
            real_files=real_box_files,
            output_dir=output_base / "ft_box",
            seed=pcfg.experiment.seed,
        )
    console.print()

    console.print("[bold yellow]Fine-Tuned (FT) Strategy - SLOPE[/bold yellow]")
    console.print(
        f"Available: {len(synthetic_slope_files)} synth, {len(real_slope_files)} real"
    )
    for exp in FINETUNE_EXPERIMENTS_SLOPE:
        create_splits_for_experiment(
            experiment=exp,
            synthetic_files=synthetic_slope_files,
            real_files=real_slope_files,
            output_dir=output_base / "ft_slope",
            seed=pcfg.experiment.seed,
        )

    console.print()
    console.print("[bold green]✓ All splits created successfully![/bold green]")
    console.print(f"\nSplits saved to: {output_base}")
    console.print(
        "\n[cyan]Generated:[/cyan]"
        "\n  • 14 Simple Mixed (SM) experiments: sm_box/ and sm_slope/"
        "\n  • 14 Fine-Tuned (FT) experiments: ft_box/ and ft_slope/"
        "\n\n[cyan]Next steps:[/cyan]"
        "\n  1. Register splits in Azure ML (if deploying to cloud)"
        "\n  2. Run training using experiment_strategy parameter"
        "\n     - SM: simplemixed_box_10, simplemixed_slope_30, etc."
        "\n     - FT: finetune_box_10, finetune_slope_30, etc."
    )


if __name__ == "__main__":
    main()
