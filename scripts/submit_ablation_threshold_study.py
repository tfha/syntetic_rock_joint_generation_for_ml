"""Submit ablation study for prediction threshold sensitivity analysis.

This script submits 28 parallel Azure ML jobs (4 models × 7 thresholds)
to study how prediction threshold affects model performance.

Models (4 total):
  - deeplabv3plus-finetune_box_50
  - deeplabv3plus-simplemixed_box_50
  - deeplabv3plus-finetune_slope_50
  - deeplabv3plus-simplemixed_slope_50

Thresholds (7 total):
  - 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

Total: 28 jobs submitted in parallel
Experiment: ablation_prediction_threshold
Compute: Standard-NC6s-v3 (GPU)
Metrics: Both val and test logged separately

Usage:
    poetry run python scripts/submit_ablation_threshold_study.py
    poetry run python scripts/submit_ablation_threshold_study.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Setup logging and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

# Ablation study configuration
MODELS = [
    "deeplabv3plus-finetune_box_50",
    "deeplabv3plus-simplemixed_box_50",
    "deeplabv3plus-finetune_slope_50",
    "deeplabv3plus-simplemixed_slope_50",
]

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

EXPERIMENT_NAME = "ablation_prediction_threshold"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit ablation study for prediction threshold analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without submitting to Azure ML",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel job submissions (default: 4)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight storage permission checks",
    )
    return parser.parse_args()


def get_base_model_name(model_name: str) -> str:
    """Extract base model name (without strategy suffix).

    Examples:
        'deeplabv3plus-finetune_box_50' -> 'deeplabv3plus'
        'unet-simplemixed_slope_50' -> 'unet'
    """
    parts = model_name.split("-")
    return parts[0]


def get_experiment_strategy(model_name: str) -> str:
    """Extract experiment strategy from model name.

    Examples:
        'deeplabv3plus-finetune_box_50' -> 'finetune_box_50'
        'unet-simplemixed_slope_50' -> 'simplemixed_slope_50'
    """
    parts = model_name.split("-")
    if len(parts) >= 2:
        return "-".join(parts[1:])
    return model_name


def build_training_command(
    model_name: str,
    strategy: str,
    threshold: float,
    num_epochs: int = 100,
    loss_function: str = "dice",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
) -> str:
    """Build the training command with Hydra overrides."""
    # Determine training script based on strategy
    if strategy.startswith("finetune_"):
        training_script = "azure_train_finetune.py"
    else:
        training_script = "azure_train_eval.py"

    # Extract base model name (without strategy suffix) for Hydra config
    base_model = get_base_model_name(model_name)

    # Build command
    command_parts = [
        f"python scripts/{training_script}",
        f"model={base_model}",
        f"model.num_epochs={num_epochs}",
        f"experiment.experiment_strategy={strategy}",
        f"experiment.loss_function={loss_function}",
        f"experiment.focal_alpha={focal_alpha}",
        f"experiment.focal_gamma={focal_gamma}",
        f"experiment.ablation_threshold={threshold}",
        "+dataset.azure_images_path=${{inputs.images_data}}",
        "+dataset.azure_masks_path=${{inputs.masks_data}}",
        "+dataset.azure_splits_path=${{inputs.splits_data}}",
    ]

    return " ".join(command_parts)


def submit_single_job(
    ml_client: Any,
    model_name: str,
    threshold: float,
    job_config: dict[str, Any],
    splits_by_strategy: dict[str, Any],
    dry_run: bool = False,
) -> tuple[str, str, bool]:
    """Submit a single ablation job."""
    from azure.ai.ml import Input, command
    from azure.ai.ml.entities import ManagedIdentityConfiguration

    from ml_segmentation.azure_core import retry_azure_operation

    try:
        strategy = get_experiment_strategy(model_name)
        timestamp_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        display_name = f"ablation-{model_name}-threshold_{threshold}-{timestamp_suffix}"

        train_command = build_training_command(
            model_name=model_name,
            strategy=strategy,
            threshold=threshold,
            num_epochs=job_config["num_epochs"],
            loss_function=job_config["loss_function"],
            focal_alpha=job_config["focal_alpha"],
            focal_gamma=job_config["focal_gamma"],
        )

        if dry_run:
            return (display_name, f"[DRY-RUN] {train_command[:80]}...", True)

        # Get the correct splits dataset for this job's strategy
        if strategy not in splits_by_strategy:
            raise ValueError(
                f"No splits dataset found for strategy '{strategy}'. "
                f"Available: {list(splits_by_strategy.keys())}"
            )
        splits_dataset = splits_by_strategy[strategy]

        # Create job inputs with the correct splits for this strategy
        job_inputs = dict(job_config["inputs"])  # Copy base inputs
        job_inputs["splits_data"] = Input(
            type="uri_folder", path=splits_dataset.id, mode="ro_mount"
        )

        # Create and submit job
        job = command(
            code="./",
            command=train_command,
            environment=job_config["environment"],
            compute=job_config["compute_name"],
            display_name=display_name,
            experiment_name=job_config["experiment_name"],
            inputs=job_inputs,
            outputs=job_config["outputs"],
            tags={
                "model": model_name,
                "threshold": str(threshold),
                "ablation_study": "prediction_threshold",
            },
            identity=ManagedIdentityConfiguration(),
            environment_variables=job_config.get("environment_variables"),
        )

        submit_with_retry = retry_azure_operation(
            ml_client.jobs.create_or_update, operation_name="Job submission"
        )
        job_run = submit_with_retry(job)

        return (display_name, job_run.name, True)

    except Exception as e:
        error_msg = f"Failed: {str(e)[:100]}"
        return (f"ablation-{model_name}-threshold_{threshold}", error_msg, False)


def main() -> int:
    """Main function to orchestrate ablation study submission."""
    from azure.ai.ml import Input, Output

    from ml_segmentation.azure_authentication import (
        connect_to_azure_ml,
        setup_azure_environment_variables,
        validate_workspace_permissions,
    )
    from ml_segmentation.azure_compute import validate_and_refresh_compute
    from ml_segmentation.azure_core import configure_azure_logging_and_warning
    from ml_segmentation.azure_data_assets import (
        get_data_asset,
        retrieve_and_validate_data_assets,
    )
    from ml_segmentation.azure_diagnostics import preflight_storage_permissions

    args = parse_arguments()

    # Configure logging
    configure_azure_logging_and_warning()
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("msrest.serialization").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    warnings.filterwarnings("ignore", category=UserWarning, module="msrest")

    # Print ablation study summary
    console.print("\n" + "=" * 80)
    console.print("ABLATION STUDY: Prediction Threshold Sensitivity Analysis")
    console.print("=" * 80 + "\n")

    summary_table = Table(title="Ablation Study Configuration")
    summary_table.add_column("Parameter")
    summary_table.add_column("Value")
    summary_table.add_row("Models", str(len(MODELS)))
    summary_table.add_row("Thresholds", str(len(THRESHOLDS)))
    summary_table.add_row("Total Jobs", str(len(MODELS) * len(THRESHOLDS)))
    summary_table.add_row("Experiment", EXPERIMENT_NAME)
    summary_table.add_row("Dry-Run", "YES" if args.dry_run else "NO")
    console.print(summary_table)
    console.print()

    # Setup Azure environment
    console.print("[INFO] Setting up Azure environment...")

    ml_client = None
    images_dataset = None
    masks_dataset = None
    splits_dataset = None

    # Only authenticate to Azure if not a dry-run
    if not args.dry_run:
        # Real submission - authenticate to Azure
        _, subscription_id, resource_group, workspace_name = (
            setup_azure_environment_variables()
        )
        ml_client = connect_to_azure_ml(
            subscription_id=subscription_id,
            console=console,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )
        validate_workspace_permissions(ml_client, console)
        console.print("[OK] Connected to Azure ML workspace")

    # Retrieve data assets (unless dry-run)
    console.print("[INFO] Retrieving data assets...")
    images_dataset = None
    masks_dataset = None
    splits_by_strategy: dict[str, Any] = {}

    if not args.dry_run:
        assert ml_client is not None  # Ensure ml_client exists in non-dry-run

        # Get unique strategies from MODELS
        unique_strategies = sorted({get_experiment_strategy(m) for m in MODELS})
        console.print(f"[INFO] Retrieving splits for strategies: {unique_strategies}")

        # Retrieve splits for each strategy
        for strategy in unique_strategies:
            images_dataset, masks_dataset, splits_dataset = (
                retrieve_and_validate_data_assets(
                    ml_client=ml_client,
                    console=console,
                    experiment_strategy=strategy,
                    get_data_asset_func=get_data_asset,
                )
            )
            splits_by_strategy[strategy] = splits_dataset
            console.print(f"  [OK] Retrieved splits for {strategy}")

        console.print("[OK] Data assets retrieved")

    # Validate compute cluster (unless dry-run)
    compute_name = "Standard-NC6s-v3"
    if not args.dry_run:
        assert ml_client is not None  # Ensure ml_client exists in non-dry-run
        console.print(f"[INFO] Validating compute cluster: {compute_name}")
        validate_and_refresh_compute(ml_client, console, compute_name)
        if not args.skip_preflight:
            preflight_storage_permissions(ml_client, console, compute_name)
        console.print("[OK] Compute cluster validated")

    # Prepare shared job configuration
    job_config = {
        "compute_name": compute_name,
        "experiment_name": EXPERIMENT_NAME,
        "num_epochs": 100,
        "loss_function": "dice",
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
        "environment": (
            "rock-segmentation-env-curated-py310:3"
            if not args.dry_run
            else "mock-env:1"
        ),
        "environment_variables": None,
    }

    if not args.dry_run:
        # Setup shared job inputs (images and masks are common to all jobs)
        assert images_dataset is not None  # Guaranteed by non-dry-run check
        assert masks_dataset is not None  # Guaranteed by non-dry-run check
        assert len(splits_by_strategy) > 0  # At least one strategy's splits
        job_config["inputs"] = {
            "images_data": Input(
                type="uri_folder", path=images_dataset.id, mode="ro_mount"
            ),
            "masks_data": Input(
                type="uri_folder", path=masks_dataset.id, mode="ro_mount"
            ),
            # Note: splits_data will be set per-job in submit_single_job
        }
    else:
        job_config["inputs"] = {}

    job_config["outputs"] = {
        "model_output": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/models",
        ),
        "tensorboard_logs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/tensorboard_logs",
        ),
        "mlflow_logs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/mlruns",
        ),
        "plots": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/plots",
        ),
        "example_images": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/example_images",
        ),
        "hydra_outputs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/hydra_outputs",
        ),
    }

    # Generate all job combinations
    job_combinations = [
        (model, threshold) for model in MODELS for threshold in THRESHOLDS
    ]

    console.print(f"[INFO] Preparing {len(job_combinations)} jobs for submission...\n")

    # Submit jobs
    submitted_jobs = []
    failed_jobs = []

    if args.dry_run:
        # In dry-run mode, just show what would be submitted
        jobs_table = Table(title="Jobs to be Submitted (Dry-Run)")
        jobs_table.add_column("Model")
        jobs_table.add_column("Threshold")
        jobs_table.add_column("Strategy")

        for model, threshold in job_combinations:
            strategy = get_experiment_strategy(model)
            jobs_table.add_row(model, str(threshold), strategy)

        console.print(jobs_table)
        console.print(f"\n[DRY-RUN] Would submit {len(job_combinations)} jobs\n")
        return 0

    # Real submission with parallel execution
    assert ml_client is not None  # Ensure ml_client exists in non-dry-run
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Submitting {len(job_combinations)} jobs in parallel...",
            total=len(job_combinations),
        )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    submit_single_job,
                    ml_client,
                    model,
                    threshold,
                    job_config,
                    splits_by_strategy,
                    args.dry_run,
                ): (model, threshold)
                for model, threshold in job_combinations
            }

            for future in as_completed(futures):
                model, threshold = futures[future]
                display_name, job_id, success = future.result()

                if success:
                    submitted_jobs.append(
                        {"model": model, "threshold": threshold, "job_id": job_id}
                    )
                else:
                    failed_jobs.append(
                        {"model": model, "threshold": threshold, "error": job_id}
                    )

                progress.update(task, advance=1)

    # Print summary
    console.print("\n" + "=" * 80)
    console.print("SUBMISSION SUMMARY")
    console.print("=" * 80 + "\n")

    summary_table = Table(title="Results")
    summary_table.add_column("Status")
    summary_table.add_column("Count")
    summary_table.add_row("Submitted", str(len(submitted_jobs)))
    summary_table.add_row("Failed", str(len(failed_jobs)))
    summary_table.add_row("Total", str(len(job_combinations)))
    console.print(summary_table)

    if submitted_jobs:
        console.print("\n[bold green]Submitted Jobs:[/bold green]")
        jobs_table = Table()
        jobs_table.add_column("Model")
        jobs_table.add_column("Threshold")
        jobs_table.add_column("Job ID")

        for job in submitted_jobs[:10]:
            jobs_table.add_row(job["model"], str(job["threshold"]), job["job_id"])

        if len(submitted_jobs) > 10:
            jobs_table.add_row(
                "[dim]...[/dim]",
                "[dim]...[/dim]",
                f"[dim]+{len(submitted_jobs) - 10} more[/dim]",
            )

        console.print(jobs_table)

    if failed_jobs:
        console.print("\n[bold red]Failed Jobs:[/bold red]")
        failed_table = Table()
        failed_table.add_column("Model")
        failed_table.add_column("Threshold")
        failed_table.add_column("Error")

        for job in failed_jobs:
            failed_table.add_row(job["model"], str(job["threshold"]), job["error"])

        console.print(failed_table)

    # Experiment link
    console.print(f"\n[INFO] Experiment: {EXPERIMENT_NAME}")
    console.print(f"View results: https://ml.azure.com/experiments/{EXPERIMENT_NAME}")

    console.print("\n" + "=" * 80 + "\n")

    return 0 if len(failed_jobs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
