"""Download metrics CSV files from Azure ML jobs.

This script downloads *_metrics.csv files from completed Azure ML jobs
specified in a manifest or CSV file.

Usage:
    # From batch jobs summary CSV
    poetry run python scripts/results_analysis/download_metrics.py --csv experiments/batch_jobs_summary.csv --output-folder-name batch_20251209

    # From job manifest
    poetry run python scripts/results_analysis/download_metrics.py --manifest experiments/batch_submissions/job_manifest_20251209_215422.json --output-folder-name run1

    # From a text file with job names
    poetry run python scripts/results_analysis/download_metrics.py --jobs-file job_names.txt --output-folder-name test_run

    # Filter by model or strategy
    poetry run python scripts/results_analysis/download_metrics.py --csv experiments/batch_jobs_summary.csv --model unet --output-folder-name unet_only

    # Dry run to preview
    poetry run python scripts/results_analysis/download_metrics.py --csv experiments/batch_jobs_summary.csv --output-folder-name preview --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track


def get_job_info_from_azure(
    ml_client: MLClient,
    job_name: str,
) -> tuple[str, str]:
    """Get model and strategy information from Azure ML job.

    Args:
        ml_client: Azure ML client
        job_name: Job name

    Returns:
        Tuple of (model, strategy)
    """
    try:
        job = ml_client.jobs.get(job_name)

        # Try to get from display name first (format: model_strategy)
        if job.display_name and "_" in job.display_name:
            parts = job.display_name.split("_", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

        # Try to get from tags
        if job.tags:
            model = job.tags.get("model", "unknown")
            strategy = job.tags.get("experiment_strategy", "unknown")
            if model != "unknown" or strategy != "unknown":
                return model, strategy

        # Try to get from properties
        if job.properties:
            model = job.properties.get("model", "unknown")
            strategy = job.properties.get("experiment_strategy", "unknown")
            return model, strategy

        return "unknown", "unknown"

    except Exception:
        return "unknown", "unknown"


def load_job_names_from_csv(
    csv_path: Path,
    model: str | None = None,
    strategy: str | None = None,
) -> list[tuple[str, str, str]]:
    """Load job names from batch_jobs_summary.csv.

    Args:
        csv_path: Path to CSV file
        model: Filter by model name
        strategy: Filter by experiment strategy

    Returns:
        List of (job_name, model, strategy) tuples
    """
    df = pd.read_csv(csv_path)

    # Filter out empty job names
    df = df[df["job_name"].notna() & (df["job_name"] != "")]

    # Apply filters
    if model:
        df = df[df["model"] == model]
    if strategy:
        df = df[df["experiment_strategy"] == strategy]

    return list(
        df[["job_name", "model", "experiment_strategy"]].itertuples(
            index=False, name=None
        )
    )


def load_job_names_from_manifest(
    manifest_path: Path,
    model: str | None = None,
    strategy: str | None = None,
) -> list[tuple[str, str, str]]:
    """Load job names from job manifest JSON.

    Args:
        manifest_path: Path to manifest file
        model: Filter by model name
        strategy: Filter by experiment strategy

    Returns:
        List of (job_name, model, strategy) tuples
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    jobs = []
    for job_record in manifest["jobs"]:
        if job_record["status"] != "submitted" or not job_record.get("job_name"):
            continue

        job_model = job_record.get("model", "unknown")
        job_strategy = job_record.get("experiment_strategy", "unknown")

        # Apply filters
        if model and job_model != model:
            continue
        if strategy and job_strategy != strategy:
            continue

        jobs.append((job_record["job_name"], job_model, job_strategy))

    return jobs


def download_metrics_from_job(
    ml_client: MLClient,
    console: Console,
    job_name: str,
    model: str,
    strategy: str,
    output_dir: Path,
    pattern: str = "*_metrics.csv",
    dry_run: bool = False,
) -> bool:
    """Download metrics CSV files from a job.

    Args:
        ml_client: Azure ML client
        console: Rich console
        job_name: Job name
        model: Model name
        strategy: Experiment strategy
        output_dir: Output directory
        pattern: File pattern to match
        dry_run: Dry run mode

    Returns:
        True if successful
    """
    if dry_run:
        console.print(f"[DRY RUN] {job_name} ({model}/{strategy})", style="blue")
        return True

    try:
        # Create temporary download directory
        temp_dir = output_dir / "_temp" / job_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Download all artifacts
        ml_client.jobs.download(
            name=job_name,
            download_path=str(temp_dir),
            all=True,
        )

        # Find matching files recursively
        # Handles both:
        # - simplemixed: metrics/*.csv
        # - finetune: *.csv in root
        matching_files = list(temp_dir.rglob(pattern))

        if not matching_files:
            console.print(f"  No metrics files found for {job_name}", style="yellow")
            # Clean up
            import shutil

            shutil.rmtree(temp_dir)
            return False

        # Create job-specific output directory
        job_output_dir = output_dir / f"{model}_{strategy}" / job_name
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy matching files, preserving subdirectory structure if needed
        import shutil

        for file_path in matching_files:
            # Get relative path from temp_dir to preserve folder structure
            relative_path = file_path.relative_to(temp_dir)

            # Check if file is in a subdirectory (e.g., metrics/)
            if len(relative_path.parts) > 1:
                # Preserve the parent folder name in output
                dest_path = job_output_dir / relative_path.parts[-2] / file_path.name
            else:
                # File is in root, save directly
                dest_path = job_output_dir / file_path.name

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)

        # Clean up
        shutil.rmtree(temp_dir)

        console.print(
            f"  [OK] {job_name}: {len(matching_files)} file(s)", style="green"
        )
        return True

    except Exception as e:
        console.print(f"  [ERROR] {job_name}: {e}", style="red")
        return False


def main() -> None:
    """Download metrics from Azure ML jobs."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download metrics CSV files from Azure ML jobs"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to batch_jobs_summary.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to job manifest JSON",
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help="Path to text file with job names (one per line)",
    )
    parser.add_argument(
        "--model",
        choices=["unet", "deeplabv3plus"],
        help="Filter by model",
    )
    parser.add_argument(
        "--strategy",
        help="Filter by experiment strategy",
    )
    parser.add_argument(
        "--pattern",
        default="*_metrics.csv",
        help="File pattern to match (default: *_metrics.csv)",
    )
    parser.add_argument(
        "--output-folder-name",
        required=True,
        help="Output folder name (e.g., 'run1', 'batch_20251209')",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("experiments/results/metrics"),
        help="Base output directory (default: experiments/results/metrics)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without downloading",
    )
    args = parser.parse_args()

    console = Console()

    # Load job names
    jobs = []
    if args.csv:
        if not args.csv.exists():
            console.print(f"Error: CSV not found: {args.csv}", style="red")
            sys.exit(1)
        jobs = load_job_names_from_csv(args.csv, args.model, args.strategy)
        console.print(f"Loaded {len(jobs)} jobs from CSV", style="green")

    elif args.manifest:
        if not args.manifest.exists():
            console.print(f"Error: Manifest not found: {args.manifest}", style="red")
            sys.exit(1)
        jobs = load_job_names_from_manifest(args.manifest, args.model, args.strategy)
        console.print(f"Loaded {len(jobs)} jobs from manifest", style="green")

    elif args.jobs_file:
        if not args.jobs_file.exists():
            console.print(f"Error: File not found: {args.jobs_file}", style="red")
            sys.exit(1)
        with open(args.jobs_file) as f:
            job_names = [line.strip() for line in f if line.strip()]

        # We'll get model and strategy from Azure ML later
        jobs = [(name, "unknown", "unknown") for name in job_names]
        console.print(f"Loaded {len(jobs)} job names from file", style="green")
        console.print("Will retrieve model and strategy from Azure ML...", style="blue")

    else:
        console.print("Error: Specify --csv, --manifest, or --jobs-file", style="red")
        sys.exit(1)

    if not jobs:
        console.print("No jobs to process", style="yellow")
        sys.exit(0)

    # Construct output directory with manual name
    output_dir = args.output_base / args.output_folder_name

    # Create base directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()):
        console.print(
            f"Output directory exists, will add to existing results: {output_dir}",
            style="blue",
        )
    else:
        console.print(f"Creating new output directory: {output_dir}", style="blue")

    console.print(f"Pattern: {args.pattern}")
    console.print(f"Output: {output_dir}\n")

    # Connect to Azure ML
    ml_client = None
    if not args.dry_run:
        console.print("Connecting to Azure ML...", style="blue")
        try:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            workspace_name = os.getenv("AZURE_ML_WORKSPACE")

            if not all([subscription_id, resource_group, workspace_name]):
                console.print("Error: Azure environment variables not set", style="red")
                sys.exit(1)

            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
            console.print("[OK] Connected\n", style="green")

        except Exception as e:
            console.print(f"Error: {e}", style="red")
            sys.exit(1)

    # Download metrics
    success_count = 0
    for job_name, model, strategy in track(jobs, description="Downloading..."):
        # If model/strategy is unknown, fetch from Azure ML
        if model == "unknown" or strategy == "unknown":
            if not args.dry_run:
                model, strategy = get_job_info_from_azure(ml_client, job_name)
            else:
                model, strategy = "unknown", "unknown"

        success = download_metrics_from_job(
            ml_client=ml_client,
            console=console,
            job_name=job_name,
            model=model,
            strategy=strategy,
            output_dir=output_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
        if success:
            success_count += 1

    console.print(f"\n✓ Downloaded metrics from {success_count}/{len(jobs)} jobs")
    if not args.dry_run:
        console.print(f"📁 {output_dir}")


if __name__ == "__main__":
    main()
