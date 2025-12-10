"""Download metrics CSV files from Azure ML jobs.

This script uses Azure Storage Blob SDK to selectively download only *_metrics.csv files
from completed Azure ML jobs. Performance optimizations include:
- Direct blob access bypasses Azure ML SDK (7-8x faster)
- Reuses blob client across jobs (eliminates per-job overhead)
- Skips unnecessary job metadata fetches
- Typical performance: 3-4 seconds per job vs 90+ seconds with Azure ML SDK

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

    # Show detailed timing for debugging
    poetry run python scripts/results_analysis/download_metrics.py --jobs-file job_names.txt --output-folder-name test --timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track


def download_metrics_from_job(
    ml_client: MLClient,
    console: Console,
    job_name: str,
    model: str,
    strategy: str,
    output_dir: Path,
    pattern: str = "*_metrics.csv",
    dry_run: bool = False,
    blob_service_client: BlobServiceClient | None = None,
    show_timing: bool = False,
) -> bool:
    """Download metrics CSV files from a job using fast selective download.

    Uses Azure Storage Blob SDK to directly access blobs in the azureml container.
    Optimized to avoid expensive API calls and reuse connections across jobs.

    Performance: ~3-4 seconds per job (vs 90+ seconds with Azure ML SDK).

    Args:
        ml_client: Azure ML client (used only to get datastore info)
        console: Rich console for output
        job_name: Job name
        model: Model name
        strategy: Experiment strategy
        output_dir: Output directory (flat structure, no subdirectories)
        pattern: File pattern to match (default: *_metrics.csv)
        dry_run: Dry run mode (preview without downloading)
        blob_service_client: Reusable blob service client (improves performance)
        show_timing: Show detailed timing breakdown for debugging

    Returns:
        True if successful
    """
    if dry_run:
        console.print(f"[DRY RUN] {job_name} ({model}/{strategy})", style="blue")
        return True

    try:
        # Create output directory (no subdirectories - all CSVs in one folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use Azure Storage Blob SDK for fast selective downloads
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient

            t0 = time.time() if show_timing else None

            # Get datastore info (we can skip getting full job details)
            datastore = ml_client.datastores.get_default()
            account_name = datastore.account_name
            container_name = "azureml"

            t1: float
            t2: float
            if show_timing:
                t1 = time.time()
                console.print(f"  [TIME] Get datastore: {t1 - t0:.1f}s", style="dim")  # type: ignore[operator]

            # Reuse blob client if provided, otherwise create new one
            if blob_service_client is None:
                account_url = f"https://{account_name}.blob.core.windows.net"
                credential = DefaultAzureCredential()
                blob_service_client = BlobServiceClient(
                    account_url=account_url, credential=credential
                )
                if show_timing:
                    t2 = time.time()
                    console.print(
                        f"  [TIME] Create blob client: {t2 - t1:.1f}s", style="dim"
                    )
            else:
                if show_timing:
                    t1_val = t1 if show_timing else time.time()
                    t2 = t1_val
                    console.print("  [TIME] Reuse blob client: 0.0s", style="dim")

            container_client = blob_service_client.get_container_client(container_name)

            # List blobs under ExperimentRun/dcid.{job_name}
            base_prefix = f"ExperimentRun/dcid.{job_name}"

            downloaded_files = []
            seen_filenames = set()  # Track filenames to avoid duplicates

            # Download only *_metrics.csv files
            blob_count = 0
            for blob in container_client.list_blobs(name_starts_with=base_prefix):
                blob_count += 1
                if blob.name.endswith("_metrics.csv"):
                    filename = Path(blob.name).name

                    # Skip if we've already downloaded a file with this name
                    if filename in seen_filenames:
                        continue

                    seen_filenames.add(filename)
                    dest_path = output_dir / filename

                    blob_client = container_client.get_blob_client(blob.name)
                    with open(dest_path, "wb") as f:
                        f.write(blob_client.download_blob().readall())

                    downloaded_files.append(dest_path)

            if show_timing:
                t3 = time.time()
                console.print(
                    f"  [TIME] List and download ({blob_count} blobs): {t3 - t2:.1f}s",
                    style="dim",
                )
                console.print(f"  [TIME] Total: {t3 - t0:.1f}s", style="dim")  # type: ignore[operator]

            if downloaded_files:
                console.print(
                    f"  [OK] {job_name}: {len(downloaded_files)} file(s)", style="green"
                )
                return True
            else:
                console.print(
                    f"  [SKIP] {job_name}: No metrics files found", style="yellow"
                )
                return False

        except Exception as e:
            console.print(f"  [ERROR] Failed to download {job_name}: {e}", style="red")
            return False

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
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show detailed timing information for debugging",
    )
    args = parser.parse_args()

    console = Console()

    # Load job names - inline simple logic instead of helper functions
    jobs: list[tuple[str, str, str]] = []
    if args.csv:
        if not args.csv.exists():
            console.print(f"Error: CSV not found: {args.csv}", style="red")
            sys.exit(1)
        df = pd.read_csv(args.csv)
        df = df[df["job_name"].notna() & (df["job_name"] != "")]
        if args.model:
            df = df[df["model"] == args.model]
        if args.strategy:
            df = df[df["experiment_strategy"] == args.strategy]
        jobs = list(
            df[["job_name", "model", "experiment_strategy"]].itertuples(
                index=False, name=None
            )
        )
        console.print(f"Loaded {len(jobs)} jobs from CSV", style="green")

    elif args.manifest:
        if not args.manifest.exists():
            console.print(f"Error: Manifest not found: {args.manifest}", style="red")
            sys.exit(1)
        with open(args.manifest) as f:
            manifest = json.load(f)
        for job_record in manifest["jobs"]:
            if job_record["status"] == "submitted" and job_record.get("job_name"):
                job_model = job_record.get("model", "unknown")
                job_strategy = job_record.get("experiment_strategy", "unknown")
                if (not args.model or job_model == args.model) and (
                    not args.strategy or job_strategy == args.strategy
                ):
                    jobs.append((job_record["job_name"], job_model, job_strategy))
        console.print(f"Loaded {len(jobs)} jobs from manifest", style="green")

    elif args.jobs_file:
        if not args.jobs_file.exists():
            console.print(f"Error: File not found: {args.jobs_file}", style="red")
            sys.exit(1)
        with open(args.jobs_file) as f:
            job_names = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        jobs = [(name, "unknown", "unknown") for name in job_names]
        console.print(f"Loaded {len(jobs)} job names from file", style="green")

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

            # Create BlobServiceClient once for reuse across all jobs
            datastore = ml_client.datastores.get_default()
            account_url = f"https://{datastore.account_name}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(
                account_url=account_url, credential=credential
            )

        except Exception as e:
            console.print(f"Error: {e}", style="red")
            sys.exit(1)
    else:
        blob_service_client = None

    # Download metrics using fast selective download
    success_count = 0
    for job_name, model, strategy in track(jobs, description="Downloading..."):
        if ml_client is None:
            console.print("Error: ML client not initialized", style="red")
            continue

        success = download_metrics_from_job(
            ml_client=ml_client,
            console=console,
            job_name=job_name,
            model=model,
            strategy=strategy,
            output_dir=output_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
            blob_service_client=blob_service_client,
            show_timing=args.timing,
        )
        if success:
            success_count += 1
    console.print(f"\n[OK] Downloaded metrics from {success_count}/{len(jobs)} jobs")
    if not args.dry_run:
        console.print(f"[FILE] {output_dir}")


if __name__ == "__main__":
    main()
