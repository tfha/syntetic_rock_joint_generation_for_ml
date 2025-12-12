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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


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


def get_blob_client(ml_client: MLClient) -> BlobServiceClient:
    """Get blob service client using Azure ML workspace's storage account.

    Args:
        ml_client: Azure ML client

    Returns:
        BlobServiceClient for the workspace storage
    """
    # Get workspace info
    workspace = ml_client.workspaces.get(ml_client.workspace_name)

    # Get the storage account from workspace
    storage_account = workspace.storage_account

    # Extract account name from full resource ID
    # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{name}
    account_name = storage_account.split("/")[-1]

    # Create blob service client with same credential as ML client
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=ml_client._credential
    )

    return blob_service_client


def construct_blob_paths(
    job_name: str,
    model: str,
    strategy: str,
    experiment: str,
    proportion: str,
) -> list[str]:
    """Construct possible blob storage paths for metrics files.

    Args:
        job_name: Azure ML job name
        model: Model name (e.g., "unet", "deeplabv3plus")
        strategy: Strategy name (e.g., "finetune", "simplemixed")
        experiment: Experiment name (e.g., "box_10", "slope_30")
        proportion: Proportion value (e.g., "10", "30")

    Returns:
        List of possible blob paths to try
    """
    paths = []

    if "finetune" in strategy.lower():
        # Finetune: ExperimentRun/dcid.{job_name}/{model}_finetune_{experiment}_{proportion}_metrics.csv
        filename = f"{model}_finetune_{experiment}_{proportion}_metrics.csv"
        paths.append(f"ExperimentRun/dcid.{job_name}/{filename}")
    else:
        # Simplemixed: ExperimentRun/dcid.{job_name}/metrics/{model}-simplemixed_{experiment}_{proportion}-{timestamp}_metrics.csv
        # We don't know the timestamp, so we'll need to list files in the metrics/ folder
        paths.append(f"ExperimentRun/dcid.{job_name}/metrics/")

    return paths


def download_blob_to_file(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_path: str,
    output_path: Path,
) -> bool:
    """Download a single blob to a file.

    Args:
        blob_service_client: Azure Blob Service client
        container_name: Container name
        blob_path: Blob path within container
        output_path: Local file path to save to

    Returns:
        True if successful
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download blob
        with open(output_path, "wb") as f:
            blob_data = blob_client.download_blob()
            blob_data.readinto(f)

        return True
    except Exception:
        return False


def find_and_download_metrics(
    blob_service_client: BlobServiceClient,
    container_name: str,
    job_name: str,
    model: str,
    strategy: str,
    output_dir: Path,
    debug: bool = False,
) -> tuple[bool, int, str]:
    """Find and download metrics files for a job using direct blob access.

    Args:
        blob_service_client: Azure Blob Service client
        container_name: Container name (usually "azureml-blobstore-...")
        job_name: Job name
        model: Model name
        strategy: Strategy name
        output_dir: Output directory
        debug: Print debug information

    Returns:
        Tuple of (success, file_count, debug_info)
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)

        # Create output directory (flat structure - all files in one folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try multiple path patterns
        patterns_to_try = [
            f"ExperimentRun/dcid.{job_name}/",
            f"azureml/{job_name}/",
        ]

        metrics_files = []
        debug_info = []

        for prefix in patterns_to_try:
            try:
                blobs = list(container_client.list_blobs(name_starts_with=prefix))
                if debug:
                    debug_info.append(
                        f"Checking prefix '{prefix}': {len(blobs)} total blobs"
                    )

                # Filter for metrics CSV files
                csv_blobs = [
                    blob for blob in blobs if blob.name.endswith("_metrics.csv")
                ]

                if debug and csv_blobs:
                    debug_info.append(f"  Found {len(csv_blobs)} metrics files:")
                    for blob in csv_blobs:
                        debug_info.append(f"    - {blob.name}")

                metrics_files.extend([blob.name for blob in csv_blobs])

                # If we found files, stop searching
                if metrics_files:
                    break
            except Exception as e:
                if debug:
                    debug_info.append(
                        f"Error listing blobs with prefix '{prefix}': {e}"
                    )
                continue

        if not metrics_files:
            return (
                False,
                0,
                "\n".join(debug_info) if debug_info else "No metrics files found",
            )

        # Download each metrics file directly to output folder
        downloaded = 0
        for blob_path in metrics_files:
            # Get filename from path
            filename = Path(blob_path).name

            # Save directly to output folder (flat structure)
            output_path = output_dir / filename

            if download_blob_to_file(
                blob_service_client, container_name, blob_path, output_path
            ):
                downloaded += 1

        return downloaded > 0, downloaded, "\n".join(debug_info) if debug_info else ""

    except Exception as e:
        return False, 0, f"Exception: {e}"


def download_metrics_from_job(
    blob_service_client: BlobServiceClient,
    container_name: str,
    job_name: str,
    model: str,
    strategy: str,
    output_dir: Path,
    dry_run: bool = False,
    debug: bool = False,
) -> tuple[bool, str]:
    """Download metrics CSV files from a job.

    Args:
        blob_service_client: Azure Blob Service client
        container_name: Container name
        job_name: Job name
        model: Model name
        strategy: Experiment strategy
        output_dir: Output directory
        dry_run: Dry run mode
        debug: Print debug information

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        return True, f"[DRY RUN] {job_name} ({model}/{strategy})"

    try:
        success, file_count, debug_info = find_and_download_metrics(
            blob_service_client,
            container_name,
            job_name,
            model,
            strategy,
            output_dir,
            debug=debug,
        )

        if success:
            return True, f"[OK] {job_name}: {file_count} file(s)"
        else:
            msg = f"[SKIP] {job_name}: No metrics files found"
            if debug and debug_info:
                msg += f"\n{debug_info}"
            return False, msg

    except Exception as e:
        return False, f"[ERROR] {job_name}: {e}"


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
        "--debug",
        action="store_true",
        help="Print debug information about blob paths",
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
    blob_service_client = None
    container_name: str | None = None

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
            console.print("[OK] Connected to Azure ML", style="green")

            # Get blob service client
            blob_service_client = get_blob_client(ml_client)

            # Get workspaceartifactstore (where ExperimentRun files are stored)
            datastore = ml_client.datastores.get("workspaceartifactstore")
            container_name = datastore.container_name

            console.print(
                f"[OK] Connected to blob storage (container: {container_name})\n",
                style="green",
            )

        except Exception as e:
            console.print(f"Error: {e}", style="red")
            sys.exit(1)

    # Fetch model/strategy for jobs that need it
    console.print("Fetching job information...", style="blue")
    jobs_with_info = []
    for job_name, model, strategy in jobs:
        if model == "unknown" or strategy == "unknown":
            if not args.dry_run:
                model, strategy = get_job_info_from_azure(ml_client, job_name)
                if args.debug:
                    console.print(
                        f"  {job_name}: model={model}, strategy={strategy}", style="dim"
                    )
        jobs_with_info.append((job_name, model, strategy))

    console.print("[OK] Job information ready\n", style="green")

    # Download metrics in parallel
    success_count = 0
    failed_count = 0
    max_workers = 10  # Adjust based on your needs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading metrics...", total=len(jobs_with_info))

        def download_job(job_info):
            job_name, model, strategy = job_info
            success, message = download_metrics_from_job(
                blob_service_client=blob_service_client,
                container_name=container_name or "",
                job_name=job_name,
                model=model,
                strategy=strategy,
                output_dir=output_dir,
                dry_run=args.dry_run,
                debug=args.debug,
            )
            return success, message

        if args.dry_run:
            # Sequential for dry run
            for job_info in jobs_with_info:
                success, message = download_job(job_info)
                console.print(message, style="blue" if success else "yellow")
                if success:
                    success_count += 1
                progress.update(task, advance=1)
        else:
            # Parallel download
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(download_job, job_info): job_info
                    for job_info in jobs_with_info
                }

                for future in as_completed(futures):
                    success, message = future.result()

                    if "[OK]" in message:
                        console.print(message, style="green")
                        success_count += 1
                    elif "[SKIP]" in message:
                        console.print(message, style="yellow")
                    else:
                        console.print(message, style="red")
                        failed_count += 1

                    progress.update(task, advance=1)

    console.print(
        f"\n[OK] Downloaded metrics from {success_count}/{len(jobs_with_info)} jobs"
    )
    if failed_count > 0:
        console.print(f"[WARNING] {failed_count} jobs failed", style="yellow")
    if not args.dry_run:
        console.print(f"[FILE] {output_dir}")


if __name__ == "__main__":
    main()
