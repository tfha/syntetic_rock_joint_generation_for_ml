"""Download artifacts from Azure ML jobs using batch submission manifest.

This script reads a job manifest created by submit_batch_jobs.py and downloads
outputs/artifacts from the completed jobs. Supports filtering by model, strategy,
and status.

Usage:
    # Download all successful jobs from latest manifest
    poetry run python scripts/download_batch_artifacts.py

    # Download from specific manifest
    poetry run python scripts/download_batch_artifacts.py --manifest experiments/batch_submissions/job_manifest_20251209_143022.json

    # Download only specific model
    poetry run python scripts/download_batch_artifacts.py --model unet

    # Download only specific strategy
    poetry run python scripts/download_batch_artifacts.py --strategy finetune_box_50

    # Download only failed jobs (for debugging)
    poetry run python scripts/download_batch_artifacts.py --status failed

    # Dry run to see what would be downloaded
    poetry run python scripts/download_batch_artifacts.py --dry-run

    # Download specific outputs (default: all)
    poetry run python scripts/download_batch_artifacts.py --outputs final/test models

Output structure:
    experiments/batch_downloads/<manifest_timestamp>/
        <job_name>/
            <model>_<strategy>/
                final/
                    test/
                models/
                ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from rich.console import Console
from rich.progress import track


def find_latest_manifest(project_root: Path) -> Path | None:
    """Find the most recent job manifest file.

    Args:
        project_root: Project root directory

    Returns:
        Path to latest manifest, or None if not found
    """
    manifest_dir = project_root / "experiments" / "batch_submissions"
    if not manifest_dir.exists():
        return None

    manifests = list(manifest_dir.glob("job_manifest_*.json"))
    if not manifests:
        return None

    # Sort by filename (timestamp) and return latest
    return sorted(manifests, reverse=True)[0]


def load_manifest(manifest_path: Path, console: Console) -> dict:
    """Load and validate manifest file.

    Args:
        manifest_path: Path to manifest JSON file
        console: Rich console for output

    Returns:
        Manifest dictionary
    """
    console.print(f"Loading manifest: {manifest_path.name}")

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        console.print(f"[ERROR] Loading manifest: {e}")
        sys.exit(1)

    # Validate manifest structure
    required_keys = ["submission_time", "total_jobs", "jobs"]
    for key in required_keys:
        if key not in manifest:
            console.print(f"[ERROR] Invalid manifest: missing '{key}' field")
            sys.exit(1)

    console.print(
        f"[OK] Manifest loaded: {len(manifest['jobs'])} jobs from {manifest['submission_time']}"
    )
    return manifest


def filter_jobs(
    manifest: dict,
    model: str | None = None,
    strategy: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Filter jobs based on criteria.

    Args:
        manifest: Job manifest dictionary
        model: Filter by model name (unet, deeplabv3plus)
        strategy: Filter by experiment strategy
        status: Filter by status (submitted, failed)

    Returns:
        List of filtered job records
    """
    jobs = manifest["jobs"]

    if model:
        jobs = [j for j in jobs if j.get("model") == model]

    if strategy:
        jobs = [j for j in jobs if j.get("experiment_strategy") == strategy]

    if status:
        jobs = [j for j in jobs if j.get("status") == status]

    return jobs


def download_job_artifacts(
    ml_client: MLClient,
    console: Console,
    job_name: str,
    output_dir: Path,
    outputs: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Download artifacts from a single Azure ML job.

    Args:
        ml_client: Azure ML client
        console: Rich console for output
        job_name: Azure ML job name
        output_dir: Local directory to save artifacts
        outputs: Specific output paths to download (None = all)
        dry_run: If True, print what would be downloaded without executing

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        console.print(f"[DRY RUN] Would download job: {job_name}")
        console.print(f"[DRY RUN] Output directory: {output_dir}")
        return True

    try:
        # Get job details
        job = ml_client.jobs.get(job_name)
        console.print(f"Downloading artifacts for job: {job_name}")
        console.print(f"  Status: {job.status}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download outputs
        if outputs:
            # Download specific outputs
            for output_name in outputs:
                try:
                    console.print(f"  Downloading output: {output_name}")
                    ml_client.jobs.download(
                        name=job_name,
                        download_path=str(output_dir),
                        output_name=output_name,
                    )
                except Exception as e:
                    console.print(f"  [WARN] Could not download '{output_name}': {e}")
        else:
            # Download all outputs
            console.print("  Downloading all outputs...")
            ml_client.jobs.download(
                name=job_name,
                download_path=str(output_dir),
                all=True,
            )

        console.print(f"[OK] Downloaded to: {output_dir}")
        return True

    except Exception as e:
        console.print(f"[ERROR] Error downloading job {job_name}: {e}")
        return False


def main() -> None:
    """Download artifacts from batch-submitted Azure ML jobs."""
    parser = argparse.ArgumentParser(
        description="Download artifacts from Azure ML jobs using batch manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to job manifest JSON file (default: latest manifest)",
    )
    parser.add_argument(
        "--model",
        choices=["unet", "deeplabv3plus"],
        help="Filter by model name",
    )
    parser.add_argument(
        "--strategy",
        help="Filter by experiment strategy (e.g., finetune_box_50)",
    )
    parser.add_argument(
        "--status",
        choices=["submitted", "failed"],
        help="Filter by job status (default: submitted only)",
        default="submitted",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        help="Specific outputs to download (e.g., final/test models). Default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without executing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for downloads (default: experiments/batch_downloads/<timestamp>)",
    )
    args = parser.parse_args()

    console = Console()
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent

    # Find or load manifest
    if args.manifest:
        manifest_path = args.manifest
        if not manifest_path.exists():
            console.print(f"[ERROR] Manifest not found: {manifest_path}")
            sys.exit(1)
    else:
        manifest_path = find_latest_manifest(project_root)
        if not manifest_path:
            console.print(
                "[ERROR] No manifest files found in experiments/batch_submissions/"
            )
            console.print("Run submit_batch_jobs.py first to create a manifest.")
            sys.exit(1)

    # Load manifest
    manifest = load_manifest(manifest_path, console)

    # Filter jobs
    console.print("\nFiltering jobs...")
    filtered_jobs = filter_jobs(
        manifest,
        model=args.model,
        strategy=args.strategy,
        status=args.status,
    )

    if not filtered_jobs:
        console.print("[WARN] No jobs match the filter criteria")
        sys.exit(0)

    console.print(f"[OK] Found {len(filtered_jobs)} jobs matching filters:")
    if args.model:
        console.print(f"  Model: {args.model}")
    if args.strategy:
        console.print(f"  Strategy: {args.strategy}")
    if args.status:
        console.print(f"  Status: {args.status}")

    # Setup output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        manifest_timestamp = manifest["submission_time"]
        output_base = (
            project_root / "experiments" / "batch_downloads" / manifest_timestamp
        )

    console.print(f"\nOutput directory: {output_base}")

    # Connect to Azure ML (skip for dry run)
    ml_client = None
    if not args.dry_run:
        console.print("\nConnecting to Azure ML...")
        try:
            import os

            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

            if not all([subscription_id, resource_group, workspace_name]):
                console.print("[ERROR] Azure environment variables not set")
                console.print(
                    "Required: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME"
                )
                sys.exit(1)

            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
            console.print("[OK] Connected to Azure ML workspace")

        except Exception as e:
            console.print(f"[ERROR] Connecting to Azure ML: {e}")
            sys.exit(1)

    # Download artifacts
    console.print(f"\n{'=' * 80}")
    console.print(f"Downloading artifacts from {len(filtered_jobs)} jobs")
    console.print(f"{'=' * 80}\n")

    success_count = 0
    failed_jobs = []

    for job_record in track(filtered_jobs, description="Downloading..."):
        job_name = job_record.get("job_name")
        if not job_name:
            console.print(f"[WARN] Job {job_record.get('index')} has no job_name")
            continue

        # Create job-specific output directory
        model = job_record.get("model", "unknown")
        strategy = job_record.get("experiment_strategy", "unknown")
        job_output_dir = output_base / job_name / f"{model}_{strategy}"

        success = download_job_artifacts(
            ml_client=ml_client,
            console=console,
            job_name=job_name,
            output_dir=job_output_dir,
            outputs=args.outputs,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
        else:
            failed_jobs.append((job_name, model, strategy))

    # Summary
    console.print(f"\n{'=' * 80}")
    console.print("Download complete")
    console.print(f"{'=' * 80}")
    console.print(f"✓ Successful: {success_count}/{len(filtered_jobs)}")

    if failed_jobs:
        console.print(f"✗ Failed: {len(failed_jobs)}")
        console.print("\nFailed downloads:")
        for job_name, model, strategy in failed_jobs:
            console.print(f"  - {job_name} ({model} / {strategy})")
        sys.exit(1)
    else:
        console.print("\n✅ All artifacts downloaded successfully!")
        if not args.dry_run:
            console.print(f"📁 Output directory: {output_base}")


if __name__ == "__main__":
    main()
