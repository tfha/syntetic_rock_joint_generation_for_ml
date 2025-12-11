"""Download example prediction images from Azure ML jobs.

This script downloads example images showing model predictions (original, ground truth, predicted)
from Azure ML blob storage, organized by job name and epoch.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_job_display_name(ml_client: MLClient, job_name: str) -> str:
    """Get the display name for a job."""
    try:
        job = ml_client.jobs.get(job_name)
        return job.display_name if job.display_name else job_name
    except Exception as e:
        print(f"Warning: Could not get display name for {job_name}: {e}")
        return job_name


def parse_epoch_number(epoch_folder: str) -> int | None:
    """Extract epoch number from folder name like 'epoch_5' or 'epoch_31'."""
    if epoch_folder.startswith("epoch_"):
        try:
            return int(epoch_folder.split("_")[1])
        except (IndexError, ValueError):
            return None
    return None


def get_stage_epochs_from_metrics(
    blob_service_client: BlobServiceClient,
    job_name: str,
) -> dict[int, int]:
    """Read metrics CSV and return mapping of epoch -> stage number.

    Returns:
        Dictionary mapping epoch number to stage number (1 or 2)
    """
    container_client = blob_service_client.get_container_client("azureml")

    # Look for metrics CSV
    job_blob_prefix = f"ExperimentRun/dcid.{job_name}/"

    try:
        blobs = container_client.list_blobs(name_starts_with=job_blob_prefix)
        metrics_blob = None

        for blob in blobs:
            if blob.name.endswith("_metrics.csv"):
                metrics_blob = blob.name
                break

        if not metrics_blob:
            print(f"  Warning: No metrics CSV found for {job_name}")
            return {}

        # Download and parse metrics CSV
        blob_client = container_client.get_blob_client(metrics_blob)
        csv_data = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(csv_data))

        # Build epoch -> stage mapping
        epoch_to_stage = {}
        for _, row in df.iterrows():
            epoch = int(row["epoch"])
            stage = int(row["stage"])
            epoch_to_stage[epoch] = stage

        return epoch_to_stage

    except Exception as e:
        print(f"  Warning: Could not read metrics for {job_name}: {e}")
        return {}


def download_job_images(
    blob_service_client: BlobServiceClient,
    job_name: str,
    display_name: str,
    output_dir: Path,
    strategy: str,
) -> int:
    """Download example images from a specific job.

    Args:
        blob_service_client: Azure blob service client
        job_name: Azure ML job name
        display_name: Display name for organizing downloads
        output_dir: Base output directory
        strategy: Training strategy ('simplemixed' or 'finetune')

    Returns:
        Number of images downloaded
    """
    container_client = blob_service_client.get_container_client("azureml")

    # For finetune, get stage information from metrics CSV
    epoch_to_stage = {}
    if strategy == "finetune":
        epoch_to_stage = get_stage_epochs_from_metrics(blob_service_client, job_name)

    # Determine target epochs for finetune
    target_epochs = set()
    stage2_first_epoch = None
    stage2_fifth_epoch = None
    if strategy == "finetune" and epoch_to_stage:
        # Find epochs for stage 1: get epochs 5 and 10
        stage1_epochs = [ep for ep, st in epoch_to_stage.items() if st == 1]
        stage1_epochs.sort()

        if 5 in stage1_epochs:
            target_epochs.add(5)
        if 10 in stage1_epochs:
            target_epochs.add(10)

        # Find epochs for stage 2: get 1st and 5th
        stage2_epochs = [ep for ep, st in epoch_to_stage.items() if st == 2]
        stage2_epochs.sort()

        if len(stage2_epochs) > 0:
            stage2_first_epoch = stage2_epochs[0]  # First stage 2 epoch
            target_epochs.add(stage2_first_epoch)
            print(f"  Stage 2 starts at epoch {stage2_first_epoch}")
        if len(stage2_epochs) >= 5:
            stage2_fifth_epoch = stage2_epochs[4]  # 5th stage 2 epoch
            target_epochs.add(stage2_fifth_epoch)
            print(f"  Stage 2 fifth epoch is {stage2_fifth_epoch}")

    # Base path for this job
    job_blob_prefix = f"ExperimentRun/dcid.{job_name}/outputs/models/example_images/"

    # List all blobs under this prefix
    try:
        blobs = list(container_client.list_blobs(name_starts_with=job_blob_prefix))
    except Exception as e:
        print(f"Error listing blobs for {job_name}: {e}")
        return 0

    downloaded = 0

    for blob in blobs:
        blob_name = blob.name

        # Only download .png files
        if not blob_name.endswith(".png"):
            continue

        # Parse the blob path to understand structure
        # Format: ExperimentRun/dcid.{job}/outputs/models/example_images/{timestamp}/{epoch_or_final}/{stage}/{sample}.png
        parts = blob_name.split("/")

        if len(parts) < 8:
            continue

        epoch_folder = parts[6]  # e.g., "epoch_5", "final"
        stage_folder = parts[7]  # e.g., "test", "stage1_val", "stage2_val"
        filename = parts[8]  # e.g., "sample_0.png"

        should_download = False

        # Check download criteria based on strategy
        if epoch_folder == "final" and stage_folder == "test":
            should_download = True
        elif strategy == "simplemixed":
            # SimpleMixed: test stage, epochs 5, 10, 15, and 20
            if stage_folder == "test" and epoch_folder in [
                "epoch_5",
                "epoch_10",
                "epoch_15",
                "epoch_20",
            ]:
                should_download = True
        elif strategy == "finetune":
            # For finetune, use the target epochs we determined from metrics
            epoch_num = parse_epoch_number(epoch_folder)
            if epoch_num in target_epochs:
                # Download val images only (not train)
                if "val" in stage_folder:
                    should_download = True

        if not should_download:
            continue

        # Create flattened output directory structure
        # Structure: {output_dir}/{display_name}/{epoch_folder}/
        epoch_num = parse_epoch_number(epoch_folder)
        if (
            strategy == "finetune"
            and stage_folder == "stage2_val"
            and epoch_num is not None
        ):
            # Use descriptive names for stage 2 epochs
            if epoch_num == stage2_first_epoch:
                renamed_epoch_folder = f"epoch_{epoch_num}_stage2_first_epoch"
            elif epoch_num == stage2_fifth_epoch:
                renamed_epoch_folder = f"epoch_{epoch_num}_stage2_fifth_epoch"
            else:
                renamed_epoch_folder = epoch_folder
        else:
            renamed_epoch_folder = epoch_folder

        output_subdir = output_dir / display_name / renamed_epoch_folder
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_path = output_subdir / filename

        # Download the blob
        try:
            blob_client = container_client.get_blob_client(blob_name)
            with open(output_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            downloaded += 1
            print(f"  Downloaded: {renamed_epoch_folder}/{filename}")
        except Exception as e:
            print(f"  Error downloading {blob_name}: {e}")

    return downloaded


def main() -> None:
    """Download example images from Azure ML jobs."""
    parser = argparse.ArgumentParser(
        description="Download example prediction images from Azure ML jobs"
    )
    parser.add_argument(
        "--jobs-file",
        type=str,
        required=True,
        help="Path to text file containing job names (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/images",
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--use-display-names",
        action="store_true",
        help="Use job display names instead of job names for organizing downloads (slower, requires API calls)",
    )

    args = parser.parse_args()

    # Load job names
    jobs_file = Path(args.jobs_file)
    if not jobs_file.exists():
        print(f"Error: Jobs file not found: {jobs_file}")
        return

    with open(jobs_file) as f:
        job_names = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(job_names)} job names from {jobs_file}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Azure clients
    print("\nConnecting to Azure ML...")
    credential = DefaultAzureCredential()

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE")

    if not all([subscription_id, resource_group, workspace_name]):
        print("Error: Azure environment variables not set")
        print(
            "Required: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE"
        )
        return

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Get datastore info to construct blob service client
    datastore = ml_client.datastores.get_default()
    account_url = f"https://{datastore.account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url, credential=credential)

    print(f"Connected to storage account: {datastore.account_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Download images for each job
    total_downloaded = 0
    total_jobs = len(job_names)

    for i, job_name in enumerate(job_names, 1):
        # Get display name if requested
        if args.use_display_names:
            display_name = get_job_display_name(ml_client, job_name)
        else:
            display_name = job_name

        # Determine strategy from job name or display name
        # SimpleMixed jobs have "simplemixed" in the name, finetune have "finetune"
        if "simplemixed" in display_name.lower():
            strategy = "simplemixed"
        elif "finetune" in display_name.lower():
            strategy = "finetune"
        else:
            # Try to infer from job metadata
            try:
                job = ml_client.jobs.get(job_name)
                if job.display_name and "simplemixed" in job.display_name.lower():
                    strategy = "simplemixed"
                elif job.display_name and "finetune" in job.display_name.lower():
                    strategy = "finetune"
                else:
                    print(
                        f"Warning: Cannot determine strategy for {job_name}, skipping..."
                    )
                    continue
            except Exception:
                print(f"Warning: Cannot determine strategy for {job_name}, skipping...")
                continue

        print(
            f"[{i}/{total_jobs}] Processing {job_name} (strategy: {strategy}, as: {display_name})..."
        )

        downloaded = download_job_images(
            blob_service_client,
            job_name,
            display_name,
            output_dir,
            strategy,
        )

        total_downloaded += downloaded
        print(f"  → Downloaded {downloaded} images")

    print(f"\n✓ Complete! Downloaded {total_downloaded} images from {total_jobs} jobs")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
