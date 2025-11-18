"""Quick script to check Azure ML job status and errors."""

import sys

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from ml_segmentation.azure_authentication import (
    setup_azure_environment_variables,
)

job_name = sys.argv[1] if len(sys.argv) > 1 else "bubbly_van_96xw6m0jk3"

# Connect to Azure ML
console, subscription_id, resource_group, workspace_name = (
    setup_azure_environment_variables()
)
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group or "rg-rock-joint-detection",
    workspace_name or "ws-rock-joint-det",
)

# Get job details
job = ml_client.jobs.get(job_name)
print(f"Job Name: {job.name}")
print(f"Job Status: {job.status}")
print(f"Job Type: {job.type}")

if hasattr(job, "display_name"):
    print(f"Display Name: {job.display_name}")

# For sweep jobs, check child jobs
if job.type == "sweep":
    print("\n" + "=" * 60)
    print("Checking child jobs...")
    print("=" * 60)

    child_jobs = list(ml_client.jobs.list(parent_job_name=job_name))
    print(f"Total child jobs: {len(child_jobs)}")

    status_counts: dict[str, int] = {}
    failed_jobs = []

    for child in child_jobs:
        status = child.status
        status_counts[status] = status_counts.get(status, 0) + 1

        if status in ["Failed", "Canceled"]:
            failed_jobs.append(child)

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    if failed_jobs:
        print(f"\n{'=' * 60}")
        print(f"Failed/Canceled jobs ({len(failed_jobs)}):")
        print("=" * 60)

        for i, child in enumerate(failed_jobs[:5], 1):
            print(f"\n{i}. Job: {child.name}")
            print(f"   Status: {child.status}")

            # Try to get error details
            try:
                full_job = ml_client.jobs.get(child.name)
                if hasattr(full_job, "error") and full_job.error:
                    print(f"   Error Message: {full_job.error.get('message', 'N/A')}")
                    print(f"   Error Code: {full_job.error.get('code', 'N/A')}")
            except Exception as e:
                print(f"   Could not retrieve error details: {e}")

        if len(failed_jobs) > 5:
            print(f"\n... and {len(failed_jobs) - 5} more failed jobs")

        # Try to download logs for the first failed job
        print(f"\n{'=' * 60}")
        print("Downloading logs for first failed job...")
        print("=" * 60)
        try:
            first_failed = failed_jobs[0].name
            ml_client.jobs.download(
                name=first_failed, download_path="./temp_logs", all=True
            )
            print(f"Logs downloaded to ./temp_logs for job {first_failed}")
        except Exception as e:
            print(f"Could not download logs: {e}")

    # Show recent running/completed jobs
    print(f"\n{'=' * 60}")
    print("Recent child jobs:")
    print("=" * 60)
    for i, child in enumerate(child_jobs[:10], 1):
        print(f"{i}. {child.name} - Status: {child.status}")

else:
    # Single job
    if hasattr(job, "error") and job.error:
        print(f"\nError: {job.error}")
