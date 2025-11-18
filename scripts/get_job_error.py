"""Get detailed error information from Azure ML job."""

import sys

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from ml_segmentation.azure_authentication import (
    setup_azure_environment_variables,
)

job_name = sys.argv[1] if len(sys.argv) > 1 else "bubbly_van_96xw6m0jk3_0"

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
print(f"Job: {job.name}")
print(f"Status: {job.status}")
print(f"Type: {job.type}")

# Get properties
if hasattr(job, "properties") and job.properties:
    print("\nProperties:")
    for key, value in job.properties.items():
        print(f"  {key}: {value}")

# Get error details
if hasattr(job, "error") and job.error:
    print("\nError:")
    if isinstance(job.error, dict):
        for key, value in job.error.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {job.error}")

# Try to stream logs
print(f"\nAttempting to stream logs for {job.name}...")
try:
    ml_client.jobs.stream(job.name)
except Exception as e:
    print(f"Could not stream logs: {e}")

    # Try to get output logs
    print("\nTrying alternative method to get logs...")
    try:
        # Download to a specific location
        ml_client.jobs.download(
            name=job.name, download_path=f"./temp_logs/{job.name}", all=True
        )
        print(f"Downloaded logs to ./temp_logs/{job.name}")
    except Exception as e2:
        print(f"Could not download logs: {e2}")
