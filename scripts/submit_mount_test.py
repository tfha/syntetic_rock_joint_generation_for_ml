import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_ML_WORKSPACE"]

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

job = command(
    code="./",  # repo root
    command="python scripts/azure_mount_test.py",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",  # or your preferred env
    compute="NC64as-T4-v3",
    inputs={
        "images_data": Input(
            type="uri_folder", path="azureml:rock_images:20250507.1543"
        ),
        "masks_data": Input(type="uri_folder", path="azureml:rock_masks:20250507.1543"),
        "splits_data": Input(
            type="uri_folder", path="azureml:split_verification_box:20250522.1451"
        ),
    },
    experiment_name="mount-test-experiment",  # <-- Add this line
)

ml_client.jobs.create_or_update(job)
