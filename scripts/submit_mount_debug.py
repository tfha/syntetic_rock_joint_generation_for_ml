import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables from repo .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Read required workspace settings (fail fast if missing)
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_ML_WORKSPACE"]
compute_name = os.environ.get("AZURE_ML_COMPUTE", "Standard-NC6s-v3")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

job = command(
    code="./",
    command=(
        'bash -lc "env | grep AZUREML || true; '
        "ls -la /mnt || true; ls -la /mnt/azureml || true; "
        'ls -la /mnt/azureml/inputs || true; sleep 5"'
    ),
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",
    compute=compute_name,
    inputs={
        "images_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:rock_images:20250507.1543",
            mode=InputOutputModes.RO_MOUNT,
        ),
    },
    identity=ManagedIdentityConfiguration(),
    experiment_name="mount-debug",
)

ml_client.jobs.create_or_update(job)
print("Submitted mount-debug job")
