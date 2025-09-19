import os

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"],
)

job = command(
    code="./",
    command='bash -lc "env | grep AZUREML || true; ls -la /mnt || true; ls -la /mnt/azureml || true; ls -la /mnt/azureml/inputs || true; sleep 5"',
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",
    compute=os.environ.get("AZ_ML_COMPUTE", "NC64as-T4-v3"),
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
ml.jobs.create_or_update(job)
