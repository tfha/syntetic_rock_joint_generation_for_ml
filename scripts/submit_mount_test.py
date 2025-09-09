import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, UserIdentityConfiguration, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_ML_WORKSPACE"]

ml_client = MLClient(
    AzureCliCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

job = command(
    code="./",
    command="python scripts/azure_mount_test.py",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",
    compute="Standard-NC6s-v3",
    inputs={
        "images_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:rock_images:20250507.1543",
            mode=InputOutputModes.DOWNLOAD,  # temp to prove the asset resolves
        ),
        "masks_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:rock_masks:20250507.1543",
            mode=InputOutputModes.RO_MOUNT,
        ),
        "splits_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:split_verification_box:20250522.1451",
            mode=InputOutputModes.RO_MOUNT,
        ),
    },
    experiment_name="mount-test-experiment",
    identity=UserIdentityConfiguration(),
)

print("IDENTITY OBJ:", job.identity)
rest = job._to_rest_object()
rest_identity = (
    rest.get("identity") if isinstance(rest, dict) else getattr(rest, "identity", None)
)
print("REST HAS ID:", rest_identity)

ml_client.jobs.create_or_update(job)
