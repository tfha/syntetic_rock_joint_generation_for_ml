import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# ---------- Pre-reqs ----------
# 1) Enable System-assigned Managed Identity on the *compute cluster* you use.
# 2) Grant that MI "Storage Blob Data Reader" on the storage account(s) backing your datastores.
# 3) If ADLS Gen2 (HNS=true), ensure container/path ACLs allow traverse/read for the MI.

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_ML_WORKSPACE"]

# Use your CLI login only to TALK to AML; the *run itself* will use the cluster's Managed Identity.
ml_client = MLClient(
    AzureCliCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# ⚠️ Use your *real* cluster name here (not the SKU label).
# e.g., "nc6v3-cluster" not "Standard-NC6s-v3"
COMPUTE = os.environ.get("AZ_ML_COMPUTE", "Standard-NC6s-v3")

job = command(
    code="./",
    command="python scripts/azure_mount_test.py",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",
    compute=COMPUTE,
    # All inputs mount-readonly to exercise the MI path
    inputs={
        "images_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:rock_images:20250507.1543",
            mode=InputOutputModes.RO_MOUNT,
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
    identity=ManagedIdentityConfiguration(),  # <-- the key line
)

# Quick sanity: show identity in the REST payload
rest = job._to_rest_object()
rest_identity = (
    rest.get("identity") if isinstance(rest, dict) else getattr(rest, "identity", None)
)
print("REST identity payload:", rest_identity)

submitted = ml_client.jobs.create_or_update(job)
print("Submitted job name:", submitted.name)
print(
    "Studio URL:",
    submitted.services.get("Studio").endpoint if submitted.services else "(no svc)",
)

# Optional: stream logs so you immediately see the mount output
ml_client.jobs.stream(submitted.name)
