import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# --- env / workspace ---
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

SUB = os.environ["AZURE_SUBSCRIPTION_ID"]
RG = os.environ["AZURE_RESOURCE_GROUP"]
WS = os.environ["AZURE_ML_WORKSPACE"]
COMPUTE = os.environ.get("AZ_ML_COMPUTE", "YOUR-CLUSTER-NAME")  # real cluster name

ml_client = MLClient(
    AzureCliCredential(), subscription_id=SUB, resource_group_name=RG, workspace_name=WS
)

print(f"Connected to workspace: {ml_client.workspace_name} (rg={RG}, sub={SUB})")


# --- resolve assets here (fail early if wrong) ---
def ensure_asset(name: str, version: str) -> str:
    try:
        da = ml_client.data.get(name=name, version=version)
        print(f"✓ asset resolved: {da.name}:{da.version}  short_uri={da.short_uri}")
        return f"azureml:{da.name}:{da.version}"
    except ResourceNotFoundError:
        raise SystemExit(
            f"[ERROR] Data asset not found in this workspace: {name}:{version}"
        ) from None


IMAGES = ensure_asset("rock_images", "20250507.1543")
MASKS = ensure_asset("rock_masks", "20250507.1543")
SPLITS = ensure_asset("split_verification_box", "20250522.1451")

# --- job ---
job = command(
    code="./",
    command="python scripts/azure_mount_test.py",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/41",
    compute=COMPUTE,
    inputs={
        "images_data": Input(
            type=AssetTypes.URI_FOLDER, path=IMAGES, mode=InputOutputModes.RO_MOUNT
        ),
        "masks_data": Input(
            type=AssetTypes.URI_FOLDER, path=MASKS, mode=InputOutputModes.RO_MOUNT
        ),
        "splits_data": Input(
            type=AssetTypes.URI_FOLDER, path=SPLITS, mode=InputOutputModes.RO_MOUNT
        ),
    },
    experiment_name="mount-test-experiment",
    identity=ManagedIdentityConfiguration(),
)

rest = job._to_rest_object()
print(
    "REST identity:",
    rest.get("identity") if isinstance(rest, dict) else getattr(rest, "identity", None),
)
print(
    "REST inputs:",
    list((rest.get("inputs") or {}).keys()) if isinstance(rest, dict) else "unknown",
)

submitted = ml_client.jobs.create_or_update(job)
print("Submitted job:", submitted.name)
ml_client.jobs.stream(submitted.name)
