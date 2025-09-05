"""
Quick check: instantiate MLClient and attempt list(ml_client.datastores.list()).

Reads AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE from env/.env.
Prints success or detailed error to help diagnose RBAC/credential issues.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_core import configure_azure_logging_and_warning

# Validate config using our Pydantic schema (repository convention)
from ml_segmentation.schema_config import ConfigSchema


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> int:
    # Reduce noisy Azure SDK and HTTP logs for clearer permission diagnostics
    configure_azure_logging_and_warning()

    # Initialize config the same way as in azure_submit_job.py
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)

    load_dotenv()

    sub = os.getenv("AZURE_SUBSCRIPTION_ID")
    rg = os.getenv("AZURE_RESOURCE_GROUP")
    ws = os.getenv("AZURE_ML_WORKSPACE")

    if not sub or not rg or not ws:
        print(
            "Missing required env vars: AZURE_SUBSCRIPTION_ID/AZURE_RESOURCE_GROUP/AZURE_ML_WORKSPACE",
            file=sys.stderr,
        )
        return 2

    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except Exception as ie:
        print(
            "Failed to import Azure SDK packages. Ensure azure-ai-ml and azure-identity are installed.",
            file=sys.stderr,
        )
        print(f"Import error: {ie}", file=sys.stderr)
        return 3

    print(f"Creating MLClient for workspace '{ws}' in resource group '{rg}'...")
    try:
        cred = DefaultAzureCredential()
        ml_client = MLClient(
            credential=cred,
            subscription_id=sub,
            resource_group_name=rg,
            workspace_name=ws,
        )
        print("Connected to Azure ML workspace.")
    except Exception as e:
        print(
            "Failed to create MLClient (authentication/workspace access issue):",
            file=sys.stderr,
        )
        print(repr(e), file=sys.stderr)
        traceback.print_exc()
        return 4

    print("Attempting to list datastores (permission probe)...")
    try:
        stores = list(ml_client.datastores.list())
        print(f"Success. Found {len(stores)} datastores.")
        if stores:
            # print a couple of names as a sanity check
            for s in stores[:3]:
                try:
                    print(f"- {getattr(s, 'name', '<noname>')}")
                except Exception:
                    pass
    except Exception as e:
        print("Permission/Access error when listing datastores:", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        traceback.print_exc()
        return 5

    # Additional check: does the configured compute (from config) have storage access?
    compute_name: str | None = pcfg.azure_ml.compute_name

    print(f"\nChecking compute-to-storage access for compute: {compute_name} ...")
    try:
        comp = ml_client.compute.get(compute_name)
    except Exception as e:
        print(f"[Warn] Unable to get compute '{compute_name}': {e}", file=sys.stderr)
        return 0

    # Extract principal IDs from compute identity
    principal_ids: list[str] = []
    try:
        identity = getattr(comp, "identity", None)
        if identity is None:
            print("[Warn] Compute has no managed identity configured.")
        else:
            # System-assigned MI
            sa_principal = getattr(identity, "principal_id", None)
            if sa_principal:
                principal_ids.append(sa_principal)
            # User-assigned MIs
            uamis = getattr(identity, "user_assigned_identities", None)
            if isinstance(uamis, dict):
                for _, ident in uamis.items():
                    pid = (
                        ident.get("principal_id")
                        if isinstance(ident, dict)
                        else getattr(ident, "principal_id", None)
                    )
                    if pid:
                        principal_ids.append(pid)
    except Exception:
        pass

    if not principal_ids:
        print(
            "[Warn] No principal IDs found for compute identity; cannot verify storage RBAC."
        )
        return 0

    # Determine storage account for default datastore
    try:
        default_store_name = "workspaceblobstore"
        try:
            ds = ml_client.datastores.get(default_store_name)
        except Exception:
            # Fallback: pick first AzureBlob datastore
            ds = next(
                (d for d in stores if getattr(d, "type", "").lower() == "azure_blob"),
                None,
            )
        account_name = getattr(ds, "account_name", None) if ds else None
        if not account_name:
            print(
                "[Info] Could not determine default storage account from datastore; skipping RBAC verification."
            )
            return 0
    except Exception as e:
        print(f"[Info] Failed to resolve storage account from datastore: {e}")
        return 0

    # Use Azure CLI if available to query role assignments
    az_path = shutil.which("az") or shutil.which("az.cmd")
    if not az_path:
        print(
            "[Info] Azure CLI not found; manual check required. Suggested steps:\n"
            f"- Get storage account id: az storage account show -n {account_name} -g {rg} --query id -o tsv\n"
            "- For each principalId (compute identity), run:\n"
            "  az role assignment list --assignee <principalId> --scope <storageId> --query '[].roleDefinitionName' -o tsv\n"
            "  Ensure role includes 'Storage Blob Data Reader' or 'Storage Blob Data Contributor'."
        )
        return 0

    try:
        # Get storage resource ID
        cmd = [
            az_path,
            "storage",
            "account",
            "show",
            "-n",
            account_name,
            "-g",
            rg,
            "--query",
            "id",
            "-o",
            "tsv",
        ]
        storage_id = subprocess.check_output(cmd, text=True).strip()
        if not storage_id:
            print(
                "[Warn] Could not resolve storage account resource id; skipping RBAC verification."
            )
            return 0

        required_roles = {"Storage Blob Data Reader", "Storage Blob Data Contributor"}
        any_ok = False
        for pid in principal_ids:
            try:
                roles_cmd = [
                    az_path,
                    "role",
                    "assignment",
                    "list",
                    "--assignee",
                    pid,
                    "--scope",
                    storage_id,
                    "--query",
                    "[].roleDefinitionName",
                    "-o",
                    "tsv",
                ]
                out = subprocess.check_output(roles_cmd, text=True)
                assigned = {r.strip() for r in out.splitlines() if r.strip()}
                has_required = bool(required_roles & assigned)
                print(
                    f"Principal {pid}: roles on storage → {', '.join(sorted(assigned)) or '<none>'}"
                )
                if has_required:
                    any_ok = True
            except subprocess.CalledProcessError as ce:
                print(f"[Warn] Failed to list role assignments for {pid}: {ce}")
                continue

        if any_ok:
            print(
                "[OK] At least one compute identity has required Blob Data role on the storage account."
            )
        else:
            print(
                "[Warn] None of the compute identities appear to have 'Storage Blob Data Reader/Contributor' on the storage account."
                " Jobs may fail to read/write data. Assign a Blob Data role to the compute's managed identity at the storage account scope."
            )
    except FileNotFoundError as fnf:
        print(
            f"[Warn] Azure CLI executable not found when invoking: {fnf}. Manual steps:\n"
            f"- Get storage account id: az storage account show -n {account_name} -g {rg} --query id -o tsv\n"
            "- az role assignment list --assignee <principalId> --scope <storageId> --query '[].roleDefinitionName' -o tsv"
        )
    except subprocess.CalledProcessError as ce:
        print(f"[Warn] Azure CLI check failed: {ce}")
    except Exception as e:
        print(f"[Warn] Unexpected error during RBAC verification: {e}")

    return 0


if __name__ == "__main__":
    main()
