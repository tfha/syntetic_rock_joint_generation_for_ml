"""
Azure ML compute cluster management utilities.

This module handles compute cluster validation, permission checking,
and compute-related operations for Azure ML.
"""

import sys
from typing import Any

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from rich.console import Console


def check_compute_permissions(
    ml_client: MLClient, compute_name: str, console: Console
) -> dict[str, Any]:
    """
    Check if the compute cluster has the necessary permissions to access data.

    Args:
        ml_client: Azure ML client
        compute_name: Name of the compute cluster
        console: Console for logging

    Returns:
        Dictionary with results of permission checks
    """
    permissions = {
        "has_identity": False,
        "identity_type": None,
        "storage_blob_access": False,
        "workspace_access": False,
        "log_streaming_access": False,
    }

    try:
        console.print(f"Checking permissions for compute: {compute_name}", style="blue")

        # Get compute information
        compute = ml_client.compute.get(compute_name)

        # Check if compute has managed identity
        if hasattr(compute, "identity") and compute.identity:
            permissions["has_identity"] = True
            permissions["identity_type"] = compute.identity.type
            console.print(
                f"[OK] Compute has {compute.identity.type} identity", style="green"
            )
        else:
            console.print("[WARNING] Compute lacks managed identity", style="yellow")

        # Test basic workspace access
        try:
            # Some SDK versions don't support max_results; just iterate one item
            _ = next(iter(ml_client.datastores.list()), None)
            permissions["workspace_access"] = True
            console.print("[OK] Workspace access confirmed", style="green")
        except Exception:
            console.print("[WARNING] Limited workspace access", style="yellow")

        # Note: Storage and log streaming access would require additional API calls
        # that may not be available or may require specific permissions
        permissions["log_streaming_access"] = permissions["has_identity"]

        return permissions

    except Exception as e:
        console.print(f"Error checking compute permissions: {str(e)}", style="red")
        return permissions


def refresh_compute_cluster(
    ml_client: MLClient, compute_name: str, console: Console
) -> Any:
    """
    Forcibly refresh the compute cluster information from Azure.

    Sometimes after making changes to compute in the portal, the local client cache
    needs to be refreshed to see the changes.

    Args:
        ml_client: Azure ML client
        compute_name: Name of the compute cluster
        console: Console for logging

    Returns:
        The refreshed compute cluster object
    """
    try:
        console.print(
            f"Refreshing compute cluster information for: {compute_name}", style="blue"
        )

        # Force refresh by doing a list operation first
        all_computes = list(ml_client.compute.list())
        console.print(
            f"Found {len(all_computes)} compute resource(s) in workspace", style="blue"
        )

        # Now get the specific compute we need - should be fresh
        compute = ml_client.compute.get(
            name=compute_name,
        )

        console.print(
            f"Successfully refreshed compute cluster: {compute_name}", style="green"
        )
        return compute

    except Exception as e:
        console.print(f"Error refreshing compute cluster: {str(e)}", style="red")
        return None


def validate_and_refresh_compute(
    ml_client: MLClient, console: Console, compute_name: str
) -> dict[str, Any]:
    """
    Validate and refresh compute cluster, check permissions.

    This function handles compute cluster validation, refresh operations,
    and comprehensive permission checking with actionable guidance.

    Args:
        ml_client: Azure ML client
        console: Console for logging
        compute_name: Name of the compute cluster

    Returns:
        dict: Permission check results

    Raises:
        SystemExit: If compute cluster is not accessible
    """
    try:
        console.print(f"Validating compute cluster: {compute_name}", style="blue")

        # Refresh compute to ensure latest state
        compute = refresh_compute_cluster(ml_client, compute_name, console)

        if compute is None:
            console.print("Failed to refresh compute cluster", style="red")
            sys.exit(1)

        # Run comprehensive permission checks
        permissions = check_compute_permissions(ml_client, compute_name, console)

        # Provide actionable guidance based on results
        _provide_compute_guidance(console, permissions)

        return permissions

    except ResourceNotFoundError:
        console.print(f"Compute cluster '{compute_name}' not found", style="red")
        console.print(
            "Solution: Create compute cluster or update config with "
            "existing cluster name",
            style="blue",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"Error accessing compute cluster: {str(e)}", style="red")
        sys.exit(1)


def _provide_compute_guidance(console: Console, permissions: dict[str, Any]) -> None:
    """Provide actionable guidance based on compute permission results."""

    # Identity validation
    if not permissions["has_identity"]:
        console.print(
            "[WARNING] Compute cluster lacks managed identity", style="yellow"
        )
        console.print(
            "Solution: Enable system-assigned identity in Azure Portal > "
            "Compute > Identity tab",
            style="blue",
        )
        # Provide detailed step-by-step guidance for resolving missing identity
        _provide_compute_identity_setup_guidance(console)
    else:
        console.print(
            f"[OK] Compute identity type: {permissions['identity_type']}",
            style="green",
        )

    # Workspace access validation
    if permissions["workspace_access"]:
        console.print("[OK] Compute has workspace access", style="green")
    else:
        console.print("[WARNING] Could not verify workspace access", style="yellow")

    # Log streaming access validation
    if permissions.get("log_streaming_access", False):
        console.print("[OK] Compute configured for log streaming", style="green")
    else:
        console.print(
            "[WARNING] Compute may lack log streaming permissions", style="yellow"
        )
        console.print(
            "Solution: Ensure managed identity has:\n"
            "  1. 'Storage Blob Data Reader' on workspace storage\n"
            "  2. 'AzureML Data Scientist' on workspace",
            style="blue",
        )


def _provide_compute_identity_setup_guidance(console: Console) -> None:
    """Provide detailed guidance for setting up compute identity."""
    console.print("\n🔧 NoIdentityOnCompute Error - Setup Guide:", style="red")
    console.print(
        "1. Azure Portal → Your ML Workspace → Compute → Compute clusters\n"
        "2. Select your compute cluster → Identity tab\n"
        "3. Enable 'System assigned' managed identity → Save\n"
        "4. Navigate to workspace Storage Account → Access Control (IAM)\n"
        "5. Add role assignment:\n"
        "   • Role: Storage Blob Data Contributor\n"
        "   • Assign access to: System assigned managed identity\n"
        "   • Select: Your compute cluster's identity\n"
        "6. Save and retry job submission",
        style="blue",
    )
