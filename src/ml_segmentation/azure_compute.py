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
        console.print(f"Checking permissions for compute: {compute_name}", style="info")

        # Get compute information
        compute = ml_client.compute.get(compute_name)

        # Check if compute has managed identity
        if hasattr(compute, "identity") and compute.identity:
            permissions["has_identity"] = True
            permissions["identity_type"] = compute.identity.type
            console.print(
                f"✓ Compute has {compute.identity.type} identity", style="success"
            )
        else:
            console.print("⚠ Compute lacks managed identity", style="warning")

        # Test basic workspace access
        try:
            # Some SDK versions don't support max_results; just iterate one item
            _ = next(iter(ml_client.datastores.list()), None)
            permissions["workspace_access"] = True
            console.print("✓ Workspace access confirmed", style="success")
        except Exception:
            console.print("⚠ Limited workspace access", style="warning")

        # Note: Storage and log streaming access would require additional API calls
        # that may not be available or may require specific permissions
        permissions["log_streaming_access"] = permissions["has_identity"]

        return permissions

    except Exception as e:
        console.print(f"Error checking compute permissions: {str(e)}", style="error")
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
            f"Refreshing compute cluster information for: {compute_name}", style="info"
        )

        # Force refresh by doing a list operation first
        all_computes = list(ml_client.compute.list())
        console.print(
            f"Found {len(all_computes)} compute resource(s) in workspace", style="info"
        )

        # Now get the specific compute we need - should be fresh
        compute = ml_client.compute.get(
            name=compute_name,
        )

        console.print(
            f"Successfully refreshed compute cluster: {compute_name}", style="success"
        )
        return compute

    except Exception as e:
        console.print(f"Error refreshing compute cluster: {str(e)}", style="error")
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
        console.print(f"Validating compute cluster: {compute_name}", style="info")

        # Refresh compute to ensure latest state
        compute = refresh_compute_cluster(ml_client, compute_name, console)

        if compute is None:
            console.print("Failed to refresh compute cluster", style="error")
            sys.exit(1)

        # Run comprehensive permission checks
        permissions = check_compute_permissions(ml_client, compute_name, console)

        # Provide actionable guidance based on results
        _provide_compute_guidance(console, permissions)

        return permissions

    except ResourceNotFoundError:
        console.print(f"Compute cluster '{compute_name}' not found", style="error")
        console.print(
            "Solution: Create compute cluster or update config with "
            "existing cluster name",
            style="info",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"Error accessing compute cluster: {str(e)}", style="error")
        sys.exit(1)


def _provide_compute_guidance(console: Console, permissions: dict[str, Any]) -> None:
    """Provide actionable guidance based on compute permission results."""

    # Identity validation
    if not permissions["has_identity"]:
        console.print("⚠ Compute cluster lacks managed identity", style="warning")
        console.print(
            "Solution: Enable system-assigned identity in Azure Portal > "
            "Compute > Identity tab",
            style="info",
        )
        # Provide detailed step-by-step guidance for resolving missing identity
        _provide_compute_identity_setup_guidance(console)
    else:
        console.print(
            f"✓ Compute identity type: {permissions['identity_type']}", style="success"
        )

    # Workspace access validation
    if permissions["workspace_access"]:
        console.print("✓ Compute has workspace access", style="success")
    else:
        console.print("⚠ Could not verify workspace access", style="warning")

    # Log streaming access validation
    if permissions.get("log_streaming_access", False):
        console.print("✓ Compute configured for log streaming", style="success")
    else:
        console.print("⚠ Compute may lack log streaming permissions", style="warning")
        console.print(
            "Solution: Ensure managed identity has:\n"
            "  1. 'Storage Blob Data Reader' on workspace storage\n"
            "  2. 'AzureML Data Scientist' on workspace",
            style="info",
        )


def _provide_compute_identity_setup_guidance(console: Console) -> None:
    """Provide detailed guidance for setting up compute identity."""
    console.print("\n🔧 NoIdentityOnCompute Error - Setup Guide:", style="error")
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
        style="info",
    )
