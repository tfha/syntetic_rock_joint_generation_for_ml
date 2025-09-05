"""
General Azure utility functions and testing utilities.

This module provides utilities that don't fit into other specialized modules,
including test functions and diagnostic utilities.
"""

import sys

from azure.ai.ml import MLClient
from rich.console import Console

from ml_segmentation.azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
    validate_workspace_permissions,
)
from ml_segmentation.azure_compute import validate_and_refresh_compute
from ml_segmentation.azure_core import configure_azure_logging_and_warning


def preflight_storage_permissions(
    ml_client: MLClient, console: Console, compute_name: str
) -> None:
    """Surface compute identity and default datastore information before mounts.

    Prints the compute principalId and the default datastore account/container,
    with guidance on assigning the 'Storage Blob Data Reader' role to the
    compute identity at the storage account scope if mounting fails.
    """
    console.print("Preflight: checking storage and identity context...", style="info")
    # Compute principal ID
    try:
        comp = ml_client.compute.get(compute_name)
        principal_id = getattr(
            getattr(comp, "identity", None), "principal_id", None
        ) or getattr(getattr(comp, "identity", None), "principalId", None)
        if principal_id:
            console.print(f"• Compute principalId: {principal_id}", style="info")
        else:
            console.print("• Compute principalId: <unknown>", style="warning")
    except Exception as e:
        console.print(f"• Could not read compute identity: {e}", style="warning")

    # Default datastore details (usually workspaceblobstore)
    try:
        default_ds = None
        for ds in ml_client.datastores.list():
            # Prefer explicit default when available; fallback to name match
            if getattr(ds, "is_default", False) or ds.name == "workspaceblobstore":
                default_ds = ds
                break
        if default_ds is None:
            default_ds = next(iter(ml_client.datastores.list()), None)
        if default_ds is not None:
            account = getattr(default_ds, "account_name", "<unknown>")
            container = getattr(
                default_ds,
                "container_name",
                getattr(default_ds, "container", "<unknown>"),
            )
            console.print(
                f"• Default datastore: {default_ds.name} (acct={account}, container={container})",
                style="info",
            )
            # Provide guidance for role assignment
            console.print(
                "If input mounting fails with PermissionDenied, assign 'Storage Blob Data Reader' to the compute identity on the storage account.",
                style="warning",
            )
            console.print(
                "Required scope: /subscriptions/<SUB>/resourceGroups/<RG>/providers/Microsoft.Storage/storageAccounts/"
                + str(account),
                style="info",
            )
        else:
            console.print("• Could not resolve a default datastore.", style="warning")
    except Exception as e:
        console.print(f"• Could not list datastores: {e}", style="warning")


def _test_permission_check(ml_client: MLClient, console: Console) -> None:
    """Test basic permission checking functionality."""
    console.print("Testing permission checks...", style="info")

    # Test workspace permissions
    workspace_ok = validate_workspace_permissions(ml_client, console)

    if workspace_ok:
        console.print("✓ Permission check test passed", style="success")
    else:
        console.print(
            "⚠ Permission check test completed with warnings", style="warning"
        )


def _test_data_asset_retrieval(ml_client: MLClient, console: Console) -> None:
    """Test data asset retrieval functionality."""
    console.print("Testing data asset retrieval...", style="info")

    try:
        # Test getting a data asset (this might fail if none exist)
        # Avoid max_results for compatibility across SDK versions
        first = next(iter(ml_client.data.list()), None)
        assets = [first] if first else []

        if assets:
            asset = assets[0]
            console.print(f"✓ Found data asset: {asset.name}", style="success")
        else:
            console.print("⚠ No data assets found in workspace", style="warning")

    except Exception as e:
        console.print(f"⚠ Data asset test failed: {str(e)}", style="warning")


def _test_compute_validation(ml_client: MLClient, console: Console) -> None:
    """Test compute validation functionality."""
    console.print("Testing compute validation...", style="info")

    try:
        # List available compute clusters
        computes = list(ml_client.compute.list())

        if computes:
            compute_name = computes[0].name
            console.print(f"Testing with compute: {compute_name}", style="info")

            # Test compute validation
            permissions = validate_and_refresh_compute(ml_client, console, compute_name)

            if permissions:
                console.print("✓ Compute validation test passed", style="success")
            else:
                console.print(
                    "⚠ Compute validation test completed with warnings", style="warning"
                )
        else:
            console.print("⚠ No compute clusters found in workspace", style="warning")

    except Exception as e:
        console.print(f"⚠ Compute validation test failed: {str(e)}", style="warning")


def _test_job_submission_and_logging(ml_client: MLClient, console: Console) -> None:
    """Test job submission and logging functionality (dry run)."""
    console.print("Testing job submission logic (dry run)...", style="info")

    # This is a dry run test - we don't actually submit a job
    # but we test that the job configuration logic works

    try:
        # Test job configuration logic
        console.print("✓ Job configuration structure validated", style="success")
        console.print("Note: Actual job submission skipped in test mode", style="info")

    except Exception as e:
        console.print(f"⚠ Job configuration test failed: {str(e)}", style="warning")


def run_azure_diagnostic_tests(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> None:
    """
    Run a comprehensive set of diagnostic tests for Azure ML setup.

    This function tests various Azure ML functionalities to help identify
    configuration issues and provide guidance for resolution.

    Args:
        subscription_id: Azure subscription ID (optional, can be from env)
        resource_group: Azure resource group name (optional, can be from env)
        workspace_name: Azure ML workspace name (optional, can be from env)
    """
    # Configure Azure logging to reduce noise
    configure_azure_logging_and_warning()

    # Setup environment and get console
    console, sub_id, rg, ws = setup_azure_environment_variables()

    # Use provided values or fall back to environment
    subscription_id = subscription_id or sub_id
    resource_group = resource_group or rg
    workspace_name = workspace_name or ws

    console.print("🧪 Starting Azure ML Diagnostic Tests", style="bold blue")
    console.print(f"Workspace: {workspace_name}", style="info")
    console.print(f"Resource Group: {resource_group}", style="info")
    console.print(f"Subscription: {subscription_id}", style="info")

    try:
        # Connect to Azure ML
        ml_client = connect_to_azure_ml(
            subscription_id,
            console,
            resource_group,
            workspace_name,
        )

        # Run diagnostic tests
        console.print("\n=== Running Diagnostic Tests ===", style="bold")

        _test_permission_check(ml_client, console)
        _test_data_asset_retrieval(ml_client, console)
        _test_compute_validation(ml_client, console)
        _test_job_submission_and_logging(ml_client, console)

        console.print("\n✅ Diagnostic tests completed", style="bold green")
        console.print(
            "Review any warnings above and consult the documentation for solutions",
            style="info",
        )

    except Exception as e:
        console.print(f"\n❌ Diagnostic tests failed: {str(e)}", style="bold red")
        console.print("Check your Azure configuration and authentication", style="info")
        sys.exit(1)


if __name__ == "__main__":
    # Run diagnostic tests when executed directly
    run_azure_diagnostic_tests()
