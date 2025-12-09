"""
Azure ML job management utilities.

This module handles Azure ML job monitoring, output management,
and error handling for Azure ML operations.
"""

from typing import Any

from rich.console import Console

from .azure_compute import (
    _provide_compute_identity_setup_guidance as _provide_identity_setup_guidance,
)


def download_job_outputs(
    ml_client, console: Console, job_run: Any, display_name: str
) -> None:
    """
    Download outputs from a completed Azure ML job.

    Args:
        ml_client: Azure ML client
        console: Console for logging
        job_run: Azure ML job run object
        display_name: Display name for the job
    """
    try:
        console.print("Downloading job outputs...", style="info")

        # Download job outputs
        ml_client.jobs.download(
            name=job_run.name,
            download_path="./outputs",
        )

        console.print("[OK] Job outputs downloaded to ./outputs", style="success")

    except Exception as e:
        console.print(f"Failed to download job outputs: {str(e)}", style="warning")
        console.print(
            f"You can manually download outputs from: {job_run.studio_url}",
            style="info",
        )


def get_job_details(ml_client, console: Console, job_run: Any) -> None:
    """
    Get and display detailed information about a job run.

    Args:
        ml_client: Azure ML client
        console: Console for logging
        job_run: Azure ML job run object
    """
    try:
        console.print("Getting job details...", style="info")

        # Get fresh job details
        job_details = ml_client.jobs.get(job_run.name)

        console.print(f"Job Name: {job_details.name}", style="info")
        console.print(f"Status: {job_details.status}", style="info")
        console.print(f"Compute: {job_details.compute}", style="info")
        console.print(f"Environment: {job_details.environment}", style="info")

        if hasattr(job_details, "studio_url"):
            console.print(f"Studio URL: {job_details.studio_url}", style="info")

    except Exception as e:
        console.print(f"Failed to get job details: {str(e)}", style="warning")


def handle_job_submission_error(console: Console, error: Exception) -> None:
    """
    Handle and provide guidance for job submission errors.

    Args:
        console: Console for logging
        error: The exception that occurred
    """
    error_str = str(error).lower()

    if "noidentityoncompute" in error_str:
        console.print("NoIdentityOnCompute Error detected", style="error")
        _provide_identity_setup_guidance(console)
    elif "insufficient permissions" in error_str:
        console.print("Insufficient permissions error detected", style="error")
        console.print(
            "Solution: Ensure compute identity has required permissions", style="info"
        )
    else:
        console.print(f"Job submission failed: {str(error)}", style="error")
        console.print(
            "Check job configuration and Azure ML workspace setup", style="info"
        )


def handle_log_streaming_error(
    console: Console, error: Exception, job_run: Any = None, attempt: int = 1
) -> None:
    """
    Handle and provide guidance for log streaming errors.

    Args:
        console: Console for logging
        error: The exception that occurred
        job_run: Azure ML job run object (optional)
        attempt: Current attempt number
    """
    error_str = str(error).lower()

    if "authentication" in error_str or "unauthorized" in error_str:
        console.print(
            f"Authentication error on attempt {attempt}: {str(error)}", style="warning"
        )
        console.print(
            "This may be due to insufficient permissions or token expiry", style="info"
        )
    elif "timeout" in error_str:
        console.print(
            f"Timeout error on attempt {attempt}: {str(error)}", style="warning"
        )
        console.print("This may be a temporary network issue", style="info")
    else:
        console.print(
            f"Log streaming error on attempt {attempt}: {str(error)}", style="warning"
        )
