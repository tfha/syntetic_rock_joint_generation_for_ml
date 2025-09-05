"""
Submit Azure ML training job for rock mass segmentation.
Uses data assets for improved data management.

Architecture:
- Uses Azure ML SDK v2 for job submission
- Leverages data assets for efficient data versioning and management
- Implements best practices for error handling and retry logic
- Supports environment versioning and reproducibility
- Centralized configuration using Hydra
- MLflow integration for experiment tracking

This script handles:
1. Setup and validation of Azure ML environment
2. Data asset retrieval and validation
3. Compute target verification
4. Environment setup and versioning
5. Job configuration and submission
6. Output management and retrieval
"""

from datetime import datetime
from typing import Any

import hydra
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import ManagedIdentityConfiguration
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
    validate_workspace_permissions,
)
from ml_segmentation.azure_compute import validate_and_refresh_compute
from ml_segmentation.azure_core import (
    configure_azure_logging_and_warning,
    retry_azure_operation,
)
from ml_segmentation.azure_data_assets import (
    get_data_asset,
    retrieve_and_validate_data_assets,
)
from ml_segmentation.azure_diagnostics import preflight_storage_permissions
from ml_segmentation.azure_jobs import (
    download_job_outputs,
    get_job_details,
    handle_job_submission_error,
    handle_log_streaming_error,
)
from ml_segmentation.schema_config import ConfigSchema


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Submit Azure ML job with managed data assets using Command approach."""

    # Configure logging and warnings to reduce verbose Azure client output
    configure_azure_logging_and_warning()

    # 1. Initialize configuration
    ###########################################
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)

    # Setup environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = (
        setup_azure_environment_variables()
    )

    console.print("Starting Azure ML job submission", style="info")

    # Set job display name based on model and strategy
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = (
        f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-{timestamp}"
    )

    # 2. Connect to Azure ML workspace and validate permissions
    ###########################################
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        console=console,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    # Validate workspace access
    validate_workspace_permissions(ml_client, console)

    # 3. Retrieve and validate data assets
    ###########################################
    images_dataset, masks_dataset, splits_dataset = retrieve_and_validate_data_assets(
        ml_client=ml_client,
        console=console,
        experiment_strategy=pcfg.experiment.experiment_strategy,
        get_data_asset_func=get_data_asset,
    )

    # 4. Validate compute cluster and permissions
    ###########################################
    compute_cluster_name = pcfg.azure_ml.compute_name
    validate_and_refresh_compute(ml_client, console, compute_cluster_name)

    # Only run preflight if we plan to mount inputs (skip for minimal smoke)
    if not (
        pcfg.experiment.smoke_test
        and getattr(pcfg.experiment, "smoke_test_minimal", False)
    ):
        preflight_storage_permissions(ml_client, console, compute_cluster_name)

    # 5. Configure job parameters
    ###########################################
    azure_experiment_name = pcfg.azure_ml.experiment_name

    # Command to execute (smoke test or full training)
    if pcfg.experiment.smoke_test:
        if getattr(pcfg.experiment, "smoke_test_minimal", False):
            console.print(
                "Smoke test (minimal) is set: submitting azure_smoke_min.py",
                style="warning",
            )
            train_command = "python scripts/azure_smoke_min.py"
        else:
            console.print(
                "Smoke test flag is set: submitting azure_smoke_test.py",
                style="warning",
            )
            train_command = "python scripts/azure_smoke_test.py"
    else:
        train_command = (
            f"python scripts/azure_train_eval.py "
            f"model={pcfg.model.name} "
            f"experiment.experiment_strategy={pcfg.experiment.experiment_strategy} "
        )

    # Job inputs
    if pcfg.experiment.smoke_test and getattr(
        pcfg.experiment, "smoke_test_minimal", False
    ):
        # Minimal smoke does not require dataset mounts; avoid triggering mount permissions
        job_inputs: dict[str, Any] = {}
    else:
        job_inputs = {
            "images_data": Input(type="uri_folder", path=images_dataset.id),
            "masks_data": Input(type="uri_folder", path=masks_dataset.id),
            "splits_data": Input(type="uri_folder", path=splits_dataset.id),
        }

    # Job outputs
    job_outputs = {
        "model_output": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/models",
        ),
        "tensorboard_logs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/tensorboard_logs",
        ),
        "mlflow_logs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/mlruns",
        ),
        "plots": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/plots",
        ),
        "example_images": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/example_images",
        ),
        "hydra_outputs": Output(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/outputs/hydra_outputs",
        ),
    }

    # Run metadata for tracking
    run_metadata = {
        "model_name": pcfg.model.name,
        "strategy": pcfg.experiment.experiment_strategy,
        "images_dataset_version": images_dataset.version,
        "masks_dataset_version": masks_dataset.version,
        "splits_dataset_version": splits_dataset.version,
    }

    # 6. Create and submit job
    ###########################################
    env_ref = f"{pcfg.azure_ml.environment_name}:{pcfg.azure_ml.environment_version}"

    console.print("Creating Azure ML job...", style="info")

    # Create job using Azure ML SDK command()
    # Optional environment variables for smoke tests
    env_vars: dict[str, str] | None = None
    if pcfg.experiment.smoke_test:
        # Signal minimal smoke whether to try CUDA; default is CPU only
        env_vars = {
            "SMOKE_USE_CUDA": ("1" if pcfg.experiment.smoke_test_use_cuda else "0")
        }
        # Also proactively limit CUDA visibility when CPU-only to avoid early driver init
        if not pcfg.experiment.smoke_test_use_cuda:
            env_vars["CUDA_VISIBLE_DEVICES"] = ""

    job = command(
        code="./",
        command=train_command,
        environment=env_ref,
        compute=compute_cluster_name,
        display_name=display_name,
        experiment_name=azure_experiment_name,
        inputs=job_inputs,
        outputs=job_outputs,
        tags=run_metadata,
        identity=ManagedIdentityConfiguration(),
        environment_variables=env_vars,
    )

    console.print("Submitting job to Azure ML...", style="info")

    # Submit job with retry logic
    submit_job_with_retry = retry_azure_operation(
        ml_client.jobs.create_or_update, operation_name="Job submission"
    )

    try:
        job_run = submit_job_with_retry(job)
        console.print(f"✓ Job submitted: {job_run.name}", style="success")
        job_url = job_run.services.get("Studio").endpoint
        console.print(f"Job URL: {job_url}", style="info")
    except Exception as e:
        handle_job_submission_error(console, e)
        return

    # 7. Stream job logs
    ###########################################
    console.print("Streaming job logs...", style="info")

    try:
        ml_client.jobs.stream(job_run.name)
        console.print("✓ Log streaming completed", style="success")
    except KeyboardInterrupt:
        console.print("Log streaming interrupted by user", style="warning")
    except Exception as stream_error:
        handle_log_streaming_error(console, stream_error, job_run)

    # 8. Download outputs if requested
    ###########################################
    if pcfg.experiment.download_outputs:
        download_job_outputs(ml_client, console, job_run, display_name)

    # 9. Get final job details
    ###########################################
    get_job_details(ml_client, console, job_run)

    console.print("✅ Azure ML job completed successfully!", style="success")


if __name__ == "__main__":
    main()
