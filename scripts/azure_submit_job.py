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

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import Environment, ManagedIdentityConfiguration
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_data_assets import (
    get_data_asset,
)
from ml_segmentation.azure_utility import (
    check_compute_permissions,
    configure_azure_logging,
    connect_to_azure_ml,
    export_poetry_to_environment_yml,
    refresh_compute_cluster,
    retry_azure_operation,
    setup_azure_environment_variables,
)
from ml_segmentation.schema_config import ConfigSchema


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Submit Azure ML job with managed data assets using Command approach."""

    # Configure logging and warnings to reduce verbose Azure client output
    configure_azure_logging()
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    warnings.filterwarnings("ignore", category=UserWarning, module="msrest")

    # 1. Initialize configuration
    ###########################################
    # Convert the DictConfig to Python dictionary, then to Pydantic model
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)

    # Setup print-console, environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = (
        setup_azure_environment_variables()
    )

    console.print("Starting Azure ML job submission with config:", style="info")
    console.print(OmegaConf.to_yaml(cfg), style="info")

    # Set job display name based on model and strategy
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = (
        f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-{timestamp}"
    )

    # 2. Export poetry environment to environment.yml
    ###########################################
    if not pcfg.azure_ml.use_curated_env:
        try:
            environment_file = export_poetry_to_environment_yml()
            console.print(f"Export environment to {environment_file}", style="info")
        except Exception as e:
            console.print(f"Error exporting environment: {str(e)}", style="error")
            sys.exit(1)

    # 3. Connect to Azure ML workspace
    ###########################################
    try:
        ml_client = connect_to_azure_ml(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )
        console.print("Successfully connected to Azure ML workspace", style="success")
        console.print("Veryfying workspace permissions...", style="info")

        # Check workspace permissions - verify the client has proper access
        try:
            # Try a simple operation to validate permissions
            workspace = ml_client.workspaces.get(workspace_name)
            console.print(
                f"Authenticated to workspace: {workspace.name} "
                f"(location: {workspace.location})",
                style="info",
            )
        except Exception as perm_error:
            console.print(
                "Warning: Connected to workspace but may have limited permissions. "
                f"Error: {str(perm_error)}",
                style="warning",
            )
    except Exception as e:
        console.print(
            f"Error connecting to Azure ML workspace: {str(e)}. "
            "Contact administrator to get access",
            style="error",
        )
        sys.exit(1)

    # 4. Get Azure ML configuration from config
    ###########################################
    compute_cluster_name = pcfg.azure_ml.compute_name
    azure_experiment_name = pcfg.azure_ml.experiment_name
    # not to be confused with mlflow experiment name

    # 5. Retrieve data assets from Azure ML
    ###########################################
    try:
        console.print("Retrieving latest data assets from Azure ML...", style="info")

        # Apply retry logic to data asset retrieval
        get_data_asset_with_retry = retry_azure_operation(
            get_data_asset, operation_name="Data asset retrieval"
        )  # Get base image and mask datasets
        images_dataset = get_data_asset_with_retry(ml_client, console, "rock_images")
        masks_dataset = get_data_asset_with_retry(ml_client, console, "rock_masks")

        # Select the correct split asset based on experiment strategy
        strategy = pcfg.experiment.experiment_strategy.lower()
        split_asset_name = f"split_{strategy.replace('.', '_').replace(' ', '_')}"
        splits_dataset = get_data_asset_with_retry(
            ml_client,
            console,
            split_asset_name,
        )

        console.print(
            f"Using split asset: {split_asset_name} (version {splits_dataset.version})",
            style="info",
        )
        console.print(
            f"Using images dataset version: {images_dataset.version}", style="info"
        )
        console.print(
            f"Using masks dataset version: {masks_dataset.version}", style="info"
        )

        # Log data asset types and paths for easier debugging
        console.print(
            "Data asset details (for authentication validation):",
            style="info",
        )
        console.print(f"Images asset path: {images_dataset.id}", style="info")
        console.print(f"Masks asset path: {masks_dataset.id}", style="info")
        console.print(f"Splits asset path: {splits_dataset.id}", style="info")

        # Verify data asset permissions explicitly
        console.print("Verifying data asset permissions...", style="info")
        try:
            # Try to list the data asset to verify permissions
            # This can help identify permission issues early
            assets = ml_client.data.list(max_results=5)
            asset_count = 0
            for _ in assets:
                asset_count += 1
                if asset_count >= 5:
                    break
            console.print(
                f"Successfully verified access to {asset_count} data assets",
                style="success",
            )
        except Exception as perm_e:
            console.print(
                f"Warning: Limited data asset permissions detected: {str(perm_e)}",
                style="warning",
            )
            console.print(
                "You may need to grant additional permissions to the compute cluster",
                style="warning",
            )

    except ResourceNotFoundError as e:
        console.print(f"Data asset not found: {str(e)}", style="error")
        msg = "Make sure you have registered data assets"
        console.print(msg, style="warning")
        sys.exit(1)
    except Exception as e:
        console.print(f"Error retrieving data assets: {str(e)}", style="error")
        msg = "Make sure you have registered data assets"
        console.print(msg, style="warning")
        sys.exit(1)

    # 6. Get and validate compute cluster
    ###########################################
    try:
        # First refresh compute to make sure we have the latest state
        console.print(
            f"Refreshing compute cluster: {compute_cluster_name}", style="info"
        )

        # Refresh compute to ensure we have up-to-date identity information
        compute = refresh_compute_cluster(ml_client, compute_cluster_name, console)

        if compute is None:
            # Fall back to retry logic if refresh fails
            get_compute_with_retry = retry_azure_operation(
                ml_client.compute.get, operation_name="Compute cluster retrieval"
            )
            _ = get_compute_with_retry(compute_cluster_name)

        # Run detailed permission checks
        console.print("Checking compute cluster permissions...", style="info")
        permissions = check_compute_permissions(
            ml_client, compute_cluster_name, console
        )

        # Validate identity and provide guidance based on results
        if not permissions["has_identity"]:
            console.print(
                "Warning: Compute cluster does not have a managed identity. "
                "The job will use system-assigned identity instead. "
                "Consider attaching a managed identity to the compute cluster in the "
                "Azure portal and assign it Storage Blob Data Contributor role.",
                style="warning",
            )
        else:
            console.print(
                f"Compute cluster has identity type: {permissions['identity_type']}",
                style="info",
            )

        # Verify workspace access
        if permissions["workspace_access"]:
            console.print("Compute cluster has access to the workspace.", style="info")
        else:
            console.print(
                "Warning: Could not verify compute's workspace access.",
                style="warning",
            )

        # Verify log streaming access - NEW CHECK
        if permissions.get("log_streaming_access", False):
            console.print(
                "Compute appears to have permissions needed for log streaming.",
                style="info",
            )
        else:
            console.print(
                "Warning: Compute cluster may not have all permissions needed for "
                "log streaming. This might cause 'ScriptExecution.StreamAccess."
                "Authentication' errors later.",
                style="warning",
            )
            console.print(
                "To ensure log streaming works correctly, make sure the compute's "
                "managed identity has:",
                style="info",
            )
            console.print(
                "1. 'Storage Blob Data Reader' role on the workspace storage account",
                style="info",
            )
            console.print(
                "2. 'AzureML Data Scientist' role on the workspace", style="info"
            )

    except ResourceNotFoundError:
        console.print(
            f"Compute cluster '{compute_cluster_name}' not found", style="error"
        )
        msg = (
            "Please create the compute cluster in the Azure ML workspace "
            "or update the configuration."
        )
        console.print(msg, style="warning")
        sys.exit(1)
    except Exception as e:
        console.print(f"Error accessing compute cluster: {str(e)}", style="error")
        sys.exit(1)

    # 7. Create or retrieve Azure ML environment
    ###########################################
    console.print("Creating or retrieving Azure ML environment...", style="info")

    # Use environment name and version from config
    env_name = pcfg.azure_ml.environment_name
    env_version = pcfg.azure_ml.environment_version

    # Warning for version mismatches with curated environment
    curated_env_warning = (
        "[INFO] Using 'acpt-pytorch-2.2-cuda12.1:37' curated environment. "
        "This environment provides CUDA 12.1 which is compatible with your "
        "PyTorch cu121 build. However, your project uses Python 3.11 while "
        "this environment uses Python 3.10. This may cause minor version "
        "compatibility issues, but should work for most use cases."
    )

    if pcfg.azure_ml.use_curated_env:
        console.print(
            f"Using curated AzureML environment: {pcfg.azure_ml.curated_env_name}",
            style="info",
        )
        console.print(
            f"Environment name: {pcfg.azure_ml.curated_env_name}", style="info"
        )
        # Warn if requirements are not met
        console.print(curated_env_warning, style="warning")
        env = pcfg.azure_ml.curated_env_name
    elif pcfg.azure_ml.use_new_version:
        console.print(
            f"Registering new version of environment: {env_name}", style="info"
        )
        env = Environment(
            name=env_name,
            description="Environment for rock segmentation model training. "
            "Environment created from a Docker image plus Conda environment.",
            conda_file="environment.yml",  # Use the exported environment file
            image=pcfg.azure_ml.base_docker_image,  # Use the base image from config
        )
        console.print(
            f"Using base Docker image: {pcfg.azure_ml.base_docker_image}", style="info"
        )
        try:
            # Apply retry logic to environment creation
            create_env_with_retry = retry_azure_operation(
                ml_client.environments.create_or_update,
                operation_name="Environment registration",
            )

            # Create environment with retry
            env = create_env_with_retry(env)
            console.print(
                "Environment registered successfully as a new version", style="success"
            )
        except Exception as e:
            console.print(f"Error registering environment: {str(e)}", style="error")
            sys.exit(1)
    else:
        console.print(
            f"Using existing environment: {env_name}:{env_version}", style="info"
        )
        try:
            # Apply retry logic to environment retrieval
            get_env_with_retry = retry_azure_operation(
                ml_client.environments.get, operation_name="Environment retrieval"
            )

            # Get environment with retry
            env = get_env_with_retry(name=env_name, version=env_version)
        except ResourceNotFoundError:
            console.print(
                f"Environment {env_name}:{env_version} not found", style="error"
            )
            sys.exit(1)
        except Exception as e:
            console.print(f"Error retrieving environment: {str(e)}", style="error")
            sys.exit(1)

    # 8. Set up run metadata
    ###########################################
    # Store run metadata for tracking
    run_metadata = {
        "model_name": pcfg.model.name,
        "strategy": pcfg.experiment.experiment_strategy,
        "images_dataset_version": images_dataset.version,
        "masks_dataset_version": masks_dataset.version,
        "splits_dataset_version": splits_dataset.version,
    }

    # 9. Define training command and parameters
    ###########################################
    # Define the training command - fixed script name
    # First run setup script to ensure local package is available, then run training
    train_command = (
        f"python scripts/setup_azure_environment.py && "
        f"python scripts/azure_train_eval.py "
        f"model={pcfg.model.name} "
        f"experiment.experiment_strategy={pcfg.experiment.experiment_strategy} "
    )  # 10. Configure job inputs and outputs folders with experiment information
    ###########################################
    # Define job inputs and outputs for remote execution
    # The directories will be created by the azure_train_eval.py script
    console.print("Configuring job inputs and outputs...", style="info")
    job_inputs = {
        "images_data": Input(type="uri_folder", path=images_dataset.id),
        "masks_data": Input(type="uri_folder", path=masks_dataset.id),
        "splits_data": Input(type="uri_folder", path=splits_dataset.id),
    }

    # Use proper Azure output paths with azureml:// prefix as required by the service
    # This is needed to support persisting outputs correctly in the Azure ML workspace
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

    # 11. Create and submit the job
    ###########################################
    # Create a single Command job
    job = command(
        code="./",
        command=train_command,
        environment=env,
        compute=compute_cluster_name,
        display_name=display_name,
        experiment_name=azure_experiment_name,
        inputs=job_inputs,
        outputs=job_outputs,
        tags=run_metadata,  # Add metadata as tags
        # Use system-assigned managed identity on the compute cluster
        # This uses the identity you set up on the compute node
        identity=ManagedIdentityConfiguration(),
    )
    # Submit the job
    console.print("Submitting Azure ML job...", style="info")
    # Print identity information to help with debugging
    console.print(
        "Job identity configuration: System-assigned managed identity",
        style="info",
    )
    console.print(
        "This uses the managed identity you configured on the compute cluster",
        style="info",
    )

    # Add information about required storage account permissions
    console.print(
        "\n[bold]Important:[/bold] For system-assigned identity to work, you need to:",
        style="warning",
    )
    console.print(
        "1. Find the storage account associated with your Azure ML workspace",
        style="info",
    )
    console.print(
        "2. Go to the storage account's 'Access Control (IAM)' section",
        style="info",
    )
    console.print(
        "3. Add a role assignment with:",
        style="info",
    )
    console.print(
        "   - Role: 'Storage Blob Data Contributor'",
        style="info",
    )
    console.print(
        "   - Assign access to: 'Managed identity'",
        style="info",
    )
    console.print(
        "   - Members: Your compute's system-assigned managed identity",
        style="info",
    )
    console.print(
        "   - Principal ID: 6b64c8b6-82db-4894-b6ea-3d60dc2292de",
        style="info",
    )
    console.print(
        "If you haven't done this yet, the job may still fail with NoIdentityOnCompute",
        style="warning",
    )

    try:
        # Use retry decorator for job submission
        submit_job_with_retry = retry_azure_operation(
            ml_client.jobs.create_or_update, operation_name="Job submission"
        )
        job_run = submit_job_with_retry(job)
        console.print(
            f"Job successfully submitted! Job name: {job_run.name}",
            style="success",
        )
        console.print(
            f"Job run URL: {job_run.services.get('Studio').endpoint}",
            style="info",
        )

        # Add troubleshooting guidance for NoIdentityOnCompute errors
        console.print(
            "\n[bold]If you encounter NoIdentityOnCompute errors:[/bold]",
            style="warning",
        )
        console.print(
            "1. In Azure Portal, navigate to your Azure ML workspace",
            style="info",
        )
        console.print(
            "2. Go to 'Compute > Compute clusters > Select your cluster'",
            style="info",
        )
        console.print(
            "3. Click 'Identity' tab",
            style="info",
        )
        console.print(
            "4. Enable 'System assigned' managed identity",
            style="info",
        )
        console.print(
            "5. After saving, go to your workspace's Storage Account",
            style="info",
        )
        console.print(
            "6. Under 'Access Control (IAM)', add a role assignment:",
            style="info",
        )
        console.print(
            "   - Role: Storage Blob Data Contributor",
            style="info",
        )
        console.print(
            "   - Assign access to: System assigned managed identity",
            style="info",
        )
        console.print(
            "   - Select: Your compute cluster's managed identity",
            style="info",
        )
        console.print(
            "7. Save and retry the job submission",
            style="info",
        )

    except ServiceRequestError as e:
        console.print(
            f"Error submitting job (service request error): {str(e)}", style="error"
        )
        console.print(
            "Check your Azure ML workspace configuration and connectivity.",
            style="warning",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"Error submitting job: {str(e)}", style="error")
        sys.exit(1)

    # 12. Stream logs to console
    ###########################################
    # Stream the logs
    console.print("Job submitted. Streaming logs...", style="info")

    # Add explicit retry logic with backoff for streaming
    max_retries = 3
    retry_count = 0
    stream_success = False

    while retry_count < max_retries and not stream_success:
        try:
            if retry_count > 0:
                console.print(
                    f"Retry attempt {retry_count}/{max_retries} for log streaming...",
                    style="warning",
                )
                # Add a short delay before retry with exponential backoff
                time.sleep(2**retry_count)

            # Try to stream with explicit timeout
            ml_client.jobs.stream(job_run.name)
            stream_success = True

        except KeyboardInterrupt:
            console.print(
                "Log streaming interrupted. Job is still running.", style="warning"
            )
            break

        except Exception as stream_error:
            retry_count += 1
            error_msg = str(stream_error)

            # Check for authentication/permission errors
            auth_error = (
                "Authentication failed" in error_msg or "PermissionDenied" in error_msg
            )

            if auth_error:
                console.print(
                    f"Authentication error (attempt {retry_count}): "
                    f"{error_msg[:100]}...",
                    style="error",
                )
                console.print(
                    "This is likely due to insufficient permissions "
                    "on the managed identity.",
                    style="warning",
                )

                # On final retry, give detailed guidance
                if retry_count == max_retries:
                    console.print(
                        "\n[bold]Permission error details:[/bold]", style="error"
                    )
                    console.print(
                        "The system-assigned managed identity doesn't have "
                        "proper permissions to access the data stream.",
                        style="error",
                    )
                    console.print("To fix this issue:", style="info")
                    console.print(
                        "1. Make sure the managed identity has "
                        "'Storage Blob Data Reader' role",
                        style="info",
                    )
                    console.print(
                        "2. Make sure the managed identity has "
                        "'AzureML Data Scientist' role",
                        style="info",
                    )
                    console.print(
                        "3. Check if there are any network restrictions "
                        "blocking access",
                        style="info",
                    )
                    portal_url = job_run.services.get("Studio").endpoint
                    console.print(
                        f"You can still view the job logs through the "
                        f"Azure ML portal: {portal_url}",
                        style="info",
                    )
            else:
                # For non-permission errors, retry with different handling
                console.print(
                    f"Error streaming logs (attempt {retry_count}): "
                    f"{error_msg[:100]}...",
                    style="error",
                )
                if retry_count == max_retries:
                    console.print(
                        "Max retries reached. Job is still running in Azure ML.",
                        style="warning",
                    )
                    portal_url = job_run.services.get("Studio").endpoint
                    console.print(
                        f"You can view the job logs through the "
                        f"Azure ML portal: {portal_url}",
                        style="info",
                    )

    # If all retries failed but job is still running, print a message
    if not stream_success:
        console.print(
            "Unable to stream logs after multiple attempts. "
            "The job is still running in Azure ML.",
            style="warning",
        )

    # 13. Download job outputs from Azure ML
    ###########################################
    if pcfg.experiment.download_outputs:
        # Create a directory for downloading outputs
        download_dir = Path(f"./downloaded_remote_runs/{display_name}")
        download_dir.mkdir(parents=True, exist_ok=True)

        # Download outputs after completion
        console.print(
            "Downloading job outputs after training in Azure ML...", style="info"
        )
        try:
            # Use retry logic for downloads to handle potential network issues
            download_with_retry = retry_azure_operation(
                ml_client.jobs.download, operation_name="Output download"
            )

            # Download outputs with retry
            download_with_retry(
                name=job_run.name,
                output_name="model_output",
                download_path=download_dir / "models",
            )
            download_with_retry(
                name=job_run.name,
                output_name="plots",
                download_path=download_dir / "plots",
            )
            download_with_retry(
                name=job_run.name,
                output_name="example_images",
                download_path=download_dir / "examples",
            )

            # Log the download location
            console.print(
                f"Downloaded outputs to: {download_dir.absolute()}", style="success"
            )
        except Exception as e:
            console.print(f"Error downloading job outputs: {str(e)}", style="error")
            msg = (
                "You can download the outputs manually from the Azure ML portal: "
                f"{job_run.services.get('Studio').endpoint}"
            )
            console.print(msg, style="warning")

    # 14. Print job output information
    ###########################################
    # Print outputs information
    try:
        job_details = ml_client.jobs.get(job_run.name)
        console.print("Job outputs:", style="info")
        console.print(job_details.outputs)
    except Exception as e:
        console.print(f"Error getting job details: {str(e)}", style="error")

    console.print("Azure ML job completed successfully!", style="success")


if __name__ == "__main__":
    main()
