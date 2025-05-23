"""
Submit Azure ML training job for rock mass segmentation.
Uses data assets for improved data management.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any  # Only import Any as it doesn't have a built-in equivalent

import hydra
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_data_assets import (
    get_data_asset,
)
from ml_segmentation.azure_utility import (
    configure_azure_logging,
    connect_to_azure_ml,
    export_poetry_to_environment_yml,
    setup_azure_environment,
)
from ml_segmentation.schema_config import ConfigSchema


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Submit Azure ML job with managed data assets using Command approach."""

    # Configure logging to reduce verbose Azure client output
    configure_azure_logging()

    # 1. Initialize configuration
    ###########################################
    # Convert the DictConfig to Python dictionary, then to Pydantic model
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)

    # Setup print-console, environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = setup_azure_environment()

    console.print("Starting Azure ML job submission with config:", style="info")
    console.print(OmegaConf.to_yaml(cfg), style="info")

    # Set job display name based on model and strategy
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = (
        f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-{timestamp}"
    )

    # 2. Export poetry environment to environment.yml
    ###########################################
    try:
        environment_file = export_poetry_to_environment_yml()
        console.print(f"Environment exported to {environment_file}", style="info")
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
    except Exception as e:
        console.print(
            f"Error connecting to Azure ML workspace: {str(e)}", style="error"
        )
        sys.exit(1)

    # 4. Get Azure ML configuration from config
    ###########################################
    compute_cluster_name = pcfg.azure_ml.compute_name
    azure_experiment_name = pcfg.azure_ml.experiment_name

    # 5. Retrieve data assets from Azure ML
    ###########################################
    try:
        console.print("Retrieving latest data assets from Azure ML...", style="info")

        # Get base image and mask datasets
        images_dataset = get_data_asset(ml_client, "rock_images")
        masks_dataset = get_data_asset(ml_client, "rock_masks")

        # Select the correct split asset based on experiment strategy
        strategy = pcfg.experiment.experiment_strategy.lower()
        split_asset_name = f"split_{strategy.replace('.', '_').replace(' ', '_')}"
        splits_dataset = get_data_asset(ml_client, split_asset_name)

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

    # 6. Get and validate compute cluster exists
    ###########################################
    try:
        ml_client.compute.get(compute_cluster_name)
        console.print(f"Using compute cluster: {compute_cluster_name}", style="info")
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
    console.print("Creating Azure ML environment...", style="info")
    env = Environment(
        name="rock-segmentation-env",
        description="Environment for rock segmentation model training",
        build=BuildContext(path="."),
    )

    # Register the environment if needed
    try:
        registered_env = ml_client.environments.get(
            name="rock-segmentation-env", label="latest"
        )
        console.print("Using existing environment", style="info")
        env = registered_env
    except ResourceNotFoundError:
        console.print("Registering new environment", style="info")
        try:
            env = ml_client.environments.create_or_update(env)
            console.print("Environment registered successfully", style="success")
        except Exception as e:
            console.print(f"Error registering environment: {str(e)}", style="error")
            sys.exit(1)
    except Exception as e:
        console.print(
            f"Error checking for existing environment: {str(e)}", style="error"
        )
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
    train_command = (
        f"python scripts/azure_train_eval.py "
        f"model={pcfg.model.name} "
        f"experiment.experiment_strategy={pcfg.experiment.experiment_strategy} "
    )

    # 10. Configure job inputs and outputs
    ###########################################
    # Define job inputs and outputs
    job_inputs = {
        "images_data": Input(type="uri_folder", path=images_dataset.id),
        "masks_data": Input(type="uri_folder", path=masks_dataset.id),
        "splits_data": Input(type="uri_folder", path=splits_dataset.id),
    }

    job_outputs = {
        "model_output": Output(type="uri_folder", path="./outputs/models"),
        "tensorboard_logs": Output(
            type="uri_folder", path="./outputs/tensorboard_logs"
        ),
        "mlflow_logs": Output(type="uri_folder", path="./outputs/mlruns"),
        "plots": Output(type="uri_folder", path="./outputs/plots"),
        "example_images": Output(type="uri_folder", path="./outputs/example_images"),
        "hydra_outputs": Output(type="uri_folder", path="./outputs/hydra_outputs"),
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
    )

    # Submit the job
    console.print("Submitting Azure ML job...", style="info")
    try:
        job_run = ml_client.jobs.create_or_update(job)
        console.print(
            f"Job successfully submitted! Job name: {job_run.name}",
            style="success",
        )
        console.print(
            f"Job run URL: {job_run.services.get('Studio').endpoint}",
            style="info",
        )
    except ServiceRequestError as e:
        console.print(
            f"Error submitting job (service request error): {str(e)}", style="error"
        )
        console.print(
            "Check your Azure ML workspace configuration and network connectivity.",
            style="warning",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"Error submitting job: {str(e)}", style="error")
        sys.exit(1)

    # 12. Stream logs
    ###########################################
    # Stream the logs
    console.print("Job submitted. Streaming logs...", style="info")
    try:
        ml_client.jobs.stream(job_run.name)
    except KeyboardInterrupt:
        console.print(
            "Log streaming interrupted. Job is still running.", style="warning"
        )
    except Exception as e:
        console.print(f"Error streaming logs: {str(e)}", style="error")
        console.print("Job is still running in Azure ML.", style="warning")

    # 13. Download job outputs
    ###########################################
    # Create a directory for downloading outputs
    download_dir = Path(f"./downloaded_runs/{display_name}")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download outputs after completion
    console.print("Downloading job outputs...", style="info")
    try:
        ml_client.jobs.download(
            name=job_run.name,
            output_name="model_output",
            download_path=download_dir / "models",
        )
        ml_client.jobs.download(
            name=job_run.name,
            output_name="plots",
            download_path=download_dir / "plots",
        )
        ml_client.jobs.download(
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
