"""
Submit Azure ML training job for rock mass segmentation.
Uses data assets for improved data management.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any  # Only import Any as it doesn't have a built-in equivalent

import hydra
from azure.ai.ml import Input, dsl
from azure.ai.ml.entities import BuildContext, Environment
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_data_assets import connect_to_azure_ml, get_data_asset
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.utility import export_poetry_to_environment_yml, get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Submit Azure ML job with managed data assets."""
    # Create a console for pretty printing
    console = get_custom_console()

    # Convert the DictConfig to Python dictionary, then to Pydantic model
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)

    # Load environment variables from .env file
    load_dotenv()

    console.print("Starting Azure ML job submission with config:", style="info")
    console.print(OmegaConf.to_yaml(cfg), style="info")

    # Set job display name based on model and strategy
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = (
        f"{pcfg.model.name}-{pcfg.experiment.experiment_strategy}-{timestamp}"
    )

    # Export Poetry environment to environment.yml
    environment_file = export_poetry_to_environment_yml()

    # Get Azure credentials from environment variables
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.environ.get(
        "AZURE_RESOURCE_GROUP", "rg-rock-joint-detection"
    )
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "ws-rock-joint-det")

    if not subscription_id:
        console.print(
            "Error: AZURE_SUBSCRIPTION_ID environment variable not set", style="error"
        )
        console.print(
            "Please set it with: export AZURE_SUBSCRIPTION_ID='your-subscription-id'",
            style="error",
        )
        return

    # Connect to Azure ML workspace
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        resource_group=resource_group_name,
        workspace_name=workspace_name,
    )

    # Get Azure ML configuration from config file
    compute_cluster_name = pcfg.azure_ml.compute_name
    azure_experiment_name = pcfg.azure_ml.experiment_name

    # Get the latest versions of our data assets
    try:
        console.print("Retrieving latest data assets from Azure ML...", style="info")

        # Get base image and mask datasets
        images_dataset = get_data_asset(ml_client, "rock_images")
        masks_dataset = get_data_asset(ml_client, "rock_masks")

        # Get data splits if available
        try:
            splits_dataset = get_data_asset(ml_client, "rock_segmentation_splits")
            console.print(
                f"Using dataset splits version: {splits_dataset.version}", style="info"
            )
            has_splits = True
        except Exception:
            console.print(
                "No dataset splits found, will use strategy-based splitting",
                style="warning",
            )
            has_splits = False

        console.print(
            f"Using images dataset version: {images_dataset.version}", style="info"
        )
        console.print(
            f"Using masks dataset version: {masks_dataset.version}", style="info"
        )

    except Exception as e:
        console.print(f"Error retrieving data assets: {str(e)}", style="error")
        console.print(
            "Make sure you have registered data assets using manage_azure_data_assets.py",
            style="warning",
        )
        return

    # Create an environment from local dependencies
    console.print("Creating Azure ML environment...", style="info")
    env = Environment(
        name="rock-segmentation-env",
        description="Environment for rock segmentation model training",
        build=BuildContext(path="."),
        conda_file=environment_file,
    )

    # Register the environment if needed
    try:
        registered_env = ml_client.environments.get(
            name="rock-segmentation-env", label="latest"
        )
        console.print("Using existing environment", style="info")
        env = registered_env
    except Exception:
        console.print("Registering new environment", style="info")
        env = ml_client.environments.create_or_update(env)

    # Store run metadata for tracking
    run_metadata = {
        "model_name": pcfg.model.name,
        "strategy": pcfg.experiment.experiment_strategy,
        "images_dataset_version": images_dataset.version,
        "masks_dataset_version": masks_dataset.version,
    }

    if has_splits:
        run_metadata["splits_dataset_version"] = splits_dataset.version

    # Define pipeline using components
    @dsl.pipeline(
        name=display_name,
        description="Rock mass segmentation training pipeline with data assets",
        experiment_name=azure_experiment_name,
        tags=run_metadata,
    )
    def rock_segmentation_pipeline():
        """Define the ML pipeline with proper data asset inputs."""
        # Define the training component
        train_command = (
            f"python scripts/train_eval_azure_native.py "
            f"model={pcfg.model.name} "
            f"experiment.log_mlflow=True "
            f"experiment.experiment_strategy={pcfg.experiment.experiment_strategy} "
            f"model.num_epochs={pcfg.model.num_epochs} "
            f"model.batch_size={pcfg.model.batch_size} "
            f"model.learning_rate={pcfg.model.learning_rate}"
        )

        # Add data split info if available
        if has_splits:
            train_command += " experiment.use_registered_splits=True"

        train_step = dsl.command(
            name="train_model",
            display_name="Train Segmentation Model",
            code="./",
            command=train_command,
            environment=env,
            compute=compute_cluster_name,
            inputs={
                "images_data": Input(type="uri_folder", path=images_dataset.id),
                "masks_data": Input(type="uri_folder", path=masks_dataset.id),
            },
            outputs={
                "model_output": dsl.Output(type="uri_folder", path="./outputs/models"),
                "tensorboard_logs": dsl.Output(
                    type="uri_folder", path="./outputs/tensorboard_logs"
                ),
                "mlflow_logs": dsl.Output(type="uri_folder", path="./outputs/mlruns"),
                "plots": dsl.Output(type="uri_folder", path="./outputs/plots"),
                "example_images": dsl.Output(
                    type="uri_folder", path="./outputs/example_images"
                ),
            },
        )

        # Add splits dataset if available
        if has_splits:
            train_step.inputs["splits_data"] = Input(
                type="uri_folder", path=splits_dataset.id
            )

        # Define the model registration component
        register_step = dsl.command(
            name="register_model",
            display_name="Register Model",
            code="./",
            command=(
                'python -c "import mlflow; '
                "from azure.ai.ml import MLClient; "
                "from azure.identity import DefaultAzureCredential; "
                f"mlflow.register_model('runs:/$(cat ./outputs/models/best_run_id.txt)/best_model', '{pcfg.model.name}-{pcfg.experiment.experiment_strategy}')\""
            ),
            environment=env,
            compute=compute_cluster_name,
            inputs={"model_data": train_step.outputs.model_output},
            outputs={
                "registered_model": dsl.Output(
                    type="uri_folder", path="./outputs/registered_model"
                )
            },
        )

        # Execute the registration step after the training step
        register_step.inputs["model_data"] = train_step.outputs.model_output

        return {
            "model_output": train_step.outputs.model_output,
            "registered_model": register_step.outputs.registered_model,
            "plots": train_step.outputs.plots,
            "example_images": train_step.outputs.example_images,
        }

    # Create the pipeline
    pipeline = rock_segmentation_pipeline()

    # Submit the pipeline
    console.print("Submitting Azure ML pipeline...", style="info")
    pipeline_run = ml_client.jobs.create_or_update(pipeline)
    console.print(
        f"Pipeline run URL: {pipeline_run.services.get('Studio').endpoint}",
        style="info",
    )

    # Stream the logs
    console.print("Pipeline submitted. Streaming logs...", style="info")
    ml_client.jobs.stream(pipeline_run.name)

    # Create a directory for downloading outputs
    download_dir = Path(f"./downloaded_runs/{display_name}")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download outputs after completion
    console.print("Downloading pipeline outputs...", style="info")
    ml_client.jobs.download(
        name=pipeline_run.name,
        output_name="model_output",
        download_path=download_dir / "models",
    )
    ml_client.jobs.download(
        name=pipeline_run.name,
        output_name="plots",
        download_path=download_dir / "plots",
    )
    ml_client.jobs.download(
        name=pipeline_run.name,
        output_name="example_images",
        download_path=download_dir / "examples",
    )
    ml_client.jobs.download(
        name=pipeline_run.name,
        output_name="registered_model",
        download_path=download_dir / "registered_model",
    )

    # Log the download location
    console.print(f"Downloaded outputs to: {download_dir.absolute()}", style="success")

    # Print outputs information
    job = ml_client.jobs.get(pipeline_run.name)
    console.print("Pipeline outputs:", style="info")
    console.print(job.outputs)

    console.print("Azure ML pipeline completed successfully!", style="success")


if __name__ == "__main__":
    main()
