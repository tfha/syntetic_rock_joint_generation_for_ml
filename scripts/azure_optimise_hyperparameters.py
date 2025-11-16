"""
Hyperparameter optimization for rock mass segmentation models in Azure ML.

This script submits a hyperparameter optimization job to Azure ML, using Bayesian
optimization to find the best hyperparameters for segmentation models.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any  # Only import Any as it doesn't have a built-in equivalent

from azure.ai.ml import Input, command
from azure.ai.ml.entities import (
    BuildContext,
    Environment,
    ManagedIdentityConfiguration,
)
from azure.ai.ml.sweep import BanditPolicy, Choice, Uniform

from ml_segmentation.azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
)
from ml_segmentation.azure_core import configure_azure_logging_and_warning
from ml_segmentation.azure_data_assets import (
    get_data_asset,
)
from ml_segmentation.azure_environment import export_poetry_to_environment_yml
from ml_segmentation.azure_hyperparameter_spaces import (
    get_bayesian_sampling_params,
    get_model_search_space,
)


def get_model_config_from_args() -> tuple[str, dict[str, Any]]:
    """Parse command line arguments to get model configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run hyperparameter optimization for segmentation models in Azure ML"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "deeplabv3plus"],
        help="Model architecture to optimize (unet or deeplabv3plus)",
    )

    parser.add_argument(
        "--experiment-strategy",
        type=str,
        default="verification_box",
        help="Experiment strategy (e.g., verification_box, verification_dfn)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to train each trial",
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=30,
        help="Maximum number of optimization trials",
    )

    parser.add_argument(
        "--concurrent-trials",
        type=int,
        default=4,
        help="Maximum number of concurrent trials",
    )

    parser.add_argument(
        "--compute-cluster",
        type=str,
        default="Standard-NC6s-v3",
        help="Azure ML compute cluster name",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2 for Azure ML)",
    )

    args = parser.parse_args()

    # Compile optimization config
    opt_config = {
        "model": args.model,
        "experiment_strategy": args.experiment_strategy,
        "epochs": args.epochs,
        "max_trials": args.max_trials,
        "concurrent_trials": args.concurrent_trials,
        "compute_cluster": args.compute_cluster,
        "num_workers": args.num_workers,
    }

    return args.model, opt_config


def main():
    """Main entry point for the script."""
    # Configure Azure logging
    configure_azure_logging_and_warning()

    # Setup environment and load configuration
    console, subscription_id, resource_group, workspace_name = (
        setup_azure_environment_variables()
    )

    # Create config for the run
    azure_config = {
        "subscription_id": subscription_id,
        "resource_group": resource_group or "rg-rock-joint-detection",
        "workspace_name": workspace_name or "ws-rock-joint-det",
    }

    # Export Poetry environment to environment.yml
    environment_file = export_poetry_to_environment_yml()

    console.print("Starting Azure ML hyperparameter optimization", style="info")

    # Connect to Azure ML
    console.print("Connecting to Azure ML...", style="info")
    ml_client = connect_to_azure_ml(
        subscription_id=subscription_id,
        console=console,
        resource_group=azure_config["resource_group"],
        workspace_name=azure_config["workspace_name"],
    )

    model_name, opt_config = get_model_config_from_args()

    console.print(
        f"Starting hyperparameter optimization for {model_name}", style="info"
    )
    console.print(f"Optimization configuration: {opt_config}", style="info")

    # Get the latest versions of our data assets
    try:
        console.print("Retrieving latest data assets from Azure ML...", style="info")

        # Get base image and mask datasets
        images_dataset = get_data_asset(ml_client, console, "rock_images")
        masks_dataset = get_data_asset(ml_client, console, "rock_masks")

        # Try to get data splits if available
        try:
            splits_dataset = get_data_asset(
                ml_client, console, "rock_segmentation_splits"
            )
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
            "Make sure you have registered data assets using "
            "manage_azure_data_assets.py",
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

    # Set up the command job for hyperparameter tuning
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_name = (
        f"hparam_opt_{model_name}_{opt_config['experiment_strategy']}_{timestamp}"
    )

    # Create the base training command
    base_command = (
        f"python scripts/train_eval_azure_native.py "
        f"model={model_name} "
        f"experiment.log_mlflow=True "
        f"experiment.experiment_strategy={opt_config['experiment_strategy']} "
        f"model.num_epochs={opt_config['epochs']} "
        f"experiment.num_workers={opt_config['num_workers']} "
    )  # Data splits are now always used in the system
    # No need to add any specific flags

    # Add output reporting for hyperparameter optimization
    base_command += " experiment.report_metrics_to_file=True"

    # Prepare inputs dictionary (use download mode for reliability)
    inputs_dict = {
        "images_data": Input(
            type="uri_folder", path=images_dataset.id, mode="download"
        ),
        "masks_data": Input(type="uri_folder", path=masks_dataset.id, mode="download"),
    }

    # Add splits dataset if available
    if has_splits:
        inputs_dict["splits_data"] = Input(
            type="uri_folder", path=splits_dataset.id, mode="download"
        )

    # Define the hyperparameter optimization command job
    command_job = command(
        code="./",
        command=base_command,
        environment=env,
        compute=opt_config["compute_cluster"],
        display_name=f"hparam_opt_{model_name}",
        experiment_name=experiment_name,
        inputs=inputs_dict,
        identity=ManagedIdentityConfiguration(),
        # Note: outputs are automatically handled by Azure ML for sweep jobs
    )

    # Get the search space for the specified model
    search_space = get_model_search_space(model_name)

    # Convert the search space to Azure ML sweep parameters
    # Azure ML requires parameter names with only letters, numbers, and underscores
    # So we replace dots with underscores for the sweep, and convert back in the command
    sweep_params: dict[str, Any] = {}
    for param_name, param_config in search_space.items():
        # Replace dots with underscores for Azure ML compatibility
        azure_param_name = param_name.replace(".", "_")

        if param_config["type"] == "choice":
            sweep_params[azure_param_name] = Choice(param_config["values"])
        elif param_config["type"] == "uniform":
            sweep_params[azure_param_name] = Uniform(
                min_value=param_config["min_value"], max_value=param_config["max_value"]
            )

    # Get Bayesian sampling parameters
    sampling_params = get_bayesian_sampling_params(
        max_total_trials=opt_config["max_trials"],
        max_concurrent_trials=opt_config["concurrent_trials"],
    )

    # Create the sweep job
    console.print(
        "Setting up hyperparameter sweep job with Bayesian optimization...",
        style="info",
    )
    sweep_job = command_job.sweep(
        sampling_algorithm=sampling_params["sampling_algorithm"],
        goal=sampling_params["goal"],
        primary_metric=sampling_params["primary_metric"],
        max_total_trials=sampling_params["max_total_trials"],
        max_concurrent_trials=sampling_params["max_concurrent_trials"],
        search_space=sweep_params,
    )

    # Add early termination if specified
    if "early_termination" in sampling_params:
        et_params = sampling_params["early_termination"]
        if et_params.get("type") == "bandit":
            sweep_job.early_termination = BanditPolicy(
                evaluation_interval=et_params["evaluation_interval"],
                delay_evaluation=et_params["delay_evaluation"],
                slack_factor=et_params["slack_factor"],
            )

    # Submit the sweep job
    console.print(
        f"Submitting hyperparameter sweep job for {model_name}...", style="info"
    )
    returned_job = ml_client.create_or_update(sweep_job)
    console.print(f"Submitted job: {returned_job.name}", style="success")
    console.print(
        f"Job URL: {returned_job.services.get('Studio').endpoint}", style="info"
    )

    # Set up tracking file to save job info
    job_tracking_dir = Path("./experiments/hyperparameters")
    job_tracking_dir.mkdir(parents=True, exist_ok=True)

    job_info = {
        "job_name": returned_job.name,
        "model": model_name,
        "experiment_strategy": opt_config["experiment_strategy"],
        "timestamp": timestamp,
        "max_trials": opt_config["max_trials"],
        "search_space": search_space,
        "images_dataset_version": images_dataset.version,
        "masks_dataset_version": masks_dataset.version,
    }

    if has_splits:
        job_info["splits_dataset_version"] = splits_dataset.version

    # Save job info to a file for later reference
    job_info_file = job_tracking_dir / f"job_info_{model_name}_{timestamp}.json"
    with open(job_info_file, "w") as f:
        json.dump(job_info, f, indent=2)

    console.print(f"Job tracking information saved to {job_info_file}", style="info")
    console.print(
        "You can monitor the job progress in the Azure ML Studio", style="info"
    )

    # Optionally stream the logs
    console.print("Do you want to stream the logs? (y/n): ", style="bold cyan")
    if input().lower() == "y":
        console.print("Streaming logs...", style="info")
        ml_client.jobs.stream(returned_job.name)

    console.print(
        f"Hyperparameter optimization job for {model_name} submitted successfully!",
        style="success",
    )
    console.print(
        "When the job completes, you can use the 'analyze_hyperparameter_results.py'"
        " script "
        "to analyze the results and generate a report of the best hyperparameters.",
        style="info",
    )


if __name__ == "__main__":
    main()
