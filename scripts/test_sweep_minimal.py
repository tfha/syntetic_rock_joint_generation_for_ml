"""
Minimal test for sweep job data inputs.

This script creates a minimal sweep job with just 2 trials and 1 epoch
to quickly verify that the data input configuration works correctly.
"""

from datetime import datetime

from azure.ai.ml import Input, command
from azure.ai.ml.entities import ManagedIdentityConfiguration
from azure.ai.ml.sweep import BanditPolicy, Choice

from ml_segmentation.azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
)
from ml_segmentation.azure_core import configure_azure_logging_and_warning
from ml_segmentation.azure_data_assets import get_data_asset


def main() -> None:
    """Run minimal sweep job test."""
    # Configure Azure logging
    configure_azure_logging_and_warning()

    # Initialize environment and connect to workspace
    console, subscription_id, resource_group, workspace_name = (
        setup_azure_environment_variables()
    )

    ml_client = connect_to_azure_ml(
        subscription_id, console, resource_group, workspace_name
    )

    console.print(
        "\n[bold]Starting minimal sweep job test[/bold]",
        style="info",
    )

    # Get data assets
    console.print("Retrieving data assets...", style="info")
    images_dataset = get_data_asset(ml_client, console, "rock_images")
    masks_dataset = get_data_asset(ml_client, console, "rock_masks")
    splits_dataset = get_data_asset(ml_client, console, "split_verification_box")

    # Log dataset information
    console.print("\n[bold]Dataset Information:[/bold]", style="info")
    console.print(
        f"  Images: {images_dataset.name}@{images_dataset.version}",
        style="info",
    )
    console.print(
        f"  Masks: {masks_dataset.name}@{masks_dataset.version}",
        style="info",
    )
    console.print(
        f"  Splits: {splits_dataset.name}@{splits_dataset.version}",
        style="info",
    )

    # Prepare inputs using azureml asset URI format
    inputs_dict = {
        "images_data": Input(
            type="uri_folder",
            path=f"azureml:{images_dataset.name}:{images_dataset.version}",
        ),
        "masks_data": Input(
            type="uri_folder",
            path=f"azureml:{masks_dataset.name}:{masks_dataset.version}",
        ),
        "splits_data": Input(
            type="uri_folder",
            path=f"azureml:{splits_dataset.name}:{splits_dataset.version}",
        ),
    }

    # Create minimal command with just 1 epoch
    # Azure ML sweep parameters must use underscores (not dots), but we need
    # to map them back to Hydra's dotted notation in the command
    base_command = (
        "python scripts/azure_train_eval.py "
        "model=unet "
        "experiment.log_mlflow=True "
        "experiment.experiment_strategy=verification_box "
        "model.num_epochs=1 "
        "experiment.num_workers=2 "
        "experiment.report_metrics_to_file=True "
        "model.encoder_name=${{search_space.encoder_name}} "
        "model.batch_size=${{search_space.batch_size}}"
    )

    # Get environment
    env = ml_client.environments.get(name="rock-segmentation-env", label="latest")

    # Create command job
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_name = f"sweep_test_minimal_{timestamp}"

    command_job = command(
        code="./",
        command=base_command,
        environment=env,
        compute="Standard-NC6s-v3",  # GPU cluster name
        display_name="sweep_test_minimal",
        experiment_name=experiment_name,
        inputs=inputs_dict,
        identity=ManagedIdentityConfiguration(),
    )

    # Define minimal search space (just 2 parameters)
    # Note: Keys must use underscores (not dots) - dots not allowed in Azure ML
    search_space = {
        "encoder_name": Choice(["resnet34", "efficientnet-b0"]),
        "batch_size": Choice([4]),
    }

    # Configure sweep with just 2 trials
    sweep_job = command_job.sweep(
        sampling_algorithm="random",
        primary_metric="val_loss",
        goal="minimize",
        max_total_trials=2,
        max_concurrent_trials=2,
        search_space=search_space,
        early_termination_policy=BanditPolicy(
            evaluation_interval=1, slack_factor=0.1, delay_evaluation=5
        ),
    )

    # Set timeout to 3 hours (180 min) for 2 trials with 1 epoch each
    # Each trial might take ~30-60 min, so 180 min provides buffer
    sweep_job.limits.timeout = 180

    # Submit job
    console.print(
        "\n[bold]Submitting minimal test job...[/bold]",
        style="info",
    )
    submitted_job = ml_client.jobs.create_or_update(sweep_job)

    console.print(
        f"\n✓ Test job submitted: {submitted_job.name}",
        style="success",
    )
    console.print(
        f"Studio URL: {submitted_job.studio_url}",
        style="info",
    )
    console.print(
        "\nThis test will run 2 trials with 1 epoch each.",
        style="info",
    )
    console.print(
        "Check the job logs to verify data inputs are accessible.",
        style="info",
    )


if __name__ == "__main__":
    main()
