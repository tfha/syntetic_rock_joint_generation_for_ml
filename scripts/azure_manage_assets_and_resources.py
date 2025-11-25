"""
Manage Azure ML data assets for rock mass segmentation.

This script demonstrates how to use Azure ML data asset management to
handle rock mass segmentation datasets properly.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.theme import Theme

from ml_segmentation.azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
)
from ml_segmentation.azure_core import configure_azure_logging_and_warning
from ml_segmentation.azure_data_assets import (
    compare_assets,
    list_data_assets,
    prepare_and_save_dataset_splits,
    register_base_datasets,
    register_split_data_asset,
    upload_base_data_to_azure_blob,
    upload_split_data_to_azure_blob,
)
from ml_segmentation.schema_config import AzureDataAssetsCommand, ConfigSchema
from ml_segmentation.utility import seed_everything


def get_command_description(command: AzureDataAssetsCommand) -> str:
    """Get a descriptive text for an Azure Data Assets command.

    This function provides human-readable descriptions of each command beyond
    the kebab-case identifiers in the enum. These descriptions are used for
    UI presentation and help text to improve usability, while maintaining
    separation between command identifiers and their presentation layer.

    Args:
        command: The AzureDataAssetsCommand enum value

    Returns:
        A description of what the command does
    """
    descriptions = {
        AzureDataAssetsCommand.REGISTER_BASE_DATASETS: (
            "Register base datasets in Azure ML"
        ),
        AzureDataAssetsCommand.REGISTER_SPLITS: ("Register dataset splits in Azure ML"),
        AzureDataAssetsCommand.GENERATE_SPLITS: ("Generate dataset splits locally"),
        AzureDataAssetsCommand.UPLOAD_SPLITS: (
            "Upload dataset splits to Azure Blob Storage"
        ),
        AzureDataAssetsCommand.PROCESS_ALL_SPLITS: (
            "Process all experiment strategies: "
            "generate, upload, and register splits in one operation"
        ),
        AzureDataAssetsCommand.LIST_ASSETS: "List data assets in Azure ML",
        AzureDataAssetsCommand.COMPARE_ASSETS: ("Compare two versions of a data asset"),
        AzureDataAssetsCommand.UPLOAD_DATA: ("Upload local data to Azure Blob storage"),
        AzureDataAssetsCommand.BUILD_ENVIRONMENT: (
            "Build and register Azure ML environment from Dockerfile"
        ),
    }
    return descriptions.get(command, "Unknown command")


def show_command_help(console: Console):
    """Display available commands and usage examples.

    Args:
        console: Console object for pretty printing
    """
    # Print available commands
    console.print("Available commands:", style="info")
    for cmd in AzureDataAssetsCommand:
        console.print(
            f"  {cmd.value}: {get_command_description(cmd)}",
            style="info",
        )

    console.print("\nUsage examples:", style="info")
    # Make long command examples more readable by splitting into multiple lines
    base_cmd = "python scripts/manage_azure_data_assets.py azure_data_assets.command="

    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.REGISTER_BASE_DATASETS.value}",
        style="info",
    )
    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.BUILD_ENVIRONMENT.value}",
        style="info",
    )
    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.UPLOAD_SPLITS.value}",
        style="info",
    )
    console.print(
        f"  {base_cmd}{AzureDataAssetsCommand.REGISTER_SPLITS.value}",
        style="info",
    )


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for the script using Hydra.

    Args:
        cfg: The Hydra configuration object
    """
    # Configure logging to reduce verbose Azure client output
    configure_azure_logging_and_warning()

    # Initialize environment and get Azure credentials
    console, subscription_id, resource_group, workspace_name = (
        setup_azure_environment_variables()
    )

    # Ensure the Console supports the semantic styles used across the codebase.
    # If setup_azure_environment_variables returned a plain Console, replace it
    # with one that provides the expected semantic color names.
    semantic_theme = Theme(
        {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            # keep any other semantic names used elsewhere
            "bold green": "bold green",
        }
    )
    console = Console(theme=semantic_theme)

    try:
        # Convert OmegaConf to a Python dictionary and validate with Pydantic
        cfg_dict = OmegaConf.to_object(cfg)
        pcfg = ConfigSchema(**cfg_dict)

        # Set random seed at the beginning to affect all operations
        seed_everything(pcfg.experiment.seed)

        console.print(
            f"Using configuration: {pcfg.azure_data_assets}",
            style="info",
        )

        # Get command from validated config
        command = pcfg.azure_data_assets.command

        # Only connect to Azure ML if the command requires it and is valid
        ml_client = None
        try:
            valid_commands_requiring_connection = [
                AzureDataAssetsCommand.REGISTER_BASE_DATASETS,
                AzureDataAssetsCommand.REGISTER_SPLITS,
                AzureDataAssetsCommand.LIST_ASSETS,
                AzureDataAssetsCommand.COMPARE_ASSETS,
                AzureDataAssetsCommand.BUILD_ENVIRONMENT,
                AzureDataAssetsCommand.PROCESS_ALL_SPLITS,
                AzureDataAssetsCommand.UPLOAD_SPLITS,
            ]

            if command in valid_commands_requiring_connection:
                ml_client = connect_to_azure_ml(
                    subscription_id=subscription_id,
                    console=console,
                    resource_group=resource_group,
                    workspace_name=workspace_name,
                )

            # Get asset parameters from validated config
            asset_name = pcfg.azure_data_assets.asset_name
            version1 = pcfg.azure_data_assets.version1
            version2 = pcfg.azure_data_assets.version2

            # Execute the appropriate command based on the enum value
            match command:
                case AzureDataAssetsCommand.BUILD_ENVIRONMENT:
                    from ml_segmentation.azure_environment import (
                        build_and_register_environment,
                    )

                    # Build/register environment using config value
                    # azure_ml.environment_name. Previously we passed an
                    # unsupported kw arg 'pcfg'. Fix to use validated name.
                    environment_name = pcfg.azure_ml.environment_name
                    build_and_register_environment(
                        ml_client=ml_client,
                        console=console,
                        environment_name=environment_name,
                        dockerfile_path="./Dockerfile",
                        context_path="./",
                    )
                case AzureDataAssetsCommand.LIST_ASSETS:
                    list_data_assets(ml_client, console, asset_name)
                case AzureDataAssetsCommand.COMPARE_ASSETS:
                    if asset_name is None or version1 is None or version2 is None:
                        console.print(
                            (
                                "asset_name, version1, and version2 must be "
                                "provided for compare-assets"
                            ),
                            style="error",
                        )
                    else:
                        compare_assets(
                            ml_client, console, asset_name, version1, version2
                        )
                case AzureDataAssetsCommand.UPLOAD_DATA:
                    # upload_base_data_to_azure_blob expects str paths
                    upload_base_data_to_azure_blob(
                        console,
                        str(pcfg.dataset.path_images),
                        str(pcfg.dataset.path_raw_mask_labels),
                        str(pcfg.dataset.path_processed_mask_labels),
                    )
                case AzureDataAssetsCommand.REGISTER_BASE_DATASETS:
                    register_base_datasets(ml_client, console)
                case AzureDataAssetsCommand.GENERATE_SPLITS:
                    prepare_and_save_dataset_splits(
                        console,
                        images_directory=pcfg.dataset.path_images,
                        labels_directory=(pcfg.dataset.path_processed_mask_labels),
                        experiment_strategy=(pcfg.experiment.experiment_strategy),
                        dataset_strategies=(pcfg.experiment.dataset_strategies),
                        dataset_prefixes=pcfg.dataset.prefixes,
                        train_fraction=pcfg.experiment.train_fraction,
                        val_fraction=pcfg.experiment.val_fraction,
                        test_fraction=pcfg.experiment.test_fraction,
                        train_count=pcfg.experiment.train_count,
                        val_count=pcfg.experiment.val_count,
                        test_count=pcfg.experiment.test_count,
                        strategy_splits=pcfg.experiment.strategy_splits,
                    )
                case AzureDataAssetsCommand.UPLOAD_SPLITS:
                    upload_split_data_to_azure_blob(
                        console,
                        pcfg.experiment.experiment_strategy,
                    )
                case AzureDataAssetsCommand.REGISTER_SPLITS:
                    register_split_data_asset(
                        ml_client,
                        console,
                        pcfg.experiment.experiment_strategy,
                    )
                case AzureDataAssetsCommand.PROCESS_ALL_SPLITS:
                    # List of experiment strategies to process
                    strategies = [
                        "verification_box",
                        "verification_dfn",
                        "main_objective_dfn_rock_slope",
                        "main_objective_dfn_box",
                        "main_objective_box_rock_slope",
                        "main_objective_box_box",
                    ]
                    for strategy in strategies:
                        console.print(
                            "\n================ Processing strategy: "
                            f"{strategy} ================",
                            style="bold green",
                        )
                        try:
                            # Step 1: Generate splits
                            console.print(
                                f"Step 1: Generating splits for {strategy}...",
                                style="yellow",
                            )
                            prepare_and_save_dataset_splits(
                                console,
                                images_directory=pcfg.dataset.path_images,
                                labels_directory=(
                                    pcfg.dataset.path_processed_mask_labels
                                ),
                                experiment_strategy=strategy,
                                dataset_strategies=(pcfg.experiment.dataset_strategies),
                                dataset_prefixes=pcfg.dataset.prefixes,
                                train_fraction=pcfg.experiment.train_fraction,
                                val_fraction=pcfg.experiment.val_fraction,
                                test_fraction=pcfg.experiment.test_fraction,
                                train_count=pcfg.experiment.train_count,
                                val_count=pcfg.experiment.val_count,
                                test_count=pcfg.experiment.test_count,
                                strategy_splits=(pcfg.experiment.strategy_splits),
                            )
                            # Step 2: Upload splits
                            console.print(
                                "Step 2: Uploading splits for "
                                + f"{strategy} to Azure Blob Storage...",
                                style="yellow",
                            )
                            upload_split_data_to_azure_blob(console, strategy)
                            # Step 3: Register splits
                            console.print(
                                "Step 3: Registering splits for "
                                + f"{strategy} in Azure ML...",
                                style="yellow",
                            )
                            register_split_data_asset(
                                ml_client,
                                console,
                                strategy,
                            )
                            console.print(
                                f"Successfully processed {strategy}",
                                style="green",
                            )
                        except Exception as e:
                            console.print(
                                f"Error processing {strategy}: {str(e)}",
                                style="red",
                            )
                    console.print(
                        ("\nAll strategies processed. Check above for any errors."),
                        style="bold green",
                    )
                case _:
                    show_command_help(console)
        except ValueError as e:
            error_str = str(e)
            # Check for specific types of errors and provide helpful messages
            if "azure_data_assets.command" in error_str:
                console.print(f"Error: {error_str}", style="error")
                show_command_help(console)
            else:
                # For other errors, show the error but still with command help
                console.print(
                    f"Configuration error: {error_str}",
                    style="error",
                )
                console.print(
                    "Check your configuration and try again.", style="warning"
                )
                show_command_help(console)
    except Exception as e:
        # Catch any other exceptions
        console.print(f"Error: {str(e)}", style="error")
        show_command_help(console)


if __name__ == "__main__":
    main()
