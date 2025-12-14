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
    upload_finetune_split_data_to_azure_blob,
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
        AzureDataAssetsCommand.REGISTER_FINETUNE_SPLITS: (
            "Register all Wachter et al. (2025) finetune experiment splits "
            "(140 experiments: 70 SM + 70 FT)"
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
                AzureDataAssetsCommand.REGISTER_FINETUNE_SPLITS,
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
                case AzureDataAssetsCommand.REGISTER_FINETUNE_SPLITS:
                    # Upload and register all Wachter et al. experiment splits
                    # 140 total: 70 Simple Mixed (SM) + 70 Fine-Tuned (FT)
                    wachter_strategies = [
                        # Simple Mixed (SM) - BOX
                        "simplemixed_box_0",
                        "simplemixed_box_10",
                        "simplemixed_box_30",
                        "simplemixed_box_50",
                        "simplemixed_box_70",
                        "simplemixed_box_90",
                        "simplemixed_box_100",
                        # Simple Mixed (SM) - SLOPE
                        "simplemixed_slope_0",
                        "simplemixed_slope_10",
                        "simplemixed_slope_30",
                        "simplemixed_slope_50",
                        "simplemixed_slope_70",
                        "simplemixed_slope_90",
                        "simplemixed_slope_100",
                        # Simple Mixed (SM) - GENERALISATION PATTERN BOX
                        "simplemixed_generalisation_pattern_box_0",
                        "simplemixed_generalisation_pattern_box_10",
                        "simplemixed_generalisation_pattern_box_30",
                        "simplemixed_generalisation_pattern_box_50",
                        "simplemixed_generalisation_pattern_box_70",
                        "simplemixed_generalisation_pattern_box_90",
                        "simplemixed_generalisation_pattern_box_100",
                        # Simple Mixed (SM) - PATTERN BOX
                        "simplemixed_pattern_box_0",
                        "simplemixed_pattern_box_10",
                        "simplemixed_pattern_box_30",
                        "simplemixed_pattern_box_50",
                        "simplemixed_pattern_box_70",
                        "simplemixed_pattern_box_90",
                        "simplemixed_pattern_box_100",
                        # Simple Mixed (SM) - GENERALISATION CARDBOARD BOX
                        "simplemixed_generalisation_cardboard_box_0",
                        "simplemixed_generalisation_cardboard_box_10",
                        "simplemixed_generalisation_cardboard_box_30",
                        "simplemixed_generalisation_cardboard_box_50",
                        "simplemixed_generalisation_cardboard_box_70",
                        "simplemixed_generalisation_cardboard_box_90",
                        "simplemixed_generalisation_cardboard_box_100",
                        # Simple Mixed (SM) - CARDBOARD BOX
                        "simplemixed_cardboard_box_0",
                        "simplemixed_cardboard_box_10",
                        "simplemixed_cardboard_box_30",
                        "simplemixed_cardboard_box_50",
                        "simplemixed_cardboard_box_70",
                        "simplemixed_cardboard_box_90",
                        "simplemixed_cardboard_box_100",
                        # Simple Mixed (SM) - GENERALISATION LARVIK
                        "simplemixed_generalisation_larvik_0",
                        "simplemixed_generalisation_larvik_10",
                        "simplemixed_generalisation_larvik_30",
                        "simplemixed_generalisation_larvik_50",
                        "simplemixed_generalisation_larvik_70",
                        "simplemixed_generalisation_larvik_90",
                        "simplemixed_generalisation_larvik_100",
                        # Simple Mixed (SM) - LARVIK
                        "simplemixed_larvik_0",
                        "simplemixed_larvik_10",
                        "simplemixed_larvik_30",
                        "simplemixed_larvik_50",
                        "simplemixed_larvik_70",
                        "simplemixed_larvik_90",
                        "simplemixed_larvik_100",
                        # Simple Mixed (SM) - GENERALISATION RV4
                        "simplemixed_generalisation_rv4_0",
                        "simplemixed_generalisation_rv4_10",
                        "simplemixed_generalisation_rv4_30",
                        "simplemixed_generalisation_rv4_50",
                        "simplemixed_generalisation_rv4_70",
                        "simplemixed_generalisation_rv4_90",
                        "simplemixed_generalisation_rv4_100",
                        # Simple Mixed (SM) - RV4
                        "simplemixed_rv4_0",
                        "simplemixed_rv4_10",
                        "simplemixed_rv4_30",
                        "simplemixed_rv4_50",
                        "simplemixed_rv4_70",
                        "simplemixed_rv4_90",
                        "simplemixed_rv4_100",
                        # Fine-Tuned (FT) - BOX
                        "finetune_box_0",
                        "finetune_box_10",
                        "finetune_box_30",
                        "finetune_box_50",
                        "finetune_box_70",
                        "finetune_box_90",
                        "finetune_box_100",
                        # Fine-Tuned (FT) - SLOPE
                        "finetune_slope_0",
                        "finetune_slope_10",
                        "finetune_slope_30",
                        "finetune_slope_50",
                        "finetune_slope_70",
                        "finetune_slope_90",
                        "finetune_slope_100",
                        # Fine-Tuned (FT) - GENERALISATION PATTERN BOX
                        "finetune_generalisation_pattern_box_0",
                        "finetune_generalisation_pattern_box_10",
                        "finetune_generalisation_pattern_box_30",
                        "finetune_generalisation_pattern_box_50",
                        "finetune_generalisation_pattern_box_70",
                        "finetune_generalisation_pattern_box_90",
                        "finetune_generalisation_pattern_box_100",
                        # Fine-Tuned (FT) - PATTERN BOX
                        "finetune_pattern_box_0",
                        "finetune_pattern_box_10",
                        "finetune_pattern_box_30",
                        "finetune_pattern_box_50",
                        "finetune_pattern_box_70",
                        "finetune_pattern_box_90",
                        "finetune_pattern_box_100",
                        # Fine-Tuned (FT) - GENERALISATION CARDBOARD BOX
                        "finetune_generalisation_cardboard_box_0",
                        "finetune_generalisation_cardboard_box_10",
                        "finetune_generalisation_cardboard_box_30",
                        "finetune_generalisation_cardboard_box_50",
                        "finetune_generalisation_cardboard_box_70",
                        "finetune_generalisation_cardboard_box_90",
                        "finetune_generalisation_cardboard_box_100",
                        # Fine-Tuned (FT) - CARDBOARD BOX
                        "finetune_cardboard_box_0",
                        "finetune_cardboard_box_10",
                        "finetune_cardboard_box_30",
                        "finetune_cardboard_box_50",
                        "finetune_cardboard_box_70",
                        "finetune_cardboard_box_90",
                        "finetune_cardboard_box_100",
                        # Fine-Tuned (FT) - GENERALISATION LARVIK
                        "finetune_generalisation_larvik_0",
                        "finetune_generalisation_larvik_10",
                        "finetune_generalisation_larvik_30",
                        "finetune_generalisation_larvik_50",
                        "finetune_generalisation_larvik_70",
                        "finetune_generalisation_larvik_90",
                        "finetune_generalisation_larvik_100",
                        # Fine-Tuned (FT) - LARVIK
                        "finetune_larvik_0",
                        "finetune_larvik_10",
                        "finetune_larvik_30",
                        "finetune_larvik_50",
                        "finetune_larvik_70",
                        "finetune_larvik_90",
                        "finetune_larvik_100",
                        # Fine-Tuned (FT) - GENERALISATION RV4
                        "finetune_generalisation_rv4_0",
                        "finetune_generalisation_rv4_10",
                        "finetune_generalisation_rv4_30",
                        "finetune_generalisation_rv4_50",
                        "finetune_generalisation_rv4_70",
                        "finetune_generalisation_rv4_90",
                        "finetune_generalisation_rv4_100",
                        # Fine-Tuned (FT) - RV4
                        "finetune_rv4_0",
                        "finetune_rv4_10",
                        "finetune_rv4_30",
                        "finetune_rv4_50",
                        "finetune_rv4_70",
                        "finetune_rv4_90",
                        "finetune_rv4_100",
                    ]
                    console.print(
                        "\n=== Uploading & Registering Wachter et al. Splits ===",
                        style="bold green",
                    )
                    console.print(
                        f"Total experiments: {len(wachter_strategies)} "
                        "(70 SM + 70 FT)\n",
                        style="info",
                    )
                    for i, strategy in enumerate(wachter_strategies, 1):
                        console.print(
                            f"\n[{i}/{len(wachter_strategies)}] "
                            f"Processing {strategy}...",
                            style="yellow",
                        )
                        try:
                            # Step 1: Upload splits to Azure Blob Storage
                            console.print(f"  Uploading {strategy}...", style="info")
                            upload_finetune_split_data_to_azure_blob(
                                console,
                                strategy,
                            )
                            # Step 2: Register as data asset
                            console.print(f"  Registering {strategy}...", style="info")
                            register_split_data_asset(
                                ml_client,
                                console,
                                strategy,
                            )
                            console.print(
                                f"[OK] Successfully processed {strategy}",
                                style="green",
                            )
                        except Exception as e:
                            console.print(
                                f"[ERROR] Error processing {strategy}: {str(e)}",
                                style="red",
                            )
                    console.print(
                        "\n=== Wachter splits upload & registration complete ===",
                        style="bold green",
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
