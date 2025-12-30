"""
Download models from Azure ML for all experiments.

Selects the best performing model (UNet or DeepLabV3+) for each experiment
based on best_val_dice_joints and average_quality_score from Appendix F.

For finetune strategy:
- best_model.pth: Best model from Stage 2 (finetuning on real data)
- stage1_best_model.pth: Best model from Stage 1 (pretraining on synthetic data)

For simplemixed strategy:
- best_metrics_model.pth: Best model based on validation metrics during training
- final_model.pth: Final model after all epochs
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()


def load_batch_jobs_summary(csv_path: Path) -> dict[tuple[str, str], str]:
    """Load batch jobs summary CSV and map (architecture, experiment_strategy) to job_name."""
    experiment_to_job = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            architecture = row["model"]
            exp_strategy = row["experiment_strategy"]
            job_name = row["job_name"]
            experiment_to_job[(architecture, exp_strategy)] = job_name

    return experiment_to_job


def load_appendix_f(csv_path: Path) -> list[dict[str, str]]:
    """Load Appendix F CSV with experiment results."""
    experiments = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            experiments.append(row)

    return experiments


def select_best_models(experiments: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select the best performing models for each experiment and strategy.

    For each (experiment, strategy) combination:
    - Selects the model with best best_val_dice_joints (regardless of architecture/proportion)
    - Selects the model with best average_quality_score (regardless of architecture/proportion)

    Args:
        experiments: List of all experiments from Appendix F

    Returns:
        List of best experiments with 'selection_reason' field added
    """
    # Group experiments by (experiment, strategy)
    grouped = defaultdict(list)

    for exp in experiments:
        experiment = exp.get("experiment", "")
        strategy = exp.get("strategy", "")

        key = (experiment, strategy)
        grouped[key].append(exp)

    # Select best models for each group
    best_experiments = []
    selected_displaynames = set()

    console.print(
        f"\nFound {len(grouped)} unique (experiment, strategy) combinations\n"
    )

    for key, group in grouped.items():
        experiment, strategy = key

        # Find maximum Dice score
        max_dice_score = max(
            (float(e.get("best_val_dice_joints", 0) or 0) for e in group), default=0
        )

        # Find ALL models with best Dice score (handle ties)
        best_dice_models = [
            e
            for e in group
            if abs(float(e.get("best_val_dice_joints", 0) or 0) - max_dice_score)
            < 0.0001
        ]

        # Find maximum Quality score
        max_quality_score = max(
            (float(e.get("average_quality_score", 0) or 0) for e in group), default=0
        )

        # Find ALL models with best Quality score (handle ties)
        best_quality_models = [
            e
            for e in group
            if abs(float(e.get("average_quality_score", 0) or 0) - max_quality_score)
            < 0.0001
        ]

        # Add all best dice models
        for best_dice in best_dice_models:
            best_dice_copy = best_dice.copy()
            best_dice_copy["selection_reason"] = "best_dice"
            best_experiments.append(best_dice_copy)
            selected_displaynames.add(best_dice.get("displayName"))

            console.print(
                f"[cyan]{experiment:30}_{strategy:12}[/cyan] - Best Dice: "
                f"{best_dice.get('architecture'):12} "
                f"Dice={best_dice.get('best_val_dice_joints'):5}, "
                f"Quality={best_dice.get('average_quality_score'):3}, "
                f"Prop={best_dice.get('proportion_real'):3}"
            )

        # Add all best quality models (skip if already added as best dice)
        quality_added = 0
        for best_quality in best_quality_models:
            if best_quality.get("displayName") not in selected_displaynames:
                best_quality_copy = best_quality.copy()
                best_quality_copy["selection_reason"] = "best_quality"
                best_experiments.append(best_quality_copy)
                selected_displaynames.add(best_quality.get("displayName"))
                quality_added += 1

                console.print(
                    f"[cyan]{experiment:30}_{strategy:12}[/cyan] - Best Quality: "
                    f"{best_quality.get('architecture'):12} "
                    f"Dice={best_quality.get('best_val_dice_joints'):5}, "
                    f"Quality={best_quality.get('average_quality_score'):3}, "
                    f"Prop={best_quality.get('proportion_real'):3}"
                )

        if quality_added == 0 and len(best_quality_models) > 0:
            console.print(
                f"[dim]{'':30} {'':12}   (Same model(s) have best Dice & Quality)[/dim]"
            )

    return best_experiments


def verify_best_models(
    best_experiments: list[dict[str, str]], all_experiments: list[dict[str, str]]
) -> bool:
    """Verify that selected models are truly the best in their groups.

    Args:
        best_experiments: Selected best models
        all_experiments: All experiments from Appendix F

    Returns:
        True if all selections are valid
    """
    console.print(
        "\n[bold cyan]Verifying selections against Appendix F...[/bold cyan]\n"
    )

    # Group all experiments by (experiment, strategy)
    grouped = defaultdict(list)
    for exp in all_experiments:
        experiment = exp.get("experiment", "")
        strategy = exp.get("strategy", "")
        key = (experiment, strategy)
        grouped[key].append(exp)

    all_valid = True
    issues = []

    for best_exp in best_experiments:
        experiment = best_exp.get("experiment", "")
        strategy = best_exp.get("strategy", "")
        selection_reason = best_exp.get("selection_reason", "")
        displayName = best_exp.get("displayName", "")

        key = (experiment, strategy)
        group = grouped.get(key, [])

        if selection_reason == "best_dice":
            # Check if this is truly the best Dice score
            max_dice = max(
                (float(e.get("best_val_dice_joints", 0) or 0) for e in group), default=0
            )
            selected_dice = float(best_exp.get("best_val_dice_joints", 0) or 0)

            if abs(selected_dice - max_dice) > 0.0001:  # Allow small float differences
                all_valid = False
                issues.append(
                    {
                        "experiment": experiment,
                        "strategy": strategy,
                        "type": "dice",
                        "selected": displayName,
                        "selected_value": selected_dice,
                        "max_value": max_dice,
                    }
                )

        elif selection_reason == "best_quality":
            # Check if this is truly the best Quality score
            max_quality = max(
                (float(e.get("average_quality_score", 0) or 0) for e in group),
                default=0,
            )
            selected_quality = float(best_exp.get("average_quality_score", 0) or 0)

            if abs(selected_quality - max_quality) > 0.0001:
                all_valid = False
                issues.append(
                    {
                        "experiment": experiment,
                        "strategy": strategy,
                        "type": "quality",
                        "selected": displayName,
                        "selected_value": selected_quality,
                        "max_value": max_quality,
                    }
                )

    if all_valid:
        console.print(
            "[green]✓ All selections verified! Each model is the best in its category.[/green]"
        )
    else:
        console.print(f"[red]✗ Found {len(issues)} issues with selections:[/red]\n")
        for issue in issues[:5]:  # Show first 5 issues
            console.print(
                f"  {issue['experiment']}_{issue['strategy']} ({issue['type']}): "
                f"Selected {issue['selected_value']}, but max is {issue['max_value']}"
            )

    return all_valid


def extract_strategy_from_displayname(displayname: str) -> tuple[tuple[str, str], str]:
    """
    Extract architecture and strategy from displayname.

    Example: 'deeplabv3plus-finetune_box_10-20251211-1646'
    Returns: ('deeplabv3plus', 'finetune'), 'finetune_box_10'
    """
    parts = displayname.split("-")
    architecture = parts[0]

    # Extract strategy and experiment identifier
    strategy_exp = parts[1]

    # Determine if finetune or simplemixed
    if "finetune" in strategy_exp:
        strategy = "finetune"
    elif "simplemixed" in strategy_exp:
        strategy = "simplemixed"
    else:
        strategy = "unknown"

    return (architecture, strategy), strategy_exp


def get_blob_client(ml_client: MLClient) -> tuple[BlobServiceClient, str]:
    """Get blob service client using Azure ML workspace's storage account.

    Args:
        ml_client: Azure ML client

    Returns:
        Tuple of (BlobServiceClient, container_name)
    """
    # Get workspace info
    workspace = ml_client.workspaces.get(ml_client.workspace_name)

    # Get the storage account from workspace
    storage_account = workspace.storage_account

    # Extract account name from full resource ID
    # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{name}
    account_name = storage_account.split("/")[-1]

    # Create blob service client with same credential as ML client
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=ml_client._credential
    )

    # Container name is typically azureml-blobstore-{guid}
    # We'll get this from workspace default datastore
    try:
        datastore = ml_client.datastores.get_default()
        container_name = datastore.container_name
    except Exception:
        # Fallback: try to find azureml container
        container_name = "azureml"

    return blob_service_client, container_name


def get_model_blob_paths(job_name: str, strategy: str) -> dict[str, str]:
    """
    Generate blob storage paths for models based on strategy.

    Args:
        job_name: Azure ML job name (e.g., 'purple_parrot_0br1ylt249')
        strategy: Training strategy ('finetune' or 'simplemixed')

    Returns:
        Dictionary mapping model type to blob path
    """
    paths = {}

    if strategy == "finetune":
        # Finetune strategy models
        paths["best_model"] = f"ExperimentRun/dcid.{job_name}/best_model.pth"
        paths["stage1_best_model"] = (
            f"ExperimentRun/dcid.{job_name}/outputs/models/stage1_best_model.pth"
        )
    elif strategy == "simplemixed":
        # Simplemixed strategy models
        paths["best_metrics_model"] = (
            f"ExperimentRun/dcid.{job_name}/outputs/models/best_metrics_model.pth"
        )
        paths["final_model"] = (
            f"ExperimentRun/dcid.{job_name}/outputs/models/final_model.pth"
        )
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")

    return paths


def download_blob_to_file(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_path: str,
    output_path: Path,
) -> tuple[bool, str]:
    """Download a single blob to a file.

    Args:
        blob_service_client: Azure Blob Service client
        container_name: Container name
        blob_path: Blob path within container
        output_path: Local file path to save to

    Returns:
        Tuple of (success, error_message)
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download blob
        with open(output_path, "wb") as f:
            blob_data = blob_client.download_blob()
            blob_data.readinto(f)

        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    """Main function to download all models."""
    # Load environment variables
    load_dotenv()

    # Setup paths
    repo_root = Path(__file__).parent.parent
    batch_jobs_csv = repo_root / "experiments" / "batch_jobs_summary.csv"
    appendix_f_csv = Path(
        r"C:\Users\KYC\OneDrive - NGI\Documents\PhD\Paper - Synthetic rock joint generation for ML\Appendices\Appendix_F.csv"
    )
    models_output_dir = repo_root / "models" / "downloaded_experiments"

    # Load data
    console.print("[bold cyan]Loading experiment data...[/bold cyan]")
    experiment_to_job = load_batch_jobs_summary(batch_jobs_csv)
    all_experiments = load_appendix_f(appendix_f_csv)

    console.print(f"Found {len(all_experiments)} total experiments")
    console.print(f"Found {len(experiment_to_job)} job mappings (240 total)")

    # Select best models for each experiment configuration
    console.print(
        "\n[bold cyan]Selecting best models for each experiment...[/bold cyan]"
    )
    experiments = select_best_models(all_experiments)
    console.print(f"Selected {len(experiments)} best models")

    # Verify selections
    if not verify_best_models(experiments, all_experiments):
        console.print(
            "\n[red]Verification failed! Please review the issues above.[/red]"
        )
        return

    # Authenticate with Azure
    try:
        console.print("\n[bold cyan]Connecting to Azure ML...[/bold cyan]")
        credential = DefaultAzureCredential()

        # Get Azure ML client
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_ML_WORKSPACE")

        if not all([subscription_id, resource_group, workspace_name]):
            console.print("[red]Error: Missing Azure configuration in .env file[/red]")
            console.print(
                "Required: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE"
            )
            return

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        # Get blob service client
        blob_service_client, container_name = get_blob_client(ml_client)

        # Get workspaceartifactstore (where ExperimentRun files are stored)
        datastore = ml_client.datastores.get("workspaceartifactstore")
        container_name = datastore.container_name

        console.print(f"[green]Connected to container: {container_name}[/green]")

    except Exception as e:
        console.print(f"[red]Azure connection failed: {e}[/red]")
        return

    # Process each experiment
    download_plan: list[dict[str, str | Path | None]] = []

    for exp in experiments:
        # Handle both displayName and displayname
        displayname = exp.get("displayName") or exp.get("displayname")
        if not displayname:
            console.print(f"[yellow]Warning: No displayName in row: {exp}[/yellow]")
            continue

        (architecture, strategy), strategy_exp = extract_strategy_from_displayname(
            displayname
        )

        # Get job name from batch_jobs_summary using both architecture and strategy
        job_key = (architecture, strategy_exp)
        if job_key not in experiment_to_job:
            console.print(
                f"[yellow]Warning: No job mapping found for {architecture}/{strategy_exp}[/yellow]"
            )
            continue

        job_name = experiment_to_job[job_key]

        # Get model blob paths
        model_paths = get_model_blob_paths(job_name, strategy)

        if not model_paths:
            continue

        # Get experiment name and selection reason
        experiment_name = exp.get("experiment", "")
        selection_reason = exp.get("selection_reason", "best_dice")

        # Map selection reason to folder name
        metric_folder = (
            "best_val_dice_joint"
            if selection_reason == "best_dice"
            else "best_quality_score"
        )

        # Plan downloads - organize by experiment/strategy/metric_type/displayname/
        for model_type, blob_path in model_paths.items():
            destination = (
                models_output_dir
                / experiment_name
                / strategy
                / metric_folder
                / displayname
                / f"{model_type}.pth"
            )
            download_plan.append(
                {
                    "displayname": displayname,
                    "architecture": architecture,
                    "strategy": strategy,
                    "strategy_exp": strategy_exp,
                    "model_type": model_type,
                    "blob_path": blob_path,
                    "destination": destination,
                    "dice": exp.get("best_val_dice_joints"),
                    "quality": exp.get("average_quality_score"),
                    "selection_reason": selection_reason,
                }
            )

    console.print(
        f"\n[bold cyan]Download plan: {len(download_plan)} model files[/bold cyan]"
    )

    # Show summary by strategy
    finetune_count = sum(1 for d in download_plan if d["strategy"] == "finetune")
    simplemixed_count = sum(1 for d in download_plan if d["strategy"] == "simplemixed")
    console.print(f"  - Finetune model files: {finetune_count}")
    console.print(f"  - Simplemixed model files: {simplemixed_count}")

    # Count unique experiments
    unique_exps = len({d["strategy_exp"] for d in download_plan})
    console.print(f"  - Unique experiments: {unique_exps}")

    # Confirm before downloading
    console.print("\n[bold yellow]Model types explanation:[/bold yellow]")
    console.print("\n[cyan]Finetune strategy:[/cyan]")
    console.print(
        "  - best_model.pth: Best model from Stage 2 (finetuning on real data)"
    )
    console.print(
        "  - stage1_best_model.pth: Best model from Stage 1 (pretraining on synthetic data)"
    )

    console.print("\n[cyan]Simplemixed strategy:[/cyan]")
    console.print(
        "  - best_metrics_model.pth: Best model based on validation metrics during training"
    )
    console.print("  - final_model.pth: Final model after all epochs")

    response = console.input("\n[bold]Proceed with downloads? (y/n): [/bold]")
    if response.lower() != "y":
        console.print("[yellow]Download cancelled[/yellow]")
        return

    # Download files with progress bar
    console.print("\n[bold cyan]Downloading models...[/bold cyan]")

    success_count = 0
    failed_count = 0
    failed_details = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(download_plan))

        for plan in download_plan:
            # Show current download - extract with type narrowing
            strategy_exp_val = plan["strategy_exp"]
            model_type_val = plan["model_type"]
            blob_path_val = plan["blob_path"]
            destination_val = plan["destination"]

            # Type narrow using assertions
            assert isinstance(strategy_exp_val, str)
            assert isinstance(model_type_val, str)
            assert isinstance(blob_path_val, str)
            assert isinstance(destination_val, Path)

            desc = f"{strategy_exp_val[:30]:30} | {model_type_val}"
            progress.update(task, description=desc)

            # Download using blob client
            success, error = download_blob_to_file(
                blob_service_client,
                container_name,
                blob_path_val,
                destination_val,
            )

            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_details.append(
                    {
                        "exp": strategy_exp_val,
                        "model": model_type_val,
                        "path": blob_path_val,
                        "error": error,
                    }
                )

            progress.advance(task)

    # Summary
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"  - Successful: {success_count}")
    console.print(f"  - Failed: {failed_count}")

    if failed_count > 0:
        console.print("\n[yellow]Failed downloads (first 5):[/yellow]")
        for fail in failed_details[:5]:
            console.print(f"  - {fail['exp']} / {fail['model']}")
            console.print(f"    Path: {fail['path']}")
            console.print(f"    Error: {fail['error'][:100]}")

    console.print(f"\nModels saved to: {models_output_dir}")

    # Check if models are identical for demonstration
    if success_count > 0:
        console.print("\n[bold cyan]Checking model differences...[/bold cyan]")
        check_model_differences(models_output_dir)


def check_model_differences(models_dir: Path):
    """
    Check if downloaded models are identical or different.
    This helps understand what each model type represents.
    """
    import torch

    console.print("\n[cyan]Comparing model files within experiments:[/cyan]")

    # Find all experiment directories
    for architecture_dir in models_dir.iterdir():
        if not architecture_dir.is_dir():
            continue

        for exp_dir in architecture_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            models = list(exp_dir.glob("*.pth"))

            if len(models) < 2:
                continue

            # Load models and compare
            console.print(f"\n[yellow]{architecture_dir.name}/{exp_dir.name}:[/yellow]")

            model_states = {}
            for model_path in models:
                try:
                    state_dict = torch.load(model_path, map_location="cpu")
                    model_states[model_path.stem] = state_dict
                    console.print(
                        f"  - {model_path.stem}: {len(state_dict)} parameters"
                    )
                except Exception as e:
                    console.print(
                        f"  - [red]{model_path.stem}: Error loading - {e}[/red]"
                    )

            # Compare if we have multiple models
            if len(model_states) >= 2:
                model_names = list(model_states.keys())

                # Compare first two models
                state1 = model_states[model_names[0]]
                state2 = model_states[model_names[1]]

                # Check if identical
                are_identical = len(state1) == len(state2)

                if are_identical:
                    for key in state1.keys():
                        if key not in state2:
                            are_identical = False
                            break
                        if not torch.equal(state1[key], state2[key]):
                            are_identical = False
                            break

                if are_identical:
                    console.print("    [green]→ Models are IDENTICAL[/green]")
                else:
                    console.print("    [yellow]→ Models are DIFFERENT[/yellow]")


if __name__ == "__main__":
    main()
