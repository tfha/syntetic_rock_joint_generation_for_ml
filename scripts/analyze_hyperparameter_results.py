#!/usr/bin/env python
"""
Analyze hyperparameter optimization results from Azure ML.

This script retrieves the results of a completed hyperparameter optimization job
from Azure ML, analyzes the performance of different hyperparameter configurations,
and outputs a report with the best parameters found.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any  # Only import Any as it doesn't have a built-in equivalent

import pandas as pd
import yaml
from dotenv import load_dotenv

from ml_segmentation.azure_authentication import connect_to_azure_ml
from ml_segmentation.utility import get_custom_console


def setup_environment():
    """Setup environment and return a console for pretty printing."""
    # Load environment variables from .env file
    load_dotenv()

    # Create a console for pretty printing
    console = get_custom_console()

    # Check if Azure environment variables are set
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        console.print(
            "Error: AZURE_SUBSCRIPTION_ID environment variable not set", style="error"
        )
        console.print(
            "Please set it with: export AZURE_SUBSCRIPTION_ID='your-subscription-id'",
            style="error",
        )
        raise ValueError("Missing Azure subscription ID")

    return console


def get_job_details_from_args() -> argparse.Namespace:
    """Parse command line arguments to get job details."""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter optimization results from Azure ML"
    )

    parser.add_argument(
        "--job-name", type=str, help="Name of the Azure ML job to analyze"
    )

    parser.add_argument(
        "--job-info-file",
        type=str,
        help="Path to the job info JSON file created during job submission",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/hyperparameters/results",
        help="Directory to save analysis results",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top configurations to show in the report",
    )

    args = parser.parse_args()

    if not args.job_name and not args.job_info_file:
        parser.print_help()
        raise ValueError("Either --job-name or --job-info-file must be provided")

    return args


def get_job_info(args, console) -> dict[str, Any]:
    """Get job information from file or Azure ML."""
    if args.job_info_file:
        try:
            job_info_path = Path(args.job_info_file)
            if not job_info_path.exists():
                console.print(
                    f"Job info file not found: {args.job_info_file}", style="error"
                )
                raise FileNotFoundError(
                    f"Job info file not found: {args.job_info_file}"
                )

            with open(job_info_path) as f:
                job_info = json.load(f)
                job_name = job_info["job_name"]
                console.print(f"Loaded job info for job: {job_name}", style="info")
                return job_info

        except Exception as e:
            console.print(f"Error loading job info file: {e}", style="error")
            raise
    else:
        # If only job name is provided, create minimal job info
        return {"job_name": args.job_name}


def get_job_results(ml_client, job_name: str, console) -> pd.DataFrame:
    """Get job results from Azure ML and convert to DataFrame."""
    try:
        console.print(f"Fetching job data for: {job_name}", style="info")
        job = ml_client.jobs.get(job_name)

        # Check if job is completed
        if job.status != "Completed":
            console.print(f"Job status: {job.status}", style="warning")
            if job.status != "Failed":
                console.print(
                    "Job not completed yet, results may be incomplete", style="warning"
                )

        # Get all child runs
        console.print("Fetching child runs...", style="info")
        child_runs = list(ml_client.jobs.list(parent_job_name=job_name))
        console.print(f"Found {len(child_runs)} child runs", style="info")

        # Extract results into a DataFrame
        results = []
        for run in child_runs:
            if hasattr(run, "properties"):
                # Extract parameters and metrics
                run_data = {"run_id": run.name, "status": run.status}

                # Extract parameters
                if hasattr(run, "inputs") and run.inputs:
                    for name, value in run.inputs.items():
                        if isinstance(value, str) and not value.startswith("azureml:"):
                            run_data[f"param_{name}"] = value

                # Extract metrics
                if hasattr(run, "metrics") and run.metrics:
                    for name, value in run.metrics.items():
                        run_data[f"metric_{name}"] = value

                # Add to results
                results.append(run_data)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df

    except Exception as e:
        console.print(f"Error fetching job results: {e}", style="error")
        raise


def analyze_results(df: pd.DataFrame, console, top_n: int = 5) -> dict[str, Any]:
    """Analyze hyperparameter optimization results."""
    try:
        console.print("Analyzing hyperparameter optimization results...", style="info")

        # Check if we have the primary metric
        if "metric_val_dice_score" not in df.columns:
            metrics = [col for col in df.columns if col.startswith("metric_")]
            console.print(
                "Primary metric 'val_dice_score' not found. Available metrics: "
                f"{metrics}",
                style="warning",
            )

            # Try to find a suitable metric
            val_metrics = [m for m in metrics if "val" in m]
            if val_metrics:
                primary_metric = val_metrics[0]
                console.print(
                    f"Using '{primary_metric}' as the primary metric", style="info"
                )
            elif metrics:
                primary_metric = metrics[0]
                console.print(
                    f"Using '{primary_metric}' as the primary metric", style="info"
                )
            else:
                console.print("No metrics found in results", style="error")
                raise ValueError("No metrics found in results")
        else:
            primary_metric = "metric_val_dice_score"

        # Extract parameter columns
        param_cols = [col for col in df.columns if col.startswith("param_")]

        # Filter out failed runs
        if "status" in df.columns:
            df_completed = df[df["status"] == "Completed"].copy()
            console.print(
                f"Using {len(df_completed)} completed runs out of {len(df)} total runs",
                style="info",
            )
        else:
            df_completed = df.copy()

        # Sort by primary metric
        df_sorted = df_completed.sort_values(by=primary_metric, ascending=False)

        # Get top N configurations
        top_configs = df_sorted.head(top_n)

        # Prepare analysis results
        analysis = {
            "total_runs": len(df),
            "completed_runs": len(df_completed),
            "primary_metric": primary_metric.replace("metric_", ""),
            "top_configs": top_configs.to_dict("records"),
            "parameter_importance": {},
        }

        # Calculate parameter importance if we have enough data
        if len(df_completed) >= 10:
            console.print("Calculating parameter importance...", style="info")
            try:
                # Simple correlation-based importance
                param_importance: dict[str, float] = {}
                for param in param_cols:
                    if df_completed[param].nunique() > 1:
                        if pd.api.types.is_numeric_dtype(df_completed[param]):
                            corr = df_completed[param].corr(
                                df_completed[primary_metric]
                            )
                            param_importance[param.replace("param_", "")] = abs(corr)
                analysis["parameter_importance"] = param_importance
            except Exception as e:
                console.print(
                    f"Warning: Could not calculate parameter importance: {e}",
                    style="warning",
                )

        return analysis

    except Exception as e:
        console.print(f"Error analyzing results: {e}", style="error")
        raise


def generate_report(analysis: dict[str, Any], job_info: dict[str, Any], args, console):
    """Generate a report from the analysis results."""
    try:
        console.print("Generating analysis report...", style="info")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine model name
        model_name = job_info.get("model", "unknown")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        # Create report file
        report_file = output_dir / f"{model_name}_hparam_report_{timestamp}.md"

        # Write report
        with open(report_file, "w") as f:
            # Header
            f.write("# Hyperparameter Optimization Report\n\n")
            f.write(f"- **Model:** {model_name}\n")
            f.write(f"- **Job Name:** {job_info.get('job_name', 'unknown')}\n")
            f.write(f"- **Date:** {timestamp}\n")
            f.write(f"- **Total Runs:** {analysis['total_runs']}\n")
            f.write(f"- **Completed Runs:** {analysis['completed_runs']}\n\n")

            # Best configuration
            f.write("## Best Configuration\n\n")
            if analysis["top_configs"]:
                best_config = analysis["top_configs"][0]

                # Extract best metric value
                primary_metric = analysis["primary_metric"]
                best_metric_value = best_config.get(f"metric_{primary_metric}", "N/A")
                f.write(f"**Best {primary_metric}:** {best_metric_value}\n\n")

                # Extract parameters
                f.write("### Parameters:\n\n")
                params = {
                    k.replace("param_", ""): v
                    for k, v in best_config.items()
                    if k.startswith("param_")
                }

                # Format parameters as a table
                f.write("| Parameter | Value |\n")
                f.write("| --- | --- |\n")
                for param, value in params.items():
                    f.write(f"| {param} | {value} |\n")
                f.write("\n")

            # Top configurations
            f.write(f"## Top {len(analysis['top_configs'])} Configurations\n\n")

            # Format as table
            if analysis["top_configs"]:
                metrics = [
                    k
                    for k in analysis["top_configs"][0].keys()
                    if k.startswith("metric_")
                ]
                param_keys = [
                    k
                    for k in analysis["top_configs"][0].keys()
                    if k.startswith("param_")
                ]

                # Create table header
                header = (
                    ["Rank", "Run ID"]
                    + [m.replace("metric_", "") for m in metrics]
                    + [p.replace("param_", "") for p in param_keys]
                )
                f.write("| " + " | ".join(header) + " |\n")
                f.write("| " + " | ".join(["---" for _ in header]) + " |\n")

                for i, config in enumerate(analysis["top_configs"]):
                    row = [str(i + 1), config.get("run_id", "N/A")]
                    # Add metrics
                    for m in metrics:
                        row.append(str(config.get(m, "N/A")))

                    # Add parameters
                    for p in param_keys:
                        row.append(str(config.get(p, "N/A")))

                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")

            # Parameter importance
            if analysis.get("parameter_importance"):
                f.write("## Parameter Importance\n\n")
                f.write("| Parameter | Importance |\n")
                f.write("| --- | --- |\n")

                # Sort by importance
                sorted_importance = sorted(
                    analysis["parameter_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for param, importance in sorted_importance:
                    f.write(f"| {param} | {importance:.4f} |\n")
                f.write("\n")

            # Hydra config for best parameters
            if analysis["top_configs"]:
                f.write("## Hydra Configuration for Best Parameters\n\n")
                f.write("```yaml\n")

                # Convert parameters to hydra format
                hydra_config: dict[str, Any] = {}
                # Re-derive params from best_config to avoid scope issues
                best_config = analysis["top_configs"][0]
                best_params = {
                    k.replace("param_", ""): v
                    for k, v in best_config.items()
                    if k.startswith("param_")
                }
                for param, value in best_params.items():
                    keys = param.split(".")
                    current = hydra_config
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = value

                # Write as yaml
                yaml_str = yaml.dump(hydra_config, default_flow_style=False)
                f.write(yaml_str)
                f.write("```\n")

        console.print(f"Report generated: {report_file}", style="success")

        # Generate plots
        _ = output_dir / f"{model_name}_hparam_plots_{timestamp}.png"

        # Save plots as separate files or return them
        return report_file

    except Exception as e:
        console.print(f"Error generating report: {e}", style="error")
        raise


def main():
    """Main entry point for the script."""
    try:
        # Set up environment and get job details
        console = setup_environment()
        args = get_job_details_from_args()

        # Get Azure ML client
        sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID") or ""
        ml_client = connect_to_azure_ml(
            subscription_id=sub_id,
            console=console,
            resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
            workspace_name=os.environ.get("AZURE_ML_WORKSPACE"),
        )

        # Get job information
        job_info = get_job_info(args, console)
        job_name = job_info["job_name"]

        # Get job results
        results_df = get_job_results(ml_client, job_name, console)

        # Analyze results
        analysis = analyze_results(results_df, console, args.top_n)

        # Generate report
        report_file = generate_report(analysis, job_info, args, console)

        # Create visualizations
        # (Additional code for plots could be added here)

        console.print("Analysis completed successfully!", style="success")
        console.print(f"Report saved to: {report_file}", style="info")

    except Exception as e:
        console = get_custom_console()
        console.print(f"Error: {e}", style="error")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
