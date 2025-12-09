"""Submit multiple Azure ML jobs in batch.

This script submits a list of predefined Azure ML training jobs sequentially.
Each job is submitted via azure_submit_job.py with specific model and experiment configurations.
Jobs are defined in a simple text file (default: batch_jobs_config.txt).

Configuration file format (one job per line):
    model experiment_strategy
    # Lines starting with # are comments
    # Empty lines are ignored

Example:
    unet finetune_box_0
    unet finetune_box_10
    deeplabv3plus finetune_slope_50

Usage:
    poetry run python scripts/submit_batch_jobs.py
    poetry run python scripts/submit_batch_jobs.py --dry-run  # Preview commands without executing
    poetry run python scripts/submit_batch_jobs.py --config custom_jobs.txt  # Use custom config
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def validate_config(
    jobs: list[tuple[str, str]], scripts_dir: Path
) -> tuple[bool, list[str]]:
    """Validate that all models and experiment strategies exist in config files.

    Args:
        jobs: List of (model, experiment_strategy) tuples
        scripts_dir: Path to scripts directory

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    config_dir = scripts_dir / "config"
    model_dir = config_dir / "model"
    main_config = config_dir / "main.yaml"

    # Load main.yaml to get valid experiment strategies
    try:
        with open(main_config, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to load main.yaml: {e}")
        return False, errors

    # Extract valid experiment strategies from dataset_strategies
    valid_strategies = set()
    if (
        "experiment" in config_data
        and "dataset_strategies" in config_data["experiment"]
    ):
        valid_strategies = set(config_data["experiment"]["dataset_strategies"].keys())

    # Get valid models from config/model/*.yaml files
    valid_models = set()
    if model_dir.exists():
        for yaml_file in model_dir.glob("*.yaml"):
            # Model name is the filename without extension
            valid_models.add(yaml_file.stem)

    # Validate each job
    for i, (model, strategy) in enumerate(jobs, 1):
        # Check model exists
        if model not in valid_models:
            errors.append(
                f"Job {i}: Invalid model '{model}'. "
                f"Available models: {', '.join(sorted(valid_models))}"
            )

        # Check experiment strategy exists
        if strategy not in valid_strategies:
            errors.append(
                f"Job {i}: Invalid experiment_strategy '{strategy}'. "
                f"Available strategies: {', '.join(sorted(valid_strategies))}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def run_command(command: list[str], dry_run: bool = False) -> tuple[bool, str | None]:
    """Run a command and return success status and job name.

    Args:
        command: Command and arguments as a list
        dry_run: If True, print command without executing

    Returns:
        Tuple of (success, job_name)
        - success: True if command succeeded (or dry_run), False otherwise
        - job_name: Azure ML job name if successful, None otherwise
    """
    cmd_str = " ".join(command)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")

    if dry_run:
        return True, "dry_run_job_name"

    try:
        import os

        env = os.environ.copy()
        env["AZURE_BATCH_MODE"] = "true"
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, env=env
        )

        # Parse job name from output (look for "✓ Job submitted: <job_name>")
        job_name = None
        for line in result.stdout.split("\n"):
            if "✓ Job submitted:" in line or "Job submitted:" in line:
                # Extract job name after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    job_name = parts[-1].strip()
                    break

        print("✓ Job submitted successfully")
        if job_name:
            print(f"  Job name: {job_name}")
        return True, job_name
    except subprocess.CalledProcessError as e:
        print(f"✗ Job submission failed with exit code {e.returncode}", file=sys.stderr)
        if e.stderr:
            print(f"  Error: {e.stderr}", file=sys.stderr)
        return False, None


def main() -> None:
    """Submit batch of Azure ML jobs."""
    parser = argparse.ArgumentParser(
        description="Submit multiple Azure ML jobs in batch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop submitting jobs if one fails",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "batch_jobs_config.txt",
        help="Path to config file (default: config/batch_jobs_config.txt)",
    )
    args = parser.parse_args()

    # Load jobs from text config file
    config_path = args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # First pass: count total lines to parse
    total_lines = 0
    try:
        with open(config_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    total_lines += 1
    except Exception as e:
        print(f"Error: Failed to read config file: {e}", file=sys.stderr)
        sys.exit(1)

    jobs = []
    try:
        with open(config_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace and skip comments/empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse: model experiment_strategy
                parts = line.split()
                if len(parts) != 2:
                    print(
                        f"Warning: Skipping invalid line {line_num}: '{line}' "
                        f"(expected format: model experiment_strategy)",
                        file=sys.stderr,
                    )
                    continue

                model, strategy = parts
                jobs.append((model, strategy))
                print(f"  Parsed job {len(jobs)}/{total_lines}: {model} | {strategy}")

    except Exception as e:
        print(f"Error: Failed to read config file: {e}", file=sys.stderr)
        sys.exit(1)

    if not jobs:
        print("Error: No valid jobs found in config file", file=sys.stderr)
        sys.exit(1)

    # Validate jobs before submission
    print(f"{'=' * 80}")
    print(f"Validating {len(jobs)} jobs from: {config_path.name}")
    print(f"{'=' * 80}")

    base_dir = Path(__file__).parent
    is_valid, validation_errors = validate_config(jobs, base_dir)

    if not is_valid:
        print("\n❌ Validation failed with the following errors:\n", file=sys.stderr)
        for error in validation_errors:
            print(f"  • {error}", file=sys.stderr)
        print(
            "\nPlease fix the errors in your config file and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("✓ All jobs validated successfully\n")

    print(f"{'=' * 80}")
    print(f"Submitting {len(jobs)} Azure ML jobs")
    print(f"{'=' * 80}")

    # Base command
    submit_script = base_dir / "azure_submit_job.py"

    # Prepare job manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = base_dir / "experiments" / "batch_submissions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"job_manifest_{timestamp}.json"

    job_manifest = {
        "submission_time": timestamp,
        "config_file": str(config_path.name),
        "total_jobs": len(jobs),
        "dry_run": args.dry_run,
        "jobs": [],
    }

    success_count = 0
    failed_jobs = []

    for i, (model, strategy) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] Model: {model}, Strategy: {strategy}")

        command = [
            "poetry",
            "run",
            "python",
            str(submit_script),
            f"model={model}",
            f"experiment.experiment_strategy={strategy}",
        ]

        success, job_name = run_command(command, dry_run=args.dry_run)

        # Record job in manifest
        job_record = {
            "index": i,
            "model": model,
            "experiment_strategy": strategy,
            "status": "submitted" if success else "failed",
            "job_name": job_name,
            "submission_time": datetime.now().isoformat(),
        }
        job_manifest["jobs"].append(job_record)

        # Save manifest incrementally (in case of interruption)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(job_manifest, f, indent=2)

        if success:
            success_count += 1
        else:
            failed_jobs.append((model, strategy))
            if args.stop_on_error:
                print("\nStopping due to error (--stop-on-error enabled)")
                break

    # Summary
    print(f"\n{'=' * 80}")
    print("Batch submission complete")
    print(f"{'=' * 80}")
    print(f"✓ Successful: {success_count}/{len(jobs)}")

    if failed_jobs:
        print(f"✗ Failed: {len(failed_jobs)}")
        print("\nFailed jobs:")
        for model, strategy in failed_jobs:
            print(f"  - model={model} experiment.experiment_strategy={strategy}")

    # Final manifest update
    job_manifest["summary"] = {
        "successful": success_count,
        "failed": len(failed_jobs),
        "completion_time": datetime.now().isoformat(),
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(job_manifest, f, indent=2)

    print(f"\n📄 Job manifest saved to: {manifest_file.relative_to(base_dir)}")
    print("   Use job names from manifest to download artifacts later")

    if failed_jobs:
        sys.exit(1)
    else:
        print("\n✅ All jobs submitted successfully!")


if __name__ == "__main__":
    main()
