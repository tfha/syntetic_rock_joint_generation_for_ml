"""Update batch_jobs_summary.csv with job names from manifest."""

import argparse
import json
from pathlib import Path

import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Update batch_jobs_summary.csv with job names from manifest"
)
parser.add_argument(
    "manifest",
    nargs="?",
    default="experiments/batch_submissions/job_manifest_20251209_152124.json",
    help="Path to job manifest JSON file",
)
parser.add_argument(
    "--csv",
    default="experiments/batch_jobs_summary.csv",
    help="Path to CSV file to update",
)
args = parser.parse_args()

# Paths
manifest_path = Path(args.manifest)
csv_path = Path(args.csv)

# Read manifest
with open(manifest_path) as f:
    manifest = json.load(f)

# Create lookup dict: (model, experiment_strategy) -> job_name
job_lookup = {}
for job in manifest["jobs"]:
    if job["status"] == "submitted" and job["job_name"] is not None:
        key = (job["model"], job["experiment_strategy"])
        job_lookup[key] = job["job_name"]

# Read CSV
df = pd.read_csv(csv_path)

# Add job_name column (empty string for failed submissions)
df["job_name"] = df.apply(
    lambda row: job_lookup.get((row["model"], row["experiment_strategy"]), ""),
    axis=1,
)

# Save updated CSV
df.to_csv(csv_path, index=False)

print(f"Updated {csv_path}")
print(f"Total jobs in CSV: {len(df)}")
print(f"Successfully submitted: {df['job_name'].ne('').sum()}")
print(f"Failed to submit (empty job_name): {df['job_name'].eq('').sum()}")
