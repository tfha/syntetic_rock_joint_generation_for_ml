"""Merge job names from multiple manifests into batch_jobs_summary.csv."""

import json
from pathlib import Path

import pandas as pd

manifests = [
    "experiments/batch_submissions/job_manifest_20251211_160228.json",
    "experiments/batch_submissions/job_manifest_20251211_170124.json",
]

job_lookup = {}
for manifest_path in manifests:
    with open(manifest_path) as f:
        manifest = json.load(f)
    for job in manifest["jobs"]:
        if job["status"] == "submitted" and job["job_name"] is not None:
            key = (job["model"], job["experiment_strategy"])
            job_lookup[key] = job["job_name"]

csv_path = Path("experiments/batch_jobs_summary.csv")
df = pd.read_csv(csv_path)

# Only update rows where we have a successfully submitted job
# Preserve existing job names for jobs not in the manifests
for idx, row in df.iterrows():
    key = (row["model"], row["experiment_strategy"])
    if key in job_lookup:
        df.at[idx, "job_name"] = job_lookup[key]

df.to_csv(csv_path, index=False)

print(f"Updated {csv_path}")
print(f"Total jobs in CSV: {len(df)}")
submitted_count = df["job_name"].ne("").sum()
failed_count = df["job_name"].eq("").sum()
print(f"Successfully submitted: {submitted_count}")
print(f"Failed to submit (empty job_name): {failed_count}")
