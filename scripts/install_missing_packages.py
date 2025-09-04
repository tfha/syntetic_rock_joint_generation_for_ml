"""
Install missing Python packages for curated Azure ML environments.
Run this script as a job step before your main entry point.
"""

import subprocess

# List only packages NOT provided by the curated environment
REQUIRED_PACKAGES = [
    "hydra-core",
    "omegaconf",
    "rich",
    "torchmetrics",
    "segmentation-models-pytorch",
    "timm",
    # Add any other non-ML packages your code imports
]


def main() -> None:
    subprocess.run(
        ["pip", "install", *REQUIRED_PACKAGES],
        check=True,
    )


if __name__ == "__main__":
    main()
