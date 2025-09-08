"""
Minimal Azure ML mount test: lists /mnt/azureml/inputs and prints Azure ML input env vars.
Run this as an Azure ML job with your usual inputs mapped.
"""

import os
from pathlib import Path


def main() -> None:
    inputs_dir = Path("/mnt/azureml/inputs")
    print("\n[Azure ML] Mounted input directories:")
    if inputs_dir.exists():
        for item in inputs_dir.iterdir():
            print(f"  {item} (exists={item.exists()})")
            if item.is_dir():
                files = list(item.glob("*"))
                print(f"    {len(files)} files/dirs inside")
    else:
        print("  /mnt/azureml/inputs does not exist")

    print("\nEnvironment variables containing 'AZUREML_INPUT':")
    for k, v in os.environ.items():
        if "AZUREML_INPUT" in k:
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
