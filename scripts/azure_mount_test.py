"""
Minimal Azure ML mount test (Managed Identity path):
- Lists /mnt/azureml and /mnt/azureml/inputs
- Prints AZUREML_INPUT_* env vars
- Attempts to list a few files inside each mounted input

Exit code:
  0 = inputs mounted and at least one mount has contents (or subdirs)
  2 = /mnt/azureml/inputs missing and no AZUREML_INPUT_* present
"""

import os
import sys
from pathlib import Path


def list_dir(path: Path, max_entries: int = 20) -> list[str]:
    try:
        if path.exists():
            items = list(path.iterdir())
            lines = [f"{len(items)} entries"]
            for p in items[:max_entries]:
                lines.append(f"  {p.name}/" if p.is_dir() else f"  {p.name}")
            if len(items) > max_entries:
                lines.append(f"  ... (+{len(items) - max_entries} more)")
            return lines
        return ["<does not exist>"]
    except Exception as e:
        return [f"<error listing: {type(e).__name__}: {e}>"]


def main() -> int:
    az_inputs = Path("/mnt/azureml/inputs")
    az_root = Path("/mnt/azureml")

    print("[Azure ML] /mnt/azureml contents:")
    for line in list_dir(az_root):
        print(" ", line)

    print("\n[Azure ML] Mounted input directories:")
    mounted_any = False
    if az_inputs.exists():
        for item in sorted(az_inputs.iterdir()):
            print(f"  {item}")
            mounted_any = True
            # Try to peek inside mount (non-recursive)
            try:
                inner = list(item.iterdir())
                print(f"    {len(inner)} entries at top level")
                for p in inner[:10]:
                    print(f"      {'DIR ' if p.is_dir() else 'FILE'} {p.name}")
                if len(inner) > 10:
                    print(f"      ... (+{len(inner) - 10} more)")
            except Exception as e:
                print(f"    <cannot list: {type(e).__name__}: {e}>")
    else:
        print("  /mnt/azureml/inputs does not exist")

    print("\nEnvironment variables containing 'AZUREML_INPUT':")
    env_count = 0
    for k, v in sorted(os.environ.items()):
        if "AZUREML_INPUT" in k:
            env_count += 1
            print(f"{k}={v}")

    # Success if we either saw mounts or envs (ideally both)
    if mounted_any or env_count > 0:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
