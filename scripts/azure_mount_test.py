"""
Azure ML mount test (Managed Identity):
- Polls /mnt/azureml/inputs briefly to wait for mounts
- Prints AZUREML_INPUT_* env vars and lists top-level entries
Exit:
  0 = saw mounts or AZUREML_INPUT_* envs
  2 = nothing mounted and no envs
"""

import os
import sys
import time
from pathlib import Path


def main() -> int:
    az_root = Path("/mnt/azureml")
    az_inputs = az_root / "inputs"

    print("[Azure ML] /mnt/azureml contents (initial):")
    try:
        if az_root.exists():
            for p in az_root.iterdir():
                print("  ", p.name + ("/" if p.is_dir() else ""))
        else:
            print("  <does not exist>")
    except Exception as e:
        print("  <error listing /mnt/azureml:", e, ">")

    # Poll up to ~30s for /mnt/azureml/inputs to show up
    for _i in range(30):
        if az_inputs.exists():
            break
        time.sleep(1)

    print("\n[Azure ML] Mounted input directories (after wait):")
    mounted_any = False
    if az_inputs.exists():
        for item in sorted(az_inputs.iterdir()):
            print(f"  {item}")
            mounted_any = True
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

    return 0 if (mounted_any or env_count > 0) else 2


if __name__ == "__main__":
    sys.exit(main())
