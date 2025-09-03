import os
import subprocess
import sys
from datetime import datetime


def run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = p.communicate(timeout=20)
        return p.returncode, out.strip(), err.strip()
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def main() -> None:
    # Optional: force CPU by clearing CUDA_VISIBLE_DEVICES if requested
    use_cuda = os.environ.get("SMOKE_USE_CUDA", "0") == "1"
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    lines: list[str] = []
    lines.append(f"[azure_smoke_min] started {datetime.utcnow().isoformat()}Z")
    lines.append(f"python: {sys.version.split()[0]}")

    # nvidia-smi
    code, out, err = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    lines.append(f"nvidia-smi exit={code}")
    if out:
        lines.append("nvidia-smi output:")
        lines.extend(["  " + ln for ln in out.splitlines()])
    if err:
        lines.append("nvidia-smi stderr:")
        lines.extend(["  " + ln for ln in err.splitlines()])

    # Import torch and print basic info
    try:
        import torch  # noqa: F401

        lines.append("torch import: OK")
        try:
            cuda_is_avail = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_is_avail else 0
            lines.append(f"torch.cuda.is_available: {cuda_is_avail}")
            lines.append(f"cuda device count: {device_count}")
            if cuda_is_avail and use_cuda and device_count > 0:
                # Tiny CUDA op to validate runtime
                x = torch.ones(1, device="cuda:0")
                y = x * 2
                lines.append(f"tiny cuda op: OK, value={y.item()}")
            else:
                lines.append("tiny cuda op: SKIPPED")
        except Exception as e:
            lines.append(f"torch cuda check error: {type(e).__name__}: {e}")
    except Exception as e:
        lines.append(f"torch import error: {type(e).__name__}: {e}")

    # Write to outputs so AML captures it
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/smoke_min_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Also print to stdout
    print("\n".join(lines))


if __name__ == "__main__":
    main()
