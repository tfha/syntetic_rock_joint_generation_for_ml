"""
Azure ML smoke test: validate Python, Torch, CUDA, dataset mounts, and a tiny
end-to-end model + dataloader forward pass.
This script runs quickly and exits with non-zero on failure.
Artifacts and logs are written to ./outputs for AML to capture.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.azure_core import configure_azure_logging
from ml_segmentation.azure_data_loading import setup_azure_dataloader
from ml_segmentation.debug_functionality import better_traceback
from ml_segmentation.define_model import choose_model
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.utility import get_custom_console


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate(timeout=30)
    return proc.returncode, out.strip(), err.strip()


def resolve_aml_input(name: str) -> Path:
    """Resolve Azure ML command input path by env var or default mount.

    Prefers AZUREML_INPUT_<name>. If missing, falls back to /mnt/azureml/inputs/<name>.
    """
    env_key = f"AZUREML_INPUT_{name}"
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)
    fallback = Path("/mnt/azureml/inputs") / name
    return fallback


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_azure_logging()
    console = get_custom_console()

    # Prepare outputs dir
    outputs = Path("./outputs")
    outputs.mkdir(exist_ok=True)
    report = outputs / "smoke_test_report.txt"

    lines: list[str] = []
    ts = datetime.utcnow().isoformat()
    lines.append(f"Smoke test timestamp (UTC): {ts}")

    # System info
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"Platform: {platform.platform()}")

    # Torch/CUDA info
    lines.append(f"torch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    lines.append(f"cuda_available: {cuda_available}")
    if cuda_available:
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
        lines.append(f"current_device: {torch.cuda.current_device()}")
        lines.append(f"device_name: {torch.cuda.get_device_name(0)}")
        # allocate a tiny tensor and do a small op
        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")
        z = (x @ y).sum().item()
        lines.append(f"tiny_cuda_op_result: {z}")

    # nvidia-smi if present
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            code, out, err = run_cmd(
                [
                    smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            )
            lines.append(f"nvidia_smi_code: {code}")
            lines.append("nvidia_smi_out:\n" + out)
            if err:
                lines.append("nvidia_smi_err:\n" + err)
        except Exception as e:  # pragma: no cover - best-effort
            lines.append(f"nvidia_smi_exception: {e}")
    else:
        lines.append("nvidia_smi: not found in PATH")

    # Azure ML input mounts (env var first, then default mount fallback)
    images_path = resolve_aml_input("images_data")
    masks_path = resolve_aml_input("masks_data")
    splits_path = resolve_aml_input("splits_data")

    lines.append(f"images_path: {images_path} exists={images_path.exists()}")
    lines.append(f"masks_path: {masks_path} exists={masks_path.exists()}")
    lines.append(f"splits_path: {splits_path} exists={splits_path.exists()}")

    # Try list a few files from inputs
    def list_head(p: Path) -> list[str]:
        if not p.exists():
            return []
        results: list[str] = []
        for root, _dirs, files in os.walk(p):
            for f in files[:5]:
                results.append(str(Path(root) / f))
            if results:
                break
        return results

    lines.append("images_head:\n" + "\n".join(list_head(images_path)))
    lines.append("masks_head:\n" + "\n".join(list_head(masks_path)))

    # Parse config (lightweight)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Expected Hydra cfg to be convertible to dict")
    pcfg = ConfigSchema(**cfg_dict)  # type: ignore[arg-type]

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines.append(f"device: {device}")

    # Try creating dataloaders with batch_size=1
    try:
        tr_loader, va_loader, te_loader = setup_azure_dataloader(
            console=console,
            images_path=images_path,
            labels_path=masks_path,
            batch_size=1,
            num_workers=min(1, pcfg.experiment.num_workers),
            optional_transforms=pcfg.experiment.optional_transforms,
            splits_path=splits_path if splits_path.exists() else None,
            device=device,
            pin_memory=None,
            persistent_workers=None,
        )
        lines.append("dataloaders: created OK (batch_size=1)")
    except Exception as e:  # pragma: no cover - diagnostics path
        lines.append(f"dataloaders_exception: {e}")
        report.write_text("\n".join(lines), encoding="utf-8")
        console.print("Smoke test failed while building dataloaders", style="danger")
        raise

    # Try building model and single forward pass
    try:
        model = choose_model(pcfg.model.name, pcfg.model.params).to(device)
        model.eval()
        lines.append(f"model_built: {pcfg.model.name}")

        # One batch from train or val
        loader = tr_loader if len(tr_loader) > 0 else va_loader
        batch = next(iter(loader))
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            images = batch[0]
        else:
            images = batch
        images = images.to(device)

        amp_ctx = (
            torch.amp.autocast(device_type=device.type)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with amp_ctx:
                outputs = model(images)
        lines.append(
            f"forward_ok: input_shape={tuple(images.shape)} output_shape={tuple(outputs.shape)}"
        )
        # Clean up tiny tensors
        del images, outputs, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        lines.append("forward_cleanup_done: True")
    except Exception as e:  # pragma: no cover - diagnostics path
        lines.append(f"forward_exception: {e}")
        report.write_text("\n".join(lines), encoding="utf-8")
        console.print("Smoke test failed during forward pass", style="danger")
        raise

    # Write report
    report.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"Smoke test completed. Report saved to {report}", style="success")


if __name__ == "__main__":
    better_traceback()
    main()
