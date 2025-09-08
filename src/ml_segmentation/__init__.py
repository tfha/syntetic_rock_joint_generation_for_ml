"""Minimal package marker for internal use.

Rationale (2025-09 internal phase):
- Keep import side effects effectively zero (no heavy deps touched at import time).
- Avoid accidental CUDA / native library initialisation when building images.
- Force explicit, stable imports (e.g. ``from ml_segmentation.lightning_module import SegmentationLightningModule``)
    so future public API design remains flexible.

When preparing for external distribution, see ``docs/EXTENDING_INIT.md`` for the
lazy re-export pattern and guidance on curating a public surface.
"""

__version__ = "2.0.0"
__author__ = "ML Segmentation Team"

# Explicitly declare an empty public surface for now.
__all__: list[str] = ["__version__", "__author__"]
