# Extending `ml_segmentation/__init__.py` for External Usage

_Current status (internal phase):_ `ml_segmentation/__init__.py` is intentionally minimal to
avoid any heavy imports or side effects at package import time. All symbols must be
imported explicitly from their submodules, e.g.:

```python
from ml_segmentation.lightning_module import SegmentationLightningModule
from ml_segmentation.azure_authentication import connect_to_azure_ml
```

This keeps container builds, lightweight tooling, and test discovery fast and reduces
risk of low-level crashes (e.g. native/CUDA initialisation or Azure SDK credential
probing) before they're needed.

---
## When to Introduce a Public Aggregated API
Add a richer top-level API only when BOTH are true:
1. You intend to publish the library externally (PyPI / internal index).
2. You can define a stable, documented surface that you commit to supporting.

If either is uncertain, keep `__init__` minimal.

---
## Design Goals for a Future Top-Level API
- **Zero (or near-zero) import cost:** Importing `ml_segmentation` should not download models,
  initialise CUDA, or open network connections.
- **Lazy loading:** Only import heavy dependencies (PyTorch, Lightning, Azure SDK) when the
  first symbol that needs them is accessed.
- **Clear boundary:** Everything re-exported is part of the supported public API. Internals
  remain reachable via explicit imports but are not guaranteed stable.
- **No side effects:** Avoid logging setup, environment mutation, thread spawning, credential
  acquisition, or GPU probing in `__init__`.

---
## Recommended Lazy Re-Export Pattern
Below is a concise template you can drop in later (adapt symbol map as needed):

```python
# ml_segmentation/__init__.py (future external form)
from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import Any

__version__ = "2.1.0"  # bump when public surface changes
__author__ = "ML Segmentation Team"

# Public API map: name -> (module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "SegmentationLightningModule": ("ml_segmentation.lightning_module", "SegmentationLightningModule"),
    "SegmentationDataModule": ("ml_segmentation.lightning_datamodule", "SegmentationDataModule"),
    "connect_to_azure_ml": ("ml_segmentation.azure_authentication", "connect_to_azure_ml"),
    # Add only *stable* symbols. Keep this list small at first.
}

__all__ = list(_LAZY.keys()) + ["__version__", "__author__"]

def _load(name: str) -> Any:
    mod_name, attr = _LAZY[name]
    module: ModuleType = import_module(mod_name)
    return getattr(module, attr)

def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _LAZY:
        value = _load(name)
        globals()[name] = value  # cache after first access
        return value
    raise AttributeError(name)

def __dir__() -> list[str]:  # pragma: no cover
    return sorted(__all__)
```

### Why This Works
- Keeps the first import cheap (almost no work executed).
- Defers potential failure points—if one symbol has a dependency issue, unrelated users
  still import the package successfully.
- Makes it trivial to trace which symbol triggered a heavy dependency (inspect stack).

---
## Choosing What to Expose
Start *narrow*:
- Core training entry points (Lightning module & datamodule)
- A small number of Azure utility functions most users need
- Version metadata

Add more only when:
- It has tests covering basic import and usage pathways
- You’re confident about long-term stability
- It doesn’t drag bulky transitive imports (or benefits clearly outweigh cost)

Avoid exposing:
- Experimental helpers
- Deep Azure environment registration utilities that may change
- Internal data transformation scaffolding not documented for end users

---
## Versioning & Stability Policy
Adopt semantic versioning for the public surface:
- Patch: bug fixes, no API changes
- Minor: additive, backwards compatible (new symbols allowed)
- Major: breaking changes (rename/remove/restructure)

Maintain a `CHANGELOG.md` entry whenever you add/remove a symbol in `_LAZY`.

---
## Preventing Regressions
Add a test similar to:
```python
def test_import_is_lightweight():
    import importlib, time, tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    import ml_segmentation  # noqa: F401
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    # Heuristic thresholds; adjust if needed
    assert dt < 0.5, f"Import too slow: {dt:.2f}s"
    assert peak < 25_000_000, f"Import too memory heavy: {peak/1e6:.1f}MB"
```
(Use relaxed thresholds—goal is detection of accidental blow‑ups, not micro-optimisation.)

---
## Migration Path (Internal → External)
1. Keep current minimal `__init__` until you decide to publish.
2. Introduce the lazy map in a PR labelled "public API bootstrap".
3. Document exposed symbols in `README.md` / `docs/API.md`.
4. Add import cost test + smoke tests for each exported symbol.
5. Publish pre-release (`0.x` or `1.0.0rc1`) for early feedback.

---
## Common Pitfalls to Avoid
| Pitfall | Avoid By |
|---------|----------|
| Adding `torch` import at top level | Use lazy accessor; import inside function/module scope only after call |
| Side-effectful logging/credential init | Leave setup to explicit functions (e.g. `connect_to_azure_ml`) |
| Huge `__all__` explosion | Start with 3–6 symbols; grow intentionally |
| Circular imports via re-export | Keep mapping data-only; don’t import modules eagerly |

---
## FAQ
**Q: Why not implicit namespace packages (omit `__init__.py`)?**
A: Simplicity + explicit control + conventional metadata exposure.

**Q: Can we partially eager import (e.g. Azure only)?**
A: Prefer uniform lazy pattern; mixed strategies become confusing.

**Q: What about runtime plugin discovery?**
A: Implement that in a dedicated module (e.g. `plugins.py`) rather than `__init__`.

---
## Summary Checklist for Externalisation
- [ ] Decide initial stable symbol list
- [ ] Add lazy map implementation
- [ ] Write API docs
- [ ] Add import performance test
- [ ] Add CHANGELOG entry
- [ ] Bump version

---
_This document will evolve once external packaging plans firm up._
