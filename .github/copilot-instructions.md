## GitHub Copilot coding agent: repository instructions

Note: These instructions are for the GitHub Copilot coding agent, Copilot Chat, and Copilot code review. They describe how to understand, build, test, and safely contribute to this repository.

### Repository overview

- Language/tooling: Python 3.11, Poetry, PyTest, Ruff, isort, mypy
- Package source: `src/ml_segmentation` (import as `ml_segmentation`)
- Entry scripts: `scripts/` (training, Azure ML submission, preprocessing)
- Data layout: `data/{raw,intermediate,model_ready}` (large artifacts are not tracked)
- Experiments/artifacts: `experiments/`, `models/`, `plots/`
- Azure integration: code under `src/ml_segmentation/azure_*` and scripts in `scripts/azure_*`

### What tasks Copilot should take on

- Bug fixes, small refactors, and code quality improvements in `src/ml_segmentation/**`
- Test improvements and added coverage in `tests/**`
- Documentation updates in `README.md` and `docs/**`
- Safe performance tweaks and type hints that don’t change public behavior

Avoid (unless explicitly requested):

- Broad cross-cutting refactors, redesigns, or changes to training pipelines
- Any action that requires live Azure access or modifies cloud resources
- Large data downloads, training runs, or long-running experiments
- Changes that alter default behavior without tests and justification

### Format, lint, and test

Prerequisites: Python 3.11 and Poetry. If a `copilot-setup-steps.yml` is present, Copilot should use it to preinstall dependencies and set up its cloud environment.

Install dependencies (include dev tools):

- `poetry install --with dev`

Format and lint (keep line length = 88, see `pyproject.toml`):

- Format: `poetry run ruff format .`
- Sort imports: `poetry run isort .`
- Lint (auto-fix where possible): `poetry run ruff check --fix .`
- Type check: `poetry run mypy`

Run tests:

- `poetry run pytest -q`
- Coverage (optional): `poetry run pytest --maxfail=1 --disable-warnings --cov=src --cov-report=term-missing`

CI expectations for PRs created by Copilot:

- Code is formatted, linted, and type-checked with no new warnings
- Tests pass locally; add/adjust tests when changing behavior

### Project conventions (must follow)

Configuration and Hydra pattern:

- Use Hydra ONLY in scripts under `scripts/` to load YAML configs from `scripts/config/**`
- Validate configuration with Pydantic models in `src/ml_segmentation/schema_config.py`
- Use the validated config object (commonly `pcfg`) inside library code; do not use raw Hydra config in `ml_segmentation`

Python style and libraries:

- Use Python 3.11+ features and type annotations everywhere
- Prefer PEP 604 unions for optionals (use `T | None` instead of `Optional[T]`); mypy supports this and it keeps annotations concise and consistent
- Prefer `pathlib.Path` over string paths
- Imports at top of file: Never place imports inside functions or deep within code blocks. All imports must be at module top-level to keep dependencies explicit, avoid repeated import overhead, and satisfy linters/type-checkers. The only narrow exceptions are:
  - Optional, heavy dependencies used solely in guarded or plugin-like code paths, with an inline comment explaining why the import is deferred.
  - Try/except ImportError guards for optional extras, again with an explanatory comment and a clear fallback.
- Use `numpy.typing.NDArray` for array types when relevant
- Prefer: `pathlib` over `os`, `httpx` over `requests`, `pytest` over `unittest`, Hydra v1.3+, Azure ML SDK v2
- Matplotlib: use the OO style (`ax.plot(...)`), not `plt.plot(...)`
- Strings via f-strings
- prefer `match/case` over long `if/elif` chains when appropriate

### Repository structure (orientation)

- `src/ml_segmentation/`: library code (keep Hydra out of here)
- `scripts/`: runnable entry points (Hydra lives here; read YAML config in `scripts/config/`)
- `tests/`: unit tests; mirror structure of `src` when practical
- `docs/`: documentation; keep it up to date with behavior changes
- `data/`: local data folders (don’t commit large files)
- `experiments/`, `models/`, `plots/`: outputs and artifacts (do not rely on these in unit tests)

### Testing guidelines

- Use `pytest` with clear Arrange–Act–Assert structure
- Prefer small, deterministic test fixtures; do not depend on cloud resources or large datasets
- Use `pytest.parametrize` and fixtures where beneficial
- Don’t retest third-party libraries; test our integration and logic
- When changing behavior, add tests that lock in the new contract

### Security, privacy, and data handling

- Never hardcode or print secrets; prefer environment variables and `.env` locally (do not commit)
- Do not interact with live Azure (auth, provisioning, resource mutation) unless explicitly asked and provided safe mock/test paths
- Do not fetch or commit large datasets; use small synthetic fixtures for tests

### Pull requests from Copilot

- Create a feature branch named `copilot/<short-topic>`
- Keep PRs focused and small; update or add tests and documentation
- Follow commit style:
  - Title: short imperative summary
  - Body: bullet list of key changes and rationale; include any follow-up items

### Helpful tips for Copilot

- If you add new config fields, update `schema_config.py` and relevant YAML under `scripts/config/`
- Prefer minimal, isolated changes; avoid altering training scripts or Azure submission flows unless requested
- If build tooling is unclear, consult `pyproject.toml` to align with Ruff, isort, and mypy settings

---

Maintainers: see `pyproject.toml` authors. If substantial design changes are needed, open an issue proposing the approach before large edits.
