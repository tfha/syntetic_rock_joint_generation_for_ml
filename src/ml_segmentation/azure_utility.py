"""
Azure utility functions - DEPRECATED

⚠️  WARNING: This module has been refactored into specialized modules
for better organization.

MIGRATION GUIDE:
The functions in this module have been moved to more appropriate locations:

- Core utilities (retry, logging) → azure_core.py
- Authentication & workspace → azure_authentication.py
- Compute management → azure_compute.py
- Job management → azure_jobs.py
- Data asset operations → azure_data_assets.py
- Environment management → azure_environment.py
- Diagnostics → azure_diagnostics.py

NEW IMPORTS:
# Old way
from ml_segmentation.azure_utility import connect_to_azure_ml

# New way - specific module
from ml_segmentation.azure_authentication import connect_to_azure_ml

# Or use package-level imports
from ml_segmentation import connect_to_azure_ml

For a complete list of moved functions and their new locations, see:
docs/refactoring/AZURE_MODULE_RESTRUCTURING.md

This file will be removed in a future version. Please update your imports.
"""

# Backwards compatibility imports - will be removed in future version
import warnings

# Issue deprecation warning
warnings.warn(
    "azure_utility.py is deprecated. Please use the new modular imports. "
    "See docs/refactoring/AZURE_MODULE_RESTRUCTURING.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Backwards compatibility re-exports
# These imports maintain compatibility for existing code
try:
    from .azure_authentication import (
        connect_to_azure_ml,
        setup_azure_environment_variables,
        validate_workspace_permissions,
    )
    from .azure_compute import (
        check_compute_permissions,
        refresh_compute_cluster,
        validate_and_refresh_compute,
    )
    from .azure_core import configure_azure_logging, retry_azure_operation
    from .azure_data_assets import (
        retrieve_and_validate_data_assets,
    )
    from .azure_diagnostics import run_azure_diagnostic_tests
    from .azure_environment import (
        export_poetry_to_environment_yml,
        test_environment_export,
    )
    from .azure_jobs import (
        download_job_outputs,
        get_job_details,
        handle_job_submission_error,
        handle_log_streaming_error,
    )

    # Legacy function names for backwards compatibility
    _test_permission_check = run_azure_diagnostic_tests
    _test_data_asset_retrieval = run_azure_diagnostic_tests
    _test_compute_validation = run_azure_diagnostic_tests
    _test_job_submission_and_logging = run_azure_diagnostic_tests

    # Private helper functions that were moved
    _provide_compute_guidance = None  # Moved to azure_compute
    _provide_identity_setup_guidance = None  # Moved to azure_compute
    _provide_streaming_guidance = None  # Moved to azure_jobs
    _provide_fallback_guidance = None  # Moved to azure_jobs

except ImportError as e:
    # Fallback error message if new modules aren't available
    raise ImportError(
        f"Could not import from new Azure modules: {e}\n"
        "Please ensure all new Azure modules are present in the package."
    )

# Module metadata
__deprecated__ = True
__version__ = "1.0.0-deprecated"
__all__ = [
    # Re-exported from new modules for backwards compatibility
    "configure_azure_logging",
    "retry_azure_operation",
    "connect_to_azure_ml",
    "setup_azure_environment_variables",
    "validate_workspace_permissions",
    "check_compute_permissions",
    "refresh_compute_cluster",
    "validate_and_refresh_compute",
    "download_job_outputs",
    "get_job_details",
    "handle_job_submission_error",
    "handle_log_streaming_error",
    "retrieve_and_validate_data_assets",
    "export_poetry_to_environment_yml",
    "test_environment_export",
    "run_azure_diagnostic_tests",
    "show_migration_guide",
]


def show_migration_guide():
    """
    Display a helpful migration guide for updating imports.
    """
    print(
        """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    AZURE UTILITY MIGRATION GUIDE                ║
    ╚══════════════════════════════════════════════════════════════════╝

    This module has been refactored for better organization.

    FUNCTION MIGRATIONS:

    Authentication & Connection:
    ├─ connect_to_azure_ml → azure_authentication.py
    ├─ setup_azure_environment_variables → azure_authentication.py
    └─ validate_workspace_permissions → azure_authentication.py

    Compute Management:
    ├─ check_compute_permissions → azure_compute.py
    ├─ refresh_compute_cluster → azure_compute.py
    └─ validate_and_refresh_compute → azure_compute.py

    Job Management:
    ├─ download_job_outputs → azure_jobs.py
    ├─ get_job_details → azure_jobs.py
    ├─ handle_job_submission_error → azure_jobs.py (for error handling)
    └─ handle_log_streaming_error → azure_jobs.py (for error handling)

    Data Assets:
    └─ retrieve_and_validate_data_assets → azure_data_assets.py

    Environment:
    ├─ export_poetry_to_environment_yml → azure_environment.py
    └─ test_environment_export → azure_environment.py

    Core Utilities:
    ├─ configure_azure_logging → azure_core.py
    └─ retry_azure_operation → azure_core.py

    EXAMPLE MIGRATION:

    # OLD - Will show deprecation warning
    from ml_segmentation.azure_utility import connect_to_azure_ml

    # NEW - Recommended approach
    from ml_segmentation.azure_authentication import connect_to_azure_ml

    # OR use package-level imports
    from ml_segmentation import connect_to_azure_ml

    For full details see: docs/refactoring/AZURE_MODULE_RESTRUCTURING.md
    """
    )


if __name__ == "__main__":
    # Show migration guide when run directly
    show_migration_guide()
