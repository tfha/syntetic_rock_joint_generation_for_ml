"""
Azure ML Segmentation Package

This package provides a well-organized set of Azure ML utilities for machine learning
segmentation projects. The modules are organized by functionality for better
maintainability and separation of concerns.

Module Organization:
- azure_core: Core utilities and decorators
- azure_authentication: Workspace connection and authentication
- azure_compute: Compute cluster management
- azure_jobs: Job creation, submission, and monitoring
- azure_data_assets: Data asset management and operations
- azure_data_loading: PyTorch data loading for Azure ML
- azure_environment: Environment and dependency management
- azure_diagnostics: Testing and diagnostic utilities
"""

# Core functionality - commonly used across modules
# Authentication and workspace connection
from .azure_authentication import (
    connect_to_azure_ml,
    setup_azure_environment_variables,
    validate_workspace_permissions,
)

# Compute cluster management
from .azure_compute import (
    check_compute_permissions,
    refresh_compute_cluster,
    validate_and_refresh_compute,
)
from .azure_core import configure_azure_logging, retry_azure_operation

# Data asset management
from .azure_data_assets import (
    get_data_asset,
    list_data_assets,
    register_base_datasets,
    register_data_asset,
    retrieve_and_validate_data_assets,
    upload_base_data_to_azure_blob,
)

# Data loading for training
from .azure_data_loading import (
    create_azure_datasets,
    setup_azure_dataloader,
)

# Diagnostics and testing
from .azure_diagnostics import run_azure_diagnostic_tests

# Environment management
from .azure_environment import (
    build_and_register_environment,
    export_poetry_to_environment_yml,
    test_environment_export,
)

# Job management
from .azure_jobs import (
    download_job_outputs,
    get_job_details,
    handle_job_submission_error,
    handle_log_streaming_error,
)

__version__ = "2.0.0"
__author__ = "ML Segmentation Team"

# Define what gets exported when using "from ml_segmentation import *"
__all__ = [
    # Core utilities
    "configure_azure_logging",
    "retry_azure_operation",
    # Authentication
    "connect_to_azure_ml",
    "setup_azure_environment_variables",
    "validate_workspace_permissions",
    # Compute management
    "check_compute_permissions",
    "refresh_compute_cluster",
    "validate_and_refresh_compute",
    # Job management
    "download_job_outputs",
    "get_job_details",
    "handle_job_submission_error",
    "handle_log_streaming_error",
    # Data assets
    "get_data_asset",
    "register_data_asset",
    "retrieve_and_validate_data_assets",
    "list_data_assets",
    "upload_base_data_to_azure_blob",
    "register_base_datasets",
    # Data loading
    "create_azure_datasets",
    "setup_azure_dataloader",
    # Environment management
    "export_poetry_to_environment_yml",
    "test_environment_export",
    "build_and_register_environment",
    # Diagnostics
    "run_azure_diagnostic_tests",
]

__version__ = "2.0.0"
__author__ = "ML Segmentation Team"
