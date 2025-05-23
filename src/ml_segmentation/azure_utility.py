"""
Azure utility functions for environment setup and common operations.

This module provides reusable utilities for Azure environments,
including environment variable handling, credential management,
and common Azure configuration functionality.
"""

import os
import sys

from dotenv import load_dotenv

from ml_segmentation.utility import get_custom_console


def setup_azure_environment(console=None):
    """Set up the Azure environment and return a console for pretty printing.

    This function loads environment variables from .env file,
    validates Azure credentials, and returns a console object.

    Args:
        console: Optional console object for pretty printing. If None, a new
        console is created.

    Returns:
        tuple: (console, subscription_id, resource_group, workspace_name)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Create a console for pretty printing if not provided
    if console is None:
        console = get_custom_console()

    # Check if Azure environment variables are set
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id:
        console.print(
            "Error: AZURE_SUBSCRIPTION_ID environment variable not set", style="error"
        )
        console.print(
            "Please set it in your .env file:"
            "AZURE_SUBSCRIPTION_ID='your-subscription-id'",
            style="error",
        )
        sys.exit(1)

    if not resource_group:
        console.print(
            "Warning: AZURE_RESOURCE_GROUP environment variable not set",
            style="warning",
        )

    if not workspace_name:
        console.print(
            "Warning: AZURE_ML_WORKSPACE environment variable not set", style="warning"
        )

    return console, subscription_id, resource_group, workspace_name
