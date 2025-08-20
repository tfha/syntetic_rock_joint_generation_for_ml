"""
Core Azure utilities and decorators.

This module contains fundamental Azure utilities that are used across
multiple Azure-related modules, including retry logic, error handling,
and logging configuration.
"""

import logging
import time
from typing import Callable, TypeVar

from azure.core.exceptions import AzureError, ServiceRequestError

# Type variable for generic retry function
T = TypeVar("T")


def configure_azure_logging():
    """
    Configure logging to reduce verbose Azure client output.

    This function sets the logging level for various Azure client libraries
    to WARNING, reducing noise in the console output. It should be called
    at the beginning of any script that interacts with Azure services.
    """
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)


def retry_azure_operation(
    operation: Callable[..., T],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_catch: tuple = (AzureError, ServiceRequestError),
    operation_name: str = "Azure operation",
) -> Callable[..., T]:
    """
    Decorator for retrying Azure operations with exponential backoff.

    Azure operations can fail due to transient issues such as network connectivity,
    service throttling, or temporary service unavailability. This retry mechanism
    helps improve resilience and reliability when working with cloud services.
    Using this functionality replaces the need for manual retry logic or error
    handling in your Azure operations, allowing you to focus on the core logic of
    your application.

    Exponential backoff is a standard retry strategy where the wait time between
    retry attempts increases exponentially. This approach prevents overwhelming
    the service with rapid-fire retries and gives the service time to recover
    from any issues. For example:
        - First retry: Wait 1 second
        - Second retry: Wait 2 seconds (1 * 2.0)
        - Third retry: Wait 4 seconds (2 * 2.0)
        - And so on...

    This pattern is essential for cloud applications as it:
        1. Reduces load on potentially stressed services
        2. Increases probability of eventual success
        3. Follows Azure service rate-limiting best practices
        4. Handles intermittent network issues gracefully

    Args:
        operation: The function to retry
        max_retries: Maximum number of retry attempts before giving up
        initial_backoff: Initial backoff time in seconds before first retry
        backoff_factor: Multiplicative factor to increase backoff time with each retry
        exceptions_to_catch: Tuple of exception classes that should trigger a retry
        operation_name: Name of the operation for logging purposes

    Returns:
        A wrapped function that implements retry logic

    Example:
        >>> @retry_azure_operation(max_retries=5, operation_name="Get data asset")
        >>> def get_data_asset_wrapper(client, name):
        >>>     return client.data.get_asset(name)
    """

    def wrapper(*args, **kwargs):
        last_exception = None
        backoff_time = initial_backoff

        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except exceptions_to_catch as e:
                last_exception = e
                if attempt < max_retries:
                    print(
                        f"{operation_name} failed (attempt {attempt + 1}/"
                        f"{max_retries + 1}): {str(e)}"
                    )
                    print(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= backoff_factor
                else:
                    print(f"{operation_name} failed after {max_retries + 1} attempts")
                    raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"{operation_name} failed for unknown reasons")

    return wrapper
