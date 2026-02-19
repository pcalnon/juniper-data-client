"""JuniperData Client - Python client library for the JuniperData REST API.

This package provides a simple, robust client for interacting with the JuniperData
dataset generation service, used by both JuniperCascor and JuniperCanopy.
"""

from juniper_data_client.client import JuniperDataClient
from juniper_data_client.exceptions import (
    JuniperDataClientError,
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
)

__version__ = "0.3.0"

__all__ = [
    "JuniperDataClient",
    "JuniperDataClientError",
    "JuniperDataConfigurationError",
    "JuniperDataConnectionError",
    "JuniperDataNotFoundError",
    "JuniperDataTimeoutError",
    "JuniperDataValidationError",
    "__version__",
]
