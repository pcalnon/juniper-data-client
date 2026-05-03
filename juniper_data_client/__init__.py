"""JuniperData Client - Python client library for the JuniperData REST API.

This package provides a simple, robust client for interacting with the JuniperData
dataset generation service, used by both JuniperCascor and juniper-canopy.
"""

from juniper_data_client.client import JuniperDataClient, RequestHook
from juniper_data_client.exceptions import JuniperDataClientError, JuniperDataConfigurationError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError

__version__ = "0.4.1"

__all__ = [
    "JuniperDataClient",
    "JuniperDataClientError",
    "JuniperDataConfigurationError",
    "JuniperDataConnectionError",
    "JuniperDataNotFoundError",
    "JuniperDataTimeoutError",
    "JuniperDataValidationError",
    # METRICS-MON R4.3 / seed-13: instrumentation hook type alias
    # exported so consumers can type their hook closures.
    "RequestHook",
    "__version__",
]
