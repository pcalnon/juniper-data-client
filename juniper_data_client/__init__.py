"""JuniperData Client - Python client library for the JuniperData REST API.

This package provides a simple, robust client for interacting with the JuniperData
dataset generation service, used by both JuniperCascor and juniper-canopy.
"""

from juniper_data_client.client import JuniperDataClient, RequestHook
from juniper_data_client.constants import ContractKind
from juniper_data_client.contract import validate_npz_contract
from juniper_data_client.exceptions import JuniperDataClientError, JuniperDataConfigurationError, JuniperDataConnectionError, JuniperDataContractError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError

__version__ = "0.4.2"

__all__ = [
    "JuniperDataClient",
    "validate_npz_contract",
    "JuniperDataClientError",
    "JuniperDataConfigurationError",
    "JuniperDataConnectionError",
    "JuniperDataContractError",
    "JuniperDataNotFoundError",
    "JuniperDataTimeoutError",
    "JuniperDataValidationError",
    # METRICS-MON R4.3 / seed-13: instrumentation hook type alias
    # exported so consumers can type their hook closures.
    "RequestHook",
    # APD-DCLIENT-006: the validate_npz_contract return Literal, exported so
    # consumers can annotate and exhaustively match on the result.
    "ContractKind",
    "__version__",
]
