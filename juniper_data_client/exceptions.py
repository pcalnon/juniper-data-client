"""Custom exceptions for the JuniperData client library."""


class JuniperDataClientError(Exception):
    """Base exception for all JuniperData client errors."""

    pass


class JuniperDataConnectionError(JuniperDataClientError):
    """Raised when connection to JuniperData service fails."""

    pass


class JuniperDataTimeoutError(JuniperDataClientError):
    """Raised when a request to JuniperData times out."""

    pass


class JuniperDataNotFoundError(JuniperDataClientError):
    """Raised when a requested resource is not found (404)."""

    pass


class JuniperDataValidationError(JuniperDataClientError):
    """Raised when request parameters fail validation (400/422)."""

    pass


class JuniperDataConfigurationError(JuniperDataClientError):
    """Raised when JuniperData configuration is missing or invalid."""

    pass
