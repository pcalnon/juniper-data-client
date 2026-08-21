"""Custom exceptions for the JuniperData client library."""

from __future__ import annotations

from typing import Any


class JuniperDataClientError(Exception):
    """Base exception for all JuniperData client errors.

    Carries the machine-readable context a caller needs to *act* on the error
    rather than re-parse its message (defect-register ``APD-DCLIENT-001``).
    Without ``status_code`` a 400 and a 422 raise the same type with the same
    text, so the only way to tell "you sent bad input" from "the service could
    not process it" was substring-matching the message.

    Every attribute is optional and keyword-only: locally raised errors
    (configuration, connection, timeout) have no HTTP response behind them, and
    existing call sites that pass only a message keep working unchanged.

    Attributes:
        message: The human-readable summary, also passed to ``Exception``.
        status_code: HTTP status of the originating response, when there was
            one. ``None`` for errors raised before or without a response.
        detail: The server's ``detail`` payload **exactly as decoded** -- a
            ``str`` for most handlers, and a ``list[dict]`` for FastAPI's 422
            validation errors. Deliberately not stringified: the structure is
            the point, and rendering it into the message is lossy
            (``APD-DCLIENT-003``).
        response: The originating ``requests.Response``, when available, for
            callers that need headers or the raw body.
    """

    def __init__(  # noqa: B042 — see __reduce__ below
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        detail: Any = None,
        response: Any = None,
    ) -> None:
        # B042 asks that an exception's ``__init__`` forward every argument to
        # ``super().__init__()`` and take no kwargs, so that pickle and copy
        # round-trip. The concern is real and is honoured by ``__reduce__``
        # below (pinned by ``test_context_survives_pickle_and_copy``); the
        # remedy it suggests is not available here, because "take no kwargs" is
        # precisely the defect this class is closing. Forwarding the extra
        # arguments to ``super()`` instead would put them in ``args`` and make
        # ``str(exc)`` a tuple repr, breaking every existing error message.
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.response = response

    def __reduce__(self) -> tuple[type[JuniperDataClientError], tuple[str], dict[str, Any]]:
        """Keep the added context across ``pickle`` and ``copy``.

        ``BaseException.__reduce__`` returns ``(cls, self.args)``, and ``args``
        holds only the message -- so a round-trip through pickle would rebuild
        the exception with ``status_code=None`` and ``detail=None``. It would
        still *look* correct, having silently dropped exactly the fields this
        class exists to carry. That is what flake8-bugbear's B042 warns about;
        restoring from ``__dict__`` is the fix, since the alternative it
        suggests (no kwargs at all) is the defect being closed here.

        Exceptions do cross process boundaries in this ecosystem -- a worker
        returning a failure to its parent is the ordinary case.
        """
        return (self.__class__, (self.message,), self.__dict__.copy())


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
