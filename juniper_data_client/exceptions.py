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

    def __init__(  # noqa: B042 — kwargs survive pickle via the default __reduce__; see below
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        detail: Any = None,
        response: Any = None,
    ) -> None:
        # B042 asks that an exception's ``__init__`` forward every argument to
        # ``super().__init__()`` and take no kwargs, so that pickle and copy
        # round-trip. The concern is real but already answered by CPython:
        # ``BaseException.__reduce__`` returns ``(cls, args, self.__dict__)``
        # whenever the instance dict is non-empty, so the keyword-only context
        # is restored automatically -- as long as ``cls(*args)`` stays
        # constructible, which is why the ``super()`` call below forwards the
        # message and nothing else. B042's own remedy is not available here:
        # "take no kwargs" is precisely the defect this class closed
        # (``APD-DCLIENT-001``), and forwarding the extras to ``super()``
        # would put them in ``args``, making ``str(exc)`` a tuple repr and
        # the pickle rebuild a ``TypeError``
        # (``test_context_survives_pickle_and_copy`` pins the latter).
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.response = response


class JuniperDataConnectionError(JuniperDataClientError):
    """Raised when connection to JuniperData service fails."""


class JuniperDataTimeoutError(JuniperDataClientError):
    """Raised when a request to JuniperData times out."""


class JuniperDataNotFoundError(JuniperDataClientError):
    """Raised when a requested resource is not found (404)."""


class JuniperDataValidationError(JuniperDataClientError):
    """Raised when request parameters fail validation (400/422)."""


class JuniperDataConfigurationError(JuniperDataClientError):
    """Raised when JuniperData configuration is missing or invalid."""


class JuniperDataContractError(JuniperDataClientError, ValueError):
    """Raised when a dataset artifact violates the NPZ data contract.

    ``validate_npz_contract`` raised bare ``ValueError``, so the one error this
    package detects *itself* was also the one escaping its own hierarchy --
    ``except JuniperDataClientError`` did not mean "anything this client
    raises" (defect-register ``APD-DCLIENT-002``).

    Deliberately also a ``ValueError``: the validator has documented ``Raises:
    ValueError`` since it shipped (0.4.2), and consumers pin that contract --
    juniper-recurrence's routers catch ``(JuniperDataClientError, ValueError)``
    and its data adapter re-documents the ``ValueError``. Dual inheritance
    joins the hierarchy without changing what any existing ``except`` clause
    catches.

    Contract violations are detected locally, after the artifact is already
    downloaded -- there is no HTTP failure behind them -- so ``status_code`` /
    ``detail`` / ``response`` stay ``None``, the base-class convention for
    locally raised errors.
    """
