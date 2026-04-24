"""Regression tests for the HTTP retry policy (XREPO-11).

After Phase 4B the client must NOT auto-retry non-idempotent HTTP
methods (POST / PATCH / DELETE) because retrying a request that has
already been applied server-side would cause duplicate dataset
creation or repeated deletions. Callers that need retry semantics for
mutations must layer their own idempotency on top.
"""

from __future__ import annotations

import pytest

from juniper_data_client import constants
from juniper_data_client.client import JuniperDataClient


class TestRetryAllowedMethods:
    """The retry allow-list must only contain RFC 9110 idempotent methods."""

    @pytest.mark.parametrize("method", ["HEAD", "GET", "PUT"])
    def test_idempotent_method_is_allowed(self, method: str) -> None:
        assert method in constants.RETRY_ALLOWED_METHODS

    @pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
    def test_non_idempotent_method_is_blocked(self, method: str) -> None:
        assert method not in constants.RETRY_ALLOWED_METHODS, f"HTTP {method} is not idempotent; auto-retry would duplicate " f"side-effects (got {constants.RETRY_ALLOWED_METHODS})"


class TestRetryableStatusCodes:
    """The status-forcelist must cover every canonical transient status."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_canonical_transient_code_is_retryable(self, code: int) -> None:
        assert code in constants.RETRYABLE_STATUS_CODES


class TestClientRetryConfiguration:
    """The retry strategy attached to the session must honor the constants."""

    def test_retry_adapter_reflects_constants(self) -> None:
        client = JuniperDataClient(base_url="http://localhost:8100", retries=2)
        try:
            adapter = client.session.get_adapter("http://localhost:8100/")
            status_forcelist = set(adapter.max_retries.status_forcelist or [])
            allowed_methods = set(adapter.max_retries.allowed_methods or [])
            assert status_forcelist == set(constants.RETRYABLE_STATUS_CODES)
            assert allowed_methods == set(constants.RETRY_ALLOWED_METHODS)
            # Explicit spot-checks for the XREPO-11 safety guarantee.
            assert "POST" not in allowed_methods
            assert "DELETE" not in allowed_methods
            assert "PATCH" not in allowed_methods
        finally:
            client.close()
