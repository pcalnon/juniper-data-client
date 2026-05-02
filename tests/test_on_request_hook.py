"""Regression tests for the optional ``on_request`` instrumentation hook.

METRICS-MON R4.3 / seed-13.

The hook is the canonical extension point for consumer-side metrics
emission (Prometheus, OpenTelemetry, structured logs). The contract:

* Default is a no-op so standalone notebook use pays nothing.
* Hook fires **once per HTTP call**, on every outcome — success,
  transport error, HTTP non-2xx (each of the 4 typed exception
  branches).
* ``error is None`` is the canonical success signal; ``status`` may be
  set even on the error path (404 / 422 / 500 produce both an HTTP
  status and a typed exception).
* Hook exceptions are swallowed so instrumentation never crashes a
  production HTTP path.

These tests pin all 6 outcome paths plus the swallow-and-log behavior.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from juniper_data_client import JuniperDataClient, JuniperDataClientError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError, RequestHook
from juniper_data_client.client import _noop_request_hook


def _make_response(status_code: int, ok: bool | None = None, text: str = "ok") -> MagicMock:
    """Build a ``requests.Response``-shaped mock for the success path."""
    response = MagicMock()
    response.status_code = status_code
    # Bool override lets non-2xx success-shaped responses test the
    # error-mapping branches separately from the success branch.
    response.ok = ok if ok is not None else (200 <= status_code < 300)
    response.text = text
    response.json.return_value = {}
    return response


class TestOnRequestHookDefault:
    """The default (no caller-supplied hook) is a no-op."""

    def test_no_hook_kwarg_uses_noop_default(self):
        client = JuniperDataClient(base_url="http://localhost:8100")
        assert client._on_request is _noop_request_hook

    def test_explicit_none_kwarg_uses_noop_default(self):
        # ``None`` falls back to the no-op rather than crashing on
        # ``self._on_request(...)`` at call time.
        client = JuniperDataClient(base_url="http://localhost:8100", on_request=None)
        assert client._on_request is _noop_request_hook


class TestOnRequestHookFires:
    """Hook fires exactly once per HTTP call, on every outcome."""

    def test_fires_on_success(self):
        calls: list[tuple[Any, ...]] = []

        def hook(method: str, url: str, status: int | None, duration_ms: float, error: BaseException | None) -> None:
            calls.append((method, url, status, duration_ms, error))

        client = JuniperDataClient(base_url="http://localhost:8100", on_request=hook)
        with patch.object(client.session, "request", return_value=_make_response(200)):
            client._request("GET", "/v1/health")

        assert len(calls) == 1
        method, url, status, duration_ms, error = calls[0]
        assert method == "GET"
        assert url == "http://localhost:8100/v1/health"
        assert status == 200
        assert duration_ms >= 0
        assert error is None

    def test_fires_on_404_with_typed_exception(self):
        """404 → JuniperDataNotFoundError; hook still sees status=404."""
        calls: list[tuple[Any, ...]] = []
        client = JuniperDataClient(
            base_url="http://localhost:8100",
            on_request=lambda *args: calls.append(args),
        )
        with patch.object(client.session, "request", return_value=_make_response(404, ok=False)):
            with pytest.raises(JuniperDataNotFoundError):
                client._request("GET", "/v1/datasets/missing")

        assert len(calls) == 1
        method, url, status, _, error = calls[0]
        assert status == 404
        assert isinstance(error, JuniperDataNotFoundError)

    def test_fires_on_422_with_validation_error(self):
        calls: list[tuple[Any, ...]] = []
        client = JuniperDataClient(
            base_url="http://localhost:8100",
            on_request=lambda *args: calls.append(args),
        )
        with patch.object(client.session, "request", return_value=_make_response(422, ok=False)):
            with pytest.raises(JuniperDataValidationError):
                client._request("POST", "/v1/datasets")

        assert len(calls) == 1
        _, _, status, _, error = calls[0]
        assert status == 422
        assert isinstance(error, JuniperDataValidationError)

    def test_fires_on_500_with_generic_client_error(self):
        calls: list[tuple[Any, ...]] = []
        client = JuniperDataClient(
            base_url="http://localhost:8100",
            on_request=lambda *args: calls.append(args),
        )
        with patch.object(client.session, "request", return_value=_make_response(500, ok=False)):
            with pytest.raises(JuniperDataClientError):
                client._request("GET", "/v1/health")

        assert len(calls) == 1
        _, _, status, _, error = calls[0]
        assert status == 500
        assert isinstance(error, JuniperDataClientError)

    def test_fires_on_connection_error_with_status_none(self):
        """Transport error → no HTTP response → ``status`` is None."""
        calls: list[tuple[Any, ...]] = []
        client = JuniperDataClient(
            base_url="http://localhost:8100",
            on_request=lambda *args: calls.append(args),
        )
        with patch.object(client.session, "request", side_effect=requests.exceptions.ConnectionError("refused")):
            with pytest.raises(JuniperDataConnectionError):
                client._request("GET", "/v1/health")

        assert len(calls) == 1
        _, _, status, _, error = calls[0]
        assert status is None
        assert isinstance(error, JuniperDataConnectionError)

    def test_fires_on_timeout_with_status_none(self):
        calls: list[tuple[Any, ...]] = []
        client = JuniperDataClient(
            base_url="http://localhost:8100",
            on_request=lambda *args: calls.append(args),
        )
        with patch.object(client.session, "request", side_effect=requests.exceptions.Timeout("slow")):
            with pytest.raises(JuniperDataTimeoutError):
                client._request("GET", "/v1/health")

        assert len(calls) == 1
        _, _, status, _, error = calls[0]
        assert status is None
        assert isinstance(error, JuniperDataTimeoutError)


class TestOnRequestHookResilience:
    """Hook exceptions must not crash the request path."""

    def test_hook_exception_swallowed_on_success(self, caplog):
        """A buggy hook that raises must not corrupt the success return."""

        def buggy_hook(*args: Any) -> None:
            raise RuntimeError("boom")

        client = JuniperDataClient(base_url="http://localhost:8100", on_request=buggy_hook)
        with patch.object(client.session, "request", return_value=_make_response(200)):
            with caplog.at_level("WARNING", logger="juniper_data_client.client"):
                response = client._request("GET", "/v1/health")

        assert response.status_code == 200
        # Suppression is logged at WARNING so operators can detect a
        # broken hook in production telemetry.
        assert any("on_request hook raised" in r.message for r in caplog.records)

    def test_hook_exception_does_not_mask_typed_error(self):
        """If the request itself fails AND the hook raises, the typed
        exception still propagates (hook failure is a side effect).
        """

        def buggy_hook(*args: Any) -> None:
            raise RuntimeError("boom")

        client = JuniperDataClient(base_url="http://localhost:8100", on_request=buggy_hook)
        with patch.object(client.session, "request", return_value=_make_response(404, ok=False)):
            # Original 404 → JuniperDataNotFoundError must still raise;
            # the hook's RuntimeError is swallowed in the finally.
            with pytest.raises(JuniperDataNotFoundError):
                client._request("GET", "/v1/datasets/missing")


class TestRequestHookTypeAlias:
    """The exported type alias is callable-compatible."""

    def test_request_hook_type_alias_accepts_well_typed_callable(self):
        """Compile-time-style sanity: the alias matches a closure of the
        documented shape. Runtime check verifies the closure satisfies
        the protocol the client expects.
        """

        def well_typed_hook(method: str, url: str, status: int | None, duration_ms: float, error: BaseException | None) -> None:
            pass

        # Annotate with the alias to assert the assignment is legal at
        # type-check time; mypy in CI catches drift if the alias signature
        # diverges from RequestHook callers.
        hook: RequestHook = well_typed_hook
        client = JuniperDataClient(base_url="http://localhost:8100", on_request=hook)
        assert client._on_request is well_typed_hook
