"""Regression tests for outbound ``X-Request-ID`` propagation.

METRICS-MON R4.6 / R3.6 sweep follow-up.

R3.6's §10 matrix sweep flagged that ``juniper_data_client._request()``
did not emit ``X-Request-ID`` on outbound HTTP calls. Result: juniper-data
``RequestIdMiddleware`` (R2.1) generated a fresh request-id on every
inbound request, breaking correlation back to the canopy/cascor →
data-client → data chain.

Fix shape (per roadmap §7 R4.6 Q6 resolution): opt-in propagation via
the shared ``juniper-observability`` ContextVar. When the calling thread
has a non-empty ``request_id_var``, the data-client copies it into an
outbound ``X-Request-ID`` header. Caller-supplied headers always win.

These tests pin:

1. With ``request_id_var`` set, outbound request carries
   ``X-Request-ID: <expected>``.
2. With ``request_id_var`` unset (LookupError), no header is added —
   data-side middleware will generate one.
3. With ``juniper-observability`` not importable (standalone use),
   propagation silently no-ops.
4. Caller-supplied ``X-Request-ID`` is not overwritten by the
   ContextVar value (caller wins).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from juniper_data_client import JuniperDataClient


class TestXRequestIdPropagation:
    """METRICS-MON R4.6: outbound X-Request-ID propagation via juniper-observability."""

    def test_propagates_request_id_when_contextvar_set(self) -> None:
        from juniper_observability import request_id_var

        client = JuniperDataClient(base_url="http://localhost:8100")
        token = request_id_var.set("test-rid-r4-6-aaaaaaaa")
        try:
            with patch.object(client.session, "request") as mock_request:
                mock_request.return_value.ok = True
                mock_request.return_value.status_code = 200
                client._request("GET", "/v1/health")
                assert mock_request.called
                call_kwargs = mock_request.call_args.kwargs
                assert "headers" in call_kwargs
                assert call_kwargs["headers"].get("X-Request-ID") == "test-rid-r4-6-aaaaaaaa"
        finally:
            request_id_var.reset(token)

    def test_no_header_added_when_contextvar_unset(self) -> None:
        # request_id_var has no value in a fresh ``contextvars.Context``;
        # the propagation path should silently no-op (LookupError
        # handled). We run the assertion inside a fresh context to defeat
        # any other test that may have set the var earlier in the
        # process — ``ContextVar`` state is per-Context, so a fresh
        # ``Context().run(callable)`` gives us a guaranteed unset state.
        import contextvars

        client = JuniperDataClient(base_url="http://localhost:8100")

        def _assert_no_header():
            with patch.object(client.session, "request") as mock_request:
                mock_request.return_value.ok = True
                mock_request.return_value.status_code = 200
                client._request("GET", "/v1/health")
                call_kwargs = mock_request.call_args.kwargs
                sent_headers = call_kwargs.get("headers") or {}
                assert "X-Request-ID" not in sent_headers

        # ``contextvars.Context()`` is a fresh, empty context — no
        # ContextVars are bound. ``copy_context()`` would inherit the
        # current bindings (defeating the test).
        contextvars.Context().run(_assert_no_header)

    def test_caller_supplied_header_wins(self) -> None:
        from juniper_observability import request_id_var

        client = JuniperDataClient(base_url="http://localhost:8100")
        token = request_id_var.set("from-context-var")
        try:
            with patch.object(client.session, "request") as mock_request:
                mock_request.return_value.ok = True
                mock_request.return_value.status_code = 200
                client._request("GET", "/v1/health", headers={"X-Request-ID": "from-caller"})
                call_kwargs = mock_request.call_args.kwargs
                # Caller-supplied value must not be overwritten by the
                # ContextVar value — explicit always beats implicit.
                assert call_kwargs["headers"]["X-Request-ID"] == "from-caller"
        finally:
            request_id_var.reset(token)

    def test_silently_no_ops_when_juniper_observability_not_installed(self) -> None:
        """Standalone use case: data-client imported in a notebook with no
        observability lib. Propagation must not raise; data-side middleware
        will generate a request-id at the inbound boundary.
        """
        client = JuniperDataClient(base_url="http://localhost:8100")
        # Block the import by inserting a None entry into sys.modules so
        # ``from juniper_observability import request_id_var`` raises
        # ImportError. Using ``import_mocking`` via a MagicMock isn't
        # sufficient because ``from X import Y`` accesses the attribute,
        # which a real ImportError model handles cleanly.
        with patch.dict(sys.modules, {"juniper_observability": None}):
            with patch.object(client.session, "request") as mock_request:
                mock_request.return_value.ok = True
                mock_request.return_value.status_code = 200
                # Must not raise.
                client._request("GET", "/v1/health")
                call_kwargs = mock_request.call_args.kwargs
                sent_headers = call_kwargs.get("headers") or {}
                assert "X-Request-ID" not in sent_headers
