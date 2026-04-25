"""ERR-01 (Phase 4C): malformed JSON response bodies should raise typed errors.

The client previously called ``response.json()`` directly; a malformed body
would surface ``requests.exceptions.JSONDecodeError`` (a ``ValueError`` subclass)
to the caller. Phase 4C wraps these calls in a ``_parse_json`` helper that
raises ``JuniperDataClientError`` with a body preview so the public API never
leaks a raw decode error.
"""

import pytest
import responses

from juniper_data_client import JuniperDataClient, JuniperDataClientError


BASE_URL = "http://localhost:8100"
API_URL = f"{BASE_URL}/v1"


@responses.activate
def test_health_check_malformed_json_raises_client_error() -> None:
    responses.add(
        responses.GET,
        f"{API_URL}/health",
        body="<!DOCTYPE html><html>not json</html>",
        status=200,
        content_type="application/json",
    )
    with JuniperDataClient(BASE_URL) as client:
        with pytest.raises(JuniperDataClientError, match="Malformed JSON response"):
            client.health_check()


@responses.activate
def test_list_generators_malformed_json_raises_client_error() -> None:
    responses.add(
        responses.GET,
        f"{API_URL}/generators",
        body="{not valid json",
        status=200,
        content_type="application/json",
    )
    with JuniperDataClient(BASE_URL) as client:
        with pytest.raises(JuniperDataClientError, match="Malformed JSON response"):
            client.list_generators()


@responses.activate
def test_is_ready_malformed_json_returns_false() -> None:
    """is_ready swallows JuniperDataClientError and returns False."""
    responses.add(
        responses.GET,
        f"{API_URL}/health/ready",
        body="not json at all",
        status=200,
        content_type="application/json",
    )
    with JuniperDataClient(BASE_URL) as client:
        assert client.is_ready() is False


@responses.activate
def test_empty_body_raises_client_error() -> None:
    responses.add(
        responses.GET,
        f"{API_URL}/health",
        body="",
        status=200,
        content_type="application/json",
    )
    with JuniperDataClient(BASE_URL) as client:
        with pytest.raises(JuniperDataClientError, match="Malformed JSON response"):
            client.health_check()
