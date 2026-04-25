"""Regression tests for ``create_dataset(..., tags=..., ttl_seconds=...)``.

Guards XREPO-09: the client previously dropped ``tags`` and
``ttl_seconds`` even though the server's ``CreateDatasetRequest``
accepted them. These tests pin both the real client (payload shape)
and the ``FakeDataClient`` (metadata round-trip + validation).
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import patch

import pytest

from juniper_data_client import JuniperDataClient
from juniper_data_client.exceptions import JuniperDataValidationError
from juniper_data_client.testing import FakeDataClient


@pytest.fixture
def fake_client() -> FakeDataClient:
    with FakeDataClient() as client:
        yield client


class TestRealClientPayload:
    """The real ``JuniperDataClient`` must forward the new fields to the server."""

    @patch.object(JuniperDataClient, "_request")
    def test_tags_included_in_post_body(self, mock_request: Any) -> None:
        mock_request.return_value.json.return_value = {"dataset_id": "stub"}
        client = JuniperDataClient(base_url="http://localhost:8100")
        try:
            client.create_dataset(
                generator="spiral",
                params={"seed": 1},
                tags=["alpha", "beta"],
            )
        finally:
            client.close()

        _, kwargs = mock_request.call_args
        payload = kwargs["json"]
        assert payload["tags"] == ["alpha", "beta"]

    @patch.object(JuniperDataClient, "_request")
    def test_ttl_seconds_included_in_post_body(self, mock_request: Any) -> None:
        mock_request.return_value.json.return_value = {"dataset_id": "stub"}
        client = JuniperDataClient(base_url="http://localhost:8100")
        try:
            client.create_dataset(
                generator="spiral",
                params={"seed": 1},
                ttl_seconds=3600,
            )
        finally:
            client.close()

        _, kwargs = mock_request.call_args
        payload = kwargs["json"]
        assert payload["ttl_seconds"] == 3600

    @patch.object(JuniperDataClient, "_request")
    def test_omitted_when_none(self, mock_request: Any) -> None:
        """Backward compat: old callers must still produce payloads without the new keys."""
        mock_request.return_value.json.return_value = {"dataset_id": "stub"}
        client = JuniperDataClient(base_url="http://localhost:8100")
        try:
            client.create_dataset(generator="spiral", params={"seed": 1})
        finally:
            client.close()

        _, kwargs = mock_request.call_args
        payload = kwargs["json"]
        assert "tags" not in payload
        assert "ttl_seconds" not in payload


class TestFakeClientRoundTrip:
    """The ``FakeDataClient`` must persist the new fields in metadata."""

    def test_tags_stored_in_meta(self, fake_client: FakeDataClient) -> None:
        result = fake_client.create_dataset(
            "spiral",
            {"seed": 1},
            tags=["canary", "small"],
        )
        assert result["meta"]["tags"] == ["canary", "small"]

        # Re-fetch via the public metadata accessor to catch shape drift.
        refetched = fake_client.get_dataset_metadata(result["dataset_id"])
        assert refetched["meta"]["tags"] == ["canary", "small"]

    def test_ttl_seconds_stored_in_meta(self, fake_client: FakeDataClient) -> None:
        result = fake_client.create_dataset(
            "spiral",
            {"seed": 1},
            ttl_seconds=120,
        )
        assert result["meta"]["ttl_seconds"] == 120

    def test_tags_defaults_to_absent(self, fake_client: FakeDataClient) -> None:
        """When not provided, ``meta`` must NOT contain a ``tags`` key."""
        result = fake_client.create_dataset("spiral", {"seed": 1})
        assert "tags" not in result["meta"]
        assert "ttl_seconds" not in result["meta"]

    def test_zero_ttl_rejected(self, fake_client: FakeDataClient) -> None:
        """Mirror the server-side Pydantic ``ge=1`` bound."""
        with pytest.raises(JuniperDataValidationError, match="ttl_seconds"):
            fake_client.create_dataset("spiral", {"seed": 1}, ttl_seconds=0)

    def test_negative_ttl_rejected(self, fake_client: FakeDataClient) -> None:
        with pytest.raises(JuniperDataValidationError, match="ttl_seconds"):
            fake_client.create_dataset("spiral", {"seed": 1}, ttl_seconds=-5)

    def test_tags_are_copied_not_aliased(self, fake_client: FakeDataClient) -> None:
        """Mutating the input list after creation must not mutate stored tags."""
        inp = ["one", "two"]
        result = fake_client.create_dataset("spiral", {"seed": 1}, tags=inp)
        inp.append("three")
        assert result["meta"]["tags"] == ["one", "two"]


class TestJSONSerialization:
    """The resulting payloads must remain JSON-serializable."""

    def test_roundtrip_through_json(self, fake_client: FakeDataClient) -> None:
        result = fake_client.create_dataset(
            "spiral",
            {"seed": 1},
            tags=["prod"],
            ttl_seconds=60,
        )
        encoded = json.dumps(result["meta"])
        decoded: Dict[str, Any] = json.loads(encoded)
        assert decoded["tags"] == ["prod"]
        assert decoded["ttl_seconds"] == 60
