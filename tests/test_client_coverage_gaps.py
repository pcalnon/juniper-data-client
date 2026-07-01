"""Targeted tests closing the ``client.py`` per-file coverage gap (Phase C / C-2).

Part of the ecosystem per-file-coverage rollout (juniper-ml
``notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md``,
§5 work-unit C-2). ``juniper_data_client/client.py`` was the single file
below the per-file 90% bar. These tests exercise the previously-uncovered
paths on real behaviour (not a threshold nudge):

- ``_resolve_api_key_from_env`` Docker-secret (``*_FILE``) failure branches
  (unreadable file -> ``OSError``; empty file -> treated as absent);
- the ``_request`` fallback when a non-2xx error body is not valid JSON;
- the ``wait_for_ready`` polling loop (ready and timeout paths);
- the optional ``create_dataset`` metadata fields
  (name / description / created_by / parent_dataset_id);
- the partial-argument branches of ``create_spiral_dataset`` (no seed) and
  ``batch_update_tags`` (no add/remove tag lists).

Uses the ``responses`` library to mock HTTP, matching the suite idiom.
"""

import json

import pytest
import responses

from juniper_data_client import JuniperDataClient, JuniperDataClientError, JuniperDataValidationError
from juniper_data_client import client as client_module

BASE_URL = "http://localhost:8100"


@pytest.mark.unit
class TestApiKeyFileFailurePaths:
    """`_resolve_api_key_from_env` Docker-secret (`*_FILE`) failure branches."""

    def test_unreadable_key_file_falls_back_to_plain_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """An unreadable ``*_FILE`` path (raises ``OSError``) falls back to the plain env var."""
        # Parent directory does not exist -> Path.read_text raises FileNotFoundError (an OSError).
        missing = tmp_path / "no_such_dir" / "juniper_data_api_key"
        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(missing))
        monkeypatch.setenv("JUNIPER_DATA_API_KEY", "plain-fallback-key")
        client = JuniperDataClient()
        assert client.session.headers.get("X-API-Key") == "plain-fallback-key"

    def test_empty_key_file_yields_no_auth_header(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """A whitespace-only ``*_FILE`` secret is treated as absent (no header when no plain env)."""
        secret = tmp_path / "empty_secret"
        secret.write_text("   \n")
        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret))
        monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)
        client = JuniperDataClient()
        assert "X-API-Key" not in client.session.headers


@pytest.mark.unit
class TestNonJsonErrorBody:
    """`_request` surfaces a non-JSON error body as the raw error detail."""

    @responses.activate
    def test_non_json_error_body_uses_raw_text(self) -> None:
        """A non-2xx response whose body is not valid JSON falls back to the raw text.

        Uses a POST (not a retryable method) with a 400 so the request is not
        retried and the ``response.ok is False`` branch is reached directly; the
        plain-text body makes ``response.json()`` raise, exercising the
        ``except (ValueError, KeyError)`` raw-text fallback.
        """
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/datasets",
            body="Bad Request: malformed generator params",
            status=400,
            content_type="text/plain",
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError, match="malformed generator params"):
            client.create_dataset("spiral", {"bad": True})


@pytest.mark.unit
class TestWaitForReady:
    """`wait_for_ready` polling loop — the ready path and the timeout path."""

    @responses.activate
    def test_returns_true_when_service_ready(self) -> None:
        """Returns True as soon as the readiness probe reports ready."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/v1/health/ready",
            json={"status": "ready"},
            status=200,
        )

        client = JuniperDataClient()
        assert client.wait_for_ready(timeout=5.0, poll_interval=0.01) is True

    @responses.activate
    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when the service never becomes ready before the deadline.

        A fake clock replaces the client module's ``time`` reference so the loop
        is fully deterministic and never sleeps in wall-clock time: ``sleep``
        advances the fake clock, so the deadline is reached after a fixed number
        of iterations regardless of host speed. The readiness probe returns a
        non-ready status (a plain 200 body, so no retry/backoff runs).
        """
        responses.add(
            responses.GET,
            f"{BASE_URL}/v1/health/ready",
            json={"status": "initializing"},
            status=200,
        )

        real_time = client_module.time

        class _FakeClock:
            """Deterministic clock: ``time`` advances only when ``sleep`` is called.

            Any other attribute (e.g. ``monotonic``, used by ``_request``) delegates
            to the real ``time`` module so the instrumentation path is unaffected.
            """

            def __init__(self) -> None:
                self.now = 1000.0

            def time(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.now += seconds

            def __getattr__(self, name: str):
                return getattr(real_time, name)

        monkeypatch.setattr(client_module, "time", _FakeClock())

        client = JuniperDataClient()
        assert client.wait_for_ready(timeout=0.05, poll_interval=0.02) is False


@pytest.mark.unit
class TestCreateDatasetOptionalMetadata:
    """`create_dataset` forwards the optional metadata fields when supplied."""

    @responses.activate
    def test_optional_metadata_fields_are_forwarded(self) -> None:
        """name / description / created_by / parent_dataset_id populate the request payload."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/datasets",
            json={"dataset_id": "ds-meta-1", "generator": "spiral"},
            status=201,
        )

        client = JuniperDataClient()
        result = client.create_dataset(
            "spiral",
            {"n_spirals": 2},
            name="my-dataset",
            description="a described dataset",
            created_by="unit-test",
            parent_dataset_id="parent-42",
        )

        assert result["dataset_id"] == "ds-meta-1"
        body = json.loads(responses.calls[0].request.body)
        assert body["name"] == "my-dataset"
        assert body["description"] == "a described dataset"
        assert body["created_by"] == "unit-test"
        assert body["parent_dataset_id"] == "parent-42"


@pytest.mark.unit
class TestCreateSpiralWithoutSeed:
    """`create_spiral_dataset` omits the seed key when no seed is supplied."""

    @responses.activate
    def test_seed_omitted_when_none(self) -> None:
        """No ``seed`` key is sent in the params when the caller does not supply one."""
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/datasets",
            json={"dataset_id": "spiral-noseed", "generator": "spiral"},
            status=201,
        )

        client = JuniperDataClient()
        result = client.create_spiral_dataset(n_spirals=3)

        assert result["dataset_id"] == "spiral-noseed"
        body = json.loads(responses.calls[0].request.body)
        assert "seed" not in body["params"]
        assert body["params"]["n_spirals"] == 3


@pytest.mark.unit
class TestBatchUpdateTagsPartialArgs:
    """`batch_update_tags` omits empty add/remove tag lists from the payload."""

    @responses.activate
    def test_no_tag_lists_sends_only_dataset_ids(self) -> None:
        """With neither add_tags nor remove_tags, only dataset_ids is sent."""
        responses.add(
            responses.PATCH,
            f"{BASE_URL}/v1/datasets/batch-tags",
            json={"updated": ["ds-1"], "not_found": [], "total_updated": 1},
            status=200,
        )

        client = JuniperDataClient()
        result = client.batch_update_tags(dataset_ids=["ds-1"])

        assert result["total_updated"] == 1
        body = json.loads(responses.calls[0].request.body)
        assert body == {"dataset_ids": ["ds-1"]}
        assert "add_tags" not in body
        assert "remove_tags" not in body
