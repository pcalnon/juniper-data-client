"""Unit tests for JuniperDataClient.

Uses the `responses` library to mock HTTP requests without requiring a live service.
"""

import io
import json
from typing import Any, Dict

import numpy as np
import pytest
import requests
import responses

from juniper_data_client import JuniperDataClient, JuniperDataClientError, JuniperDataConfigurationError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError
from juniper_data_client.client import _render_error_detail


class TestUrlNormalization:
    """Tests for URL normalization logic."""

    def test_normalize_basic_url(self) -> None:
        """Basic URL with scheme."""
        client = JuniperDataClient("http://localhost:8100")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_url_without_scheme(self) -> None:
        """URL without scheme gets http:// prefix."""
        client = JuniperDataClient("localhost:8100")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_url_with_trailing_slash(self) -> None:
        """Trailing slash is removed."""
        client = JuniperDataClient("http://localhost:8100/")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_url_with_v1_suffix(self) -> None:
        """/v1 suffix is removed."""
        client = JuniperDataClient("http://localhost:8100/v1")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_url_with_v1_and_trailing_slash(self) -> None:
        """/v1/ suffix is removed."""
        client = JuniperDataClient("http://localhost:8100/v1/")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_https_url(self) -> None:
        """HTTPS URLs are preserved."""
        client = JuniperDataClient("https://api.example.com:8100")
        assert client.base_url == "https://api.example.com:8100"

    def test_normalize_url_with_whitespace(self) -> None:
        """Whitespace is stripped."""
        client = JuniperDataClient("  http://localhost:8100  ")
        assert client.base_url == "http://localhost:8100"

    def test_normalize_uppercase_scheme_is_not_downgraded(self) -> None:
        """Scheme matching is case-insensitive (RFC 3986): 'HTTPS://' must stay
        https, not be re-prefixed into http://HTTPS://... — a silent TLS
        downgrade that would send the API key over HTTP to hostname 'https'."""
        client = JuniperDataClient("HTTPS://api.example.com:8100")
        assert client.base_url == "https://api.example.com:8100"

    def test_normalize_mixed_case_scheme(self) -> None:
        client = JuniperDataClient("Http://localhost:8100")
        assert client.base_url == "http://localhost:8100"

    @pytest.mark.parametrize(
        "hostless",
        [
            "",
            "   ",
            "http://",
            "https://",
            "/v1",
            "http:///v1",
            "http://user:secret@",
            # netloc is ":8200" (truthy) while hostname is None — the exact
            # distinction that made #166 read hostname rather than netloc.
            "http://:8200",
        ],
    )
    def test_normalize_hostless_url_raises_configuration_error(self, hostless: str) -> None:
        """A base URL with no host must fail at construction with the typed
        error, not opaquely on the first request (APD-DCLIENT-004).

        Each value normalizes to a hostless URL: the empty/whitespace forms,
        a bare scheme, path-only values (whose scheme-defaulted parse has
        an empty netloc), and authorities whose netloc is truthy but whose
        hostname is missing (userinfo-only, port-only).
        """
        with pytest.raises(JuniperDataConfigurationError, match="must include a host"):
            JuniperDataClient(hostless)

    def test_hostless_url_error_is_catchable_as_the_base_error(self) -> None:
        """The guard raises inside the package hierarchy."""
        with pytest.raises(JuniperDataClientError):
            JuniperDataClient("http://")

    def test_normalize_ipv6_host_preserves_brackets_and_port(self) -> None:
        """IPv6 reconstruction must keep RFC 3986 brackets and the port.

        Using ``hostname`` (``::1``) instead of ``netloc`` (``[::1]:8100``)
        would produce a URL requests cannot connect to.
        """
        client = JuniperDataClient("http://[::1]:8100")
        assert client.base_url == "http://[::1]:8100"

    def test_normalize_uppercase_https_ipv6(self) -> None:
        """Case-insensitive scheme matching plus IPv6 together.

        The #166 TLS-downgrade bug would re-prefix this into
        ``http://HTTPS://[::1]:443`` and send the API key to hostname ``https``.
        """
        client = JuniperDataClient("HTTPS://[::1]:443")
        assert client.base_url == "https://[::1]:443"

    def test_normalize_userinfo_and_host_are_preserved(self) -> None:
        """Reconstruction uses netloc, so userinfo stays on the URL.

        Dropping credentials here would silently de-auth a consumer that puts
        basic-auth in the base URL. ``http://user:secret@`` (no host) is the
        reject arm; this is the accept arm.
        """
        client = JuniperDataClient("http://user:secret@localhost:8100")
        assert client.base_url == "http://user:secret@localhost:8100"

    def test_normalize_reverse_proxy_v1_suffix_is_stripped(self) -> None:
        """A path prefix in front of /v1 is a reverse-proxy deployment.

        Only the trailing API-version suffix is removed; the prefix stays so
        subsequent requests hit ``{prefix}/v1/...``.
        """
        client = JuniperDataClient("http://localhost:8100/proxy/v1")
        assert client.base_url == "http://localhost:8100/proxy"

    def test_normalize_v1_in_the_middle_of_the_path_is_kept(self) -> None:
        """``endswith('/v1')`` must not strip a /v1 that is not the final segment."""
        client = JuniperDataClient("http://localhost:8100/v1/extra")
        assert client.base_url == "http://localhost:8100/v1/extra"


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_default_configuration(self) -> None:
        """Default configuration values."""
        client = JuniperDataClient()
        assert client.base_url == "http://localhost:8100"
        assert client.timeout == 30
        assert client.retries == 3
        assert client.backoff_factor == 0.5

    def test_custom_configuration(self) -> None:
        """Custom configuration values."""
        client = JuniperDataClient(
            base_url="http://custom:9000",
            timeout=60,
            retries=5,
            backoff_factor=1.0,
        )
        assert client.base_url == "http://custom:9000"
        assert client.timeout == 60
        assert client.retries == 5
        assert client.backoff_factor == 1.0

    def test_context_manager(self) -> None:
        """Context manager properly closes session."""
        with JuniperDataClient() as client:
            assert client.session is not None
        # Session should be closed after context exit

    def test_api_key_from_parameter(self) -> None:
        """API key from parameter is set in session headers."""
        client = JuniperDataClient(api_key="test-api-key-123")
        assert client.session.headers.get("X-API-Key") == "test-api-key-123"

    def test_api_key_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key from environment variable is set in session headers."""
        monkeypatch.setenv("JUNIPER_DATA_API_KEY", "env-api-key-456")
        client = JuniperDataClient()
        assert client.session.headers.get("X-API-Key") == "env-api-key-456"

    def test_api_key_parameter_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key from parameter takes precedence over environment variable."""
        monkeypatch.setenv("JUNIPER_DATA_API_KEY", "env-api-key")
        client = JuniperDataClient(api_key="param-api-key")
        assert client.session.headers.get("X-API-Key") == "param-api-key"

    def test_no_api_key_header_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No X-API-Key header when API key is not provided (neither plain env nor _FILE)."""
        monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)
        monkeypatch.delenv("JUNIPER_DATA_API_KEY_FILE", raising=False)
        client = JuniperDataClient()
        assert "X-API-Key" not in client.session.headers

    def test_api_key_from_file_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """API key resolves from the Docker-secret JUNIPER_DATA_API_KEY_FILE indirection."""
        secret = tmp_path / "juniper_data_api_key"
        secret.write_text("file-api-key-789\n")
        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret))
        monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)
        client = JuniperDataClient()
        assert client.session.headers.get("X-API-Key") == "file-api-key-789"

    def test_api_key_file_takes_precedence_over_plain_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """The _FILE secret wins over the plain env var (the mounted file is the source of truth)."""
        secret = tmp_path / "juniper_data_api_key"
        secret.write_text("file-key\n")
        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret))
        monkeypatch.setenv("JUNIPER_DATA_API_KEY", "plain-key")
        client = JuniperDataClient()
        assert client.session.headers.get("X-API-Key") == "file-key"

    def test_api_key_parameter_beats_file_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """An explicit api_key= still takes precedence over the _FILE secret."""
        secret = tmp_path / "juniper_data_api_key"
        secret.write_text("file-key\n")
        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret))
        client = JuniperDataClient(api_key="explicit-key")
        assert client.session.headers.get("X-API-Key") == "explicit-key"


@pytest.mark.unit
class TestConfigurationError:
    """Tests for JuniperDataConfigurationError exception."""

    def test_configuration_error_is_subclass_of_client_error(self) -> None:
        """ConfigurationError is a subclass of ClientError."""
        assert issubclass(JuniperDataConfigurationError, JuniperDataClientError)

    def test_configuration_error_can_be_raised_and_caught(self) -> None:
        """ConfigurationError can be raised and caught."""
        with pytest.raises(JuniperDataConfigurationError) as exc_info:
            raise JuniperDataConfigurationError("Missing required configuration")
        assert "Missing required configuration" in str(exc_info.value)


@pytest.mark.unit
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @responses.activate
    def test_health_check_success(self) -> None:
        """Successful health check."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health",
            json={"status": "ok", "version": "0.3.0"},
            status=200,
        )

        client = JuniperDataClient()
        result = client.health_check()
        assert result["status"] == "ok"
        assert result["version"] == "0.3.0"

    @responses.activate
    def test_is_ready_true(self) -> None:
        """Service is ready."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health/ready",
            json={"status": "ready", "version": "0.3.0"},
            status=200,
        )

        client = JuniperDataClient()
        assert client.is_ready() is True

    @responses.activate
    def test_is_ready_false_on_error(self) -> None:
        """Service not ready on error."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health/ready",
            json={"detail": "Service unavailable"},
            status=503,
        )

        client = JuniperDataClient()
        assert client.is_ready() is False

    @responses.activate
    def test_is_ready_false_on_connection_error(self) -> None:
        """Service not ready on connection error."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health/ready",
            body=requests.exceptions.ConnectionError("Connection refused"),
        )

        client = JuniperDataClient()
        assert client.is_ready() is False


@pytest.mark.unit
class TestGeneratorEndpoints:
    """Tests for generator endpoints."""

    @responses.activate
    def test_list_generators(self) -> None:
        """List available generators."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/generators",
            json=[{"name": "spiral", "version": "1.0.0", "description": "Spiral dataset"}],
            status=200,
        )

        client = JuniperDataClient()
        result = client.list_generators()
        assert len(result) == 1
        assert result[0]["name"] == "spiral"

    @responses.activate
    def test_get_generator_schema(self) -> None:
        """Get generator parameter schema."""
        schema = {
            "properties": {
                "n_spirals": {"type": "integer", "default": 2},
                "n_points_per_spiral": {"type": "integer", "default": 100},
            }
        }
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/generators/spiral/schema",
            json=schema,
            status=200,
        )

        client = JuniperDataClient()
        result = client.get_generator_schema("spiral")
        assert "properties" in result
        assert "n_spirals" in result["properties"]

    @responses.activate
    def test_get_generator_schema_not_found(self) -> None:
        """Generator not found raises exception."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/generators/nonexistent/schema",
            json={"detail": "Generator not found"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.get_generator_schema("nonexistent")


@pytest.mark.unit
class TestDatasetCreation:
    """Tests for dataset creation."""

    @responses.activate
    def test_create_dataset_success(self) -> None:
        """Successful dataset creation."""
        response_data = {
            "dataset_id": "test-dataset-123",
            "generator": "spiral",
            "meta": {
                "dataset_id": "test-dataset-123",
                "generator": "spiral",
                "n_samples": 200,
            },
            "artifact_url": "/v1/datasets/test-dataset-123/artifact",
        }
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json=response_data,
            status=201,
        )

        client = JuniperDataClient()
        result = client.create_dataset("spiral", {"n_spirals": 2, "seed": 42})
        assert result["dataset_id"] == "test-dataset-123"
        assert result["generator"] == "spiral"

    @responses.activate
    def test_create_spiral_dataset_convenience(self) -> None:
        """Convenience method for spiral datasets."""
        response_data = {
            "dataset_id": "spiral-123",
            "generator": "spiral",
            "meta": {"n_samples": 200},
            "artifact_url": "/v1/datasets/spiral-123/artifact",
        }
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json=response_data,
            status=201,
        )

        client = JuniperDataClient()
        result = client.create_spiral_dataset(
            n_spirals=2,
            n_points_per_spiral=100,
            noise=0.1,
            seed=42,
        )
        assert result["dataset_id"] == "spiral-123"

        request_body = responses.calls[0].request.body
        assert request_body is not None
        body_bytes = request_body if isinstance(request_body, bytes) else request_body.encode()
        assert b"spiral" in body_bytes
        assert b"42" in body_bytes

    @responses.activate
    def test_create_dataset_validation_error(self) -> None:
        """Invalid parameters raise validation error."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": "n_spirals must be >= 2"},
            status=400,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError, match="n_spirals"):
            client.create_dataset("spiral", {"n_spirals": -1})

    @responses.activate
    def test_create_dataset_422_error(self) -> None:
        """422 validation error."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": "Invalid parameter type"},
            status=422,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError):
            client.create_dataset("spiral", {"n_spirals": "not_an_int"})


@pytest.mark.unit
class TestDatasetRetrieval:
    """Tests for dataset retrieval."""

    @responses.activate
    def test_list_datasets(self) -> None:
        """List dataset IDs."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets",
            json=["dataset-1", "dataset-2", "dataset-3"],
            status=200,
        )

        client = JuniperDataClient()
        result = client.list_datasets(limit=10, offset=0)
        assert len(result) == 3
        assert "dataset-1" in result

    @responses.activate
    def test_get_dataset_metadata(self) -> None:
        """Get dataset metadata."""
        metadata = {
            "dataset_id": "test-123",
            "generator": "spiral",
            "n_samples": 200,
            "n_features": 2,
            "n_classes": 2,
        }
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/test-123",
            json=metadata,
            status=200,
        )

        client = JuniperDataClient()
        result = client.get_dataset_metadata("test-123")
        assert result["dataset_id"] == "test-123"
        assert result["n_samples"] == 200

    @responses.activate
    def test_get_dataset_metadata_not_found(self) -> None:
        """Dataset not found raises exception."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/nonexistent",
            json={"detail": "Dataset not found"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.get_dataset_metadata("nonexistent")


@pytest.mark.unit
class TestArtifactDownload:
    """Tests for NPZ artifact download."""

    def _create_npz_bytes(self) -> bytes:
        """Create mock NPZ file bytes."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        X_test = np.array([[5.0, 6.0]], dtype=np.float32)
        y_test = np.array([[1.0, 0.0]], dtype=np.float32)
        X_full = np.vstack([X_train, X_test])
        y_full = np.vstack([y_train, y_test])

        buffer = io.BytesIO()
        np.savez(
            buffer,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_full=X_full,
            y_full=y_full,
        )
        buffer.seek(0)
        return buffer.read()

    @responses.activate
    def test_download_artifact_npz(self) -> None:
        """Download and parse NPZ artifact."""
        npz_bytes = self._create_npz_bytes()
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/test-123/artifact",
            body=npz_bytes,
            status=200,
            content_type="application/octet-stream",
        )

        client = JuniperDataClient()
        result = client.download_artifact_npz("test-123")

        assert "X_train" in result
        assert "y_train" in result
        assert "X_test" in result
        assert "y_test" in result
        assert "X_full" in result
        assert "y_full" in result

        assert result["X_train"].dtype == np.float32
        assert result["X_train"].shape == (2, 2)

    @responses.activate
    def test_download_artifact_bytes(self) -> None:
        """Download raw NPZ bytes."""
        npz_bytes = self._create_npz_bytes()
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/test-123/artifact",
            body=npz_bytes,
            status=200,
            content_type="application/octet-stream",
        )

        client = JuniperDataClient()
        result = client.download_artifact_bytes("test-123")

        assert isinstance(result, bytes)
        assert len(result) > 0

        with np.load(io.BytesIO(result)) as data:
            assert "X_train" in data.files

    @responses.activate
    def test_download_artifact_not_found(self) -> None:
        """Artifact not found raises exception."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/nonexistent/artifact",
            json={"detail": "Dataset not found"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.download_artifact_npz("nonexistent")


@pytest.mark.unit
class TestPreview:
    """Tests for dataset preview."""

    @responses.activate
    def test_get_preview(self) -> None:
        """Get dataset preview."""
        preview = {
            "n_samples": 10,
            "X_sample": [[1.0, 2.0], [3.0, 4.0]],
            "y_sample": [[1.0, 0.0], [0.0, 1.0]],
        }
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/test-123/preview",
            json=preview,
            status=200,
        )

        client = JuniperDataClient()
        result = client.get_preview("test-123", n=10)

        assert result["n_samples"] == 10
        assert len(result["X_sample"]) == 2


@pytest.mark.unit
class TestDatasetDeletion:
    """Tests for dataset deletion."""

    @responses.activate
    def test_delete_dataset(self) -> None:
        """Delete dataset successfully."""
        responses.add(
            responses.DELETE,
            "http://localhost:8100/v1/datasets/test-123",
            status=204,
        )

        client = JuniperDataClient()
        result = client.delete_dataset("test-123")
        assert result is True

    @responses.activate
    def test_delete_dataset_not_found(self) -> None:
        """Delete nonexistent dataset raises exception."""
        responses.add(
            responses.DELETE,
            "http://localhost:8100/v1/datasets/nonexistent",
            json={"detail": "Dataset not found"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.delete_dataset("nonexistent")


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling."""

    @responses.activate
    def test_connection_error(self) -> None:
        """Connection error raises JuniperDataConnectionError."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health",
            body=requests.exceptions.ConnectionError("Connection refused"),
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataConnectionError, match="Failed to connect"):
            client.health_check()

    @responses.activate
    def test_timeout_error(self) -> None:
        """Timeout raises JuniperDataTimeoutError."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health",
            body=requests.exceptions.Timeout("Request timed out"),
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataTimeoutError, match="timed out"):
            client.health_check()

    @responses.activate
    def test_generic_request_error(self) -> None:
        """Generic request error raises JuniperDataClientError."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health",
            body=requests.exceptions.RequestException("Something went wrong"),
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataClientError, match="Request failed"):
            client.health_check()

    @responses.activate
    def test_server_error(self) -> None:
        """500 error raises JuniperDataClientError."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/health",
            json={"detail": "Internal server error"},
            status=500,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataClientError, match="500"):
            client.health_check()

    @responses.activate
    def test_error_detail_extraction(self) -> None:
        """Error detail is extracted from JSON response."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": "Custom error message"},
            status=400,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError, match="Custom error message"):
            client.create_dataset("spiral", {})


@pytest.mark.unit
class TestBatchOperations:
    """Tests for batch operation endpoints."""

    @responses.activate
    def test_batch_delete_success(self) -> None:
        """Batch delete multiple datasets."""
        response_data = {
            "deleted": ["ds-1", "ds-2"],
            "not_found": ["ds-3"],
            "total_deleted": 2,
        }
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets/batch-delete",
            json=response_data,
            status=200,
        )

        client = JuniperDataClient()
        result = client.batch_delete(["ds-1", "ds-2", "ds-3"])

        assert result["total_deleted"] == 2
        assert result["deleted"] == ["ds-1", "ds-2"]
        assert result["not_found"] == ["ds-3"]

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["dataset_ids"] == ["ds-1", "ds-2", "ds-3"]

    @responses.activate
    def test_batch_create_success(self) -> None:
        """Batch create multiple datasets."""
        datasets_input = [
            {"generator": "spiral", "params": {"n_spirals": 2, "seed": 42}},
            {"generator": "spiral", "params": {"n_spirals": 3, "seed": 99}},
        ]
        response_data = {
            "results": [
                {"dataset_id": "new-1", "status": "created"},
                {"dataset_id": "new-2", "status": "created"},
            ],
            "total_created": 2,
            "total_failed": 0,
        }
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets/batch-create",
            json=response_data,
            status=200,
        )

        client = JuniperDataClient()
        result = client.batch_create(datasets_input)

        assert result["total_created"] == 2
        assert result["total_failed"] == 0
        assert len(result["results"]) == 2

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["datasets"] == datasets_input

    @responses.activate
    def test_batch_update_tags_success(self) -> None:
        """Batch update tags uses PATCH method."""
        response_data = {
            "updated": ["ds-1", "ds-2"],
            "not_found": [],
            "total_updated": 2,
        }
        responses.add(
            responses.PATCH,
            "http://localhost:8100/v1/datasets/batch-tags",
            json=response_data,
            status=200,
        )

        client = JuniperDataClient()
        result = client.batch_update_tags(
            dataset_ids=["ds-1", "ds-2"],
            add_tags=["experiment-1"],
            remove_tags=["draft"],
        )

        assert result["total_updated"] == 2
        assert result["updated"] == ["ds-1", "ds-2"]

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["dataset_ids"] == ["ds-1", "ds-2"]
        assert request_body["add_tags"] == ["experiment-1"]
        assert request_body["remove_tags"] == ["draft"]
        assert responses.calls[0].request.method == "PATCH"

    @responses.activate
    def test_batch_export_success(self) -> None:
        """Batch export returns raw ZIP bytes."""
        fake_zip_content = b"PK\x03\x04fake-zip-archive-content"
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets/batch-export",
            body=fake_zip_content,
            status=200,
            content_type="application/zip",
        )

        client = JuniperDataClient()
        result = client.batch_export(["ds-1", "ds-2"])

        assert isinstance(result, bytes)
        assert result == fake_zip_content

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["dataset_ids"] == ["ds-1", "ds-2"]


@pytest.mark.unit
class TestVersioning:
    """Tests for dataset versioning endpoints."""

    @responses.activate
    def test_list_versions_success(self) -> None:
        """List all versions of a named dataset."""
        response_data = {
            "dataset_name": "my-spiral",
            "versions": [
                {"version": 1, "dataset_id": "ds-v1", "created_at": "2026-01-01T00:00:00Z"},
                {"version": 2, "dataset_id": "ds-v2", "created_at": "2026-02-01T00:00:00Z"},
            ],
            "total": 2,
            "latest_version": 2,
        }
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/versions",
            json=response_data,
            status=200,
        )

        client = JuniperDataClient()
        result = client.list_versions("my-spiral")

        assert result["dataset_name"] == "my-spiral"
        assert result["total"] == 2
        assert result["latest_version"] == 2
        assert len(result["versions"]) == 2

    @responses.activate
    def test_get_latest_success(self) -> None:
        """Get the latest version of a named dataset."""
        response_data = {
            "dataset_id": "ds-v2",
            "name": "my-spiral",
            "version": 2,
            "generator": "spiral",
            "created_at": "2026-02-01T00:00:00Z",
        }
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/latest",
            json=response_data,
            status=200,
        )

        client = JuniperDataClient()
        result = client.get_latest("my-spiral")

        assert result["dataset_id"] == "ds-v2"
        assert result["name"] == "my-spiral"
        assert result["version"] == 2

    @responses.activate
    def test_list_versions_not_found(self) -> None:
        """List versions for nonexistent name raises not found."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/versions",
            json={"detail": "No datasets found with name: nonexistent"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.list_versions("nonexistent")

    @responses.activate
    def test_get_latest_not_found(self) -> None:
        """Get latest for nonexistent name raises not found."""
        responses.add(
            responses.GET,
            "http://localhost:8100/v1/datasets/latest",
            json={"detail": "No datasets found with name: nonexistent"},
            status=404,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataNotFoundError):
            client.get_latest("nonexistent")


@pytest.mark.unit
class TestExceptionContext:
    """Exceptions must carry machine-readable context, not just prose.

    Regression coverage for defect-register ``APD-DCLIENT-001`` (no
    ``status_code`` / ``detail`` / ``response``, so a 400 and a 422 raised the
    same type with the same text) and ``APD-DCLIENT-003`` (a FastAPI 422
    ``detail`` LIST was f-string-interpolated into an unparseable Python repr).
    """

    #: A real FastAPI 422 body: ``detail`` is a list of error objects.
    FASTAPI_422_DETAIL = [
        {"type": "missing", "loc": ["body", "seed"], "msg": "Field required"},
        {"type": "int_parsing", "loc": ["body", "n_spirals"], "msg": "Input should be a valid integer"},
    ]

    @responses.activate
    def test_status_code_separates_400_from_422(self) -> None:
        """The whole point of APD-DCLIENT-001: these were byte-identical before."""
        responses.add(responses.POST, "http://localhost:8100/v1/datasets", json={"detail": "bad"}, status=400)
        responses.add(responses.POST, "http://localhost:8100/v1/datasets", json={"detail": "bad"}, status=422)

        client = JuniperDataClient()

        with pytest.raises(JuniperDataValidationError) as first:
            client.create_dataset("spiral", {})
        with pytest.raises(JuniperDataValidationError) as second:
            client.create_dataset("spiral", {})

        assert {first.value.status_code, second.value.status_code} == {400, 422}

    @responses.activate
    def test_422_detail_list_is_preserved_as_structure(self) -> None:
        """APD-DCLIENT-003: the caller gets the list, not a repr of it."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": self.FASTAPI_422_DETAIL},
            status=422,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            client.create_dataset("spiral", {})

        assert exc_info.value.detail == self.FASTAPI_422_DETAIL
        assert exc_info.value.detail[0]["loc"] == ["body", "seed"]

    @responses.activate
    def test_422_message_is_readable_not_a_python_repr(self) -> None:
        """The message renders the same content without the repr punctuation."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": self.FASTAPI_422_DETAIL},
            status=422,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            client.create_dataset("spiral", {})

        message = str(exc_info.value)
        assert "body.seed: Field required" in message
        assert "body.n_spirals: Input should be a valid integer" in message
        # The old behaviour interpolated the list itself; these are its
        # fingerprints and none of them should survive into the message.
        assert "'type':" not in message
        assert "[{" not in message

    @responses.activate
    def test_response_is_attached_for_header_access(self) -> None:
        """``response`` lets a caller reach headers the message cannot carry."""
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": "nope"},
            status=400,
            headers={"X-Request-ID": "abc123"},
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            client.create_dataset("spiral", {})

        assert exc_info.value.response is not None
        assert exc_info.value.response.headers["X-Request-ID"] == "abc123"

    @responses.activate
    def test_not_found_and_generic_errors_also_carry_status(self) -> None:
        """Every response-derived branch populates the context, not just 400/422.

        The generic arm deliberately uses **409**, not a 5xx. Everything in
        ``RETRYABLE_STATUS_CODES`` (429/500/502/503/504) is consumed by the
        urllib3 ``Retry`` adapter, which exhausts and raises a
        ``RequestException`` -- so those never reach the response-handling
        branch at all, and the resulting error legitimately has no single
        authoritative status to report.
        """
        responses.add(responses.GET, "http://localhost:8100/v1/datasets/missing", json={"detail": "gone"}, status=404)
        responses.add(responses.GET, "http://localhost:8100/v1/datasets/boom", json={"detail": "kaboom"}, status=409)

        client = JuniperDataClient()

        with pytest.raises(JuniperDataNotFoundError) as not_found:
            client.get_dataset_metadata("missing")
        assert not_found.value.status_code == 404

        with pytest.raises(JuniperDataClientError) as generic:
            client.get_dataset_metadata("boom")
        assert generic.value.status_code == 409

    def test_locally_raised_errors_have_no_status_code(self) -> None:
        """Backward compatibility: no HTTP response means the fields stay None.

        Configuration and connection failures are raised before any response
        exists, so a caller must not read ``status_code`` as "0" or crash on a
        missing attribute -- it is simply ``None``.
        """
        error = JuniperDataClientError("something local went wrong")

        assert error.status_code is None
        assert error.detail is None
        assert error.response is None
        assert str(error) == "something local went wrong"

    def test_positional_message_construction_still_works(self) -> None:
        """Backward compatibility: the added parameters are keyword-only.

        Consumers (and this library's own fake) construct these with a single
        positional message. That must keep working, or adding context to the
        hierarchy would be a breaking change for every downstream caller.
        """
        for factory in (JuniperDataClientError, JuniperDataNotFoundError, JuniperDataValidationError):
            error = factory("plain message")
            assert str(error) == "plain message"
            assert error.status_code is None

    def test_context_survives_pickle_and_copy(self) -> None:
        """A round-trip must not silently drop the context (flake8-bugbear B042).

        ``BaseException.__reduce__`` returns ``(cls, args, self.__dict__)``
        whenever the instance dict is non-empty, so the keyword-only context
        survives on the default path — but only while ``args`` stays exactly
        the constructor's positional parameters. This test pins that
        invariant against an ``__init__``/``args`` refactor: forwarding the
        keyword-only extras into ``super().__init__`` (B042's own remedy)
        puts them in ``args``, and the rebuild's ``cls(*args)`` then raises
        ``TypeError``. Exceptions cross process boundaries here whenever a worker
        returns a failure to its parent, so this is a real path, not a
        theoretical one.
        """
        import copy as copy_module

        # Bandit blacklists ``pickle`` (B403/B301) because deserializing
        # UNTRUSTED data executes arbitrary code (CWE-502). Nothing untrusted is
        # involved here: the payload is produced by ``pickle.dumps`` below, in
        # this process, from an exception this test just constructed. The
        # round-trip IS the assertion, so the import cannot be dropped without
        # losing the coverage that pins ``__reduce__``.
        #
        # The suppressions are the trailing inline markers only. A comment line
        # that *begins* with the marker word is itself parsed as a directive,
        # and bandit then reads the following prose as test IDs.
        import pickle  # nosec B403

        original = JuniperDataValidationError(
            "Validation error (422): body.seed: Field required",
            status_code=422,
            detail=[{"loc": ["body", "seed"], "msg": "Field required"}],
        )

        # Same reasoning as the import: the bytes come from the ``dumps`` in
        # this very expression, never from a caller, a file, or the network.
        round_tripped = pickle.loads(pickle.dumps(original))  # nosec B301

        for rebuilt in (round_tripped, copy_module.copy(original), copy_module.deepcopy(original)):
            assert isinstance(rebuilt, JuniperDataValidationError)
            assert rebuilt.status_code == 422
            assert rebuilt.detail == [{"loc": ["body", "seed"], "msg": "Field required"}]
            assert str(rebuilt) == str(original)


@pytest.mark.unit
class TestRenderErrorDetailDegenerateShapes:
    """APD-DCLIENT-003 follow-up: unexpected 422 shapes must not crash the renderer.

    Well-formed FastAPI lists are pinned in ``TestExceptionContext``. These arms
    cover the helper's remaining branches, which every HTTP error path shares —
    a proxy or a re-raised ``RequestValidationError.errors()`` payload can hit
    them, and an ``AttributeError``/``TypeError`` inside ``_request`` would
    leak out of the typed hierarchy.
    """

    def test_non_dict_list_items_are_stringified(self) -> None:
        assert _render_error_detail(["plain", 3]) == "plain; 3"

    def test_missing_loc_uses_msg(self) -> None:
        assert _render_error_detail([{"msg": "Field required"}]) == "Field required"

    def test_loc_as_tuple_joins_like_a_list(self) -> None:
        """FastAPI's ``loc`` is a tuple on the Python object before JSON encoding."""
        assert _render_error_detail([{"loc": ("body", "seed"), "msg": "Field required"}]) == "body.seed: Field required"

    def test_empty_list_stays_visible(self) -> None:
        # An empty list is a degenerate but legal payload; collapsing it to
        # "" would make ``Validation error (422): `` look like a missing body.
        assert _render_error_detail([]) == "[]"

    @responses.activate
    def test_mixed_422_list_does_not_crash_the_request_path(self) -> None:
        """A dict + a raw string in ``detail`` still raises the typed 422.

        The list stays on ``exc.detail`` unmodified; the message renders both
        arms. This is the blast-radius pin: ``_request`` must not assume every
        list item is a mapping with ``loc``/``msg``.
        """
        mixed = [
            {"type": "missing", "loc": ["body", "seed"], "msg": "Field required"},
            "not a dict",
        ]
        responses.add(
            responses.POST,
            "http://localhost:8100/v1/datasets",
            json={"detail": mixed},
            status=422,
        )

        client = JuniperDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            client.create_dataset("spiral", {})

        assert exc_info.value.detail == mixed
        assert exc_info.value.status_code == 422
        message = str(exc_info.value)
        assert "body.seed: Field required" in message
        assert "not a dict" in message


@pytest.mark.unit
class TestFakeClientMatchesRealExceptionContext:
    """``FakeDataClient`` is documented as a drop-in replacement, so it must
    populate the same context the real client does.

    A double that raises the right *type* with ``status_code=None`` lets a
    consumer's test pass against behaviour production does not have -- the
    masked-seam failure mode. These pin the two statuses that differ.
    """

    def test_fake_not_found_carries_404(self) -> None:
        from juniper_data_client.testing import FakeDataClient

        fake = FakeDataClient()
        with pytest.raises(JuniperDataNotFoundError) as exc_info:
            fake.get_dataset_metadata("no-such-dataset")

        assert exc_info.value.status_code == 404

    def test_fake_unknown_generator_carries_400_like_the_service(self) -> None:
        """juniper-data raises an explicit HTTPException(400) for this."""
        from juniper_data_client.testing import FakeDataClient

        fake = FakeDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            fake.create_dataset("no-such-generator", {})

        assert exc_info.value.status_code == 400

    def test_fake_ttl_violation_carries_422_like_pydantic(self) -> None:
        """``ttl_seconds`` is a pydantic ``Field(ge=1)``, which FastAPI answers 422."""
        from juniper_data_client.testing import FakeDataClient

        fake = FakeDataClient()
        with pytest.raises(JuniperDataValidationError) as exc_info:
            fake.create_dataset("spiral", {}, ttl_seconds=0)

        assert exc_info.value.status_code == 422

    def test_fake_unknown_schema_carries_404(self) -> None:
        """``get_generator_schema`` 404 must carry status_code like the real client."""
        from juniper_data_client.testing import FakeDataClient

        fake = FakeDataClient()
        with pytest.raises(JuniperDataNotFoundError) as exc_info:
            fake.get_generator_schema("no-such-generator")

        assert exc_info.value.status_code == 404
