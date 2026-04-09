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
        """No X-API-Key header when API key is not provided."""
        monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)
        client = JuniperDataClient()
        assert "X-API-Key" not in client.session.headers


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
