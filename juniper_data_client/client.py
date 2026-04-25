"""REST API client for JuniperData service integration.

Provides dataset creation, artifact download, and preview functionality
for JuniperCascor and juniper-canopy applications.
"""

import io
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from juniper_data_client.constants import (
    API_KEY_ENV_VAR,
    API_KEY_HEADER_NAME,
    API_VERSION_PATH_SUFFIX,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BASE_URL,
    DEFAULT_LIST_LIMIT,
    DEFAULT_LIST_OFFSET,
    DEFAULT_PREVIEW_N,
    DEFAULT_READY_POLL_INTERVAL,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_URL_SCHEME_PREFIX,
    ENDPOINT_BATCH_CREATE,
    ENDPOINT_BATCH_DELETE,
    ENDPOINT_BATCH_EXPORT,
    ENDPOINT_BATCH_TAGS,
    ENDPOINT_DATASET_ARTIFACT_TEMPLATE,
    ENDPOINT_DATASET_BY_ID_TEMPLATE,
    ENDPOINT_DATASET_PREVIEW_TEMPLATE,
    ENDPOINT_DATASETS,
    ENDPOINT_DATASETS_LATEST,
    ENDPOINT_DATASETS_VERSIONS,
    ENDPOINT_GENERATOR_SCHEMA_TEMPLATE,
    ENDPOINT_GENERATORS,
    ENDPOINT_HEALTH,
    ENDPOINT_HEALTH_READY,
    GENERATOR_SPIRAL,
    HEALTH_READY_STATUS,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_MAXSIZE,
    RETRY_ALLOWED_METHODS,
    RETRYABLE_STATUS_CODES,
    SPIRAL_ALGORITHM_DEFAULT,
    SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
    SPIRAL_N_SPIRALS_DEFAULT,
    SPIRAL_NOISE_DEFAULT,
    SPIRAL_TRAIN_RATIO_DEFAULT,
    URL_SCHEME_PREFIXES,
)
from juniper_data_client.exceptions import JuniperDataClientError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError


class JuniperDataClient:
    """Client for interacting with the JuniperData REST API.

    Provides methods for dataset creation, artifact retrieval, and service health
    checking with automatic retry logic and connection pooling.

    Example:
        >>> client = JuniperDataClient("http://localhost:8100")
        >>> result = client.create_dataset("spiral", {"n_spirals": 2, "seed": 42})
        >>> arrays = client.download_artifact_npz(result["dataset_id"])
        >>> X_train, y_train = arrays["X_train"], arrays["y_train"]
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        api_key: Optional[str] = None,
    ):
        """Initialize the JuniperData client.

        Args:
            base_url: Base URL for the JuniperData API (default: http://localhost:8100)
            timeout: Request timeout in seconds (default: 30)
            retries: Number of retry attempts for failed requests (default: 3)
            backoff_factor: Backoff factor for retry delays (default: 0.5)
            api_key: API key for authentication. If not provided, reads from
                JUNIPER_DATA_API_KEY environment variable.
        """
        self.base_url = self._normalize_url(base_url)
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()

        resolved_api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if resolved_api_key:
            self.session.headers[API_KEY_HEADER_NAME] = resolved_api_key

    def _normalize_url(self, url: str) -> str:
        """Normalize the base URL for consistent API calls.

        Args:
            url: Raw URL string to normalize

        Returns:
            Normalized URL with scheme, no trailing slash, no /v1 suffix
        """
        url = url.strip()

        if not url.startswith(URL_SCHEME_PREFIXES):
            url = f"{DEFAULT_URL_SCHEME_PREFIX}{url}"

        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        normalized = normalized.rstrip("/")

        if normalized.endswith(API_VERSION_PATH_SUFFIX):
            normalized = normalized[: -len(API_VERSION_PATH_SUFFIX)]

        return normalized

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic and connection pooling.

        Returns:
            Configured requests.Session with retry adapter
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=RETRYABLE_STATUS_CODES,
            allowed_methods=RETRY_ALLOWED_METHODS,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE,
        )

        for scheme in URL_SCHEME_PREFIXES:
            session.mount(scheme, adapter)

        return session

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Make an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object

        Raises:
            JuniperDataConnectionError: On connection failure
            JuniperDataTimeoutError: On request timeout
            JuniperDataNotFoundError: On 404 response
            JuniperDataValidationError: On 400/422 response
            JuniperDataClientError: On other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        try:
            response = self.session.request(method, url, **kwargs)
        except requests.exceptions.ConnectionError as e:
            raise JuniperDataConnectionError(f"Failed to connect to JuniperData at {self.base_url}: {e}") from e
        except requests.exceptions.Timeout as e:
            raise JuniperDataTimeoutError(f"Request to {url} timed out after {self.timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            raise JuniperDataClientError(f"Request failed: {e}") from e

        if response.ok:
            return response

        error_detail = response.text
        try:
            error_json = response.json()
            if "detail" in error_json:
                error_detail = error_json["detail"]
        except (ValueError, KeyError):
            # If the response body is not valid JSON or lacks a 'detail' field,
            # fall back to using the raw response text as the error detail.
            error_detail = response.text

        if response.status_code == HTTP_404_NOT_FOUND:
            raise JuniperDataNotFoundError(f"Resource not found: {error_detail}")
        elif response.status_code in (HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY):
            raise JuniperDataValidationError(f"Validation error: {error_detail}")
        else:
            raise JuniperDataClientError(f"Request failed ({response.status_code}): {error_detail}")

    @staticmethod
    def _parse_json(response: requests.Response) -> Any:
        """Parse a successful response body as JSON.

        ERR-01 (Phase 4C): callers previously invoked ``response.json()`` directly,
        which raises ``requests.exceptions.JSONDecodeError`` (a ``ValueError`` subclass)
        on malformed bodies. Surface a typed ``JuniperDataClientError`` instead so the
        public API never leaks a raw ``json.JSONDecodeError``.
        """
        try:
            return response.json()
        except ValueError as e:
            preview = (response.text or "")[:200]
            raise JuniperDataClientError(f"Malformed JSON response from {response.url}: {e}: {preview!r}") from e

    def health_check(self) -> Dict[str, Any]:
        """Check if the JuniperData service is healthy.

        Returns:
            Health status response from the service

        Raises:
            JuniperDataConnectionError: If service is unreachable
        """
        response = self._request("GET", ENDPOINT_HEALTH)
        return self._parse_json(response)

    def is_ready(self) -> bool:
        """Check if the JuniperData service is ready to accept requests.

        Returns:
            True if service is ready, False otherwise
        """
        try:
            response = self._request("GET", ENDPOINT_HEALTH_READY)
            return self._parse_json(response).get("status") == HEALTH_READY_STATUS
        except JuniperDataClientError:
            return False

    def wait_for_ready(self, timeout: float = DEFAULT_READY_TIMEOUT, poll_interval: float = DEFAULT_READY_POLL_INTERVAL) -> bool:
        """Wait for the JuniperData service to become ready.

        Args:
            timeout: Maximum time to wait in seconds (default: 30)
            poll_interval: Time between readiness checks in seconds (default: 0.5)

        Returns:
            True if service became ready within timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                return True
            time.sleep(poll_interval)
        return False

    def list_generators(self) -> List[Dict[str, Any]]:
        """List available dataset generators.

        Returns:
            List of generator information dictionaries
        """
        response = self._request("GET", ENDPOINT_GENERATORS)
        return self._parse_json(response)

    def get_generator_schema(self, name: str) -> Dict[str, Any]:
        """Get the parameter schema for a generator.

        Args:
            name: Generator name (e.g., "spiral")

        Returns:
            JSON schema for generator parameters

        Raises:
            JuniperDataNotFoundError: If generator not found
        """
        response = self._request("GET", ENDPOINT_GENERATOR_SCHEMA_TEMPLATE.format(name=name))
        return self._parse_json(response)

    def create_dataset(
        self,
        generator: str,
        params: Dict[str, Any],
        persist: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        parent_dataset_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new dataset via the JuniperData API.

        If a dataset with the same parameters already exists, the existing
        dataset is returned (caching behavior).

        Args:
            generator: Name of the dataset generator to use (e.g., "spiral")
            params: Parameters to pass to the generator
            persist: Whether to persist the dataset (default: True)
            name: Optional dataset name for versioning. When provided, the service
                automatically assigns an incrementing version number.
            description: Optional human-readable description of the dataset.
            created_by: Optional identifier for the creator (user or system).
            parent_dataset_id: Optional ID of the parent dataset this was derived from.
            tags: Optional list of tag strings to attach to the dataset. Forwarded
                to the server's ``CreateDatasetRequest.tags`` field. Server-side
                tags are searchable via ``list_datasets`` filters.
            ttl_seconds: Optional time-to-live in seconds (must be positive when
                provided). After the TTL elapses the server is free to expire the
                dataset via the cleanup-expired route. Mirrors
                ``CreateDatasetRequest.ttl_seconds``.

        Returns:
            Parsed JSON response containing dataset_id, generator, meta, and artifact_url

        Raises:
            JuniperDataValidationError: If parameters are invalid
            JuniperDataNotFoundError: If generator not found
        """
        payload: Dict[str, Any] = {
            "generator": generator,
            "params": params,
            "persist": persist,
        }
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if created_by is not None:
            payload["created_by"] = created_by
        if parent_dataset_id is not None:
            payload["parent_dataset_id"] = parent_dataset_id
        # XREPO-09 (2026-04-24): previously dropped by the client; the
        # server's CreateDatasetRequest has accepted these fields since
        # juniper-data v0.6.0.
        if tags is not None:
            payload["tags"] = list(tags)
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds

        response = self._request("POST", ENDPOINT_DATASETS, json=payload)
        return self._parse_json(response)

    def list_versions(self, name: str) -> Dict[str, Any]:
        """List all versions of a named dataset.

        Args:
            name: Dataset name to list versions for.

        Returns:
            Dict with dataset_name, versions list, total count, and latest_version.
        """
        response = self._request("GET", ENDPOINT_DATASETS_VERSIONS, params={"name": name})
        return self._parse_json(response)

    def get_latest(self, name: str) -> Dict[str, Any]:
        """Get the latest version of a named dataset.

        Args:
            name: Dataset name to get latest version of.

        Returns:
            Dataset metadata for the latest version.

        Raises:
            JuniperDataNotFoundError: If no versions exist for the given name.
        """
        response = self._request("GET", ENDPOINT_DATASETS_LATEST, params={"name": name})
        return self._parse_json(response)

    def list_datasets(self, limit: int = DEFAULT_LIST_LIMIT, offset: int = DEFAULT_LIST_OFFSET) -> List[str]:
        """List dataset IDs.

        Args:
            limit: Maximum number of dataset IDs to return (default: 100)
            offset: Number of dataset IDs to skip (default: 0)

        Returns:
            List of dataset ID strings
        """
        response = self._request("GET", ENDPOINT_DATASETS, params={"limit": limit, "offset": offset})
        return self._parse_json(response)

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset.

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            Dataset metadata dictionary

        Raises:
            JuniperDataNotFoundError: If dataset not found
        """
        response = self._request("GET", ENDPOINT_DATASET_BY_ID_TEMPLATE.format(dataset_id=dataset_id))
        return self._parse_json(response)

    def download_artifact_bytes(self, dataset_id: str) -> bytes:
        """Download the raw NPZ artifact bytes for a dataset.

        Args:
            dataset_id: ID of the dataset whose artifact to download

        Returns:
            Raw bytes of the NPZ file

        Raises:
            JuniperDataNotFoundError: If dataset not found
        """
        response = self._request("GET", ENDPOINT_DATASET_ARTIFACT_TEMPLATE.format(dataset_id=dataset_id))
        return response.content

    def download_artifact_npz(self, dataset_id: str) -> Dict[str, np.ndarray]:
        """Download and load an NPZ artifact for a dataset.

        The returned dictionary contains numpy arrays with the standard keys:
        - X_train, y_train: Training features and one-hot labels
        - X_test, y_test: Test features and one-hot labels
        - X_full, y_full: Full dataset features and one-hot labels

        All arrays are float32 dtype.

        Args:
            dataset_id: ID of the dataset whose artifact to download

        Returns:
            Dictionary mapping array names to numpy arrays

        Raises:
            JuniperDataNotFoundError: If dataset not found
        """
        content = self.download_artifact_bytes(dataset_id)
        npz_file = np.load(io.BytesIO(content))
        return {key: npz_file[key] for key in npz_file.files}

    def get_preview(self, dataset_id: str, n: int = DEFAULT_PREVIEW_N) -> Dict[str, Any]:
        """Get a preview of dataset samples as JSON.

        Args:
            dataset_id: ID of the dataset to preview
            n: Number of samples to include in preview (default: 100, max: 1000)

        Returns:
            Dictionary containing n_samples, X_sample, and y_sample

        Raises:
            JuniperDataNotFoundError: If dataset not found
        """
        response = self._request("GET", ENDPOINT_DATASET_PREVIEW_TEMPLATE.format(dataset_id=dataset_id), params={"n": n})
        return self._parse_json(response)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset.

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            True if dataset was deleted

        Raises:
            JuniperDataNotFoundError: If dataset not found
        """
        self._request("DELETE", ENDPOINT_DATASET_BY_ID_TEMPLATE.format(dataset_id=dataset_id))
        return True

    def create_spiral_dataset(
        self,
        n_spirals: int = SPIRAL_N_SPIRALS_DEFAULT,
        n_points_per_spiral: int = SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
        noise: float = SPIRAL_NOISE_DEFAULT,
        seed: Optional[int] = None,
        algorithm: str = SPIRAL_ALGORITHM_DEFAULT,
        train_ratio: float = SPIRAL_TRAIN_RATIO_DEFAULT,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience method for creating spiral datasets.

        Args:
            n_spirals: Number of spiral arms (default: 2)
            n_points_per_spiral: Points per spiral arm (default: 100)
            noise: Noise level (default: 0.1)
            seed: Random seed for reproducibility (optional)
            algorithm: Generation algorithm - "modern" or "legacy_cascor" (default: "modern")
            train_ratio: Fraction of data for training (default: 0.8)
            **kwargs: Additional parameters passed to the generator

        Returns:
            Dataset creation response with dataset_id and metadata
        """
        params: Dict[str, Any] = {
            "n_spirals": n_spirals,
            "n_points_per_spiral": n_points_per_spiral,
            "noise": noise,
            "algorithm": algorithm,
            "train_ratio": train_ratio,
        }
        if seed is not None:
            params["seed"] = seed
        params.update(kwargs)

        return self.create_dataset(GENERATOR_SPIRAL, params)

    # ─── Batch Operations ────────────────────────────────────────────────

    def batch_delete(self, dataset_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple datasets in a single request.

        Args:
            dataset_ids: List of dataset IDs to delete (1-100).

        Returns:
            Dictionary with deleted, not_found, and total_deleted.
        """
        response = self._request("POST", ENDPOINT_BATCH_DELETE, json={"dataset_ids": dataset_ids})
        return self._parse_json(response)

    def batch_create(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple datasets in a single request.

        Each item should have 'generator' and 'params' keys, and optionally
        'persist', 'tags', and 'ttl_seconds'.

        Args:
            datasets: List of dataset specifications (1-50).

        Returns:
            Dictionary with results, total_created, and total_failed.
        """
        response = self._request("POST", ENDPOINT_BATCH_CREATE, json={"datasets": datasets})
        return self._parse_json(response)

    def batch_update_tags(
        self,
        dataset_ids: List[str],
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add or remove tags from multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to update (1-100).
            add_tags: Tags to add to all specified datasets.
            remove_tags: Tags to remove from all specified datasets.

        Returns:
            Dictionary with updated, not_found, and total_updated.
        """
        payload: Dict[str, Any] = {"dataset_ids": dataset_ids}
        if add_tags:
            payload["add_tags"] = add_tags
        if remove_tags:
            payload["remove_tags"] = remove_tags
        response = self._request("PATCH", ENDPOINT_BATCH_TAGS, json=payload)
        return self._parse_json(response)

    def batch_export(self, dataset_ids: List[str]) -> bytes:
        """Export multiple datasets as a ZIP archive of NPZ files.

        Args:
            dataset_ids: List of dataset IDs to export (1-50).

        Returns:
            Raw bytes of the ZIP archive.

        Raises:
            JuniperDataNotFoundError: If none of the datasets exist.
        """
        response = self._request("POST", ENDPOINT_BATCH_EXPORT, json={"dataset_ids": dataset_ids})
        return response.content

    def close(self) -> None:
        """Close the client session and release resources."""
        self.session.close()

    def __enter__(self) -> "JuniperDataClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - closes the session."""
        self.close()
