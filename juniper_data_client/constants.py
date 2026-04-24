"""Constants for the JuniperData REST client and testing utilities.

Centralizes all hardcoded literals used by ``client.py``,
``testing/fake_client.py``, and ``testing/generators.py`` so that consumers
can override them and so that protocol-level identifiers (endpoint paths,
generator names, header names) are discoverable in one place.

Project: Juniper
Sub-Project: juniper-data-client
Application: JuniperDataClient
Author: Paul Calnon
Version: 0.4.0
License: MIT License
"""

from typing import List, Tuple

# ─── Service Configuration ───────────────────────────────────────────────────

DEFAULT_BASE_URL: str = "http://localhost:8100"
FAKE_BASE_URL: str = "http://fake-data:8100"

# ─── HTTP Configuration ──────────────────────────────────────────────────────

DEFAULT_TIMEOUT: int = 30
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
RETRYABLE_STATUS_CODES: List[int] = [429, 500, 502, 503, 504]
# XREPO-11 (2026-04-24): auto-retry is now restricted to idempotent
# HTTP methods per RFC 9110 §9.2.2. POST, PATCH and DELETE were
# previously included, which could cause duplicate dataset creation
# (on POST) or repeated side-effects (on DELETE) when transient 5xx
# responses retried a request that had already been applied
# server-side. Callers that need retry for mutations must implement
# their own idempotency layer (e.g., use client-supplied dataset
# names so POST collapses server-side via the existing dedupe path).
RETRY_ALLOWED_METHODS: List[str] = ["HEAD", "GET", "PUT"]
HTTP_POOL_CONNECTIONS: int = 10
HTTP_POOL_MAXSIZE: int = 10

# URL normalization helpers used by ``JuniperDataClient._normalize_url``.
URL_SCHEME_PREFIXES: Tuple[str, ...] = ("http://", "https://")
DEFAULT_URL_SCHEME_PREFIX: str = "http://"
API_VERSION_PATH_SUFFIX: str = "/v1"

# ─── Readiness Polling ───────────────────────────────────────────────────────

DEFAULT_READY_TIMEOUT: float = 30.0
DEFAULT_READY_POLL_INTERVAL: float = 0.5
HEALTH_READY_STATUS: str = "ready"

# ─── Authentication ──────────────────────────────────────────────────────────

API_KEY_HEADER_NAME: str = "X-API-Key"
API_KEY_ENV_VAR: str = "JUNIPER_DATA_API_KEY"

# ─── REST Endpoints ──────────────────────────────────────────────────────────

ENDPOINT_HEALTH: str = "/v1/health"
ENDPOINT_HEALTH_READY: str = "/v1/health/ready"
ENDPOINT_GENERATORS: str = "/v1/generators"
ENDPOINT_GENERATOR_SCHEMA_TEMPLATE: str = "/v1/generators/{name}/schema"
ENDPOINT_DATASETS: str = "/v1/datasets"
ENDPOINT_DATASETS_VERSIONS: str = "/v1/datasets/versions"
ENDPOINT_DATASETS_LATEST: str = "/v1/datasets/latest"
ENDPOINT_DATASET_BY_ID_TEMPLATE: str = "/v1/datasets/{dataset_id}"
ENDPOINT_DATASET_ARTIFACT_TEMPLATE: str = "/v1/datasets/{dataset_id}/artifact"
ENDPOINT_DATASET_PREVIEW_TEMPLATE: str = "/v1/datasets/{dataset_id}/preview"
ENDPOINT_BATCH_CREATE: str = "/v1/datasets/batch-create"
ENDPOINT_BATCH_DELETE: str = "/v1/datasets/batch-delete"
ENDPOINT_BATCH_TAGS: str = "/v1/datasets/batch-tags"
ENDPOINT_BATCH_EXPORT: str = "/v1/datasets/batch-export"

# ─── HTTP Status Codes ───────────────────────────────────────────────────────

HTTP_400_BAD_REQUEST: int = 400
HTTP_404_NOT_FOUND: int = 404
HTTP_422_UNPROCESSABLE_ENTITY: int = 422

# ─── Listing Defaults ────────────────────────────────────────────────────────

DEFAULT_LIST_LIMIT: int = 100
DEFAULT_LIST_OFFSET: int = 0

# ─── Preview Defaults ────────────────────────────────────────────────────────

DEFAULT_PREVIEW_N: int = 100
MAX_PREVIEW_N: int = 1000

# ─── Data Type Contract ──────────────────────────────────────────────────────

DEFAULT_ARRAY_DTYPE: str = "float32"

# ─── Generator Names ─────────────────────────────────────────────────────────

# Generator identifiers MUST match the keys in the server-side
# ``GENERATOR_REGISTRY`` (juniper_data/api/routes/generators.py). The
# parity test ``tests/test_generator_parity.py`` enforces this invariant.
# DC-01/XREPO-01 fix (2026-04-24): ``GENERATOR_CIRCLE`` changed from
# ``"circle"`` to ``"circles"`` to match the server. The legacy
# ``GENERATOR_CIRCLE_LEGACY`` alias is retained for one release cycle so
# downstream callers have time to migrate.
GENERATOR_SPIRAL: str = "spiral"
GENERATOR_XOR: str = "xor"
GENERATOR_CIRCLE: str = "circles"
GENERATOR_CIRCLE_LEGACY: str = "circle"  # deprecated — use GENERATOR_CIRCLE
GENERATOR_MOON: str = "moon"
# DC-03/XREPO-01c (2026-04-24): added constants for the 5 server
# generators the client previously lacked, so downstream code can avoid
# hardcoding string literals.
GENERATOR_GAUSSIAN: str = "gaussian"
GENERATOR_CHECKERBOARD: str = "checkerboard"
GENERATOR_CSV_IMPORT: str = "csv_import"
GENERATOR_MNIST: str = "mnist"
GENERATOR_ARC_AGI: str = "arc_agi"

# ─── Generator Catalog Metadata ──────────────────────────────────────────────

GENERATOR_VERSION: str = "1.0.0"
GENERATOR_DESCRIPTION_SPIRAL: str = "Multi-arm Archimedean spiral dataset"
GENERATOR_DESCRIPTION_XOR: str = "XOR classification dataset with four corner clusters"
GENERATOR_DESCRIPTION_CIRCLE: str = "Concentric circles classification dataset"
GENERATOR_DESCRIPTION_MOON: str = "Two interleaving half-moon classification dataset"
GENERATOR_DESCRIPTION_GAUSSIAN: str = "Gaussian blobs classification dataset"
GENERATOR_DESCRIPTION_CHECKERBOARD: str = "Checkerboard pattern classification dataset"
GENERATOR_DESCRIPTION_CSV_IMPORT: str = "CSV/JSON import for custom datasets"
GENERATOR_DESCRIPTION_MNIST: str = "MNIST and Fashion-MNIST digit classification dataset"
GENERATOR_DESCRIPTION_ARC_AGI: str = "ARC-AGI visual reasoning tasks dataset"

# ─── Generator Defaults — Spiral ─────────────────────────────────────────────

SPIRAL_N_SPIRALS_DEFAULT: int = 2
SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT: int = 100
SPIRAL_NOISE_DEFAULT: float = 0.1
SPIRAL_ALGORITHM_DEFAULT: str = "modern"
SPIRAL_TRAIN_RATIO_DEFAULT: float = 0.8

# Spiral schema validation bounds.
SPIRAL_N_SPIRALS_MIN: int = 2
SPIRAL_N_POINTS_PER_SPIRAL_MIN: int = 10
SPIRAL_NOISE_MIN: float = 0.0
SPIRAL_TRAIN_RATIO_MIN: float = 0.1
SPIRAL_TRAIN_RATIO_MAX: float = 0.99

# Spiral algorithm enum values.
SPIRAL_ALGORITHM_MODERN: str = "modern"
SPIRAL_ALGORITHM_LEGACY: str = "legacy_cascor"
SPIRAL_ALGORITHMS: List[str] = ["modern", "legacy_cascor"]

# Spiral generation math (``radius = t * 5.0``, ``angle = t * 4.0 * pi``).
SPIRAL_RADIUS_SCALE: float = 5.0
SPIRAL_ANGLE_TURNS: float = 4.0

# ─── Generator Defaults — XOR ────────────────────────────────────────────────

XOR_N_POINTS_DEFAULT: int = 100
XOR_NOISE_DEFAULT: float = 0.1
XOR_TRAIN_RATIO_DEFAULT: float = 0.8

# XOR schema validation bounds.
XOR_N_POINTS_MIN: int = 4
XOR_NOISE_MIN: float = 0.0
XOR_TRAIN_RATIO_MIN: float = 0.1
XOR_TRAIN_RATIO_MAX: float = 0.99

# XOR corner coordinates and class labels.
XOR_CORNERS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)
XOR_CORNER_LABELS: Tuple[int, ...] = (0, 1, 1, 0)
XOR_NUM_CORNERS: int = 4
XOR_NUM_CLASSES: int = 2

# ─── Generator Defaults — Circle ─────────────────────────────────────────────

CIRCLE_N_POINTS_DEFAULT: int = 200
CIRCLE_NOISE_DEFAULT: float = 0.1
CIRCLE_FACTOR_DEFAULT: float = 0.5
CIRCLE_TRAIN_RATIO_DEFAULT: float = 0.8

# Circle schema validation bounds.
CIRCLE_N_POINTS_MIN: int = 10
CIRCLE_NOISE_MIN: float = 0.0
CIRCLE_FACTOR_MIN: float = 0.01
CIRCLE_FACTOR_MAX: float = 0.99
CIRCLE_TRAIN_RATIO_MIN: float = 0.1
CIRCLE_TRAIN_RATIO_MAX: float = 0.99
CIRCLE_NUM_CLASSES: int = 2

# ─── Generator Defaults — Moon ───────────────────────────────────────────────

MOON_N_POINTS_DEFAULT: int = 200
MOON_NOISE_DEFAULT: float = 0.1
MOON_TRAIN_RATIO_DEFAULT: float = 0.8

# Moon schema validation bounds.
MOON_N_POINTS_MIN: int = 10
MOON_NOISE_MIN: float = 0.0
MOON_TRAIN_RATIO_MIN: float = 0.1
MOON_TRAIN_RATIO_MAX: float = 0.99
MOON_NUM_CLASSES: int = 2

# Moon generation math
# (``lower_x = 1.0 - cos``, ``lower_y = 1.0 - sin - 0.5``).
MOON_LOWER_X_OFFSET: float = 1.0
MOON_LOWER_Y_OFFSET: float = 1.0
MOON_LOWER_Y_SHIFT: float = 0.5

# ─── Fake Service Identity ───────────────────────────────────────────────────

FAKE_SERVICE_STATUS: str = "ok"
FAKE_SERVICE_NAME: str = "juniper-data"
FAKE_SERVICE_VERSION: str = "fake"
FAKE_SERVICE_UPTIME_SECONDS: float = 0.0
