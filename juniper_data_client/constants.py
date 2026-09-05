"""Constants for the JuniperData REST client and testing utilities.

Centralizes all hardcoded literals used by ``client.py``,
``testing/fake_client.py``, and ``testing/generators.py`` so that consumers
can override them and so that protocol-level identifiers (endpoint paths,
generator names, header names) are discoverable in one place.

Project: Juniper
Sub-Project: juniper-data-client
Application: JuniperDataClient
Author: Paul Calnon
Version: 0.4.2
License: MIT License
"""

from typing import Final, List, Literal, Tuple

# Complete export surface: every public module-level constant, in file order,
# grouped by the section headers below. Completeness is enforced by
# tests/test_constants_all_drift.py — a new constant that is not added here
# fails that gate (and, on any module WITH __all__, CodeQL flags touched
# non-exported globals as py/unused-global-variable, blocking the merge).
__all__ = [
    # Service Configuration
    "DEFAULT_BASE_URL",
    "FAKE_BASE_URL",
    # HTTP Configuration
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_BACKOFF_FACTOR",
    "DEFAULT_BACKOFF_JITTER",
    "RETRYABLE_STATUS_CODES",
    "RETRY_ALLOWED_METHODS",
    "HTTP_POOL_CONNECTIONS",
    "HTTP_POOL_MAXSIZE",
    "URL_SCHEME_PREFIXES",
    "DEFAULT_URL_SCHEME_PREFIX",
    "API_VERSION_PATH_SUFFIX",
    # Readiness Polling
    "DEFAULT_READY_TIMEOUT",
    "DEFAULT_READY_POLL_INTERVAL",
    "HEALTH_READY_STATUS",
    # Authentication
    "API_KEY_HEADER_NAME",
    "API_KEY_ENV_VAR",
    "API_KEY_FILE_ENV_VAR",
    # REST Endpoints
    "ENDPOINT_HEALTH",
    "ENDPOINT_HEALTH_READY",
    "ENDPOINT_GENERATORS",
    "ENDPOINT_GENERATOR_SCHEMA_TEMPLATE",
    "ENDPOINT_DATASETS",
    "ENDPOINT_DATASETS_VERSIONS",
    "ENDPOINT_DATASETS_LATEST",
    "ENDPOINT_DATASET_BY_ID_TEMPLATE",
    "ENDPOINT_DATASET_ARTIFACT_TEMPLATE",
    "ENDPOINT_DATASET_PREVIEW_TEMPLATE",
    "ENDPOINT_BATCH_CREATE",
    "ENDPOINT_BATCH_DELETE",
    "ENDPOINT_BATCH_TAGS",
    "ENDPOINT_BATCH_EXPORT",
    # HTTP Status Codes
    "HTTP_400_BAD_REQUEST",
    "HTTP_404_NOT_FOUND",
    "HTTP_422_UNPROCESSABLE_ENTITY",
    # Listing Defaults
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_LIST_OFFSET",
    # Preview Defaults
    "DEFAULT_PREVIEW_N",
    "MAX_PREVIEW_N",
    # Data Type Contract
    "DEFAULT_ARRAY_DTYPE",
    # Generator Names
    "GENERATOR_SPIRAL",
    "GENERATOR_XOR",
    "GENERATOR_CIRCLE",
    "GENERATOR_CIRCLE_LEGACY",
    "GENERATOR_MOON",
    "GENERATOR_GAUSSIAN",
    "GENERATOR_CHECKERBOARD",
    "GENERATOR_CSV_IMPORT",
    "GENERATOR_MNIST",
    "GENERATOR_ARC_AGI",
    "GENERATOR_EQUITIES",
    "GENERATOR_EQUITIES_SEQ",
    "GENERATOR_MULTI_SINE",
    "GENERATOR_MACKEY_GLASS",
    "GENERATOR_AR_P",
    "GENERATOR_IRREGULAR_SINE",
    "GENERATOR_DELAY_PRODUCT",
    # Generator Catalog Metadata
    "GENERATOR_VERSION",
    "GENERATOR_DESCRIPTION_SPIRAL",
    "GENERATOR_DESCRIPTION_XOR",
    "GENERATOR_DESCRIPTION_CIRCLE",
    "GENERATOR_DESCRIPTION_MOON",
    "GENERATOR_DESCRIPTION_GAUSSIAN",
    "GENERATOR_DESCRIPTION_CHECKERBOARD",
    "GENERATOR_DESCRIPTION_CSV_IMPORT",
    "GENERATOR_DESCRIPTION_MNIST",
    "GENERATOR_DESCRIPTION_ARC_AGI",
    "GENERATOR_DESCRIPTION_EQUITIES",
    "GENERATOR_DESCRIPTION_EQUITIES_SEQ",
    "GENERATOR_DESCRIPTION_MULTI_SINE",
    "GENERATOR_DESCRIPTION_MACKEY_GLASS",
    "GENERATOR_DESCRIPTION_AR_P",
    "GENERATOR_DESCRIPTION_IRREGULAR_SINE",
    "GENERATOR_DESCRIPTION_DELAY_PRODUCT",
    # Generator Defaults — Spiral
    "SPIRAL_N_SPIRALS_DEFAULT",
    "SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT",
    "SPIRAL_NOISE_DEFAULT",
    "SPIRAL_ALGORITHM_DEFAULT",
    "SPIRAL_TRAIN_RATIO_DEFAULT",
    "SPIRAL_N_SPIRALS_MIN",
    "SPIRAL_N_POINTS_PER_SPIRAL_MIN",
    "SPIRAL_NOISE_MIN",
    "SPIRAL_TRAIN_RATIO_MIN",
    "SPIRAL_TRAIN_RATIO_MAX",
    "SPIRAL_ALGORITHM_MODERN",
    "SPIRAL_ALGORITHM_LEGACY",
    "SPIRAL_ALGORITHMS",
    "SPIRAL_RADIUS_SCALE",
    "SPIRAL_ANGLE_TURNS",
    # Generator Defaults — XOR
    "XOR_N_POINTS_DEFAULT",
    "XOR_NOISE_DEFAULT",
    "XOR_TRAIN_RATIO_DEFAULT",
    "XOR_N_POINTS_MIN",
    "XOR_NOISE_MIN",
    "XOR_TRAIN_RATIO_MIN",
    "XOR_TRAIN_RATIO_MAX",
    "XOR_CORNERS",
    "XOR_CORNER_LABELS",
    "XOR_NUM_CORNERS",
    "XOR_NUM_CLASSES",
    # Generator Defaults — Circle
    "CIRCLE_N_POINTS_DEFAULT",
    "CIRCLE_NOISE_DEFAULT",
    "CIRCLE_FACTOR_DEFAULT",
    "CIRCLE_TRAIN_RATIO_DEFAULT",
    "CIRCLE_N_POINTS_MIN",
    "CIRCLE_NOISE_MIN",
    "CIRCLE_FACTOR_MIN",
    "CIRCLE_FACTOR_MAX",
    "CIRCLE_TRAIN_RATIO_MIN",
    "CIRCLE_TRAIN_RATIO_MAX",
    "CIRCLE_NUM_CLASSES",
    # Generator Defaults — Moon
    "MOON_N_POINTS_DEFAULT",
    "MOON_NOISE_DEFAULT",
    "MOON_TRAIN_RATIO_DEFAULT",
    "MOON_N_POINTS_MIN",
    "MOON_NOISE_MIN",
    "MOON_TRAIN_RATIO_MIN",
    "MOON_TRAIN_RATIO_MAX",
    "MOON_NUM_CLASSES",
    "MOON_LOWER_X_OFFSET",
    "MOON_LOWER_Y_OFFSET",
    "MOON_LOWER_Y_SHIFT",
    # Fake Service Identity
    "FAKE_SERVICE_STATUS",
    "FAKE_SERVICE_NAME",
    "FAKE_SERVICE_VERSION",
    "FAKE_SERVICE_UPTIME_SECONDS",
    # NPZ Artifact Contract (WS-1 / juniper-data#168)
    "NPZ_SPLITS",
    "FAKE_VAL_RATIO_DEFAULT",
    "NPZ_KEY_X",
    "NPZ_KEY_Y",
    "NPZ_KEY_Y_REG",
    "NPZ_KEY_T",
    "NPZ_KEY_DT",
    "NPZ_KEY_TARGET_DT",
    "NPZ_KEY_OBSERVED_MASK",
    "NPZ_KEY_PADDING_MASK",
    "NPZ_KEY_SEQ_LENGTHS",
    "ContractKind",
    "CONTRACT_KIND_TABULAR",
    "CONTRACT_KIND_SEQUENCE",
    "TASK_TYPE_CLASSIFICATION",
    "TASK_TYPE_REGRESSION",
]

# ─── Service Configuration ───────────────────────────────────────────────────

DEFAULT_BASE_URL: str = "http://localhost:8100"
FAKE_BASE_URL: str = "http://fake-data:8100"

# ─── HTTP Configuration ──────────────────────────────────────────────────────

DEFAULT_TIMEOUT: int = 30
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
# APD-ECO-002: urllib3 applies this as an ABSOLUTE additive term --
# ``backoff_value += random.random() * backoff_jitter`` -- not a proportional
# one. Without it every client that trips the same transient outage retries on
# an identical schedule, so a service that is already failing is hit by a
# synchronised herd. Matched to DEFAULT_BACKOFF_FACTOR so the spread is a full
# window on the first retry, which is the step that carries the most callers.
DEFAULT_BACKOFF_JITTER: float = 0.5
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
# Docker-secret indirection: when set, points at a file whose stripped contents are the
# API key (e.g. /run/secrets/juniper_data_api_keys). The client honors this before the
# plain API_KEY_ENV_VAR, so a deployment that mounts the key as a file (and sets only the
# _FILE form) still authenticates -- mirrors how the Juniper services read their secrets.
API_KEY_FILE_ENV_VAR: str = f"{API_KEY_ENV_VAR}_FILE"

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
# W-9 (CLI experimentation plan §11, 2026-08-08): added constants for the 7
# server generators the client still lacked — the equities pair plus the five
# sequence generators (juniper-data ≥ 0.8/0.9). The parity test now also
# cross-checks the pinned mirror against the LIVE server registry whenever
# juniper-data is importable, so this list can no longer drift silently.
GENERATOR_EQUITIES: str = "equities"
GENERATOR_EQUITIES_SEQ: str = "equities_seq"
GENERATOR_MULTI_SINE: str = "multi_sine"
GENERATOR_MACKEY_GLASS: str = "mackey_glass"
GENERATOR_AR_P: str = "ar_p"
GENERATOR_IRREGULAR_SINE: str = "irregular_sine"
GENERATOR_DELAY_PRODUCT: str = "delay_product"

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
GENERATOR_DESCRIPTION_EQUITIES: str = "Real equities OHLCV tabular dataset (next-day direction)"
GENERATOR_DESCRIPTION_EQUITIES_SEQ: str = "Real equities irregular-Δt sequence dataset (calendar-gap Δt)"
GENERATOR_DESCRIPTION_MULTI_SINE: str = "Regular-Δt superimposed-sinusoid sequence dataset"
GENERATOR_DESCRIPTION_MACKEY_GLASS: str = "Regular-Δt Mackey-Glass chaotic sequence dataset"
GENERATOR_DESCRIPTION_AR_P: str = "Regular-Δt stable AR(p) linear-stochastic sequence dataset"
GENERATOR_DESCRIPTION_IRREGULAR_SINE: str = "Irregular-Δt superimposed-sinusoid sequence dataset"
GENERATOR_DESCRIPTION_DELAY_PRODUCT: str = "Irregular-Δt bilinear delay-product capacity sequence dataset"

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

# ─── NPZ Artifact Contract (WS-1 / juniper-data#168) ─────────────────────────

# Split suffixes used for every per-split NPZ key (suffix a stem with
# f"{stem}_{split}", e.g. f"{NPZ_KEY_X}_train").
#
# "val" is the in-loop validation partition of the three-way train/val/test
# contract (design decision O-1). Membership here is presence-conditional --
# validate_npz_contract skips any split a given artifact does not carry -- so
# listing it does not require an artifact to provide it, and two-partition
# legacy artifacts keep validating unchanged.
#
# "full" is GONE (decision 11, 2026-09-05). The *_full family is no longer part of
# the contract: generators do not emit it and no consumer may require it.
#
# Removing it from this tuple does NOT stop a legacy artifact loading. Every stored
# artifact still carries X_full, and validate_npz_contract skips any split it does
# not list -- so an X_full that is present is simply not validated, which is what
# "tolerate, do not require" means in practice. Adding it back to get it validated
# would re-create the requirement this decision removed.
NPZ_SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Validation share used by the synthetic generators in juniper_data_client.testing.
# Applied alongside each generator's train ratio, so the default 0.8 / 0.1 carve
# leaves the remainder to test.
FAKE_VAL_RATIO_DEFAULT: float = 0.1

# Canonical per-split key stems.
NPZ_KEY_X: str = "X"  # features: (N, F) tabular or (W, L, F) sequence
NPZ_KEY_Y: str = "y"  # one-hot classification target
NPZ_KEY_Y_REG: str = "y_reg"  # regression target (e.g. next-day close)
NPZ_KEY_T: str = "t"  # absolute per-step time (sequence)
NPZ_KEY_DT: str = "dt"  # per-step elapsed time / Δt (sequence)
NPZ_KEY_TARGET_DT: str = "target_dt"  # irregular forecast horizon (sequence)
NPZ_KEY_OBSERVED_MASK: str = "observed_mask"  # 1=real, 0=imputed (sequence)
NPZ_KEY_PADDING_MASK: str = "padding_mask"  # 1=valid, 0=structural padding (sequence)
NPZ_KEY_SEQ_LENGTHS: str = "seq_lengths"  # valid step count per window (sequence)

# Contract discriminators returned by ``validate_npz_contract``. The Literal
# lets a caller exhaustively match on the result (APD-DCLIENT-006); annotating
# the constants with Final[ContractKind] keeps their values and the Literal's
# members from drifting apart -- a mismatch fails type checking.
ContractKind = Literal["tabular", "sequence"]
CONTRACT_KIND_TABULAR: Final[ContractKind] = "tabular"
CONTRACT_KIND_SEQUENCE: Final[ContractKind] = "sequence"

# ``task_type`` values carried in dataset metadata (meta.json, not the NPZ).
TASK_TYPE_CLASSIFICATION: str = "classification"
TASK_TYPE_REGRESSION: str = "regression"
