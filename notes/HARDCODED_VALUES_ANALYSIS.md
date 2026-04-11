# Hardcoded Values Analysis — juniper-data-client

**Version**: 0.4.0
**Analysis Date**: 2026-04-08
**Analyst**: Claude Code (Automated Code Review)
**Status**: PLANNING ONLY — No source code modifications

---

## Executive Summary

The juniper-data-client codebase contains approximately **89 hardcoded values** across 6 source files. The library has **no existing constants module**. Three values are defined as class-level constants on `JuniperDataClient` (`DEFAULT_TIMEOUT`, `DEFAULT_RETRIES`, `DEFAULT_BACKOFF_FACTOR`), but all other values — including 16 API endpoints, connection pool sizes, HTTP status codes, and generator defaults — are inline literals.

---

## 1. Existing Constants Infrastructure

| Pattern | Status |
|---------|--------|
| `constants.py` | **Does not exist** |
| `config.py` | **Does not exist** |
| `defaults.py` | **Does not exist** |
| `settings.py` | **Does not exist** |
| Class-level constants | 3 on `JuniperDataClient`: `DEFAULT_TIMEOUT=30`, `DEFAULT_RETRIES=3`, `DEFAULT_BACKOFF_FACTOR=0.5` |

---

## 2. Hardcoded Values Inventory

### 2.1 HTTP Configuration (`client.py`) — PARTIALLY COVERED

| Line | Value | Type | Context | Proposed Constant Name | Status |
|------|-------|------|---------|----------------------|--------|
| 33 | `"http://localhost:8100"` | str | Default base URL | `DEFAULT_BASE_URL` | NOT_COVERED |
| 34 | `30` | int | Default timeout (sec) | `DEFAULT_TIMEOUT` | COVERED (class attr) |
| 45 | `3` | int | Retry count | `DEFAULT_RETRIES` | COVERED (class attr) |
| 46 | `0.5` | float | Backoff factor | `DEFAULT_BACKOFF_FACTOR` | COVERED (class attr) |
| 47 | `[429, 500, 502, 503, 504]` | list | Retryable status codes | `RETRYABLE_STATUS_CODES` | NOT_COVERED |
| 51 | `10` | int | Pool connections | `HTTP_POOL_CONNECTIONS` | NOT_COVERED |
| 51 | `10` | int | Pool max size | `HTTP_POOL_MAX_SIZE` | NOT_COVERED |

### 2.2 API Endpoints (`client.py`) — NOT COVERED

| Line(s) | Value | Proposed Constant Name |
|---------|-------|----------------------|
| Various | `"/v1/health"` | `ENDPOINT_HEALTH` |
| Various | `"/v1/health/ready"` | `ENDPOINT_HEALTH_READY` |
| Various | `"/v1/generators"` | `ENDPOINT_GENERATORS` |
| Various | `"/v1/generators/{name}/schema"` | `ENDPOINT_GENERATOR_SCHEMA` |
| Various | `"/v1/datasets"` | `ENDPOINT_DATASETS` |
| Various | `"/v1/datasets/{dataset_id}"` | `ENDPOINT_DATASET_BY_ID` |
| Various | `"/v1/datasets/{dataset_id}/artifact"` | `ENDPOINT_DATASET_ARTIFACT` |
| Various | `"/v1/datasets/{dataset_id}/preview"` | `ENDPOINT_DATASET_PREVIEW` |
| Various | `"/v1/datasets/batch-create"` | `ENDPOINT_BATCH_CREATE` |
| Various | `"/v1/datasets/batch-delete"` | `ENDPOINT_BATCH_DELETE` |
| Various | `"/v1/datasets/batch-tags"` | `ENDPOINT_BATCH_TAGS` |
| Various | `"/v1/datasets/batch-export"` | `ENDPOINT_BATCH_EXPORT` |
| Various | `"/v1/datasets/versions"` | `ENDPOINT_DATASET_VERSIONS` |
| Various | `"/v1/datasets/latest"` | `ENDPOINT_DATASET_LATEST` |

**Files requiring import**: `client.py`, `testing/fake_client.py`
**Target location**: `juniper_data_client/constants.py`

### 2.3 HTTP Status Codes (`client.py`) — NOT COVERED

| Line | Value | Context | Proposed Constant Name |
|------|-------|---------|----------------------|
| Various | `404` | Not found | `HTTP_404_NOT_FOUND` |
| Various | `400` | Bad request | `HTTP_400_BAD_REQUEST` |
| Various | `422` | Validation error | `HTTP_422_UNPROCESSABLE_ENTITY` |

### 2.4 Polling Configuration (`client.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| Various | `30.0` | float | Ready wait timeout | `DEFAULT_READY_TIMEOUT` |
| Various | `0.5` | float | Ready poll interval | `DEFAULT_READY_POLL_INTERVAL` |

### 2.5 Authentication (`client.py`) — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| Various | `"X-API-Key"` | str | API key header name | `API_KEY_HEADER_NAME` |

### 2.6 Data Types — NOT COVERED

| Line | Value | Type | Context | Proposed Constant Name |
|------|-------|------|---------|----------------------|
| Various | `"float32"` | str | NumPy dtype descriptor | `DEFAULT_ARRAY_DTYPE` |

### 2.7 Generator Defaults (`testing/fake_client.py`, `testing/generators.py`) — NOT COVERED

**Spiral defaults**:
- `n_spirals=2`, `n_points=100`, `noise=0.1`, `algorithm="modern"`, `train_ratio=0.8`

**XOR defaults**:
- `n_points=100`, 4-corner coordinates hardcoded

**Circle defaults**:
- `n_points=200`, `factor=0.5`, `noise=0.1`

**Moon defaults**:
- `n_points=200`, `noise=0.1`

**Mathematical constants in generators**:
- `radius_multiplier=5.0`, `angle_turns=4.0`, `phase_factor=2.0`
- `full_rotation=2*pi`, semicircle operations
- Shift values: `1.0`, `0.5`

**Target location**: `juniper_data_client/constants.py` (generator defaults section)

---

## 3. Coverage Summary

| Category | Total | Covered | Not Covered | Priority |
|----------|-------|---------|-------------|----------|
| HTTP Configuration | 7 | 3 | 4 | **HIGH** |
| API Endpoints | 16 | 0 | 16 | **HIGH** |
| HTTP Status Codes | 3 | 0 | 3 | **MEDIUM** |
| Polling Config | 2 | 0 | 2 | **MEDIUM** |
| Authentication | 1 | 0 | 1 | **MEDIUM** |
| Generator Defaults | 15+ | 0 | 15+ | **MEDIUM** |
| Generator Math | 10+ | 0 | 10+ | **LOW** |
| Data Types | 1 | 0 | 1 | **LOW** |
| **TOTAL** | **~89** | **3** | **~86** | — |

---

## 4. Remediation Approach

### Recommended: Create `juniper_data_client/constants.py`

Create a single centralized constants module organized by section:

```
# Sections:
# 1. HTTP Configuration (base URL, pool sizes, retryable codes)
# 2. API Endpoints (all 16 endpoint paths)
# 3. Timeouts & Polling (ready timeout, poll interval)
# 4. Authentication (header names)
# 5. Generator Defaults (spiral, xor, circle, moon)
# 6. Generator Mathematics (radii, angles, shifts)
# 7. Data Types (dtype descriptors)
```

**Strengths**:
- Simple library, single file is appropriate
- Easy to discover and audit
- Consistent with client library conventions
- ~60 constants replacing 89+ hardcoded values

**Weaknesses**:
- File may grow if more generators are added
- Mathematical constants may seem out of place alongside HTTP config

**Risks**:
- Existing class-level constants (`DEFAULT_TIMEOUT`, etc.) should be migrated to module-level constants with class attributes referencing them for backward compatibility
- Generator defaults must exactly match current values to avoid behavioral changes

**Guardrails**:
- Add backward compatibility aliases for the 3 existing class-level constants
- Validate all generator outputs match current behavior with integration tests
- Keep mathematical constants clearly separated from configuration constants

---

## 5. Files Requiring Modification

| File | Action | Constants Count |
|------|--------|-----------------|
| `juniper_data_client/constants.py` | **NEW** | ~60 |
| `juniper_data_client/client.py` | **MODIFY** — import constants, replace inline literals | ~25 replacements |
| `juniper_data_client/testing/fake_client.py` | **MODIFY** | ~20 replacements |
| `juniper_data_client/testing/generators.py` | **MODIFY** | ~30 replacements |

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking backward compat (class-level constants) | Low | Medium | Keep class attrs referencing module constants |
| Generator output changes | Very Low | High | Constants preserve exact values; run integration tests |
| Import errors | Very Low | Low | Simple flat module structure |
| API endpoint typos | Low | High | Copy exact strings; add endpoint validation test |
