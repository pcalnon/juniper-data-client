# Hardcoded Values Refactor Plan — juniper-data-client

**Version**: 0.4.0
**Created**: 2026-04-08
**Status**: PLANNING — No source code modifications
**Companion Document**: `HARDCODED_VALUES_ANALYSIS.md`

---

## Phase 1: Constants Infrastructure (Priority: HIGH)

### Step 1.1: Create Constants Module

**Task**: Create `juniper_data_client/constants.py` (~60 constants)

**Sections**:
1. HTTP Configuration (base URL, pool sizes, retryable codes)
2. API Endpoints (16 endpoint paths)
3. Timeouts & Polling (ready timeout, poll interval)
4. Authentication (header names)
5. Generator Defaults (spiral, xor, circle, moon parameters)
6. Generator Mathematics (radii, angles, shifts)
7. Data Types (dtype descriptors)

### Step 1.2: Backward Compatibility

Maintain existing class-level constants on `JuniperDataClient` as aliases referencing module constants:
```python
class JuniperDataClient:
    DEFAULT_TIMEOUT = constants.DEFAULT_TIMEOUT
    DEFAULT_RETRIES = constants.DEFAULT_RETRIES
    DEFAULT_BACKOFF_FACTOR = constants.DEFAULT_BACKOFF_FACTOR
```

---

## Phase 2: Source File Refactor (Priority: HIGH)

### Step 2.1: Refactor Client

**File**: `client.py` — ~25 replacements (URL, endpoints, pool config, status codes)

### Step 2.2: Refactor Fake Client

**File**: `testing/fake_client.py` — ~20 replacements (URLs, training defaults, worker data)

### Step 2.3: Refactor Generators

**File**: `testing/generators.py` — ~30 replacements (math constants, dataset defaults)

---

## Phase 3: Validation (Priority: HIGH)

### Step 3.1: Run Full Test Suite

```bash
pytest tests/ -v
```

### Step 3.2: Pre-commit Hooks

```bash
pre-commit run --all-files
```

### Step 3.3: Verify Generator Outputs

Run each generator (spiral, xor, circle, moon) with default parameters. Verify outputs match pre-refactor results.

---

## Phase 4: Documentation & Release (Priority: MEDIUM)

### Step 4.1: Update AGENTS.md — Document new constants module
### Step 4.2: Update CHANGELOG.md
### Step 4.3: Create Release Description
