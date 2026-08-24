# Quick Start Guide

## Get juniper-data-client Working in 5 Minutes

**Version:** 0.4.2
**Status:** Active
**Last Updated:** August 24, 2026
**Project:** Juniper - Dataset Service Client Library

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Install](#1-install)
- [Basic Usage](#2-basic-usage)
- [Download Data](#3-download-data)
- [Error Handling](#4-error-handling)
- [Testing](#5-testing)
- [Next Steps](#6-next-steps)

---

## Prerequisites

- **Python 3.12+** (`python --version`)
- **juniper-data** service running on port 8100 (`curl http://localhost:8100/v1/health`)

---

## 1. Install

```bash
pip install juniper-data-client
```

Or install from source for development:

```bash
cd juniper-data-client
pip install -e ".[dev]"
```

---

## 2. Basic Usage

```python
from juniper_data_client import JuniperDataClient

# Create client (default: localhost:8100). Hostless values such as "http://"
# raise JuniperDataConfigurationError at construction, not on the first request.
client = JuniperDataClient("http://localhost:8100")

# Check service health
health = client.health_check()
print(f"Service: {health['status']}")  # "ok"

# Create a spiral dataset
result = client.create_spiral_dataset(
    n_spirals=2,
    n_points_per_spiral=100,
    noise=0.1,
    seed=42,
)
print(f"Dataset ID: {result['dataset_id']}")
```

---

## 3. Download Data

```python
# Download as numpy arrays
arrays = client.download_artifact_npz(result["dataset_id"])

X_train = arrays["X_train"]  # (160, 2) float32
y_train = arrays["y_train"]  # (160, 2) float32 one-hot
X_test = arrays["X_test"]    # (40, 2) float32
y_test = arrays["y_test"]    # (40, 2) float32 one-hot

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# Optional: classify tabular vs sequence (WS-1) and enforce Δt / mask rules
from juniper_data_client import validate_npz_contract

kind = validate_npz_contract(arrays)  # "tabular" or "sequence"
```

### Context Manager

```python
with JuniperDataClient("http://localhost:8100") as client:
    result = client.create_spiral_dataset(seed=42)
    arrays = client.download_artifact_npz(result["dataset_id"])
# Session automatically closed
```

### Wait for Service

```python
client = JuniperDataClient("http://localhost:8100")
if client.wait_for_ready(timeout=30):
    result = client.create_spiral_dataset(seed=42)
else:
    print("Service not available")
```

---

## 4. Error Handling

```python
from juniper_data_client import (
    JuniperDataClient,
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataValidationError,
)

try:
    client = JuniperDataClient("http://localhost:8100")
except JuniperDataConfigurationError as e:
    # Hostless base_url (empty string, "http://", "/v1", "http://user:secret@")
    # status_code is None — this is raised at construction, not on the first request.
    print(f"Bad client config: {e}")

try:
    result = client.create_dataset("spiral", {"n_spirals": -1})
except JuniperDataValidationError as e:
    # 400 and 422 share this type; e.status_code is the discriminator.
    # FastAPI 422 keeps the list on e.detail; str(e) renders "body.seed: Field required".
    print(f"Invalid parameters ({e.status_code}): {e}")
except JuniperDataConnectionError as e:
    print(f"Service unreachable: {e}")
```

---

## 5. Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/ -m unit -v

# Run with coverage
pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80
```

The test suite includes a `FakeDataClient` for testing consumers without a running service. See [REFERENCE.md](REFERENCE.md) for details.

---

## 6. Next Steps

- [Documentation Overview](DOCUMENTATION_OVERVIEW.md) -- navigation index
- [API Reference](REFERENCE.md) -- complete method and configuration reference
- [README.md](../README.md) -- project overview with more examples
- [AGENTS.md](../AGENTS.md) -- development conventions and commands

---

**Last Updated:** August 24, 2026
**Version:** 0.4.2
**Status:** Active
