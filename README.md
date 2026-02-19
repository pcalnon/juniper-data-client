# juniper-data-client

Python client library for the JuniperData REST API.

## Overview

`juniper-data-client` provides a simple, robust client for interacting with the JuniperData dataset generation service. It is the official client library used by both JuniperCascor (neural network backend) and JuniperCanopy (web dashboard).

## Installation

```bash
pip install juniper-data-client
```

Or install from source:

```bash
cd juniper-data-client
pip install -e .
```

## Quick Start

```python
from juniper_data_client import JuniperDataClient

# Create client (default: localhost:8100)
client = JuniperDataClient("http://localhost:8100")

# Check service health
health = client.health_check()
print(f"Service status: {health['status']}")

# Create a spiral dataset
result = client.create_spiral_dataset(
    n_spirals=2,
    n_points_per_spiral=100,
    noise=0.1,
    seed=42,
)
dataset_id = result["dataset_id"]
print(f"Created dataset: {dataset_id}")

# Download as numpy arrays
arrays = client.download_artifact_npz(dataset_id)
X_train = arrays["X_train"]  # (160, 2) float32
y_train = arrays["y_train"]  # (160, 2) float32 one-hot
X_test = arrays["X_test"]    # (40, 2) float32
y_test = arrays["y_test"]    # (40, 2) float32 one-hot

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

## Features

- **Simple API**: Easy-to-use methods for all JuniperData endpoints
- **Automatic Retries**: Built-in retry logic for transient failures (429, 5xx)
- **Connection Pooling**: Efficient HTTP connection reuse
- **Type Hints**: Full type annotations for IDE support
- **Context Manager**: Resource cleanup with `with` statement
- **Custom Exceptions**: Granular error handling

## Usage Examples

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

# Wait up to 30 seconds for service to be ready
if client.wait_for_ready(timeout=30):
    result = client.create_spiral_dataset(seed=42)
else:
    print("Service not available")
```

### Custom Parameters

```python
# Using the general create_dataset method
result = client.create_dataset(
    generator="spiral",
    params={
        "n_spirals": 3,
        "n_points_per_spiral": 200,
        "noise": 0.2,
        "seed": 12345,
        "algorithm": "legacy_cascor",
        "radius": 10.0,
    }
)
```

### Error Handling

```python
from juniper_data_client import (
    JuniperDataClient,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataValidationError,
)

client = JuniperDataClient()

try:
    result = client.create_dataset("spiral", {"n_spirals": -1})
except JuniperDataValidationError as e:
    print(f"Invalid parameters: {e}")
except JuniperDataConnectionError as e:
    print(f"Service unreachable: {e}")
```

### PyTorch Integration

```python
import torch
from juniper_data_client import JuniperDataClient

client = JuniperDataClient()
result = client.create_spiral_dataset(seed=42)
arrays = client.download_artifact_npz(result["dataset_id"])

# Convert to PyTorch tensors
X_train = torch.from_numpy(arrays["X_train"])  # torch.float32
y_train = torch.from_numpy(arrays["y_train"])  # torch.float32
```

## API Reference

### JuniperDataClient

| Method                              | Description                            |
| ----------------------------------- | -------------------------------------- |
| `health_check()`                    | Get service health status              |
| `is_ready()`                        | Check if service is ready (boolean)    |
| `wait_for_ready(timeout)`           | Wait for service to become ready       |
| `list_generators()`                 | List available generators              |
| `get_generator_schema(name)`        | Get parameter schema for generator     |
| `create_dataset(generator, params)` | Create dataset with generator          |
| `create_spiral_dataset(**kwargs)`   | Convenience method for spiral datasets |
| `list_datasets(limit, offset)`      | List dataset IDs                       |
| `get_dataset_metadata(id)`          | Get dataset metadata                   |
| `download_artifact_npz(id)`         | Download NPZ as dict of arrays         |
| `download_artifact_bytes(id)`       | Download raw NPZ bytes                 |
| `get_preview(id, n)`                | Get JSON preview of samples            |
| `delete_dataset(id)`                | Delete a dataset                       |
| `close()`                           | Close the client session               |

### Exceptions

| Exception                    | Description                   |
| ---------------------------- | ----------------------------- |
| `JuniperDataClientError`     | Base exception for all errors |
| `JuniperDataConnectionError` | Connection to service failed  |
| `JuniperDataTimeoutError`    | Request timed out             |
| `JuniperDataNotFoundError`   | Resource not found (404)      |
| `JuniperDataValidationError` | Invalid parameters (400/422)  |

## NPZ Artifact Schema

Downloaded artifacts contain the following numpy arrays (all `float32`):

| Key       | Shape                  | Description                   |
| --------- | ---------------------- | ----------------------------- |
| `X_train` | `(n_train, 2)`         | Training features             |
| `y_train` | `(n_train, n_classes)` | Training labels (one-hot)     |
| `X_test`  | `(n_test, 2)`          | Test features                 |
| `y_test`  | `(n_test, n_classes)`  | Test labels (one-hot)         |
| `X_full`  | `(n_total, 2)`         | Full dataset features         |
| `y_full`  | `(n_total, n_classes)` | Full dataset labels (one-hot) |

## Configuration

| Parameter        | Default                 | Description                        |
| ---------------- | ----------------------- | ---------------------------------- |
| `base_url`       | `http://localhost:8100` | JuniperData service URL            |
| `timeout`        | `30`                    | Request timeout in seconds         |
| `retries`        | `3`                     | Number of retry attempts           |
| `backoff_factor` | `0.5`                   | Backoff multiplier between retries |

## Requirements

- Python >=3.11
- numpy >=1.24.0
- requests >=2.28.0
- urllib3 >=2.0.0

## License

MIT License - Copyright (c) 2024-2026 Paul Calnon

## See Also

- [JuniperData](https://github.com/pcalnon/Juniper/tree/main/JuniperData)
- [JuniperCascor](https://github.com/pcalnon/Juniper/tree/main/JuniperCascor)
- [JuniperCanopy](https://github.com/pcalnon/Juniper/tree/main/JuniperCanopy)
