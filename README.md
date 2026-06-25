# juniper-data-client

[![PyPI](https://img.shields.io/pypi/v/juniper-data-client)](https://pypi.org/project/juniper-data-client/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**Python HTTP client for the juniper-data dataset-generation service.**

`juniper-data-client` is a thin, synchronous HTTP client for fetching datasets from a `juniper-data`
instance. It exposes a single `JuniperDataClient` class whose methods correspond to the REST surface:
generator schemas, dataset creation, tag-based filtering, batch operations, the named-version
registry, and download of the resulting NPZ artifacts as ready-to-train `numpy` arrays. It is the
canonical dataset-fetch interface used by `juniper-cascor` (training) and `juniper-canopy`
(visualisation), and it reads the platform's shared `X_train` / `y_train` / `X_test` / `y_test` /
`X_full` / `y_full` NPZ schema (all `float32`).

> **Part of the Juniper platform.** juniper-data-client is the Python HTTP client for the juniper-data
> dataset service in [Juniper](https://github.com/pcalnon/juniper-ml) — a multi-package ML research
> platform built around constructive (Cascade-Correlation) and recurrent neural networks. Point it at
> any running juniper-data instance.

## Install

```bash
pip install juniper-data-client
```

Optional extras: `[observability]` adds X-Request-ID propagation via `juniper-observability`; `[test]`
and `[dev]` install the test and lint/type-check toolchains.

## Quick start

```python
from juniper_data_client import JuniperDataClient

with JuniperDataClient("http://localhost:8100") as client:
    print(client.health_check()["status"])

    result = client.create_spiral_dataset(n_spirals=2, n_points_per_spiral=100, noise=0.1, seed=42)
    arrays = client.download_artifact_npz(result["dataset_id"])

    print("train:", len(arrays["X_train"]), "test:", len(arrays["X_test"]))
```

The artifacts come back as a dict of `float32` `numpy` arrays, ready for `torch.from_numpy(...)`.

## API

| Method | Purpose |
|--------|---------|
| `health_check()` / `is_ready()` / `wait_for_ready(...)` | Liveness / readiness probes |
| `list_generators()` / `get_generator_schema(name)` | Discover available generators + their schemas |
| `create_dataset(name, params, ...)` | Create a dataset from any generator |
| `create_spiral_dataset(...)` | Convenience wrapper for the spiral generator |
| `download_artifact_npz(dataset_id)` / `download_artifact_bytes(dataset_id)` | Download as numpy arrays / raw bytes |
| `get_dataset_metadata(id)` / `get_preview(id, n)` / `delete_dataset(id)` | Inspect / preview / delete |
| `list_datasets(limit, offset)` | Paginated dataset listing |
| `list_versions(name)` / `get_latest(name)` | Named-version registry |
| `batch_create(...)` / `batch_delete(...)` / `batch_update_tags(...)` / `batch_export(...)` | Batch operations |

## Status

**Beta** on PyPI. The current version is shown by the badge above; see [`CHANGELOG.md`](./CHANGELOG.md).

## Documentation

- [`docs/QUICK_START.md`](./docs/QUICK_START.md) — installation and verification guide
- [`docs/REFERENCE.md`](./docs/REFERENCE.md) — full API reference, configuration, and error model

## License

MIT — see [LICENSE](./LICENSE).
