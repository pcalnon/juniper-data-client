#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     fake_client.py
# Author:        Paul Calnon
# Version:       0.4.0
#
# Date Created:  2026-02-27
# Last Modified: 2026-02-27
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    In-memory fake implementation of JuniperDataClient for testing.
#    Provides the same public interface as the real client but uses
#    synthetic dataset generators instead of making HTTP calls.
#    Suitable for unit testing consumers like JuniperCascor and
#    juniper-canopy without requiring a running JuniperData service.
#####################################################################################################################################################################################################

"""FakeDataClient — drop-in test replacement for JuniperDataClient.

All state is held in memory. No network calls are made. Datasets are
generated using synthetic generators from ``juniper_data_client.testing.generators``.

Usage::

    from juniper_data_client.testing import FakeDataClient

    with FakeDataClient() as client:
        result = client.create_spiral_dataset(n_spirals=2, seed=42)
        arrays = client.download_artifact_npz(result["dataset_id"])
        X_train = arrays["X_train"]
"""

import inspect
import io
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from juniper_data_client.constants import (
    CIRCLE_FACTOR_DEFAULT,
    CIRCLE_FACTOR_MAX,
    CIRCLE_FACTOR_MIN,
    CIRCLE_N_POINTS_DEFAULT,
    CIRCLE_N_POINTS_MIN,
    CIRCLE_NOISE_DEFAULT,
    CIRCLE_NOISE_MIN,
    CIRCLE_TRAIN_RATIO_DEFAULT,
    CIRCLE_TRAIN_RATIO_MAX,
    CIRCLE_TRAIN_RATIO_MIN,
    DEFAULT_ARRAY_DTYPE,
    DEFAULT_LIST_LIMIT,
    DEFAULT_LIST_OFFSET,
    DEFAULT_PREVIEW_N,
    DEFAULT_READY_POLL_INTERVAL,
    DEFAULT_READY_TIMEOUT,
    ENDPOINT_DATASET_ARTIFACT_TEMPLATE,
    FAKE_BASE_URL,
    FAKE_SERVICE_NAME,
    FAKE_SERVICE_STATUS,
    FAKE_SERVICE_UPTIME_SECONDS,
    FAKE_SERVICE_VERSION,
    GENERATOR_CIRCLE,
    GENERATOR_CIRCLE_LEGACY,
    GENERATOR_DESCRIPTION_CIRCLE,
    GENERATOR_DESCRIPTION_MOON,
    GENERATOR_DESCRIPTION_SPIRAL,
    GENERATOR_DESCRIPTION_XOR,
    GENERATOR_MOON,
    GENERATOR_SPIRAL,
    GENERATOR_VERSION,
    GENERATOR_XOR,
    MAX_PREVIEW_N,
    MOON_N_POINTS_DEFAULT,
    MOON_N_POINTS_MIN,
    MOON_NOISE_DEFAULT,
    MOON_NOISE_MIN,
    MOON_TRAIN_RATIO_DEFAULT,
    MOON_TRAIN_RATIO_MAX,
    MOON_TRAIN_RATIO_MIN,
    SPIRAL_ALGORITHM_DEFAULT,
    SPIRAL_ALGORITHMS,
    SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
    SPIRAL_N_POINTS_PER_SPIRAL_MIN,
    SPIRAL_N_SPIRALS_DEFAULT,
    SPIRAL_N_SPIRALS_MIN,
    SPIRAL_NOISE_DEFAULT,
    SPIRAL_NOISE_MIN,
    SPIRAL_TRAIN_RATIO_DEFAULT,
    SPIRAL_TRAIN_RATIO_MAX,
    SPIRAL_TRAIN_RATIO_MIN,
    XOR_N_POINTS_DEFAULT,
    XOR_N_POINTS_MIN,
    XOR_NOISE_DEFAULT,
    XOR_NOISE_MIN,
    XOR_TRAIN_RATIO_DEFAULT,
    XOR_TRAIN_RATIO_MAX,
    XOR_TRAIN_RATIO_MIN,
)
from juniper_data_client.exceptions import JuniperDataNotFoundError, JuniperDataValidationError
from juniper_data_client.testing.generators import generate_circle, generate_moon, generate_spiral, generate_xor

# ---------------------------------------------------------------------------
# Generator catalog — mirrors the real JuniperData /v1/generators response
# ---------------------------------------------------------------------------

_GENERATOR_CATALOG: List[Dict[str, Any]] = [
    {
        "name": GENERATOR_SPIRAL,
        "description": GENERATOR_DESCRIPTION_SPIRAL,
        "version": GENERATOR_VERSION,
        "parameters": ["n_spirals", "n_points_per_spiral", "noise", "seed", "algorithm", "train_ratio"],
    },
    {
        "name": GENERATOR_XOR,
        "description": GENERATOR_DESCRIPTION_XOR,
        "version": GENERATOR_VERSION,
        "parameters": ["n_points", "noise", "seed", "train_ratio"],
    },
    {
        "name": GENERATOR_CIRCLE,
        "description": GENERATOR_DESCRIPTION_CIRCLE,
        "version": GENERATOR_VERSION,
        "parameters": ["n_points", "noise", "factor", "seed", "train_ratio"],
    },
    {
        "name": GENERATOR_MOON,
        "description": GENERATOR_DESCRIPTION_MOON,
        "version": GENERATOR_VERSION,
        "parameters": ["n_points", "noise", "seed", "train_ratio"],
    },
]

_GENERATOR_SCHEMAS: Dict[str, Dict[str, Any]] = {
    GENERATOR_SPIRAL: {
        "type": "object",
        "properties": {
            "n_spirals": {"type": "integer", "default": SPIRAL_N_SPIRALS_DEFAULT, "minimum": SPIRAL_N_SPIRALS_MIN, "description": "Number of spiral arms"},
            "n_points_per_spiral": {
                "type": "integer",
                "default": SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
                "minimum": SPIRAL_N_POINTS_PER_SPIRAL_MIN,
                "description": "Points per spiral arm",
            },
            "noise": {"type": "number", "default": SPIRAL_NOISE_DEFAULT, "minimum": SPIRAL_NOISE_MIN, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "algorithm": {
                "type": "string",
                "default": SPIRAL_ALGORITHM_DEFAULT,
                "enum": SPIRAL_ALGORITHMS,
                "description": "Generation algorithm",
            },
            "train_ratio": {
                "type": "number",
                "default": SPIRAL_TRAIN_RATIO_DEFAULT,
                "minimum": SPIRAL_TRAIN_RATIO_MIN,
                "maximum": SPIRAL_TRAIN_RATIO_MAX,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    GENERATOR_XOR: {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": XOR_N_POINTS_DEFAULT, "minimum": XOR_N_POINTS_MIN, "description": "Total number of points"},
            "noise": {"type": "number", "default": XOR_NOISE_DEFAULT, "minimum": XOR_NOISE_MIN, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": XOR_TRAIN_RATIO_DEFAULT,
                "minimum": XOR_TRAIN_RATIO_MIN,
                "maximum": XOR_TRAIN_RATIO_MAX,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    GENERATOR_CIRCLE: {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": CIRCLE_N_POINTS_DEFAULT, "minimum": CIRCLE_N_POINTS_MIN, "description": "Total number of points"},
            "noise": {"type": "number", "default": CIRCLE_NOISE_DEFAULT, "minimum": CIRCLE_NOISE_MIN, "description": "Gaussian noise level"},
            "factor": {
                "type": "number",
                "default": CIRCLE_FACTOR_DEFAULT,
                "minimum": CIRCLE_FACTOR_MIN,
                "maximum": CIRCLE_FACTOR_MAX,
                "description": "Inner/outer circle radius ratio",
            },
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": CIRCLE_TRAIN_RATIO_DEFAULT,
                "minimum": CIRCLE_TRAIN_RATIO_MIN,
                "maximum": CIRCLE_TRAIN_RATIO_MAX,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    GENERATOR_MOON: {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": MOON_N_POINTS_DEFAULT, "minimum": MOON_N_POINTS_MIN, "description": "Total number of points"},
            "noise": {"type": "number", "default": MOON_NOISE_DEFAULT, "minimum": MOON_NOISE_MIN, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": MOON_TRAIN_RATIO_DEFAULT,
                "minimum": MOON_TRAIN_RATIO_MIN,
                "maximum": MOON_TRAIN_RATIO_MAX,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
}

# Maps generator names to their callable functions. Keys here MUST
# match the server-side ``GENERATOR_REGISTRY`` so the fake cannot
# silently accept names that the real service rejects (DC-04 adjacent).
_GENERATOR_FUNCTIONS = {
    GENERATOR_SPIRAL: generate_spiral,
    GENERATOR_XOR: generate_xor,
    GENERATOR_CIRCLE: generate_circle,
    GENERATOR_MOON: generate_moon,
}

# DC-01/XREPO-01 (2026-04-24): one-release-cycle backward-compat map
# so callers still passing the old ``"circle"`` literal get their
# requests transparently routed to the correct ``"circles"`` generator
# with a ``DeprecationWarning``. Remove in the release after v0.5.
_GENERATOR_LEGACY_ALIASES: Dict[str, str] = {
    GENERATOR_CIRCLE_LEGACY: GENERATOR_CIRCLE,
}


def _resolve_generator_alias(name: str) -> str:
    """Map a legacy generator name to its canonical server-side key.

    Emits a ``DeprecationWarning`` on every legacy use so downstream
    callers are prompted to migrate to the new constant before the
    alias is removed.
    """
    canonical = _GENERATOR_LEGACY_ALIASES.get(name)
    if canonical is None:
        return name
    warnings.warn(
        f"Generator name {name!r} is deprecated; use {canonical!r} instead. "
        "The legacy alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )
    return canonical


class FakeDataClient:
    """In-memory fake of JuniperDataClient for testing.

    Provides the same public interface as ``JuniperDataClient`` without making any
    network calls. Datasets are generated locally using synthetic generators and
    stored in an internal dictionary keyed by dataset ID (UUID).

    Supports context-manager usage::

        with FakeDataClient() as client:
            result = client.create_dataset("spiral", {"n_spirals": 2})
            arrays = client.download_artifact_npz(result["dataset_id"])
    """

    def __init__(self, base_url: str = FAKE_BASE_URL, api_key: Optional[str] = None) -> None:
        """Initialize the FakeDataClient.

        Args:
            base_url: Fake base URL (stored but never contacted). Default: ``http://fake-data:8100``.
            api_key: Optional API key (stored but not validated).
        """
        self.base_url = base_url
        self.api_key = api_key
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._version_counters: Dict[str, int] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return a healthy status (always succeeds).

        Returns:
            Health status dictionary matching the real service format.
        """
        return {
            "status": FAKE_SERVICE_STATUS,
            "service": FAKE_SERVICE_NAME,
            "version": FAKE_SERVICE_VERSION,
            "uptime_seconds": FAKE_SERVICE_UPTIME_SECONDS,
        }

    def is_ready(self) -> bool:
        """Check if the fake service is ready (always True).

        Returns:
            True
        """
        return True

    def wait_for_ready(self, timeout: float = DEFAULT_READY_TIMEOUT, poll_interval: float = DEFAULT_READY_POLL_INTERVAL) -> bool:
        """Wait for readiness (returns immediately).

        Args:
            timeout: Ignored — the fake is always ready.
            poll_interval: Ignored.

        Returns:
            True
        """
        return True

    # ------------------------------------------------------------------
    # Generator catalog
    # ------------------------------------------------------------------

    def list_generators(self) -> List[Dict[str, Any]]:
        """List available dataset generators.

        Returns:
            List of generator info dictionaries (spiral, xor, circle, moon).
        """
        return list(_GENERATOR_CATALOG)

    def get_generator_schema(self, name: str) -> Dict[str, Any]:
        """Get the parameter schema for a generator.

        Args:
            name: Generator name (e.g., ``"spiral"``).

        Returns:
            JSON schema dictionary for the requested generator.

        Raises:
            JuniperDataNotFoundError: If the generator name is not recognized.
        """
        name = _resolve_generator_alias(name)
        if name not in _GENERATOR_SCHEMAS:
            raise JuniperDataNotFoundError(f"Generator not found: {name}")
        return dict(_GENERATOR_SCHEMAS[name])

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        generator: str,
        params: Dict[str, Any],
        persist: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        parent_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a synthetic dataset using an in-memory generator.

        Args:
            generator: Name of the generator (``"spiral"``, ``"xor"``, ``"circle"``, ``"moon"``).
            params: Parameters forwarded to the generator function. Unknown keys are silently
                ignored to match the real service behavior.
            persist: Accepted for API compatibility; has no effect in the fake.
            name: Optional dataset name for versioning. When provided, the fake
                automatically assigns an incrementing version number.
            description: Optional human-readable description of the dataset.
            created_by: Optional identifier for the creator (user or system).
            parent_dataset_id: Optional ID of the parent dataset this was derived from.

        Returns:
            Dictionary with ``dataset_id``, ``generator``, ``params``, ``meta``, and ``artifact_url``.

        Raises:
            JuniperDataValidationError: If the generator name is not recognized.
        """
        generator = _resolve_generator_alias(generator)
        if generator not in _GENERATOR_FUNCTIONS:
            raise JuniperDataValidationError(f"Unknown generator: {generator}")

        gen_func = _GENERATOR_FUNCTIONS[generator]

        # Filter params to only those accepted by the generator function
        sig = inspect.signature(gen_func)
        accepted_keys = set(sig.parameters.keys())
        filtered_params = {k: v for k, v in params.items() if k in accepted_keys}

        arrays = gen_func(**filtered_params)

        dataset_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        n_train = arrays["X_train"].shape[0]
        n_test = arrays["X_test"].shape[0]
        n_full = arrays["X_full"].shape[0]
        n_features = arrays["X_train"].shape[1]
        n_classes = arrays["y_train"].shape[1]

        meta: Dict[str, Any] = {
            "n_train": n_train,
            "n_test": n_test,
            "n_full": n_full,
            "n_features": n_features,
            "n_classes": n_classes,
            "dtype": DEFAULT_ARRAY_DTYPE,
            "created_at": now,
        }

        # Versioning fields — only populated when a name is provided
        if name is not None:
            version_num = self._version_counters.get(name, 0) + 1
            self._version_counters[name] = version_num
            meta["dataset_name"] = name
            meta["dataset_version"] = version_num
        if description is not None:
            meta["description"] = description
        if created_by is not None:
            meta["created_by"] = created_by
        if parent_dataset_id is not None:
            meta["parent_dataset_id"] = parent_dataset_id

        metadata = {
            "dataset_id": dataset_id,
            "generator": generator,
            "params": params,
            "meta": meta,
            "artifact_url": ENDPOINT_DATASET_ARTIFACT_TEMPLATE.format(dataset_id=dataset_id),
            "created_at": now,
        }

        self._datasets[dataset_id] = {
            "metadata": metadata,
            "arrays": arrays,
        }

        return metadata

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

        Delegates to ``create_dataset("spiral", ...)``.

        Args:
            n_spirals: Number of spiral arms (default: 2).
            n_points_per_spiral: Points per spiral arm (default: 100).
            noise: Noise level (default: 0.1).
            seed: Random seed for reproducibility (optional).
            algorithm: Generation algorithm — ``"modern"`` or ``"legacy_cascor"`` (default: ``"modern"``).
            train_ratio: Fraction of data for training (default: 0.8).
            **kwargs: Additional parameters passed to the generator.

        Returns:
            Dataset creation response with dataset_id and metadata.
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

    # ------------------------------------------------------------------
    # Dataset versioning
    # ------------------------------------------------------------------

    def list_versions(self, name: str) -> Dict[str, Any]:
        """List all versions of a named dataset.

        Args:
            name: Dataset name to list versions for.

        Returns:
            Dict with ``dataset_name``, ``versions`` list, ``total`` count,
            and ``latest_version``.
        """
        versions = []
        for entry in self._datasets.values():
            meta = entry["metadata"].get("meta", {})
            if meta.get("dataset_name") == name:
                versions.append(entry["metadata"])

        # Sort by dataset_version ascending
        versions.sort(key=lambda m: m.get("meta", {}).get("dataset_version", 0))

        latest_version = versions[-1]["meta"]["dataset_version"] if versions else None
        return {
            "dataset_name": name,
            "versions": versions,
            "total": len(versions),
            "latest_version": latest_version,
        }

    def get_latest(self, name: str) -> Dict[str, Any]:
        """Get the latest version of a named dataset.

        Args:
            name: Dataset name to get latest version of.

        Returns:
            Dataset metadata for the latest version.

        Raises:
            JuniperDataNotFoundError: If no versions exist for the given name.
        """
        result = self.list_versions(name)
        if result["total"] == 0:
            raise JuniperDataNotFoundError(f"No versions found for dataset name: {name}")
        return result["versions"][-1]

    # ------------------------------------------------------------------
    # Dataset listing / metadata / deletion
    # ------------------------------------------------------------------

    def list_datasets(self, limit: int = DEFAULT_LIST_LIMIT, offset: int = DEFAULT_LIST_OFFSET) -> List[str]:
        """List dataset IDs stored in the fake.

        Args:
            limit: Maximum number of dataset IDs to return (default: 100).
            offset: Number of dataset IDs to skip (default: 0).

        Returns:
            List of dataset ID strings.
        """
        all_ids = list(self._datasets.keys())
        return all_ids[offset : offset + limit]

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset.

        Args:
            dataset_id: Unique dataset identifier.

        Returns:
            Dataset metadata dictionary.

        Raises:
            JuniperDataNotFoundError: If the dataset does not exist.
        """
        if dataset_id not in self._datasets:
            raise JuniperDataNotFoundError(f"Dataset not found: {dataset_id}")
        return dict(self._datasets[dataset_id]["metadata"])

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from the in-memory store.

        Args:
            dataset_id: Unique dataset identifier.

        Returns:
            True if the dataset was deleted.

        Raises:
            JuniperDataNotFoundError: If the dataset does not exist.
        """
        if dataset_id not in self._datasets:
            raise JuniperDataNotFoundError(f"Dataset not found: {dataset_id}")
        del self._datasets[dataset_id]
        return True

    # ------------------------------------------------------------------
    # Artifact download
    # ------------------------------------------------------------------

    def download_artifact_bytes(self, dataset_id: str) -> bytes:
        """Serialize the stored arrays to NPZ format bytes.

        Args:
            dataset_id: ID of the dataset whose artifact to download.

        Returns:
            Raw bytes of the NPZ file.

        Raises:
            JuniperDataNotFoundError: If the dataset does not exist.
        """
        if dataset_id not in self._datasets:
            raise JuniperDataNotFoundError(f"Dataset not found: {dataset_id}")

        arrays = self._datasets[dataset_id]["arrays"]
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    def download_artifact_npz(self, dataset_id: str) -> Dict[str, np.ndarray]:
        """Return the stored numpy arrays directly.

        Args:
            dataset_id: ID of the dataset whose artifact to download.

        Returns:
            Dictionary mapping array names to numpy arrays (float32).

        Raises:
            JuniperDataNotFoundError: If the dataset does not exist.
        """
        if dataset_id not in self._datasets:
            raise JuniperDataNotFoundError(f"Dataset not found: {dataset_id}")

        return dict(self._datasets[dataset_id]["arrays"])

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def get_preview(self, dataset_id: str, n: int = DEFAULT_PREVIEW_N) -> Dict[str, Any]:
        """Get a JSON-serializable preview of dataset samples.

        Returns the first ``n`` samples from the full dataset (X_full / y_full).

        Args:
            dataset_id: ID of the dataset to preview.
            n: Number of samples to include in the preview (default: 100, max: 1000).

        Returns:
            Dictionary with ``n_samples``, ``X_sample``, and ``y_sample``.

        Raises:
            JuniperDataNotFoundError: If the dataset does not exist.
        """
        if dataset_id not in self._datasets:
            raise JuniperDataNotFoundError(f"Dataset not found: {dataset_id}")

        arrays = self._datasets[dataset_id]["arrays"]
        X_full = arrays["X_full"]
        y_full = arrays["y_full"]

        # Cap at available samples and the requested count
        n_available = min(n, X_full.shape[0], MAX_PREVIEW_N)

        return {
            "n_samples": int(n_available),
            "X_sample": X_full[:n_available].tolist(),
            "y_sample": y_full[:n_available].tolist(),
        }

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_delete(self, dataset_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple datasets from the in-memory store.

        Args:
            dataset_ids: List of dataset IDs to delete.

        Returns:
            Dictionary with deleted, not_found, and total_deleted.
        """
        deleted: List[str] = []
        not_found: List[str] = []
        for dataset_id in dataset_ids:
            if dataset_id in self._datasets:
                del self._datasets[dataset_id]
                deleted.append(dataset_id)
            else:
                not_found.append(dataset_id)
        return {"deleted": deleted, "not_found": not_found, "total_deleted": len(deleted)}

    def batch_create(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple datasets in the in-memory store.

        Args:
            datasets: List of dataset specifications with generator and params.

        Returns:
            Dictionary with results, total_created, and total_failed.
        """
        results: List[Dict[str, Any]] = []
        total_created = 0
        total_failed = 0

        for idx, item in enumerate(datasets):
            try:
                resp = self.create_dataset(
                    generator=item["generator"],
                    params=item.get("params", {}),
                    persist=item.get("persist", True),
                )
                results.append(
                    {
                        "index": idx,
                        "dataset_id": resp["dataset_id"],
                        "generator": item["generator"],
                        "success": True,
                        "artifact_url": ENDPOINT_DATASET_ARTIFACT_TEMPLATE.format(dataset_id=resp["dataset_id"]),
                    }
                )
                total_created += 1
            except Exception as e:
                results.append(
                    {
                        "index": idx,
                        "generator": item.get("generator", "unknown"),
                        "success": False,
                        "error": str(e),
                    }
                )
                total_failed += 1

        return {"results": results, "total_created": total_created, "total_failed": total_failed}

    def batch_update_tags(
        self,
        dataset_ids: List[str],
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add or remove tags from multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to update.
            add_tags: Tags to add.
            remove_tags: Tags to remove.

        Returns:
            Dictionary with updated, not_found, and total_updated.
        """
        updated: List[str] = []
        not_found: List[str] = []
        for dataset_id in dataset_ids:
            if dataset_id not in self._datasets:
                not_found.append(dataset_id)
                continue
            meta = self._datasets[dataset_id]["metadata"]
            current_tags = set(meta.get("tags", []))
            if add_tags:
                current_tags.update(add_tags)
            if remove_tags:
                current_tags -= set(remove_tags)
            meta["tags"] = sorted(current_tags)
            updated.append(dataset_id)
        return {"updated": updated, "not_found": not_found, "total_updated": len(updated)}

    def batch_export(self, dataset_ids: List[str]) -> bytes:
        """Export multiple datasets as a ZIP archive of NPZ files.

        Args:
            dataset_ids: List of dataset IDs to export.

        Returns:
            Raw bytes of the ZIP archive.

        Raises:
            JuniperDataNotFoundError: If none of the datasets exist.
        """
        import zipfile

        buf = io.BytesIO()
        found = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for dataset_id in dataset_ids:
                if dataset_id in self._datasets:
                    arrays = self._datasets[dataset_id]["arrays"]
                    npz_buf = io.BytesIO()
                    np.savez(npz_buf, **arrays)
                    zf.writestr(f"{dataset_id}.npz", npz_buf.getvalue())
                    found += 1
        if found == 0:
            raise JuniperDataNotFoundError("None of the requested datasets were found")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the fake client (clears internal state)."""
        self._datasets.clear()
        self._version_counters.clear()
        self._closed = True

    def __enter__(self) -> "FakeDataClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — closes the client."""
        self.close()
