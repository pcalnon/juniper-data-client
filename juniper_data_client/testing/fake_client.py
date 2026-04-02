#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     fake_client.py
# Author:        Paul Calnon
# Version:       0.3.1
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from juniper_data_client.exceptions import JuniperDataNotFoundError, JuniperDataValidationError
from juniper_data_client.testing.generators import generate_circle, generate_moon, generate_spiral, generate_xor

# ---------------------------------------------------------------------------
# Generator catalog — mirrors the real JuniperData /v1/generators response
# ---------------------------------------------------------------------------

_GENERATOR_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "spiral",
        "description": "Multi-arm Archimedean spiral dataset",
        "version": "1.0.0",
        "parameters": ["n_spirals", "n_points_per_spiral", "noise", "seed", "algorithm", "train_ratio"],
    },
    {
        "name": "xor",
        "description": "XOR classification dataset with four corner clusters",
        "version": "1.0.0",
        "parameters": ["n_points", "noise", "seed", "train_ratio"],
    },
    {
        "name": "circle",
        "description": "Concentric circles classification dataset",
        "version": "1.0.0",
        "parameters": ["n_points", "noise", "factor", "seed", "train_ratio"],
    },
    {
        "name": "moon",
        "description": "Two interleaving half-moon classification dataset",
        "version": "1.0.0",
        "parameters": ["n_points", "noise", "seed", "train_ratio"],
    },
]

_GENERATOR_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "spiral": {
        "type": "object",
        "properties": {
            "n_spirals": {"type": "integer", "default": 2, "minimum": 2, "description": "Number of spiral arms"},
            "n_points_per_spiral": {
                "type": "integer",
                "default": 100,
                "minimum": 10,
                "description": "Points per spiral arm",
            },
            "noise": {"type": "number", "default": 0.1, "minimum": 0.0, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "algorithm": {
                "type": "string",
                "default": "modern",
                "enum": ["modern", "legacy_cascor"],
                "description": "Generation algorithm",
            },
            "train_ratio": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.1,
                "maximum": 0.99,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    "xor": {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": 100, "minimum": 4, "description": "Total number of points"},
            "noise": {"type": "number", "default": 0.1, "minimum": 0.0, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.1,
                "maximum": 0.99,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    "circle": {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": 200, "minimum": 10, "description": "Total number of points"},
            "noise": {"type": "number", "default": 0.1, "minimum": 0.0, "description": "Gaussian noise level"},
            "factor": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.01,
                "maximum": 0.99,
                "description": "Inner/outer circle radius ratio",
            },
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.1,
                "maximum": 0.99,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
    "moon": {
        "type": "object",
        "properties": {
            "n_points": {"type": "integer", "default": 200, "minimum": 10, "description": "Total number of points"},
            "noise": {"type": "number", "default": 0.1, "minimum": 0.0, "description": "Gaussian noise level"},
            "seed": {"type": "integer", "default": None, "description": "Random seed for reproducibility"},
            "train_ratio": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.1,
                "maximum": 0.99,
                "description": "Fraction of data for training",
            },
        },
        "required": [],
    },
}

# Maps generator names to their callable functions
_GENERATOR_FUNCTIONS = {
    "spiral": generate_spiral,
    "xor": generate_xor,
    "circle": generate_circle,
    "moon": generate_moon,
}


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

    def __init__(self, base_url: str = "http://fake-data:8100", api_key: Optional[str] = None) -> None:
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
            "status": "ok",
            "service": "juniper-data",
            "version": "fake",
            "uptime_seconds": 0.0,
        }

    def is_ready(self) -> bool:
        """Check if the fake service is ready (always True).

        Returns:
            True
        """
        return True

    def wait_for_ready(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
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
            "dtype": "float32",
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
            "artifact_url": f"/v1/datasets/{dataset_id}/artifact",
            "created_at": now,
        }

        self._datasets[dataset_id] = {
            "metadata": metadata,
            "arrays": arrays,
        }

        return metadata

    def create_spiral_dataset(
        self,
        n_spirals: int = 2,
        n_points_per_spiral: int = 100,
        noise: float = 0.1,
        seed: Optional[int] = None,
        algorithm: str = "modern",
        train_ratio: float = 0.8,
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

        return self.create_dataset("spiral", params)

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

    def list_datasets(self, limit: int = 100, offset: int = 0) -> List[str]:
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

    def get_preview(self, dataset_id: str, n: int = 100) -> Dict[str, Any]:
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
        n_available = min(n, X_full.shape[0], 1000)

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
                results.append({
                    "index": idx,
                    "dataset_id": resp["dataset_id"],
                    "generator": item["generator"],
                    "success": True,
                    "artifact_url": f"/v1/datasets/{resp['dataset_id']}/artifact",
                })
                total_created += 1
            except Exception as e:
                results.append({
                    "index": idx,
                    "generator": item.get("generator", "unknown"),
                    "success": False,
                    "error": str(e),
                })
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
