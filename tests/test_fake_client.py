#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     test_fake_client.py
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
#    Comprehensive tests for FakeDataClient — the in-memory test
#    replacement for JuniperDataClient. Covers health, generators,
#    dataset CRUD, artifact download, previews, data contract
#    validation, context manager lifecycle, and direct generator tests.
#####################################################################################################################################################################################################

"""Comprehensive tests for FakeDataClient and synthetic dataset generators.

Task 6.10 — Tests verify that FakeDataClient provides a faithful drop-in
replacement for JuniperDataClient, including the NPZ data contract
(X_train, y_train, X_test, y_test, X_full, y_full — all float32,
one-hot encoded labels).
"""

import io
from typing import Any, Dict

import numpy as np
import pytest

from juniper_data_client.exceptions import (
    JuniperDataNotFoundError,
    JuniperDataValidationError,
)
from juniper_data_client.testing import FakeDataClient
from juniper_data_client.testing.generators import (
    generate_circle,
    generate_moon,
    generate_spiral,
    generate_xor,
)

# ======================================================================
# Standard NPZ keys used by the Juniper data contract
# ======================================================================
_NPZ_KEYS = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}


# ======================================================================
# Health & readiness tests
# ======================================================================


@pytest.mark.unit
class TestHealthAndReadiness:
    """Tests for FakeDataClient health and readiness endpoints."""

    def test_health_check_returns_valid_dict(self, fake_client: FakeDataClient) -> None:
        """health_check() returns a dict with expected status fields."""
        result = fake_client.health_check()
        assert isinstance(result, dict), "health_check should return a dict"
        assert result["status"] == "healthy"
        assert result["service"] == "juniper-data"
        assert "version" in result
        assert "uptime_seconds" in result

    def test_is_ready_returns_true(self, fake_client: FakeDataClient) -> None:
        """is_ready() always returns True for the fake client."""
        assert fake_client.is_ready() is True

    def test_wait_for_ready_returns_true(self, fake_client: FakeDataClient) -> None:
        """wait_for_ready() returns True immediately without blocking."""
        assert fake_client.wait_for_ready(timeout=1.0) is True


# ======================================================================
# Generator catalog tests
# ======================================================================


@pytest.mark.unit
class TestGeneratorCatalog:
    """Tests for the generator listing and schema endpoints."""

    def test_list_generators_returns_four_generators(self, fake_client: FakeDataClient) -> None:
        """list_generators() returns exactly four generator entries."""
        generators = fake_client.list_generators()
        assert len(generators) == 4, f"Expected 4 generators, got {len(generators)}"

    def test_list_generators_has_expected_names(self, fake_client: FakeDataClient) -> None:
        """list_generators() includes spiral, xor, circle, and moon."""
        generators = fake_client.list_generators()
        names = {g["name"] for g in generators}
        expected = {"spiral", "xor", "circle", "moon"}
        assert names == expected, f"Expected generators {expected}, got {names}"

    def test_get_generator_schema_spiral(self, fake_client: FakeDataClient) -> None:
        """get_generator_schema('spiral') returns a schema with expected properties."""
        schema = fake_client.get_generator_schema("spiral")
        assert "properties" in schema
        props = schema["properties"]
        for key in ("n_spirals", "n_points_per_spiral", "noise", "seed", "algorithm", "train_ratio"):
            assert key in props, f"Spiral schema missing property: {key}"

    def test_get_generator_schema_xor(self, fake_client: FakeDataClient) -> None:
        """get_generator_schema('xor') returns a schema with expected properties."""
        schema = fake_client.get_generator_schema("xor")
        assert "properties" in schema
        props = schema["properties"]
        for key in ("n_points", "noise", "seed", "train_ratio"):
            assert key in props, f"XOR schema missing property: {key}"

    def test_get_generator_schema_unknown_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """get_generator_schema() raises JuniperDataNotFoundError for unknown generators."""
        with pytest.raises(JuniperDataNotFoundError, match="Generator not found"):
            fake_client.get_generator_schema("nonexistent_generator")


# ======================================================================
# Dataset creation tests
# ======================================================================


@pytest.mark.unit
class TestDatasetCreation:
    """Tests for creating datasets via FakeDataClient."""

    def test_create_dataset_spiral(self, fake_client: FakeDataClient) -> None:
        """create_dataset('spiral', ...) succeeds and returns valid metadata."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "seed": 42})
        assert "dataset_id" in result
        assert result["generator"] == "spiral"

    def test_create_dataset_xor(self, fake_client: FakeDataClient) -> None:
        """create_dataset('xor', ...) succeeds and returns valid metadata."""
        result = fake_client.create_dataset("xor", {"n_points": 100, "seed": 42})
        assert "dataset_id" in result
        assert result["generator"] == "xor"

    def test_create_dataset_circle(self, fake_client: FakeDataClient) -> None:
        """create_dataset('circle', ...) succeeds and returns valid metadata."""
        result = fake_client.create_dataset("circle", {"n_points": 200, "seed": 42})
        assert "dataset_id" in result
        assert result["generator"] == "circle"

    def test_create_dataset_moon(self, fake_client: FakeDataClient) -> None:
        """create_dataset('moon', ...) succeeds and returns valid metadata."""
        result = fake_client.create_dataset("moon", {"n_points": 200, "seed": 42})
        assert "dataset_id" in result
        assert result["generator"] == "moon"

    def test_create_dataset_unknown_generator_raises_validation_error(self, fake_client: FakeDataClient) -> None:
        """create_dataset() raises JuniperDataValidationError for unknown generators."""
        with pytest.raises(JuniperDataValidationError, match="Unknown generator"):
            fake_client.create_dataset("nonexistent", {})

    def test_create_spiral_dataset_convenience_method(self, fake_client: FakeDataClient) -> None:
        """create_spiral_dataset() convenience method delegates to create_dataset."""
        result = fake_client.create_spiral_dataset(seed=42)
        assert "dataset_id" in result
        assert result["generator"] == "spiral"

    def test_create_spiral_dataset_with_custom_params(self, fake_client: FakeDataClient) -> None:
        """create_spiral_dataset() accepts custom parameters and produces correct metadata."""
        result = fake_client.create_spiral_dataset(
            n_spirals=3,
            n_points_per_spiral=50,
            noise=0.2,
            seed=99,
            train_ratio=0.7,
        )
        assert result["params"]["n_spirals"] == 3
        assert result["params"]["n_points_per_spiral"] == 50
        assert result["params"]["noise"] == 0.2
        assert result["meta"]["n_classes"] == 3, "3-spiral dataset should have 3 classes"
        n_total = 3 * 50
        assert result["meta"]["n_full"] == n_total, f"Expected {n_total} total points"

    def test_create_dataset_returns_dataset_id(self, fake_client: FakeDataClient) -> None:
        """create_dataset() returns a string dataset_id in UUID format."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        dataset_id = result["dataset_id"]
        assert isinstance(dataset_id, str)
        # UUID has 36 chars with 4 hyphens: 8-4-4-4-12
        assert len(dataset_id) == 36, f"Expected UUID-length string, got length {len(dataset_id)}"

    def test_create_dataset_returns_metadata(self, fake_client: FakeDataClient) -> None:
        """create_dataset() response includes meta with array shapes and dtype."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        meta = result["meta"]
        assert "n_train" in meta
        assert "n_test" in meta
        assert "n_full" in meta
        assert "n_features" in meta
        assert "n_classes" in meta
        assert meta["dtype"] == "float32"
        assert meta["n_features"] == 2, "Spiral datasets have 2 features (x, y)"
        assert meta["n_classes"] == 2, "2-spiral dataset has 2 classes"
        assert meta["n_full"] == meta["n_train"] + meta["n_test"]


# ======================================================================
# Dataset listing tests
# ======================================================================


@pytest.mark.unit
class TestDatasetListing:
    """Tests for listing datasets."""

    def test_list_datasets_empty(self, fake_client: FakeDataClient) -> None:
        """list_datasets() returns an empty list when no datasets have been created."""
        result = fake_client.list_datasets()
        assert result == [], "Expected empty list for fresh client"

    def test_list_datasets_after_creation(self, fake_client: FakeDataClient) -> None:
        """list_datasets() includes IDs of all created datasets."""
        ids = []
        for gen in ("spiral", "xor", "circle"):
            r = fake_client.create_dataset(gen, {"seed": 1})
            ids.append(r["dataset_id"])

        listed = fake_client.list_datasets()
        assert len(listed) == 3
        for dataset_id in ids:
            assert dataset_id in listed, f"Dataset {dataset_id} not found in listing"

    def test_list_datasets_with_limit(self, fake_client: FakeDataClient) -> None:
        """list_datasets(limit=N) returns at most N dataset IDs."""
        for i in range(5):
            fake_client.create_dataset("spiral", {"seed": i})

        result = fake_client.list_datasets(limit=3)
        assert len(result) == 3, f"Expected 3 datasets with limit=3, got {len(result)}"

    def test_list_datasets_with_offset(self, fake_client: FakeDataClient) -> None:
        """list_datasets(offset=N) skips the first N dataset IDs."""
        all_ids = []
        for i in range(5):
            r = fake_client.create_dataset("spiral", {"seed": i})
            all_ids.append(r["dataset_id"])

        result = fake_client.list_datasets(offset=2)
        assert len(result) == 3, f"Expected 3 datasets with offset=2 from 5, got {len(result)}"
        # The returned IDs should be the last 3
        assert result == all_ids[2:]


# ======================================================================
# Dataset metadata tests
# ======================================================================


@pytest.mark.unit
class TestDatasetMetadata:
    """Tests for retrieving dataset metadata."""

    def test_get_dataset_metadata_valid_id(self, fake_client: FakeDataClient) -> None:
        """get_dataset_metadata() returns correct metadata for a valid dataset ID."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "seed": 42})
        dataset_id = result["dataset_id"]

        metadata = fake_client.get_dataset_metadata(dataset_id)
        assert metadata["dataset_id"] == dataset_id
        assert metadata["generator"] == "spiral"
        assert "meta" in metadata
        assert "artifact_url" in metadata
        assert dataset_id in metadata["artifact_url"]

    def test_get_dataset_metadata_invalid_id_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """get_dataset_metadata() raises JuniperDataNotFoundError for unknown IDs."""
        with pytest.raises(JuniperDataNotFoundError, match="Dataset not found"):
            fake_client.get_dataset_metadata("nonexistent-id-00000000")


# ======================================================================
# Dataset deletion tests
# ======================================================================


@pytest.mark.unit
class TestDatasetDeletion:
    """Tests for deleting datasets."""

    def test_delete_dataset_valid_id(self, fake_client: FakeDataClient) -> None:
        """delete_dataset() returns True for a valid dataset ID."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        dataset_id = result["dataset_id"]

        assert fake_client.delete_dataset(dataset_id) is True

    def test_delete_dataset_invalid_id_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """delete_dataset() raises JuniperDataNotFoundError for unknown IDs."""
        with pytest.raises(JuniperDataNotFoundError, match="Dataset not found"):
            fake_client.delete_dataset("nonexistent-id-00000000")

    def test_delete_then_list_shows_removal(self, fake_client: FakeDataClient) -> None:
        """Deleted datasets no longer appear in list_datasets()."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        r2 = fake_client.create_dataset("xor", {"seed": 2})

        fake_client.delete_dataset(r1["dataset_id"])

        remaining = fake_client.list_datasets()
        assert r1["dataset_id"] not in remaining, "Deleted dataset should not appear in listing"
        assert r2["dataset_id"] in remaining, "Non-deleted dataset should still appear"


# ======================================================================
# Artifact download tests
# ======================================================================


@pytest.mark.unit
class TestArtifactDownload:
    """Tests for downloading dataset artifacts."""

    def test_download_artifact_npz_returns_all_keys(self, fake_client: FakeDataClient) -> None:
        """download_artifact_npz() returns a dict containing all standard NPZ keys."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        assert set(arrays.keys()) == _NPZ_KEYS, f"Expected keys {_NPZ_KEYS}, got {set(arrays.keys())}"

    def test_download_artifact_npz_dtypes_are_float32(self, fake_client: FakeDataClient) -> None:
        """All arrays returned by download_artifact_npz() are float32."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        for key in _NPZ_KEYS:
            assert arrays[key].dtype == np.float32, f"Array '{key}' has dtype {arrays[key].dtype}, expected float32"

    def test_download_artifact_npz_shapes_consistent(self, fake_client: FakeDataClient) -> None:
        """Train and test feature arrays have the same number of features."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        n_features_train = arrays["X_train"].shape[1]
        n_features_test = arrays["X_test"].shape[1]
        n_features_full = arrays["X_full"].shape[1]

        assert n_features_train == n_features_test == n_features_full, (
            f"Feature dimensions inconsistent: train={n_features_train}, "
            f"test={n_features_test}, full={n_features_full}"
        )

        n_classes_train = arrays["y_train"].shape[1]
        n_classes_test = arrays["y_test"].shape[1]
        n_classes_full = arrays["y_full"].shape[1]

        assert n_classes_train == n_classes_test == n_classes_full, (
            f"Class dimensions inconsistent: train={n_classes_train}, "
            f"test={n_classes_test}, full={n_classes_full}"
        )

    def test_download_artifact_bytes_returns_valid_npz(self, fake_client: FakeDataClient) -> None:
        """download_artifact_bytes() returns bytes that can be loaded as a valid NPZ file."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        raw_bytes = fake_client.download_artifact_bytes(result["dataset_id"])

        assert isinstance(raw_bytes, bytes), "Expected bytes from download_artifact_bytes"
        assert len(raw_bytes) > 0, "NPZ bytes should not be empty"

        with np.load(io.BytesIO(raw_bytes)) as data:
            loaded_keys = set(data.files)
            assert loaded_keys == _NPZ_KEYS, f"Loaded NPZ keys {loaded_keys} != expected {_NPZ_KEYS}"
            for key in _NPZ_KEYS:
                assert data[key].dtype == np.float32, f"Loaded array '{key}' has wrong dtype"

    def test_download_artifact_npz_invalid_id_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """download_artifact_npz() raises JuniperDataNotFoundError for unknown IDs."""
        with pytest.raises(JuniperDataNotFoundError, match="Dataset not found"):
            fake_client.download_artifact_npz("nonexistent-id-00000000")


# ======================================================================
# Preview tests
# ======================================================================


@pytest.mark.unit
class TestPreview:
    """Tests for dataset preview endpoint."""

    def test_get_preview_returns_samples(self, fake_client: FakeDataClient) -> None:
        """get_preview() returns a dict with n_samples, X_sample, and y_sample."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        preview = fake_client.get_preview(result["dataset_id"])

        assert "n_samples" in preview
        assert "X_sample" in preview
        assert "y_sample" in preview
        assert preview["n_samples"] > 0
        assert len(preview["X_sample"]) == preview["n_samples"]
        assert len(preview["y_sample"]) == preview["n_samples"]

    def test_get_preview_with_n_parameter(self, fake_client: FakeDataClient) -> None:
        """get_preview(n=N) returns at most N samples."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        preview = fake_client.get_preview(result["dataset_id"], n=10)

        assert preview["n_samples"] == 10, f"Expected 10 samples, got {preview['n_samples']}"
        assert len(preview["X_sample"]) == 10
        assert len(preview["y_sample"]) == 10

    def test_get_preview_invalid_id_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """get_preview() raises JuniperDataNotFoundError for unknown IDs."""
        with pytest.raises(JuniperDataNotFoundError, match="Dataset not found"):
            fake_client.get_preview("nonexistent-id-00000000")


# ======================================================================
# Data contract tests
# ======================================================================


@pytest.mark.unit
class TestDataContract:
    """Tests verifying the NPZ data contract (Juniper ecosystem standard)."""

    def test_npz_has_standard_keys(self, fake_client: FakeDataClient) -> None:
        """NPZ artifacts contain exactly the 6 standard keys."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])
        assert set(arrays.keys()) == _NPZ_KEYS

    def test_train_test_split_ratio(self, fake_client: FakeDataClient) -> None:
        """Default train/test split is approximately 80/20."""
        result = fake_client.create_dataset(
            "spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42}
        )
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        n_train = arrays["X_train"].shape[0]
        n_full = arrays["X_full"].shape[0]
        actual_ratio = n_train / n_full

        assert 0.75 <= actual_ratio <= 0.85, (
            f"Train ratio {actual_ratio:.3f} outside expected range [0.75, 0.85] "
            f"(n_train={n_train}, n_full={n_full})"
        )

    def test_full_dataset_is_union_of_train_test(self, fake_client: FakeDataClient) -> None:
        """X_full/y_full contain the same number of samples as X_train + X_test."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        n_train = arrays["X_train"].shape[0]
        n_test = arrays["X_test"].shape[0]
        n_full = arrays["X_full"].shape[0]

        assert n_full == n_train + n_test, (
            f"X_full ({n_full}) != X_train ({n_train}) + X_test ({n_test})"
        )

        # Same check for labels
        n_y_train = arrays["y_train"].shape[0]
        n_y_test = arrays["y_test"].shape[0]
        n_y_full = arrays["y_full"].shape[0]

        assert n_y_full == n_y_train + n_y_test, (
            f"y_full ({n_y_full}) != y_train ({n_y_train}) + y_test ({n_y_test})"
        )

    def test_y_arrays_are_one_hot_encoded(self, fake_client: FakeDataClient) -> None:
        """Label arrays are one-hot encoded: each row sums to 1.0 with values in {0, 1}."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 100, "seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        for key in ("y_train", "y_test", "y_full"):
            y = arrays[key]
            # Each row should sum to exactly 1.0
            row_sums = y.sum(axis=1)
            np.testing.assert_allclose(
                row_sums,
                np.ones(y.shape[0], dtype=np.float32),
                atol=1e-6,
                err_msg=f"{key} rows do not sum to 1.0",
            )
            # Values should be 0.0 or 1.0 only
            unique_values = set(np.unique(y).tolist())
            assert unique_values <= {0.0, 1.0}, (
                f"{key} contains values other than 0.0 and 1.0: {unique_values}"
            )

    def test_spiral_two_classes_for_two_spirals(self, fake_client: FakeDataClient) -> None:
        """A 2-spiral dataset produces labels with exactly 2 classes."""
        result = fake_client.create_dataset("spiral", {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42})
        arrays = fake_client.download_artifact_npz(result["dataset_id"])

        n_classes = arrays["y_full"].shape[1]
        assert n_classes == 2, f"Expected 2 classes for 2-spiral dataset, got {n_classes}"


# ======================================================================
# Context manager tests
# ======================================================================


@pytest.mark.unit
class TestContextManager:
    """Tests for FakeDataClient context manager lifecycle."""

    def test_context_manager_usage(self) -> None:
        """FakeDataClient works correctly as a context manager."""
        with FakeDataClient() as client:
            result = client.create_dataset("spiral", {"seed": 42})
            assert "dataset_id" in result
            # Should be able to retrieve the dataset inside the context
            arrays = client.download_artifact_npz(result["dataset_id"])
            assert "X_train" in arrays

    def test_close_clears_state(self) -> None:
        """close() clears internal datasets and sets the closed flag."""
        client = FakeDataClient()
        client.create_dataset("spiral", {"seed": 42})
        assert len(client.list_datasets()) == 1

        client.close()

        assert client._closed is True
        assert len(client.list_datasets()) == 0, "Datasets should be cleared after close()"


# ======================================================================
# Generator tests (directly testing generators.py)
# ======================================================================


@pytest.mark.unit
class TestGenerators:
    """Direct tests for the synthetic dataset generator functions."""

    def test_generate_spiral_basic(self) -> None:
        """generate_spiral() returns a dict with all standard NPZ keys."""
        arrays = generate_spiral(n_spirals=2, n_points_per_spiral=50, seed=42)
        assert set(arrays.keys()) == _NPZ_KEYS
        n_total = 2 * 50
        assert arrays["X_full"].shape[0] == n_total, f"Expected {n_total} total points"
        assert arrays["X_full"].shape[1] == 2, "Spiral features should be 2-dimensional"

    def test_generate_spiral_reproducible_with_seed(self) -> None:
        """generate_spiral() produces identical output when called with the same seed."""
        arrays_a = generate_spiral(n_spirals=2, n_points_per_spiral=50, seed=123)
        arrays_b = generate_spiral(n_spirals=2, n_points_per_spiral=50, seed=123)

        for key in _NPZ_KEYS:
            np.testing.assert_array_equal(
                arrays_a[key],
                arrays_b[key],
                err_msg=f"Array '{key}' differs between runs with same seed",
            )

    def test_generate_xor_basic(self) -> None:
        """generate_xor() returns arrays with correct structure."""
        arrays = generate_xor(n_points=100, seed=42)
        assert set(arrays.keys()) == _NPZ_KEYS
        assert arrays["X_full"].shape[0] == 100
        assert arrays["X_full"].shape[1] == 2, "XOR features should be 2-dimensional"
        assert arrays["y_full"].shape[1] == 2, "XOR should have 2 classes"

    def test_generate_circle_basic(self) -> None:
        """generate_circle() returns arrays with correct structure."""
        arrays = generate_circle(n_points=200, seed=42)
        assert set(arrays.keys()) == _NPZ_KEYS
        assert arrays["X_full"].shape[0] == 200
        assert arrays["X_full"].shape[1] == 2, "Circle features should be 2-dimensional"
        assert arrays["y_full"].shape[1] == 2, "Circle should have 2 classes"

    def test_generate_moon_basic(self) -> None:
        """generate_moon() returns arrays with correct structure."""
        arrays = generate_moon(n_points=200, seed=42)
        assert set(arrays.keys()) == _NPZ_KEYS
        assert arrays["X_full"].shape[0] == 200
        assert arrays["X_full"].shape[1] == 2, "Moon features should be 2-dimensional"
        assert arrays["y_full"].shape[1] == 2, "Moon should have 2 classes"

    def test_all_generators_return_float32(self) -> None:
        """All four generators produce exclusively float32 arrays."""
        generators = {
            "spiral": generate_spiral(seed=42),
            "xor": generate_xor(seed=42),
            "circle": generate_circle(seed=42),
            "moon": generate_moon(seed=42),
        }
        for gen_name, arrays in generators.items():
            for key in _NPZ_KEYS:
                assert arrays[key].dtype == np.float32, (
                    f"Generator '{gen_name}', array '{key}' has dtype "
                    f"{arrays[key].dtype}, expected float32"
                )
