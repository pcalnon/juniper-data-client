#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     test_fake_client_batch.py
# Author:        Paul Calnon
# Version:       0.3.1
#
# Date Created:  2026-03-31
# Last Modified: 2026-03-31
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Tests for FakeDataClient batch operations — batch_delete,
#    batch_create, batch_update_tags, and batch_export.
#####################################################################################################################################################################################################

"""Tests for FakeDataClient batch operations.

Covers batch_delete, batch_create, batch_update_tags, and batch_export
methods on the in-memory FakeDataClient.
"""

import io
import zipfile
from typing import Any, Dict

import numpy as np
import pytest

from juniper_data_client.exceptions import JuniperDataNotFoundError
from juniper_data_client.testing import FakeDataClient

# ======================================================================
# Standard NPZ keys used by the Juniper data contract
# ======================================================================
_NPZ_KEYS = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}


# ======================================================================
# batch_delete tests
# ======================================================================


@pytest.mark.unit
class TestBatchDelete:
    """Tests for FakeDataClient.batch_delete()."""

    def test_delete_existing_datasets(self, fake_client: FakeDataClient) -> None:
        """batch_delete() returns all IDs in the deleted list when all exist."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        r2 = fake_client.create_dataset("xor", {"seed": 2})
        ids = [r1["dataset_id"], r2["dataset_id"]]

        result = fake_client.batch_delete(ids)

        assert set(result["deleted"]) == set(ids)
        assert result["not_found"] == []
        assert result["total_deleted"] == 2

    def test_delete_mix_existing_and_nonexisting(self, fake_client: FakeDataClient) -> None:
        """batch_delete() separates found and not-found IDs correctly."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        real_id = r1["dataset_id"]
        fake_id = "nonexistent-id-00000000"

        result = fake_client.batch_delete([real_id, fake_id])

        assert result["deleted"] == [real_id]
        assert result["not_found"] == [fake_id]
        assert result["total_deleted"] == 1

    def test_total_deleted_count_correct(self, fake_client: FakeDataClient) -> None:
        """batch_delete() total_deleted matches the number of actually deleted datasets."""
        ids = []
        for i in range(4):
            r = fake_client.create_dataset("spiral", {"seed": i})
            ids.append(r["dataset_id"])

        # Delete only the first two plus a bogus ID
        to_delete = ids[:2] + ["does-not-exist"]
        result = fake_client.batch_delete(to_delete)

        assert result["total_deleted"] == 2
        assert len(result["deleted"]) == 2
        assert len(result["not_found"]) == 1

        # Remaining datasets should still be listed
        remaining = fake_client.list_datasets()
        assert len(remaining) == 2
        assert set(remaining) == set(ids[2:])


# ======================================================================
# batch_create tests
# ======================================================================


@pytest.mark.unit
class TestBatchCreate:
    """Tests for FakeDataClient.batch_create()."""

    def test_create_two_datasets_successfully(self, fake_client: FakeDataClient) -> None:
        """batch_create() with valid specs returns total_created=2."""
        specs = [
            {"generator": "spiral", "params": {"seed": 1}},
            {"generator": "xor", "params": {"seed": 2}},
        ]

        result = fake_client.batch_create(specs)

        assert result["total_created"] == 2
        assert result["total_failed"] == 0
        assert len(result["results"]) == 2

    def test_invalid_generator_causes_failure(self, fake_client: FakeDataClient) -> None:
        """batch_create() records a failure for an unknown generator."""
        specs = [
            {"generator": "spiral", "params": {"seed": 1}},
            {"generator": "nonexistent", "params": {}},
        ]

        result = fake_client.batch_create(specs)

        assert result["total_created"] == 1
        assert result["total_failed"] == 1

        # The failed entry should have success=False and an error message
        failed = [r for r in result["results"] if not r["success"]]
        assert len(failed) == 1
        assert "error" in failed[0]

    def test_result_fields_present(self, fake_client: FakeDataClient) -> None:
        """Each batch_create result entry has index, dataset_id, generator, and success fields."""
        specs = [
            {"generator": "circle", "params": {"seed": 10}},
            {"generator": "moon", "params": {"seed": 20}},
        ]

        result = fake_client.batch_create(specs)

        for entry in result["results"]:
            assert "index" in entry
            assert "generator" in entry
            assert "success" in entry
            if entry["success"]:
                assert "dataset_id" in entry

    def test_created_datasets_accessible_via_metadata(self, fake_client: FakeDataClient) -> None:
        """Datasets created via batch_create are retrievable with get_dataset_metadata."""
        specs = [
            {"generator": "spiral", "params": {"n_spirals": 3, "seed": 42}},
            {"generator": "xor", "params": {"n_points": 50, "seed": 7}},
        ]

        result = fake_client.batch_create(specs)

        for entry in result["results"]:
            assert entry["success"] is True
            metadata = fake_client.get_dataset_metadata(entry["dataset_id"])
            assert metadata["dataset_id"] == entry["dataset_id"]
            assert metadata["generator"] == entry["generator"]

    def test_result_index_matches_input_order(self, fake_client: FakeDataClient) -> None:
        """batch_create result indices correspond to the input list positions."""
        specs = [
            {"generator": "spiral", "params": {"seed": 1}},
            {"generator": "nonexistent", "params": {}},
            {"generator": "xor", "params": {"seed": 3}},
        ]

        result = fake_client.batch_create(specs)

        for i, entry in enumerate(result["results"]):
            assert entry["index"] == i


# ======================================================================
# batch_update_tags tests
# ======================================================================


@pytest.mark.unit
class TestBatchUpdateTags:
    """Tests for FakeDataClient.batch_update_tags()."""

    def test_add_tags_to_existing_datasets(self, fake_client: FakeDataClient) -> None:
        """batch_update_tags() adds tags to datasets that exist."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        r2 = fake_client.create_dataset("xor", {"seed": 2})
        ids = [r1["dataset_id"], r2["dataset_id"]]

        result = fake_client.batch_update_tags(ids, add_tags=["training", "v1"])

        assert set(result["updated"]) == set(ids)
        assert result["not_found"] == []
        assert result["total_updated"] == 2

    def test_remove_tags_from_existing_datasets(self, fake_client: FakeDataClient) -> None:
        """batch_update_tags() removes specified tags from datasets."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        dataset_id = r1["dataset_id"]

        # Add tags first
        fake_client.batch_update_tags([dataset_id], add_tags=["alpha", "beta", "gamma"])
        # Remove one tag
        fake_client.batch_update_tags([dataset_id], remove_tags=["beta"])

        metadata = fake_client.get_dataset_metadata(dataset_id)
        assert "beta" not in metadata["tags"]
        assert "alpha" in metadata["tags"]
        assert "gamma" in metadata["tags"]

    def test_mix_existing_and_nonexisting(self, fake_client: FakeDataClient) -> None:
        """batch_update_tags() separates updated and not_found IDs."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        real_id = r1["dataset_id"]
        fake_id = "nonexistent-id-00000000"

        result = fake_client.batch_update_tags([real_id, fake_id], add_tags=["test"])

        assert result["updated"] == [real_id]
        assert result["not_found"] == [fake_id]
        assert result["total_updated"] == 1

    def test_tags_actually_updated_via_metadata(self, fake_client: FakeDataClient) -> None:
        """Tags set by batch_update_tags are visible in get_dataset_metadata."""
        r1 = fake_client.create_dataset("circle", {"seed": 42})
        dataset_id = r1["dataset_id"]

        fake_client.batch_update_tags([dataset_id], add_tags=["production", "experiment-7"])

        metadata = fake_client.get_dataset_metadata(dataset_id)
        assert "production" in metadata["tags"]
        assert "experiment-7" in metadata["tags"]

    def test_add_and_remove_tags_simultaneously(self, fake_client: FakeDataClient) -> None:
        """batch_update_tags() can add and remove tags in a single call."""
        r1 = fake_client.create_dataset("moon", {"seed": 1})
        dataset_id = r1["dataset_id"]

        # Seed with initial tags
        fake_client.batch_update_tags([dataset_id], add_tags=["old-tag", "keep-tag"])
        # Add new and remove old in one call
        fake_client.batch_update_tags([dataset_id], add_tags=["new-tag"], remove_tags=["old-tag"])

        metadata = fake_client.get_dataset_metadata(dataset_id)
        assert "new-tag" in metadata["tags"]
        assert "keep-tag" in metadata["tags"]
        assert "old-tag" not in metadata["tags"]


# ======================================================================
# batch_export tests
# ======================================================================


@pytest.mark.unit
class TestBatchExport:
    """Tests for FakeDataClient.batch_export()."""

    def test_export_existing_datasets_returns_zip_bytes(self, fake_client: FakeDataClient) -> None:
        """batch_export() returns bytes that can be loaded as a valid ZIP archive."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        r2 = fake_client.create_dataset("xor", {"seed": 2})
        ids = [r1["dataset_id"], r2["dataset_id"]]

        raw = fake_client.batch_export(ids)

        assert isinstance(raw, bytes)
        assert len(raw) > 0
        # Verify it is a valid ZIP
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            assert zf.testzip() is None, "ZIP archive has corrupt entries"

    def test_zip_contains_npz_files_with_correct_ids(self, fake_client: FakeDataClient) -> None:
        """The exported ZIP contains one NPZ file per dataset, named by dataset ID."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1})
        r2 = fake_client.create_dataset("circle", {"seed": 2})
        ids = [r1["dataset_id"], r2["dataset_id"]]

        raw = fake_client.batch_export(ids)

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = set(zf.namelist())
            expected = {f"{did}.npz" for did in ids}
            assert names == expected, f"Expected ZIP entries {expected}, got {names}"

            # Each NPZ should contain the standard keys
            for did in ids:
                npz_bytes = zf.read(f"{did}.npz")
                with np.load(io.BytesIO(npz_bytes)) as data:
                    assert set(data.files) == _NPZ_KEYS, f"NPZ for {did} missing standard keys"

    def test_all_nonexisting_raises_not_found(self, fake_client: FakeDataClient) -> None:
        """batch_export() raises JuniperDataNotFoundError when none of the IDs exist."""
        with pytest.raises(JuniperDataNotFoundError, match="None of the requested datasets were found"):
            fake_client.batch_export(["no-such-id-1", "no-such-id-2"])

    def test_mix_existing_and_nonexisting_exports_found_only(self, fake_client: FakeDataClient) -> None:
        """batch_export() with a mix of valid/invalid IDs exports only the valid ones."""
        r1 = fake_client.create_dataset("moon", {"seed": 1})
        real_id = r1["dataset_id"]
        fake_id = "nonexistent-id-00000000"

        raw = fake_client.batch_export([real_id, fake_id])

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            assert names == [f"{real_id}.npz"], f"Expected only {real_id}.npz, got {names}"
