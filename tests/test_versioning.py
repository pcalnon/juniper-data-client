#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     test_versioning.py
# Author:        Paul Calnon
# Version:       0.3.2
#
# Date Created:  2026-04-01
# Last Modified: 2026-04-01
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Tests for dataset versioning support — create_dataset with
#    name/version fields, list_versions, and get_latest methods
#    on the in-memory FakeDataClient.
#####################################################################################################################################################################################################

"""Tests for dataset versioning (CAN-DEF-005 Phase 2).

Covers create_dataset with optional versioning parameters (name,
description, created_by, parent_dataset_id), version auto-increment,
list_versions, and get_latest on the FakeDataClient.
"""

import pytest

from juniper_data_client.exceptions import JuniperDataNotFoundError
from juniper_data_client.testing import FakeDataClient

# ======================================================================
# create_dataset versioning fields
# ======================================================================


@pytest.mark.unit
class TestCreateDatasetVersioning:
    """Tests for versioning fields in create_dataset()."""

    def test_create_with_name_assigns_version_fields(self, fake_client: FakeDataClient) -> None:
        """create_dataset with a name populates dataset_name and dataset_version in meta."""
        result = fake_client.create_dataset("spiral", {"seed": 42}, name="my-spiral")
        meta = result["meta"]
        assert meta["dataset_name"] == "my-spiral"
        assert meta["dataset_version"] == 1

    def test_multiple_creates_same_name_increment_version(self, fake_client: FakeDataClient) -> None:
        """Creating datasets with the same name auto-increments the version number."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1}, name="experiment-a")
        r2 = fake_client.create_dataset("spiral", {"seed": 2}, name="experiment-a")
        r3 = fake_client.create_dataset("spiral", {"seed": 3}, name="experiment-a")

        assert r1["meta"]["dataset_version"] == 1
        assert r2["meta"]["dataset_version"] == 2
        assert r3["meta"]["dataset_version"] == 3

    def test_create_without_name_has_no_version_fields(self, fake_client: FakeDataClient) -> None:
        """create_dataset without a name does not include dataset_name or dataset_version."""
        result = fake_client.create_dataset("spiral", {"seed": 42})
        meta = result["meta"]
        assert "dataset_name" not in meta
        assert "dataset_version" not in meta

    def test_description_stored_in_meta(self, fake_client: FakeDataClient) -> None:
        """The description field is stored in the meta dict when provided."""
        result = fake_client.create_dataset("spiral", {"seed": 42}, name="desc-test", description="A test dataset")
        assert result["meta"]["description"] == "A test dataset"

    def test_created_by_stored_in_meta(self, fake_client: FakeDataClient) -> None:
        """The created_by field is stored in the meta dict when provided."""
        result = fake_client.create_dataset("spiral", {"seed": 42}, name="author-test", created_by="unit-test")
        assert result["meta"]["created_by"] == "unit-test"

    def test_parent_dataset_id_stored_and_returned(self, fake_client: FakeDataClient) -> None:
        """parent_dataset_id is stored in meta and returned in the response."""
        parent = fake_client.create_dataset("spiral", {"seed": 1}, name="parent-ds")
        child = fake_client.create_dataset("spiral", {"seed": 2}, name="child-ds", parent_dataset_id=parent["dataset_id"])

        assert child["meta"]["parent_dataset_id"] == parent["dataset_id"]

    def test_versioning_fields_in_metadata_response(self, fake_client: FakeDataClient) -> None:
        """Versioning fields are visible via get_dataset_metadata after creation."""
        result = fake_client.create_dataset("xor", {"seed": 42}, name="versioned-xor", description="XOR v1", created_by="tester")
        metadata = fake_client.get_dataset_metadata(result["dataset_id"])

        assert metadata["meta"]["dataset_name"] == "versioned-xor"
        assert metadata["meta"]["dataset_version"] == 1
        assert metadata["meta"]["description"] == "XOR v1"
        assert metadata["meta"]["created_by"] == "tester"

    def test_different_names_have_independent_version_counters(self, fake_client: FakeDataClient) -> None:
        """Version counters are independent per dataset name."""
        r1 = fake_client.create_dataset("spiral", {"seed": 1}, name="alpha")
        r2 = fake_client.create_dataset("spiral", {"seed": 2}, name="beta")
        r3 = fake_client.create_dataset("spiral", {"seed": 3}, name="alpha")

        assert r1["meta"]["dataset_version"] == 1
        assert r2["meta"]["dataset_version"] == 1  # independent counter
        assert r3["meta"]["dataset_version"] == 2

    def test_optional_fields_omitted_when_not_provided(self, fake_client: FakeDataClient) -> None:
        """description, created_by, parent_dataset_id are absent when not supplied."""
        result = fake_client.create_dataset("spiral", {"seed": 42}, name="minimal")
        meta = result["meta"]
        assert "description" not in meta
        assert "created_by" not in meta
        assert "parent_dataset_id" not in meta


# ======================================================================
# list_versions tests
# ======================================================================


@pytest.mark.unit
class TestListVersions:
    """Tests for FakeDataClient.list_versions()."""

    def test_list_versions_returns_sorted_versions(self, fake_client: FakeDataClient) -> None:
        """list_versions returns versions sorted by version number ascending."""
        fake_client.create_dataset("spiral", {"seed": 1}, name="sorted-test")
        fake_client.create_dataset("spiral", {"seed": 2}, name="sorted-test")
        fake_client.create_dataset("spiral", {"seed": 3}, name="sorted-test")

        result = fake_client.list_versions("sorted-test")

        assert result["dataset_name"] == "sorted-test"
        assert result["total"] == 3
        assert result["latest_version"] == 3

        versions = result["versions"]
        assert len(versions) == 3
        assert versions[0]["meta"]["dataset_version"] == 1
        assert versions[1]["meta"]["dataset_version"] == 2
        assert versions[2]["meta"]["dataset_version"] == 3

    def test_list_versions_returns_empty_for_unknown_name(self, fake_client: FakeDataClient) -> None:
        """list_versions for a name with no datasets returns total=0 and empty list."""
        result = fake_client.list_versions("nonexistent-dataset")

        assert result["dataset_name"] == "nonexistent-dataset"
        assert result["total"] == 0
        assert result["versions"] == []
        assert result["latest_version"] is None

    def test_list_versions_excludes_other_names(self, fake_client: FakeDataClient) -> None:
        """list_versions only returns versions matching the requested name."""
        fake_client.create_dataset("spiral", {"seed": 1}, name="alpha")
        fake_client.create_dataset("spiral", {"seed": 2}, name="beta")
        fake_client.create_dataset("spiral", {"seed": 3}, name="alpha")

        result = fake_client.list_versions("alpha")

        assert result["total"] == 2
        for v in result["versions"]:
            assert v["meta"]["dataset_name"] == "alpha"

    def test_list_versions_excludes_unnamed_datasets(self, fake_client: FakeDataClient) -> None:
        """list_versions does not include datasets created without a name."""
        fake_client.create_dataset("spiral", {"seed": 1})  # no name
        fake_client.create_dataset("spiral", {"seed": 2}, name="named")

        result = fake_client.list_versions("named")
        assert result["total"] == 1


# ======================================================================
# get_latest tests
# ======================================================================


@pytest.mark.unit
class TestGetLatest:
    """Tests for FakeDataClient.get_latest()."""

    def test_get_latest_returns_highest_version(self, fake_client: FakeDataClient) -> None:
        """get_latest returns the dataset with the highest version number."""
        fake_client.create_dataset("spiral", {"seed": 1}, name="latest-test")
        fake_client.create_dataset("spiral", {"seed": 2}, name="latest-test")
        r3 = fake_client.create_dataset("spiral", {"seed": 3}, name="latest-test")

        latest = fake_client.get_latest("latest-test")

        assert latest["dataset_id"] == r3["dataset_id"]
        assert latest["meta"]["dataset_version"] == 3

    def test_get_latest_raises_not_found_for_unknown_name(self, fake_client: FakeDataClient) -> None:
        """get_latest raises JuniperDataNotFoundError when no versions exist."""
        with pytest.raises(JuniperDataNotFoundError, match="No versions found"):
            fake_client.get_latest("nonexistent-dataset")

    def test_get_latest_with_single_version(self, fake_client: FakeDataClient) -> None:
        """get_latest works correctly when only one version exists."""
        r1 = fake_client.create_dataset("xor", {"seed": 42}, name="single-version")

        latest = fake_client.get_latest("single-version")

        assert latest["dataset_id"] == r1["dataset_id"]
        assert latest["meta"]["dataset_version"] == 1

    def test_get_latest_returns_full_metadata(self, fake_client: FakeDataClient) -> None:
        """get_latest returns a complete metadata dict with all standard fields."""
        fake_client.create_dataset("circles", {"seed": 42}, name="full-meta", description="Latest circle", created_by="test-suite")

        latest = fake_client.get_latest("full-meta")

        assert "dataset_id" in latest
        assert latest["generator"] == "circles"
        assert "params" in latest
        assert "artifact_url" in latest
        assert latest["meta"]["dataset_name"] == "full-meta"
        assert latest["meta"]["description"] == "Latest circle"
        assert latest["meta"]["created_by"] == "test-suite"
