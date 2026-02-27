#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     conftest.py
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
#    Shared pytest fixtures for juniper-data-client tests.
#####################################################################################################################################################################################################

"""Shared pytest fixtures for juniper-data-client tests."""

from typing import Generator

import pytest

from juniper_data_client.testing import FakeDataClient


@pytest.fixture
def fake_client() -> Generator[FakeDataClient, None, None]:
    """Provide a fresh FakeDataClient instance, closed after the test."""
    client = FakeDataClient()
    yield client
    client.close()
