#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     __init__.py
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
#    Testing submodule for juniper-data-client. Provides FakeDataClient,
#    a drop-in replacement for JuniperDataClient that generates synthetic
#    datasets in-memory without requiring a running JuniperData service.
#####################################################################################################################################################################################################

"""Testing utilities for juniper-data-client.

Provides :class:`FakeDataClient` for use in unit tests of consumers
like JuniperCascor and juniper-canopy, as well as standalone synthetic
dataset generators.

Usage::

    from juniper_data_client.testing import FakeDataClient

    with FakeDataClient() as client:
        result = client.create_spiral_dataset(n_spirals=2, seed=42)
        arrays = client.download_artifact_npz(result["dataset_id"])
"""

from juniper_data_client.testing.fake_client import FakeDataClient
from juniper_data_client.testing.generators import (
    generate_circle,
    generate_moon,
    generate_spiral,
    generate_xor,
)

__all__ = [
    "FakeDataClient",
    "generate_circle",
    "generate_moon",
    "generate_spiral",
    "generate_xor",
]
