#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     test_file_header_versions.py
# Author:        Paul Calnon
# Version:       0.4.2
#
# Date Created:  2026-08-24
# Last Modified: 2026-08-24
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    The decorative ``Version:`` fields in file headers must agree with the
#    real package version (defect-register APD-DCLIENT-005).
#####################################################################################################################################################################################################

"""The file-header ``Version:`` fields must agree with the package version.

APD-DCLIENT-005: four distinct values had accumulated across nine decorative
header copies (the register row itself knew of six) while ``pyproject.toml``
and ``__init__.__version__`` agreed on the real version. Decorative copies
drift silently; this test converts the drift into a loud failure that names
the lagging files — including this file's own header.
"""

import re
from pathlib import Path

import pytest

import juniper_data_client

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HEADER_VERSION = re.compile(r"^(?:# )?Version:\s+(\S+)\s*$", re.MULTILINE)
_SCAN_ROOTS = ("juniper_data_client", "tests")


def _files_with_header_versions():
    for root in _SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            match = _HEADER_VERSION.search(path.read_text(encoding="utf-8")[:4096])
            if match:
                yield path, match.group(1)


@pytest.mark.unit
def test_every_file_header_version_matches_the_package_version():
    found = list(_files_with_header_versions())
    # Anti-vacuous guard: at least the nine files carrying the header today
    # (plus this one) must match the scan — a regex or path typo must not let
    # it quietly match nothing and pass.
    assert len(found) >= 9, f"header scan matched only {len(found)} files — scan broken?"
    lagging = {str(path.relative_to(_REPO_ROOT)): version for path, version in found if version != juniper_data_client.__version__}
    assert not lagging, f"file headers disagree with __version__={juniper_data_client.__version__}: {lagging}"
