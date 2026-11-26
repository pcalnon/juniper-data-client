"""Pin the publish docs' numbers to the publish workflow: the bounds the docs state are the bounds the steps run.

``docs/REFERENCE.md`` § Publishing and Recovery gives an operator the numbers that decide whether a failed install check
was index lag: how many times the check tries, how far apart, how long it waits first, and the error it stops with. Each
is read here from ``publish.yml`` and required in the doc, so a change to either side alone fails. A module of its own, as
juniper-cascor's is (juniper-cascor#777): ``test_publish_one_artifact.py`` reads the workflow alone.

Project: juniper-data-client
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ci.yml's unit lane runs ``pytest tests/`` with no marker filter, so this module runs there; the mark files it with
# the unit tests, and the integration lane (``-m integration``) leaves it out. Module-level, so it covers tests added later.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "publish.yml"
DOC = REPO / "docs" / "REFERENCE.md"
INSTALL = "verify-testpypi-install"
INSTALL_STEP = "Verify TestPyPI install"


def _doc() -> str:
    """The doc's text with each run of whitespace made one space: a rewrapped paragraph states the same bounds."""
    return " ".join(DOC.read_text(encoding="utf-8").split())


def _install_run() -> str:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][INSTALL]["steps"]
    found = [s for s in steps if s.get("name") == INSTALL_STEP]
    assert len(found) == 1, f"expected exactly one step named {INSTALL_STEP!r}"
    return str(found[0]["run"])


def _retry() -> tuple[int, list[int], list[int]]:
    """The install check's (tries, the waits inside its loop, the waits before it), read from its commands."""
    commands = [line.strip() for line in _install_run().splitlines() if line.strip() and not line.strip().startswith("#")]
    loops = [i for i, c in enumerate(commands) if re.fullmatch(r"for attempt in [0-9 ]+; do", c)]
    assert len(loops) == 1, "expected one retry loop"
    start, end = loops[0], commands.index("done", loops[0])
    tries = len(commands[start].split()[3:-1])
    waits = [int(c.split()[1]) for c in commands[start:end] if re.fullmatch(r"sleep [0-9]+", c)]
    before = [int(c.split()[1]) for c in commands[:start] if re.fullmatch(r"sleep [0-9]+", c)]
    return tries, waits, before


class TestPublishDocsStateTheBounds:
    def test_the_install_checks_retry(self):
        tries, waits, before = _retry()
        # One loop: its tries, a wait between them and none after the last; and one wait before the first try.
        assert len(waits) == 1
        assert len(before) == 1
        doc = _doc()
        assert f"It makes {tries} tries, {waits[0]} s apart, after a {before[0]} s wait" in doc
        # The error the check stops with after its last try, as the step prints it and as the doc quotes it.
        error = f"is not installable from TestPyPI after {tries} tries (index lag?)"
        assert error in _install_run()
        assert error in doc
