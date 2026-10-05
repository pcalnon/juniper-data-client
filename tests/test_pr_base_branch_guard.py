"""The base-branch guard is the only check that runs when a PR targets a feature branch.

Rulesets are scoped to the default branch, so a stacked PR has no required checks.
This shell is what turns that silent merge into a red one. Three outcomes are
load-bearing and easy to invert:

- ``merge_group`` has no PR base. Falling through fails every queued merge.
- An unresolved default branch fails open. Failing closed would fail every PR.
- The ``stacked-pr`` hatch is the string ``true`` from ``contains()``. ``false``
  is non-empty; treating any non-empty value as the hatch would pass every PR.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with a fixed argv
import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_NAME = "pr-base-branch-guard.yml"
JOB_ID = "guard-base-branch"
STEP_NAME = "Require PR base to be the default branch"


def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        if (cur / ".github" / "workflows").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise AssertionError(f"could not locate repo root with .github/workflows from {Path(__file__)}")


def _load_script() -> str:
    wf = _repo_root() / ".github" / "workflows" / WORKFLOW_NAME
    if not wf.is_file():
        raise AssertionError(f"{WORKFLOW_NAME} not present at {wf}")
    doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
    steps = doc.get("jobs", {}).get(JOB_ID, {}).get("steps", [])
    step = next((s for s in steps if s.get("name") == STEP_NAME), None)
    if step is None or "run" not in step:
        raise AssertionError(f"could not locate {STEP_NAME!r} in {WORKFLOW_NAME}")
    script = step["run"]
    if not isinstance(script, str) or "HAS_BYPASS" not in script:
        raise AssertionError(f"{STEP_NAME!r} is not the base-branch guard shell")
    return script


@pytest.fixture(scope="module")
def script() -> str:
    return _load_script()


def _run(script: str, *, event: str, base: str, default: str, bypass: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script_path = root / "guard.sh"
        script_path.write_text(script, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root),
            "LANG": "C",
            "EVENT_NAME": event,
            "BASE_REF": base,
            "DEFAULT_BRANCH": default,
            "HAS_BYPASS": bypass,
        }
        return subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, no network
            ["bash", str(script_path)],
            cwd=root,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


def test_default_branch_passes(script: str) -> None:
    result = _run(script, event="pull_request", base="main", default="main", bypass="false")

    assert result.returncode == 0
    assert "::notice" in result.stdout
    assert "::error" not in result.stdout


def test_exact_true_bypass_warns_and_passes(script: str) -> None:
    result = _run(script, event="pull_request", base="feature/stack", default="main", bypass="true")

    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "stacked-pr" in result.stdout
    assert "feature/stack" in result.stdout
    assert "::error" not in result.stdout


@pytest.mark.parametrize(
    ("base", "bypass"),
    [
        ("feature/stack", "false"),
        ("feature/stack", ""),
        ("feature/stack", "True"),
        ("feature/stack", "TRUE"),
        ("feature/stack", "1"),
        ("feature/stack", "yes"),
        ("feature/stack", "true "),
        ("feature/stack", " true"),
        ("feature/stack", "stacked-pr"),
        ("", "false"),
        ("Main", "false"),
        ("main ", "false"),
    ],
)
def test_a_base_that_is_not_the_default_fails_closed(script: str, base: str, bypass: str) -> None:
    """``false`` is what ``contains()`` returns when the label is absent, and it is non-empty."""
    result = _run(script, event="pull_request", base=base, default="main", bypass=bypass)

    assert result.returncode == 1
    assert "::error" in result.stdout
    assert f"'{base}'" in result.stdout
    assert "'main'" in result.stdout


def test_merge_group_passes_when_the_base_would_otherwise_fail(script: str) -> None:
    result = _run(script, event="merge_group", base="", default="main", bypass="false")

    assert result.returncode == 0
    assert "merge_group event" in result.stdout
    assert "::error" not in result.stdout


def test_merge_group_is_decided_before_the_missing_default_arm(script: str) -> None:
    """Both arms exit 0. The notice is what shows the queued-merge path still runs first."""
    result = _run(script, event="merge_group", base="feature/stack", default="", bypass="false")

    assert result.returncode == 0
    assert "merge_group event" in result.stdout
    assert "Could not resolve" not in result.stdout
    assert "::error" not in result.stdout


def test_missing_default_branch_fails_open(script: str) -> None:
    result = _run(script, event="pull_request", base="feature/stack", default="", bypass="false")

    assert result.returncode == 0
    assert "Could not resolve" in result.stdout
    assert "::warning" in result.stdout
    assert "::error" not in result.stdout
