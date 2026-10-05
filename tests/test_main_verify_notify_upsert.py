"""The main-verify failure notify dedups a red streak without trusting a public title.

``tests/test_main_verify_catchup_base.py`` and ``tests/test_main_verify_screen_verdicts.py``
pin the catch-up resolver and the verdict steps. Neither runs ``Upsert tracking issue``.

The match is the workflow's own authorship (``creator=github-actions[bot]``), an exact
title, and ``pull_request == null``. A title anyone can open is not a trust boundary:
a pull request, a longer title, or a different case must not capture the streak, and a
second matching issue must not be opened. Creating the issue is the point of the step,
so a failed create exits non-zero. Applying the ``main-verify`` label is best-effort
and is not part of the match.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with a fixed argv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_NAME = "main-verify.yml"
JOB_ID = "notify"
STEP_NAME = "Upsert tracking issue (stable title, one per red streak)"
STABLE_TITLE = "main-verify: post-merge verification failing"
REPO = "pcalnon/juniper-data-client"
SHA = "abc123def4567890"
RUN_URL = "https://github.com/pcalnon/juniper-data-client/actions/runs/4242"
SYMBOL_RESULT = "cancelled-distinct"
TOKEN = "unused-token-sentinel"  # nosec B105 - dummy value asserting the notify body never echoes GH_TOKEN
LIST_URL = f"repos/{REPO}/issues?state=open&creator=github-actions%5Bbot%5D&per_page=100"

# Decoys sit in front of the real issue. A filter that drops the pull-request
# exclusion, switches to a prefix match, or folds case returns one of these
# numbers instead of 42. A second exact issue (43) must not win over the first.
_ISSUES = [
    {"number": 7, "title": STABLE_TITLE, "pull_request": {"html_url": "https://github.com/example/pull/7"}},
    {"number": 8, "title": STABLE_TITLE + " (extra)", "pull_request": None},
    {"number": 11, "title": STABLE_TITLE.upper(), "pull_request": None},
    {"number": 42, "title": STABLE_TITLE, "pull_request": None},
    {"number": 43, "title": STABLE_TITLE, "pull_request": None},
]


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
    if not isinstance(script, str) or "gh issue create" not in script:
        raise AssertionError(f"{STEP_NAME!r} is not the notify upsert shell")
    return script


@dataclass
class NotifyRun:
    returncode: int
    stdout: str
    stderr: str
    urls: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    comment_body: str = ""
    create_body: str = ""


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run(script: str, issues: list[dict[str, object]], **flags: str) -> NotifyRun:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gh_dir = root / "gh"
        work = root / "work"
        bindir = root / "bin"
        gh_dir.mkdir()
        work.mkdir()
        bindir.mkdir()
        (gh_dir / "issues.json").write_text(json.dumps(issues), encoding="utf-8")
        stub = bindir / "gh"
        stub.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
case "$cmd" in
  api)
    url="${1:-}"
    printf '%s\\n' "$url" >> "${GH_DIR}/api_urls"
    filter=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--jq" ]; then filter="$a"; fi
      prev="$a"
    done
    if [ "${GH_API_RC:-0}" != "0" ]; then
      echo "gh api failed" >&2
      exit "${GH_API_RC}"
    fi
    jq -r "$filter" "${GH_DIR}/issues.json"
    ;;
  label)
    sub="${1:-}"
    shift || true
    printf '%s %s\\n' "$sub" "${1:-}" >> "${GH_DIR}/labels"
    if [ "${GH_LABEL_RC:-0}" != "0" ]; then
      echo "label failed" >&2
      exit "${GH_LABEL_RC}"
    fi
    ;;
  issue)
    sub="${1:-}"
    shift || true
    case "$sub" in
      comment)
        printf '%s\\n' "${1:-}" >> "${GH_DIR}/comments"
        prev=""
        for a in "$@"; do
          if [ "$prev" = "--body-file" ]; then cp "$a" "${GH_DIR}/comment-body.md"; fi
          prev="$a"
        done
        if [ "${GH_COMMENT_RC:-0}" != "0" ]; then
          echo "comment failed" >&2
          exit "${GH_COMMENT_RC}"
        fi
        ;;
      create)
        prev=""
        title=""
        for a in "$@"; do
          if [ "$prev" = "--title" ]; then title="$a"; fi
          if [ "$prev" = "--body-file" ]; then cp "$a" "${GH_DIR}/create-body.md"; fi
          prev="$a"
        done
        printf '%s\\n' "$title" >> "${GH_DIR}/create_titles"
        if [ "${GH_CREATE_RC:-0}" != "0" ]; then
          echo "create failed" >&2
          exit "${GH_CREATE_RC}"
        fi
        printf '%s\\n' "https://github.com/pcalnon/juniper-data-client/issues/99"
        ;;
      edit)
        printf '%s\\n' "${1:-}" >> "${GH_DIR}/edits"
        if [ "${GH_EDIT_RC:-0}" != "0" ]; then
          echo "edit failed" >&2
          exit "${GH_EDIT_RC}"
        fi
        ;;
      *)
        echo "unexpected issue subcommand: $sub" >&2
        exit 99
        ;;
    esac
    ;;
  *)
    echo "unexpected gh command: $cmd" >&2
    exit 99
    ;;
esac
""",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        script_path = root / "notify.sh"
        script_path.write_text(script, encoding="utf-8")
        env = {
            "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root),
            "LANG": "C",
            "GH_DIR": str(gh_dir),
            "GH_TOKEN": TOKEN,
            "REPO": REPO,
            "TITLE": STABLE_TITLE,
            "SHA": SHA,
            "RUN_URL": RUN_URL,
            "SYMBOL_RESULT": SYMBOL_RESULT,
        }
        env.update(flags)
        proc = subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, PATH-stubbed gh
            ["bash", str(script_path)],
            cwd=work,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        comment_path = gh_dir / "comment-body.md"
        create_path = gh_dir / "create-body.md"
        return NotifyRun(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            urls=_lines(gh_dir / "api_urls"),
            comments=_lines(gh_dir / "comments"),
            titles=_lines(gh_dir / "create_titles"),
            edits=_lines(gh_dir / "edits"),
            labels=_lines(gh_dir / "labels"),
            comment_body=comment_path.read_text(encoding="utf-8") if comment_path.is_file() else "",
            create_body=create_path.read_text(encoding="utf-8") if create_path.is_file() else "",
        )


@pytest.fixture(scope="module")
def script() -> str:
    return _load_script()


def test_existing_bot_issue_is_commented_and_decoys_do_not_capture_it(script: str) -> None:
    """The first exact bot issue is commented. A PR, a prefix, a case change, and a later duplicate are not."""
    result = _run(script, _ISSUES)

    assert result.returncode == 0
    assert result.urls == [LIST_URL]
    assert result.comments == ["42"]
    assert result.titles == []
    assert SHA in result.comment_body
    assert RUN_URL in result.comment_body
    assert SYMBOL_RESULT in result.comment_body
    assert TOKEN not in result.comment_body
    assert TOKEN not in result.stdout


def test_no_match_opens_one_issue_under_the_stable_title(script: str) -> None:
    """A sha in the title would open a new issue on every red push and the streak would never dedup."""
    result = _run(script, [])

    assert result.returncode == 0
    assert result.comments == []
    assert result.titles == [STABLE_TITLE]
    assert result.edits == ["99"]
    assert result.labels == ["create main-verify"]
    assert SHA in result.create_body
    assert "NOT auto-closed" in result.create_body
    assert TOKEN not in result.create_body
    assert SHA not in result.titles[0]


def test_label_create_failure_still_opens_the_issue(script: str) -> None:
    """The label is not part of the dedup match. A failed create of it must not swallow the tracker."""
    result = _run(script, [], GH_LABEL_RC="1")

    assert result.returncode == 0
    assert result.titles == [STABLE_TITLE]
    assert result.edits == ["99"]


def test_issue_create_failure_exits_nonzero_and_does_not_edit(script: str) -> None:
    result = _run(script, [], GH_CREATE_RC="1")

    assert result.returncode != 0
    assert result.edits == []
    assert "failed to open the stable-title tracking issue" in result.stdout


def test_label_edit_failure_still_exits_zero(script: str) -> None:
    """Best-effort labeling. A failed ``gh issue edit`` must not turn a filed tracker into a failed notify."""
    result = _run(script, [], GH_EDIT_RC="1")

    assert result.returncode == 0
    assert result.titles == [STABLE_TITLE]


def test_comment_failure_exits_nonzero_and_does_not_open_another_issue(script: str) -> None:
    result = _run(script, _ISSUES, GH_COMMENT_RC="1")

    assert result.returncode != 0
    assert result.comments == ["42"]
    assert result.titles == []


def test_api_list_failure_still_attempts_to_open_the_tracker(script: str) -> None:
    """A list error is not an empty streak. The step still has to try to open the issue."""
    result = _run(script, _ISSUES, GH_API_RC="1")

    assert result.returncode == 0
    assert result.urls == [LIST_URL]
    assert result.comments == []
    assert result.titles == [STABLE_TITLE]
