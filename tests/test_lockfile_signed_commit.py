#!/usr/bin/env python3
"""Extracted-shell rehearsal for the signed lockfile commit (#197).

``lockfile-update.yml`` used to ``git commit`` and ``git push``. The ruleset
that requires signatures is default-branch scoped, so the push to
``dependabot/pip/**`` succeeded and left an unsigned commit. A merge commit or
rebase replayed it onto main, where ``required_signatures`` rejected it.

The commit step now calls ``createCommitOnBranch`` (GitHub signs API commits)
with ``expectedHeadOid`` as a compare-and-swap. This repo has no
``requirements.lock``, so that path has never run in CI. These tests execute
the workflow's own shell against a stub ``gh``.

A local ``git commit`` would move HEAD. The signed path must leave HEAD where
it was and leave the worktree dirty.

Project: juniper-data-client
"""

from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404 - the workflow shell is the interface under test
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW = "lockfile-update.yml"
STEP_NAME = "Commit updated lockfile (GitHub-signed, via API)"
TOKEN = "cross-repo-dispatch-token-do-not-leak"  # nosec B105
REPO = "pcalnon/juniper-data-client"
BRANCH = "dependabot/pip/python-minor-example"
COMMIT_JSON = Path("/tmp/lockfile-commit.json")  # nosec B108 - the workflow hardcodes this path
RESULT_JSON = Path("/tmp/lockfile-commit-result.json")  # nosec B108 - same
HEADLINE = "[dependabot skip] Update requirements.lock"

GH_STUB = """#!/bin/bash
printf '%s\\0' "$@" >> "$GH_LOG"
printf '\\036' >> "$GH_LOG"
if [ "${GH_FAIL:-0}" = "1" ]; then
  printf '%s\\n' "${GH_ERR:-gh transport failed}" >&2
  exit 1
fi
printf '%s' "${GH_BODY:-}"
exit 0
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_step() -> tuple[dict, dict, str]:
    path = _repo_root() / ".github" / "workflows" / WORKFLOW
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise AssertionError(f"{WORKFLOW} did not parse to a mapping")
    job = doc.get("jobs", {}).get("update-lockfile")
    if not isinstance(job, dict):
        raise AssertionError("update-lockfile job missing")
    step = next((item for item in job.get("steps", []) if item.get("name") == STEP_NAME), None)
    if not isinstance(step, dict) or "run" not in step:
        raise AssertionError(f"{STEP_NAME!r} run step missing from {WORKFLOW}")
    return doc, step, step["run"]


def _invocations(raw: bytes) -> list[list[str]]:
    if not raw:
        return []
    calls = []
    for record in raw.split(b"\x1e"):
        if not record:
            continue
        parts = record.split(b"\0")
        if parts and parts[-1] == b"":
            parts = parts[:-1]
        calls.append([part.decode() for part in parts])
    return calls


def _child_env(bin_dir: Path, **overrides: str) -> dict[str, str]:
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git fallback only
        "LANG": "C",
        "GH_TOKEN": TOKEN,
        "GITHUB_REPOSITORY": REPO,
        "BRANCH": BRANCH,
        "GH_LOG": str(bin_dir / "gh.log"),
    }
    env.update(overrides)
    return env


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # nosec B603 B607 - fixed git argv in a temp repo
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(repo),
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def _init_repo(repo: Path, lock_text: str) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "requirements.lock").write_text(lock_text, encoding="utf-8")
    _git(repo, "add", "requirements.lock")
    _git(repo, "commit", "-qm", "seed lockfile")
    return _git(repo, "rev-parse", "HEAD")


def _clean_api_files() -> None:
    COMMIT_JSON.unlink(missing_ok=True)
    RESULT_JSON.unlink(missing_ok=True)


def _gnu_base64_available() -> bool:
    """The step runs on ubuntu-latest and encodes with GNU ``base64 -w0``.

    BSD ``base64`` (the macOS unit-test lane) has no ``-w``. The failure is inside
    ``$(...)`` used as an argument, so ``set -e`` does not stop the step and the
    mutation is built with empty ``contents``. That is a property of the host, not
    of the workflow, so the byte-level assertion is skipped where it cannot hold.
    """
    proc = subprocess.run(  # nosec B603 B607 - fixed argv probe of the host base64 on the step's PATH
        ["base64", "-w0"],
        input=b"",
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
    )
    return proc.returncode == 0


class TestLockfileSignedCommit(unittest.TestCase):
    """The regen commit is a signed compare-and-swap, or it does not commit."""

    script: str
    step: dict
    doc: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc, cls.step, cls.script = _load_step()

    def setUp(self) -> None:
        _clean_api_files()

    def tearDown(self) -> None:
        _clean_api_files()

    def _run(self, repo: Path, bin_dir: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "gh").write_text(GH_STUB, encoding="utf-8")
        (bin_dir / "gh").chmod(0o755)
        return subprocess.run(  # nosec B603 B607 - fixed bash, workflow-owned script
            ["bash", "-c", self.script],
            cwd=repo,
            capture_output=True,
            text=True,
            env=_child_env(bin_dir, **env_overrides),
            check=False,
            timeout=15,
        )

    def test_step_contract(self) -> None:
        """The PAT is the CI-retrigger identity, and the step never git-pushes."""
        triggers = self.doc.get("on", self.doc.get(True))
        self.assertEqual(triggers, {"push": {"branches": ["dependabot/pip/**"]}})
        job = self.doc["jobs"]["update-lockfile"]
        self.assertEqual(job.get("if"), "github.actor == 'dependabot[bot]'")
        self.assertEqual(self.step.get("if"), "steps.gate.outputs.proceed == 'true'")
        expected_env = {"BRANCH": "${{ github.head_ref || github.ref_name }}"}
        expected_env["GH_TOKEN"] = "${{ secrets.CROSS_REPO_DISPATCH_TOKEN }}"  # nosec B105
        self.assertEqual(self.step.get("env"), expected_env)
        self.assertNotIn("${{", self.script)
        commands = [line.strip() for line in self.script.splitlines() if line.strip() and not line.strip().startswith("#")]
        self.assertFalse(any(line.startswith("git commit") or line.startswith("git push") for line in commands))
        self.assertIn("createCommitOnBranch", self.script)
        self.assertIn("expectedHeadOid", self.script)
        self.assertIn(HEADLINE, self.script)

    def test_missing_lockfile_skips_without_calling_gh(self) -> None:
        """No lockfile is a clean skip. This repo is in that state today."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            proc = self._run(repo, root / "bin")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("No requirements.lock to commit", proc.stdout)
            self.assertFalse((root / "bin" / "gh.log").exists())
            self.assertFalse(COMMIT_JSON.exists())

    def test_unchanged_lockfile_skips_without_calling_gh(self) -> None:
        """A current lockfile must not open a commit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            proc = self._run(repo, root / "bin")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("no commit needed", proc.stdout)
            self.assertFalse((root / "bin" / "gh.log").exists())
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)

    def test_changed_lockfile_is_a_compare_and_swap(self) -> None:
        """The mutation carries HEAD, the branch, and the new lockfile bytes. HEAD stays put."""
        if not _gnu_base64_available():
            self.skipTest("host base64 has no GNU -w (BSD/macOS); the step runs on ubuntu-latest")
        success = {
            "data": {
                "createCommitOnBranch": {
                    "commit": {
                        "oid": "deadbeef",
                        "url": "https://github.com/pcalnon/juniper-data-client/commit/deadbeef",
                    }
                }
            }
        }
        new_text = "pinned==2\n# regenerated\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text(new_text, encoding="utf-8")
            proc = self._run(repo, root / "bin", GH_BODY=json.dumps(success))
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertIn(success["data"]["createCommitOnBranch"]["commit"]["url"], proc.stdout)
            self.assertNotIn(TOKEN, proc.stdout + proc.stderr)
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)
            self.assertNotEqual((repo / "requirements.lock").read_text(encoding="utf-8"), "pinned==1\n")

            mutation = json.loads(COMMIT_JSON.read_text(encoding="utf-8"))
            self.assertNotIn(TOKEN, COMMIT_JSON.read_text(encoding="utf-8"))
            change = mutation["variables"]["input"]
            self.assertEqual(change["expectedHeadOid"], head)
            self.assertEqual(change["branch"], {"repositoryNameWithOwner": REPO, "branchName": BRANCH})
            self.assertEqual(change["message"], {"headline": HEADLINE})
            additions = change["fileChanges"]["additions"]
            self.assertEqual([item["path"] for item in additions], ["requirements.lock"])
            self.assertNotIn("deletions", change["fileChanges"])
            self.assertEqual(base64.b64decode(additions[0]["contents"]).decode(), new_text)
            self.assertIn("createCommitOnBranch", mutation["query"])

            calls = _invocations((root / "bin" / "gh.log").read_bytes())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:3], ["api", "graphql", "--input"])
            self.assertEqual(calls[0][3], str(COMMIT_JSON))
            self.assertNotIn(TOKEN, calls[0])

    def test_graphql_errors_fail_and_do_not_claim_success(self) -> None:
        """A 200 body with ``errors`` is still a failed commit. gh exits 0 for that shape."""
        body = {
            "data": {"createCommitOnBranch": None},
            "errors": [{"message": "expectedHeadOid mismatch"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            proc = self._run(repo, root / "bin", GH_BODY=json.dumps(body))
            self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
            self.assertIn("expectedHeadOid mismatch", proc.stdout)
            self.assertIn("lockfile NOT committed", proc.stdout)
            self.assertNotIn("Signed lockfile commit", proc.stdout)
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)

    def test_gh_transport_failure_fails(self) -> None:
        """A non-zero gh does not fall through to the success line."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            proc = self._run(repo, root / "bin", GH_FAIL="1", GH_ERR="api unavailable")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("api unavailable", proc.stderr)
            self.assertNotIn("Signed lockfile commit", proc.stdout)

    def test_invalid_graphql_json_fails_closed(self) -> None:
        """An unparseable result must not be reported as a signed commit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            proc = self._run(repo, root / "bin", GH_BODY="{")
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertNotIn("Signed lockfile commit", proc.stdout)
