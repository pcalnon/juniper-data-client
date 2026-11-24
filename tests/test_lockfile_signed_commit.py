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

The step writes its mutation and result under the runner's ``/tmp/``. Each
test rewrites that prefix to its own scratch directory, so a run writes
nothing under ``/tmp`` and two runs never share a file.

``gh api graphql`` exits 1 on a response that carries GraphQL ``errors`` and
on an HTTP failure, after writing the body to stdout and ``gh: <message>`` to
stderr (cli/cli ``pkg/cmd/api/api.go``, ``processResponse``). The stub fails
the same way. Called bare under ``set -e``, that exit ended the step before its
own diagnostic; the step now captures gh's status and stderr, prints the
diagnostic, and fails, with the token's value masked in anything gh printed
(juniper-canopy#814).

Project: juniper-data-client
"""

from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404 - the workflow shell is the interface under test
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW = "lockfile-update.yml"
STEP_NAME = "Commit updated lockfile (GitHub-signed, via API)"
TOKEN = "cross-repo-dispatch-token-do-not-leak"  # nosec B105
REPO = "pcalnon/juniper-data-client"
BRANCH = "dependabot/pip/python-minor-example"
RUNNER_TMP = "/tmp/"  # nosec B108 - the prefix the step hardcodes; each test rewrites it to its own directory
COMMIT_JSON_NAME = "lockfile-commit.json"
RESULT_JSON_NAME = "lockfile-commit-result.json"
HEADLINE = "[dependabot skip] Update requirements.lock"
STALE = 'Expected branch to point to "abc" but it did not. Pull and try again.'

# The real CLI's order: the response body on stdout, then, when it fails, its message on stderr
# and the exit status (``GH_RC``, 1 unless a test names another).
GH_STUB = """#!/bin/bash
printf '%s\\0' "$@" >> "$GH_LOG"
printf '\\036' >> "$GH_LOG"
printf '%s' "${GH_BODY:-}"
if [ "${GH_FAIL:-0}" = "1" ]; then
  printf '%s\\n' "${GH_ERR:-gh transport failed}" >&2
  exit "${GH_RC:-1}"
fi
exit 0
"""

# Two broken encoders. ``base64`` runs in the step's own ``$(...)``; while that sat inside a jq
# argument, both shapes reached gh with an empty ``contents`` (flood-3 §4).
FAILING_BASE64 = "#!/bin/sh\necho 'base64: simulated encode failure' >&2\nexit 1\n"
SILENT_BASE64 = "#!/bin/sh\nexit 0\n"

# A GNU-shaped ``base64 -w0 FILE`` for a host whose own base64 cannot produce the step's payload
# (BSD/macOS; see ``_gnu_base64_available``). It encodes the real file with Python's ``base64``.
GNU_BASE64_STANDIN = """import base64
import sys

if len(sys.argv) != 3 or sys.argv[1] != "-w0":
    sys.exit("base64 stand-in: expected -w0 FILE, got " + " ".join(sys.argv[1:]))
with open(sys.argv[2], "rb") as handle:
    sys.stdout.write(base64.b64encode(handle.read()).decode("ascii"))
"""

# What gh answers for a commit that landed. A step that reaches gh with it claims success.
SIGNED_COMMIT_BODY = {"data": {"createCommitOnBranch": {"commit": {"oid": "deadbeef", "url": "https://github.com/pcalnon/juniper-data-client/commit/deadbeef"}}}}


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


def _relocated(script: str, scratch: Path) -> str:
    """The step with its runner-``/tmp/`` scratch files moved into ``scratch``."""
    if RUNNER_TMP not in script:
        raise AssertionError("the step no longer writes under /tmp/; drop the relocation")
    return script.replace(RUNNER_TMP, f"{scratch}/")


def _gnu_base64_available() -> bool:
    """The step runs on ubuntu-latest and encodes with GNU ``base64 -w0 requirements.lock``.

    BSD ``base64`` (the macOS unit-test lane) does not produce that payload, and the step
    refuses an empty or failed encode before it calls gh (flood-3 §4). That is a property of
    the host, not of the workflow, so the byte-level assertion is skipped where it cannot
    hold -- and never on Linux, the step's own platform, where a failing probe would be a
    real problem. The gh-side tests get past the encode there through ``_encoder_for_host``.

    The probe runs the step's exact form (``-w0`` and a positional FILE) and checks the
    OUTPUT: an exit-code-only probe of ``base64 -w0`` on empty stdin passed on the macOS
    runner while the step still produced an empty payload there.
    """
    if sys.platform.startswith("linux"):
        return True
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe"
        probe.write_bytes(b"x")
        try:
            proc = subprocess.run(  # nosec B603 B607 - fixed argv probe of the host base64 on the step's PATH
                ["base64", "-w0", str(probe)],
                capture_output=True,
                check=False,
                timeout=10,
                env={"PATH": "/usr/bin:/bin", "LANG": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return proc.returncode == 0 and proc.stdout == b"eA=="


def _put_base64(bin_dir: Path, body: str) -> None:
    """Install ``body`` as the ``base64`` the step finds first (``bin_dir`` leads its PATH)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "base64"
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o700)


def _encoder_for_host(bin_dir: Path) -> None:
    """Let the step get past its encode on a host whose ``base64`` cannot produce the payload.

    The step stops at an empty or failed encode (flood-3 §4), which is where BSD/macOS
    ``base64`` leaves it. The tests of what the step does with gh's answer need it to get
    further, so on such a host a GNU-shaped stand-in goes first on the step's PATH. On Linux,
    the step's own platform, nothing is installed and the host's GNU ``base64`` runs.
    """
    if not _gnu_base64_available():
        _put_base64(bin_dir, f"#!{sys.executable} -I\n{GNU_BASE64_STANDIN}")


class TestLockfileSignedCommit(unittest.TestCase):
    """The regen commit is a signed compare-and-swap, or it does not commit."""

    script: str
    step: dict
    doc: dict
    scratch: Path
    commit_json: Path
    result_json: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc, cls.step, cls.script = _load_step()

    def setUp(self) -> None:
        # Stands in for the runner's /tmp/: the step's scratch files land here, never under /tmp.
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.commit_json = self.scratch / COMMIT_JSON_NAME
        self.result_json = self.scratch / RESULT_JSON_NAME

    def _run(self, repo: Path, bin_dir: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "gh").write_text(GH_STUB, encoding="utf-8")
        (bin_dir / "gh").chmod(0o755)
        return subprocess.run(  # nosec B603 B607 - fixed bash, workflow-owned script
            ["bash", "-c", _relocated(self.script, self.scratch)],
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
        # The two runner paths the tests relocate; a renamed file must fail here, not silently.
        self.assertIn(RUNNER_TMP + COMMIT_JSON_NAME, self.script)
        self.assertIn(RUNNER_TMP + RESULT_JSON_NAME, self.script)

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
            self.assertFalse(self.commit_json.exists())
            self.assertFalse(self.result_json.exists())

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
            self.assertEqual(proc.stdout, "Signed lockfile commit: " + success["data"]["createCommitOnBranch"]["commit"]["url"] + "\n")
            self.assertEqual(proc.stderr, "")
            self.assertNotIn(TOKEN, proc.stdout + proc.stderr)
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)
            self.assertNotEqual((repo / "requirements.lock").read_text(encoding="utf-8"), "pinned==1\n")

            self.assertTrue(self.result_json.is_file(), "the step's GraphQL result must land in the scratch dir")
            mutation = json.loads(self.commit_json.read_text(encoding="utf-8"))
            self.assertNotIn(TOKEN, self.commit_json.read_text(encoding="utf-8"))
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
            self.assertEqual(calls[0][3], str(self.commit_json))
            self.assertNotIn(TOKEN, calls[0])

    def test_graphql_errors_fail_and_do_not_claim_success(self) -> None:
        """A 200 body with ``errors`` from a gh that exits 0 is still a failed commit.

        The real CLI exits 0 on such a body only when it finds no message to print, for
        example a single error without a ``message`` (api.go ``parseErrorResponse``); on
        this body it exits 1. That shape is
        ``test_a_graphql_error_that_fails_gh_prints_the_steps_diagnostic``.
        """
        body = {
            "data": {"createCommitOnBranch": None},
            "errors": [{"message": "expectedHeadOid mismatch"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            _encoder_for_host(root / "bin")
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
            _encoder_for_host(root / "bin")
            proc = self._run(repo, root / "bin", GH_FAIL="1", GH_ERR="api unavailable")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("api unavailable", proc.stderr)
            self.assertNotIn("Signed lockfile commit", proc.stdout)

    def _commit_with_gh(self, *, body: str, fail: bool, gh_err: str = "", gh_rc: int = 1) -> subprocess.CompletedProcess[str]:
        """Run the step on a changed lockfile against a gh that prints ``body`` and, when ``fail``, fails as the real CLI does."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            _encoder_for_host(root / "bin")
            env = {"GH_BODY": body}
            if fail:
                env.update(GH_FAIL="1", GH_ERR=gh_err, GH_RC=str(gh_rc))
            proc = self._run(repo, root / "bin", **env)
        self.assertNotIn(TOKEN, proc.stdout + proc.stderr, "the commit step printed the token")
        return proc

    def test_a_graphql_error_that_fails_gh_prints_the_steps_diagnostic(self) -> None:
        """The real CLI's shape: the ``errors`` body on stdout, ``gh: <message>`` on stderr, exit 1.

        A bare ``gh api graphql`` under ``set -e`` ended the step at that exit, so the step's
        own ``createCommitOnBranch failed`` diagnostic never printed.
        """
        body = {"data": {"createCommitOnBranch": None}, "errors": [{"type": "STALE_DATA", "message": STALE}]}
        proc = self._commit_with_gh(body=json.dumps(body), fail=True, gh_err=f"gh: {STALE}")
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn(f"gh: {STALE}", proc.stderr)
        self.assertIn("::error::createCommitOnBranch failed", proc.stdout)
        self.assertIn("lockfile NOT committed", proc.stdout)
        self.assertIn("STALE_DATA", proc.stdout)
        self.assertIn("gh exited 1", proc.stdout)
        self.assertIn("re-run this job", proc.stdout)
        self.assertNotIn("gh api graphql failed", proc.stdout + proc.stderr)
        self.assertNotIn("Signed lockfile commit", proc.stdout)

    def _assert_the_commit_is_unknown(self, proc: subprocess.CompletedProcess[str], gh_err: str, printed: str, names_the_token: bool) -> None:
        """No GraphQL ``errors``: an HTTP failure is not proof that the mutation did not run, so nothing says it was not committed."""
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn(gh_err, proc.stderr)
        self.assertIn("::error::gh api graphql failed (exit 1)", proc.stdout)
        self.assertIn("cannot show whether the lock commit landed", proc.stdout)
        self.assertIn(printed, proc.stdout)
        self.assertIn(f"Check {BRANCH} for the lock commit", proc.stdout)
        self.assertEqual("CROSS_REPO_DISPATCH_TOKEN is invalid or expired" in proc.stdout, names_the_token)
        self.assertNotIn("createCommitOnBranch failed", proc.stdout + proc.stderr)
        self.assertNotIn("Signed lockfile commit", proc.stdout)

    def test_an_http_failure_prints_that_the_commit_is_unknown_http_502_html(self) -> None:
        proc = self._commit_with_gh(body="<html>502 Bad Gateway</html>\n", fail=True, gh_err="gh: HTTP 502")
        self._assert_the_commit_is_unknown(proc, "gh: HTTP 502", "empty or not JSON", names_the_token=False)

    def test_an_http_failure_prints_that_the_commit_is_unknown_http_401_json(self) -> None:
        proc = self._commit_with_gh(body=json.dumps({"message": "Bad credentials", "status": "401"}), fail=True, gh_err="gh: Bad credentials (HTTP 401)")
        self._assert_the_commit_is_unknown(proc, "gh: Bad credentials (HTTP 401)", '"Bad credentials"', names_the_token=True)

    def test_an_http_failure_prints_that_the_commit_is_unknown_no_response(self) -> None:
        proc = self._commit_with_gh(body="", fail=True, gh_err="error connecting to api.github.com")
        self._assert_the_commit_is_unknown(proc, "error connecting to api.github.com", "empty or not JSON", names_the_token=False)

    def test_gh_output_is_printed_with_the_token_masked(self) -> None:
        """What gh printed reaches the log through the step's mask; ``_commit_with_gh`` refuses the token."""
        message = f"token {TOKEN} lacks access ({TOKEN})"
        proc = self._commit_with_gh(body=json.dumps({"errors": [{"type": "FORBIDDEN", "message": message}]}), fail=True, gh_err=f"gh: {message}")
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("gh: token *** lacks access (***)", proc.stderr)
        self.assertIn("token *** lacks access (***)", proc.stdout)
        self.assertIn("createCommitOnBranch failed", proc.stdout)

    def test_an_http_failure_body_is_printed_with_the_token_masked(self) -> None:
        """The HTTP-failure branch prints a JSON body through the same mask as gh's message."""
        proc = self._commit_with_gh(body=json.dumps({"message": f"bad token {TOKEN}", "status": "403"}), fail=True, gh_err=f"gh: bad token {TOKEN} (HTTP 403)")
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("gh: bad token *** (HTTP 403)", proc.stderr)
        self.assertIn("gh api graphql failed (exit 1)", proc.stdout)
        self.assertIn('"bad token ***"', proc.stdout)

    def test_the_diagnostic_names_the_exit_status_gh_gave_exit_4_no_token(self) -> None:
        """gh exits 4 when it has no usable token (cli/cli ``internal/ghcmd/cmd.go``, ``exitAuth``); the step exits 1."""
        proc = self._commit_with_gh(body="", fail=True, gh_err="gh: set the GH_TOKEN environment variable", gh_rc=4)
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("gh api graphql failed (exit 4)", proc.stdout)
        self.assertNotIn("Signed lockfile commit", proc.stdout)

    def test_the_diagnostic_names_the_exit_status_gh_gave_exit_0_errors_without_a_message(self) -> None:
        """gh exits 0 on an ``errors`` body that carries no message; the diagnostic says so, and the step exits 1."""
        proc = self._commit_with_gh(body=json.dumps({"errors": [{"type": "X"}]}), fail=False)
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("gh exited 0", proc.stdout)
        self.assertNotIn("Signed lockfile commit", proc.stdout)

    def test_invalid_graphql_json_fails_closed(self) -> None:
        """An unparseable result must not be reported as a signed commit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            _encoder_for_host(root / "bin")
            proc = self._run(repo, root / "bin", GH_BODY="{")
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertNotIn("Signed lockfile commit", proc.stdout)

    def test_a_failing_encode_stops_before_gh(self) -> None:
        """A non-zero ``base64`` stops the step. Inside ``"$(...)"`` as a jq argument it did not."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            _put_base64(root / "bin", FAILING_BASE64)
            proc = self._run(repo, root / "bin", GH_BODY=json.dumps(SIGNED_COMMIT_BODY))
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertIn("simulated encode failure", proc.stderr)
            self.assertNotIn("Signed lockfile commit", proc.stdout)
            self.assertFalse((root / "bin" / "gh.log").exists(), "gh was called without a payload")
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)

    def test_an_empty_encode_stops_before_gh(self) -> None:
        """An encode that exits 0 but prints nothing is refused, not committed: its exit status alone is not trusted."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            head = _init_repo(repo, "pinned==1\n")
            (repo / "requirements.lock").write_text("pinned==2\n", encoding="utf-8")
            _put_base64(root / "bin", SILENT_BASE64)
            proc = self._run(repo, root / "bin", GH_BODY=json.dumps(SIGNED_COMMIT_BODY))
            self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
            self.assertIn("::error::base64 produced an empty payload -- lockfile NOT committed", proc.stdout)
            self.assertNotIn("Signed lockfile commit", proc.stdout)
            self.assertFalse((root / "bin" / "gh.log").exists(), "gh was called with an empty payload")
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)
