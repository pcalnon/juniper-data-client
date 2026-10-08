#!/usr/bin/env python3
"""Rehearse the per-PR sequence-safety shell, including its label hatch.

Sequence Safety is a required check. The workflow demotes one screen to
``--advisory`` only when ``gh pr view`` reports the exact label
``allow-symbol-loss`` or ``docs-rewrite``. The flag is an argument to that
screen; it does not discard the screen's exit code locally.

A substring or case-insensitive match, a label that merely contains the name,
or a failed ``gh`` whose stderr mentions the label, would waive a required
screen. An empty pull-request number must not call ``gh`` at all. Exit >= 2
stays an invocation error even when the other screen only found a deletion.

Project: juniper-data-client
Author: Paul Calnon
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted screen shell with a fixed argv
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW_NAME = "sequence-safety.yml"
STEP_NAME = "Run sequence-safety screens (symbol + docs)"
SYMBOL = "juniper-symbol-loss-check"
DOCS = "juniper-docs-additions-check"
SCOPES = ("juniper_data_client/**/*.py", "tests/**/*.py")

_SCREEN_STUB = """\
#!/usr/bin/env python3
import os
import sys

log = os.environ["SCREEN_LOG"]
name = os.path.basename(sys.argv[0])
kind = "json" if "--json" in sys.argv[1:] else "human"
with open(log, "a", encoding="utf-8") as handle:
    handle.write("\\t".join([kind, name, *sys.argv[1:]]) + "\\n")
if kind == "json":
    raise SystemExit(0)
code = os.environ["SYMBOL_EXIT"] if name == "juniper-symbol-loss-check" else os.environ["DOCS_EXIT"]
raise SystemExit(int(code))
"""

_GH_STUB = """\
#!/usr/bin/env python3
import os
import sys

log = os.environ["GH_LOG"]
with open(log, "a", encoding="utf-8") as handle:
    handle.write("\\t".join(sys.argv[1:]) + "\\n")
rc = int(os.environ.get("GH_RC", "0"))
if rc != 0:
    sys.stderr.write(os.environ.get("GH_STDERR", ""))
    raise SystemExit(rc)
labels = os.environ.get("GH_LABELS", "")
if labels:
    sys.stdout.write(labels if labels.endswith("\\n") else labels + "\\n")
"""


def _child_env(**overrides: str) -> dict[str, str]:
    """Minimal child environment. The real process environment is not copied."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - fallback only; git needs a HOME
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        if (cur / ".github" / "workflows").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise AssertionError("could not locate repo root with .github/workflows")


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(  # nosec B603 B607 - fixed git argv in a temp fixture
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"),
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


class SequenceSafetyHatchTest(unittest.TestCase):
    """Extract and run the workflow's own screen step against stubbed tools."""

    script: str

    @classmethod
    def setUpClass(cls) -> None:
        wf = _repo_root() / ".github" / "workflows" / WORKFLOW_NAME
        if not wf.is_file():
            raise AssertionError(f"{WORKFLOW_NAME} missing at {wf} -- a skip here would be silently green")
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        steps = doc.get("jobs", {}).get("sequence-safety", {}).get("steps", [])
        step = next((s for s in steps if s.get("name") == STEP_NAME), None)
        if step is None or "run" not in step:
            raise AssertionError(f"{STEP_NAME!r} run step missing from {WORKFLOW_NAME} -- that is the drift this test guards")
        cls.script = step["run"]

    def _stage(self, root: Path) -> tuple[Path, str]:
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")
        _git(repo, "config", "commit.gpgsign", "false")
        (repo / "f.txt").write_text("a\n", encoding="utf-8")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-m", "base")
        return repo, _git(repo, "rev-parse", "HEAD")

    def _run(
        self,
        *,
        base: str,
        pr_number: str = "219",
        labels: str = "",
        gh_rc: int = 0,
        gh_stderr: str = "",
        symbol_exit: int = 0,
        docs_exit: int = 0,
        repo: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[tuple[str, str, list[str]]], list[list[str]]]:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            if repo is None:
                repo, _ = self._stage(td_path)
            script_path = td_path / "screens.sh"
            script_path.write_text(self.script, encoding="utf-8")
            stub_bin = td_path / "bin"
            stub_bin.mkdir()
            screen = stub_bin / "_screen.py"
            screen.write_text(_SCREEN_STUB, encoding="utf-8")
            screen.chmod(0o755)
            (stub_bin / SYMBOL).symlink_to(screen)
            (stub_bin / DOCS).symlink_to(screen)
            gh = stub_bin / "gh"
            gh.write_text(_GH_STUB, encoding="utf-8")
            gh.chmod(0o755)
            screen_log = td_path / "screens.log"
            gh_log = td_path / "gh.log"
            env = _child_env(
                PATH=str(stub_bin) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
                GH_TOKEN="unused",  # nosec B106 - dummy token for the PATH-stubbed gh, never a real credential
                PR_NUMBER=pr_number,
                PR_BASE_SHA=base,
                SCREEN_LOG=str(screen_log),
                GH_LOG=str(gh_log),
                GH_LABELS=labels,
                GH_RC=str(gh_rc),
                GH_STDERR=gh_stderr,
                SYMBOL_EXIT=str(symbol_exit),
                DOCS_EXIT=str(docs_exit),
            )
            proc = subprocess.run(  # nosec B603 B607 - the workflow's own shell, fixed argv
                ["bash", str(script_path)],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=15,
            )
            invocations = _read_invocations(screen_log)
            gh_calls = _read_rows(gh_log)
            return proc, invocations, gh_calls

    def test_empty_base_exits_2_without_calling_the_screens(self) -> None:
        proc, invocations, gh_calls = self._run(base="")
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 2, text)
        self.assertIn("could not resolve a base sha", text)
        self.assertEqual(invocations, [])
        self.assertEqual(gh_calls, [])

    def test_clean_screens_pass_and_keep_the_client_scope(self) -> None:
        """The symbol screen must see this package. The ml default scope would not."""
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, gh_calls = self._run(base=base, repo=repo)
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, text)
        self.assertIn("screens clean", text)
        human_symbol = [args for kind, name, args in invocations if kind == "human" and name == SYMBOL]
        self.assertEqual(len(human_symbol), 1, invocations)
        self.assertEqual(_scopes(human_symbol[0]), list(SCOPES))
        self.assertNotIn("--advisory", human_symbol[0])
        self.assertTrue(gh_calls, "a numbered PR reads labels")
        jq_at = gh_calls[0].index("--jq")
        self.assertIn(".labels[].name", gh_calls[0][jq_at + 1])

    def test_exact_symbol_label_advises_only_the_symbol_screen(self) -> None:
        """The hatch passes ``--advisory``. It does not ignore a failing exit."""
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, _ = self._run(base=base, repo=repo, labels="allow-symbol-loss", symbol_exit=1)
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, text)
        self.assertIn("compositional-loss finding", text)
        self.assertNotIn("invocation error", text)
        self.assertTrue(_advised(invocations, SYMBOL))
        self.assertFalse(_advised(invocations, DOCS))

    def test_exact_docs_label_advises_only_the_docs_screen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, _ = self._run(base=base, repo=repo, labels="docs-rewrite", docs_exit=1)
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, text)
        self.assertIn("compositional-loss finding", text)
        self.assertTrue(_advised(invocations, DOCS))
        self.assertFalse(_advised(invocations, SYMBOL))

    def test_both_exact_labels_advise_both_screens(self) -> None:
        labels = "allow-symbol-loss\ndocs-rewrite"
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, _ = self._run(base=base, repo=repo, labels=labels)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(_advised(invocations, SYMBOL))
        self.assertTrue(_advised(invocations, DOCS))

    def test_inexact_labels_do_not_advise(self) -> None:
        """``grep -qx`` is the whole match. A prefix, a suffix, case, or padding is not a waiver."""
        rejected = (
            "allow-symbol-loss-extra",
            "xallow-symbol-loss",
            "Allow-Symbol-Loss",
            "allow-symbol-loss ",
            " allow-symbol-loss",
            "DOCS-REWRITE",
            "docs-rewrite-please",
            "allow-symbol-loss docs-rewrite",
        )
        for label in rejected:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as td:
                    repo, base = self._stage(Path(td))
                    _, invocations, _ = self._run(base=base, repo=repo, labels=label, symbol_exit=1)
                self.assertFalse(_advised(invocations, SYMBOL), invocations)
                self.assertFalse(_advised(invocations, DOCS), invocations)

    def test_failed_gh_does_not_demote_and_stderr_is_not_a_label(self) -> None:
        """A failing ``gh`` prints the waiver name on stderr. That must not become a label."""
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, gh_calls = self._run(
                base=base,
                repo=repo,
                gh_rc=1,
                gh_stderr="allow-symbol-loss\ndocs-rewrite\n",
                labels="allow-symbol-loss",
            )
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, text)
        self.assertIn("screens clean", text)
        self.assertTrue(gh_calls)
        self.assertFalse(_advised(invocations, SYMBOL), invocations)
        self.assertFalse(_advised(invocations, DOCS), invocations)

    def test_empty_pr_number_does_not_call_gh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, base = self._stage(Path(td))
            proc, invocations, gh_calls = self._run(base=base, repo=repo, pr_number="", labels="allow-symbol-loss")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(gh_calls, [])
        self.assertFalse(_advised(invocations, SYMBOL))
        self.assertFalse(_advised(invocations, DOCS))

    def test_invocation_error_outranks_a_finding(self) -> None:
        """Exit >= 2 is reported as an invocation error, including exit 3 and a mixed pair."""
        cases = ((2, 0), (0, 2), (3, 0), (0, 3), (1, 2), (2, 1))
        for symbol_exit, docs_exit in cases:
            with self.subTest(symbol_exit=symbol_exit, docs_exit=docs_exit):
                with tempfile.TemporaryDirectory() as td:
                    repo, base = self._stage(Path(td))
                    proc, _, _ = self._run(base=base, repo=repo, symbol_exit=symbol_exit, docs_exit=docs_exit)
                text = proc.stdout + proc.stderr
                self.assertEqual(proc.returncode, 2, text)
                self.assertIn("invocation error", text)
                self.assertNotIn("compositional-loss finding", text)


def _read_invocations(path: Path) -> list[tuple[str, str, list[str]]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        kind, name, *args = line.split("\t")
        rows.append((kind, name, args))
    return rows


def _read_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]


def _scopes(args: list[str]) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args) if arg == "--scope" and i + 1 < len(args)]


def _advised(invocations: list[tuple[str, str, list[str]]], tool: str) -> bool:
    matched = [args for _kind, name, args in invocations if name == tool]
    if not matched:
        raise AssertionError(f"{tool} was not invoked: {invocations}")
    return all("--advisory" in args for args in matched)


if __name__ == "__main__":
    unittest.main()
