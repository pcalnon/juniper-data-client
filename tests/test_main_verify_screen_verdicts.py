#!/usr/bin/env python3
"""Execute the main-verify screen-verdict shells, not just their step names.

``tests/test_main_verify_catchup_base.py`` rehearses catch-up base resolution and
pins that ``Assert screens reached a verdict`` exists and precedes ``Assert screens
clean``. It never runs those two scripts. The split added in #186 is load-bearing:

- exit 0 is clean: both asserts pass, and the window is screened;
- exit 1 is a finding: the verdict assert still passes (the base may advance) and
  the clean assert fails the job;
- exit >= 2, or a missing / empty code, is not a verdict: the verdict assert fails
  so the base stays put.

Changing ``-ge 2`` to ``-ge 1`` freezes ``main`` red again. Changing the clean
assert's ``-ge 1`` to ``-ge 2`` lets a finding go green. Dropping the ``:-99``
default fail-opens an empty code into "screened".

Project: juniper-data-client
Author: Paul Calnon
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted assert shells with a fixed argv
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW_NAME = "main-verify.yml"
VERDICT_STEP = "Assert screens reached a verdict"
CLEAN_STEP = "Assert screens clean"
HEAD = "abc123deadbeef"


def _child_env(**overrides: str) -> dict[str, str]:
    """Minimal child environment. The real process environment is not copied."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git-less scripts; HOME is unused
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


def _step_script(name: str) -> str:
    wf = _repo_root() / ".github" / "workflows" / WORKFLOW_NAME
    if not wf.is_file():
        raise AssertionError(f"{WORKFLOW_NAME} missing at {wf} -- a skip here would be silently green")
    doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
    steps = doc.get("jobs", {}).get("symbol-screen", {}).get("steps", [])
    step = next((s for s in steps if s.get("name") == name), None)
    if step is None or "run" not in step:
        raise AssertionError(f"{name!r} run step missing from {WORKFLOW_NAME} -- that is the drift this test guards")
    return step["run"]


class ScreenVerdictShellTest(unittest.TestCase):
    """Drive the workflow's own assert scripts over the exit-code matrix."""

    verdict: str
    clean: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.verdict = _step_script(VERDICT_STEP)
        cls.clean = _step_script(CLEAN_STEP)

    def _run(self, script: str, **variables: str | None) -> subprocess.CompletedProcess[str]:
        env = _child_env(HEAD_SHA=HEAD)
        for key, value in variables.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "assert.sh"
            path.write_text(script, encoding="utf-8")
            return subprocess.run(  # nosec B603 B607 - the workflow's own shell, fixed argv
                ["bash", str(path)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=15,
            )

    def _output(self, proc: subprocess.CompletedProcess[str]) -> str:
        return proc.stdout + proc.stderr

    def test_clean_window_passes_both_asserts(self) -> None:
        """Exit 0 is a verdict and a clean screen."""
        verdict = self._run(self.verdict, SRC="0", DRC="0")
        clean = self._run(self.clean, SRC="0", DRC="0")
        self.assertEqual(verdict.returncode, 0, self._output(verdict))
        self.assertIn("symbol=0", self._output(verdict))
        self.assertIn("docs=0", self._output(verdict))
        self.assertIn("IS screened", self._output(verdict))
        self.assertNotIn("NOT screened", self._output(verdict))
        self.assertEqual(clean.returncode, 0, self._output(clean))
        self.assertIn(f"screens clean at {HEAD}", self._output(clean))

    def test_finding_is_screened_and_still_fails_the_job(self) -> None:
        """Exit 1 advances the base and still turns the job red.

        Either screen is enough. A threshold of ``-ge 1`` on the verdict assert
        would refuse to advance; a threshold of ``-ge 2`` on the clean assert
        would let the finding through.
        """
        cases = (("1", "0"), ("0", "1"), ("1", "1"))
        for src, drc in cases:
            with self.subTest(src=src, drc=drc):
                verdict = self._run(self.verdict, SRC=src, DRC=drc)
                clean = self._run(self.clean, SRC=src, DRC=drc)
                self.assertEqual(verdict.returncode, 0, self._output(verdict))
                self.assertIn(f"symbol={src}", self._output(verdict))
                self.assertIn(f"docs={drc}", self._output(verdict))
                self.assertIn("IS screened", self._output(verdict))
                self.assertNotIn("NOT screened", self._output(verdict))
                self.assertEqual(clean.returncode, 1, self._output(clean))
                self.assertIn(f"finding(s) at {HEAD}", self._output(clean))

    def test_invocation_error_is_not_a_verdict(self) -> None:
        """Exit >= 2 does not advance the base, including when only one screen failed.

        ``3`` is included so an ``== 2`` comparison, which would miss every other
        invocation error, fails this test. A finding on the other screen must not
        hide the invocation error (the checks are OR, and the >= 2 check comes first).
        """
        cases = (("2", "0"), ("0", "2"), ("3", "0"), ("0", "3"), ("1", "2"), ("2", "1"))
        for src, drc in cases:
            with self.subTest(src=src, drc=drc):
                verdict = self._run(self.verdict, SRC=src, DRC=drc)
                text = self._output(verdict)
                self.assertEqual(verdict.returncode, 2, text)
                self.assertIn(f"symbol={src}", text)
                self.assertIn(f"docs={drc}", text)
                self.assertIn("NOT screened", text)
                self.assertIn("will NOT advance", text)
                self.assertNotIn("IS screened", text)

    def test_missing_or_empty_codes_are_invocation_errors(self) -> None:
        """Absent outputs default to 99. An empty string is absent, not zero.

        ``${SRC:-99}`` covers both. ``${SRC-99}`` would leave an empty string
        empty, the integer test would fail open, and the window would be marked
        screened. Unset without a default aborts before the screened notice.
        """
        # The empty side renders as 99; a present 0 stays 0 and must not rescue it.
        cases = (
            ("unset", None, None, "99", "99"),
            ("empty", "", "", "99", "99"),
            ("src-empty", "", "0", "99", "0"),
            ("drc-empty", "0", "", "0", "99"),
        )
        for label, src, drc, want_src, want_drc in cases:
            with self.subTest(label=label):
                verdict = self._run(self.verdict, SRC=src, DRC=drc)
                clean = self._run(self.clean, SRC=src, DRC=drc)
                vtext = self._output(verdict)
                ctext = self._output(clean)
                self.assertEqual(verdict.returncode, 2, vtext)
                self.assertIn(f"symbol={want_src}", vtext)
                self.assertIn(f"docs={want_drc}", vtext)
                self.assertIn("NOT screened", vtext)
                self.assertNotIn("unbound variable", vtext)
                self.assertEqual(clean.returncode, 1, ctext)
                self.assertIn(f"finding(s) at {HEAD}", ctext)
                self.assertNotIn("screens clean", ctext)
                self.assertNotIn("unbound variable", ctext)


if __name__ == "__main__":
    unittest.main()
