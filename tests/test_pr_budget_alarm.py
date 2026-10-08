#!/usr/bin/env python3
"""Extracted-shell rehearsal for the open-PR budget alarm (#205).

The workflow is report-only: a breach stays green, and a failed ``gh pr list``
stays green with ``level=OK`` so a transient API blip cannot page anyone. That
failure must not look like an empty queue (which would also be OK, but would
publish ``total=0``). Only a ``cursor/`` prefix counts. The Slack payload
carries counts and the run URL, never the webhook.

Project: juniper-data-client
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - the workflow shell is the interface under test
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW = "pr-budget-alarm.yml"
COUNT_STEP = "Count open PRs and evaluate the budget"
SLACK_STEP = "Slack notification on breach (non-blocking, Q-CHANNEL)"
WEBHOOK = "https://hooks.example.test/services/T000/B000/SECRETWEBHOOK"
RUN_URL = "https://github.com/pcalnon/juniper-data-client/actions/runs/99"

GH_STUB = """#!/bin/bash
printf '%s\\0' "$@" >> "$GH_LOG"
printf '\\036' >> "$GH_LOG"
if [ "${GH_FAIL:-0}" = "1" ]; then
  printf '%s\\n' "${GH_ERR:-gh failed}" >&2
  exit 1
fi
printf '%s' "${GH_PRS_JSON:-[]}"
exit 0
"""

CURL_STUB = """#!/bin/bash
printf '%s\\0' "$@" >> "$CURL_LOG"
printf '\\036' >> "$CURL_LOG"
if [ "${CURL_FAIL:-0}" = "1" ]; then
  echo "curl: (22) HTTP failed" >&2
  exit 22
fi
exit 0
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load() -> tuple[dict, str, str]:
    path = _repo_root() / ".github" / "workflows" / WORKFLOW
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise AssertionError(f"{WORKFLOW} did not parse to a mapping")
    steps = doc.get("jobs", {}).get("budget-alarm", {}).get("steps", [])
    by_name = {step.get("name"): step for step in steps}
    for name in (COUNT_STEP, SLACK_STEP):
        if name not in by_name or "run" not in by_name[name]:
            raise AssertionError(f"{name!r} run step missing from {WORKFLOW}")
    return doc, by_name[COUNT_STEP]["run"], by_name[SLACK_STEP]["run"]


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


def _outputs(text: str) -> dict[str, str]:
    parsed = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    return parsed


def _prs(names: list[str]) -> str:
    return json.dumps([{"number": index + 1, "headRefName": name} for index, name in enumerate(names)])


class TestPrBudgetAlarm(unittest.TestCase):
    """Budget thresholds, the cursor/ prefix, and the non-paging failure modes."""

    doc: dict
    count_script: str
    slack_script: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc, cls.count_script, cls.slack_script = _load()

    def _count(self, bin_dir: Path, names: list[str] | None = None, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        (bin_dir / "gh").write_text(GH_STUB, encoding="utf-8")
        (bin_dir / "gh").chmod(0o755)
        env = {
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - bash fallback only
            "LANG": "C",
            "GH_REPO": "pcalnon/juniper-data-client",
            "GH_LOG": str(bin_dir / "gh.log"),
            "GITHUB_OUTPUT": str(bin_dir / "output.txt"),
            "GITHUB_STEP_SUMMARY": str(bin_dir / "summary.md"),
        }
        env["GH_TOKEN"] = "gh-actions-fixture-do-not-leak"  # nosec B105
        if names is not None:
            env["GH_PRS_JSON"] = _prs(names)
        env.update(env_overrides)
        return subprocess.run(  # nosec B603 B607 - fixed bash, workflow-owned script
            ["bash", "-c", self.count_script],
            cwd=bin_dir,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=15,
        )

    def _slack(self, bin_dir: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        (bin_dir / "curl").write_text(CURL_STUB, encoding="utf-8")
        (bin_dir / "curl").chmod(0o755)
        env = {
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - bash fallback only
            "LANG": "C",
            "CURL_LOG": str(bin_dir / "curl.log"),
            "RUN_URL": RUN_URL,
            "LEVEL": "WARN",
            "TOTAL": "16",
            "CURSOR": "4",
            "WARN": "15",
            "ALARM": "30",
        }
        env.update(env_overrides)
        return subprocess.run(  # nosec B603 B607 - fixed bash, workflow-owned script
            ["bash", "-c", self.slack_script],
            cwd=bin_dir,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=15,
        )

    def test_workflow_contract(self) -> None:
        """Schedule and dispatch only, read-only permissions, breach does not select Slack on OK."""
        triggers = self.doc.get("on", self.doc.get(True))
        self.assertEqual(set(triggers), {"schedule", "workflow_dispatch"})
        self.assertEqual(triggers["schedule"], [{"cron": "0 14 * * *"}])
        self.assertEqual(self.doc.get("permissions"), {"contents": "read", "pull-requests": "read"})
        self.assertEqual(self.doc.get("concurrency"), {"group": "pr-budget-alarm", "cancel-in-progress": True})
        steps = self.doc["jobs"]["budget-alarm"]["steps"]
        count = next(step for step in steps if step.get("name") == COUNT_STEP)
        slack = next(step for step in steps if step.get("name") == SLACK_STEP)
        self.assertEqual(count.get("id"), "count")
        self.assertNotIn("SLACK", self.count_script)
        self.assertNotIn("SLACK_WEBHOOK_URL", count.get("env", {}))
        self.assertEqual(slack.get("if"), "steps.count.outputs.level != 'OK'")
        self.assertIs(slack.get("continue-on-error"), True)
        self.assertEqual(slack["env"]["SLACK_WEBHOOK_URL"], "${{ secrets.SLACK_WEBHOOK_URL }}")

    def test_threshold_boundaries(self) -> None:
        """Warn and alarm are >=. A count that meets both is ALARM. Either series can trip it."""
        cases = [
            (["b"] * 14, "15", "30", "OK"),
            (["b"] * 15, "15", "30", "WARN"),
            (["b"] * 29, "15", "30", "WARN"),
            (["b"] * 30, "15", "30", "ALARM"),
            (["cursor/a"] * 14, "15", "30", "OK"),
            (["cursor/a"] * 15, "15", "30", "WARN"),
            (["cursor/a"] * 30, "15", "30", "ALARM"),
            (["b"] * 9, "10", "100", "OK"),
            (["b"] * 10, "10", "100", "WARN"),
            (["b"] * 100, "10", "100", "ALARM"),
            (["cursor/a"] * 100, "10", "90", "ALARM"),
            (["cursor/a"] * 30, "15", "30", "ALARM"),
        ]
        for names, warn, alarm, level in cases:
            with self.subTest(total=len(names), warn=warn, alarm=alarm, level=level):
                with tempfile.TemporaryDirectory() as tmp:
                    bin_dir = Path(tmp)
                    proc = self._count(bin_dir, names, PR_BUDGET_WARN=warn, PR_BUDGET_ALARM=alarm)
                    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                    parsed = _outputs((bin_dir / "output.txt").read_text(encoding="utf-8"))
                    self.assertEqual(parsed["level"], level)
                    self.assertEqual(parsed["total"], str(len(names)))
                    summary = (bin_dir / "summary.md").read_text(encoding="utf-8")
                    self.assertIn(f"**{level}**", summary)

    def test_empty_thresholds_default_to_15_and_30(self) -> None:
        """An unset repo variable arrives as an empty string, not as an unbound one."""
        bounds = [(["b"] * 14, "OK"), (["b"] * 15, "WARN"), (["b"] * 30, "ALARM")]
        for names, level in bounds:
            with self.subTest(total=len(names), level=level):
                with tempfile.TemporaryDirectory() as tmp:
                    bin_dir = Path(tmp)
                    proc = self._count(bin_dir, names, PR_BUDGET_WARN="", PR_BUDGET_ALARM="")
                    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                    parsed = _outputs((bin_dir / "output.txt").read_text(encoding="utf-8"))
                    self.assertEqual(parsed["warn"], "15")
                    self.assertEqual(parsed["alarm"], "30")
                    self.assertEqual(parsed["level"], level)

    def test_only_a_cursor_slash_prefix_counts(self) -> None:
        """``cursor-bot/``, a bare ``cursor``, and a case difference are not the fleet prefix."""
        names = ["cursor/a", "cursor-bot/b", "cursor", "Cursor/c", "feature/cursor/d", "cursor/e"]
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._count(bin_dir, names, PR_BUDGET_WARN="15", PR_BUDGET_ALARM="30")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            parsed = _outputs((bin_dir / "output.txt").read_text(encoding="utf-8"))
            self.assertEqual(parsed["total"], "6")
            self.assertEqual(parsed["cursor"], "2")
            self.assertEqual(parsed["level"], "OK")

    def test_gh_failure_stays_green_and_is_not_an_empty_queue(self) -> None:
        """The warning path writes level=OK and does not publish total=0."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._count(bin_dir, GH_FAIL="1", GH_ERR="api timeout")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            combined = proc.stdout + proc.stderr
            self.assertIn("::warning", combined)
            self.assertIn("api timeout", combined)
            written = (bin_dir / "output.txt").read_text(encoding="utf-8")
            self.assertIn("level=OK", written)
            self.assertNotIn("total=", written)
            self.assertNotIn("cursor=", written)
            summary = (bin_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Could not query open PRs", summary)
            self.assertNotIn("Open PRs (total)", summary)

    def test_invalid_pr_json_fails_closed(self) -> None:
        """jq cannot count an unreadable listing, and that is not reported as OK."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._count(bin_dir, GH_PRS_JSON="not-json")
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_missing_head_ref_fails_closed(self) -> None:
        """An element without headRefName must not be counted as a non-cursor PR."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._count(bin_dir, GH_PRS_JSON='[{"number": 1}]')
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_non_numeric_alarm_does_not_abort_the_warn_compare(self) -> None:
        """A bad alarm value is an integer-expression error inside ``if``, so warn still runs."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._count(bin_dir, ["b"] * 15, PR_BUDGET_WARN="15", PR_BUDGET_ALARM="abc")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            # bash <= 5.2 says "integer expression expected"; bash 5.3 says "integer expected".
            self.assertRegex(proc.stderr, r"integer (expression )?expected")
            parsed = _outputs((bin_dir / "output.txt").read_text(encoding="utf-8"))
            self.assertEqual(parsed["level"], "WARN")

    def test_missing_webhook_annotates_and_does_not_post(self) -> None:
        """This repo has no webhook. A breach must say so, and must not call curl."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._slack(bin_dir, LEVEL="ALARM", TOTAL="40", CURSOR="3", WARN="15", ALARM="30")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("::warning title=PR budget ALARM with no Slack webhook::", proc.stdout)
            self.assertIn("40 open PR(s), 3 on cursor/ branches", proc.stdout)
            self.assertFalse((bin_dir / "curl.log").exists())

    def test_slack_payload_omits_the_webhook(self) -> None:
        """The webhook is the POST target. It is not part of the message."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._slack(bin_dir, SLACK_WEBHOOK_URL=WEBHOOK)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            calls = _invocations((bin_dir / "curl.log").read_bytes())
            self.assertEqual(len(calls), 1)
            args = calls[0]
            self.assertEqual(args[-1], WEBHOOK)
            payload = json.loads(args[args.index("-d") + 1])
            self.assertEqual(set(payload), {"text"})
            text = payload["text"]
            self.assertIn("PR budget WARN", text)
            self.assertIn("16 open PR(s)", text)
            self.assertIn("4 on cursor/ branches", text)
            self.assertIn(RUN_URL, text)
            self.assertNotIn("SECRETWEBHOOK", text)
            self.assertNotIn(WEBHOOK, text)
            self.assertNotIn("SECRETWEBHOOK", " ".join(args[:-1]))

    def test_failed_post_exits_nonzero(self) -> None:
        """The script itself fails the POST. continue-on-error on the step is what keeps the run green."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            proc = self._slack(bin_dir, SLACK_WEBHOOK_URL=WEBHOOK, CURL_FAIL="1")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("curl: (22)", proc.stderr)
