#!/usr/bin/env python3
"""Extracted-shell rehearsal for the notify-downstream dispatches (#211 / #212).

A bare ``curl -X POST`` exits 0 on HTTP 401, 403, and 404. A revoked or
under-scoped ``CROSS_REPO_DISPATCH_TOKEN`` then reported a dispatch that never
happened, and the downstream CI this job exists to trigger never ran. The fix
is ``--fail-with-body`` plus building the payload into a variable first, so a
``jq`` failure stops the step instead of posting an empty body.

This unittest runs the workflow's own shell. The curl stub implements curl's
``--fail-with-body`` contract: a status >= 400 is exit 0 unless that flag is
present, in which case it is exit 22 and the response body is on stdout.
Dropping the flag makes a 401 look successful.

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

WORKFLOW = "ci.yml"
JOB = "notify-downstream"
TOKEN = "cross-repo-dispatch-token-do-not-leak"  # nosec B105
SHA = "0123456789abcdef0123456789abcdef01234567"
ERROR_BODY = '{"message":"Bad credentials","documentation_url":"https://docs.github.com"}'

TARGETS = {
    "Dispatch to juniper-data": "https://api.github.com/repos/pcalnon/juniper-data/dispatches",
    "Dispatch to juniper-cascor": "https://api.github.com/repos/pcalnon/juniper-cascor/dispatches",
    "Dispatch to juniper-canopy": "https://api.github.com/repos/pcalnon/juniper-canopy/dispatches",
}

CURL_STUB = """#!/bin/bash
printf '%s\\0' "$@" >> "$CURL_LOG"
printf '\\036' >> "$CURL_LOG"
has_fail=0
for arg in "$@"; do
  if [ "$arg" = "--fail-with-body" ]; then
    has_fail=1
  fi
done
status="${CURL_STATUS:-204}"
if [ "$status" -ge 400 ] && [ "$has_fail" -eq 1 ]; then
  printf '%s\\n' "${CURL_BODY:-http error}"
  printf 'curl: (22) The requested URL returned error: %s\\n' "$status" >&2
  exit 22
fi
exit 0
"""

JQ_FAIL_STUB = """#!/bin/bash
echo "jq: forced failure" >&2
exit 3
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_workflow() -> dict:
    path = _repo_root() / ".github" / "workflows" / WORKFLOW
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise AssertionError(f"{WORKFLOW} did not parse to a mapping")
    return doc


def _job(doc: dict) -> dict:
    job = doc.get("jobs", {}).get(JOB)
    if not isinstance(job, dict):
        raise AssertionError(f"{WORKFLOW} has no {JOB} job")
    return job


def _steps(doc: dict) -> dict[str, dict]:
    found = {step.get("name"): step for step in _job(doc).get("steps", [])}
    missing = [name for name in TARGETS if name not in found or "run" not in found[name]]
    if missing:
        raise AssertionError(f"{JOB} is missing dispatch steps: {missing}")
    return found


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
    """Minimal environment. Do not copy ``os.environ`` into the child."""
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git/bash fallback only
        "LANG": "C",
        "TOKEN": TOKEN,
        "SOURCE_SHA": SHA,
        "CURL_LOG": str(bin_dir / "curl.log"),
    }
    env.update(overrides)
    return env


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


class TestNotifyDownstreamDispatch(unittest.TestCase):
    """The three dispatches fail on HTTP errors and never post the token."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = _load_workflow()
        cls.steps = _steps(cls.doc)

    def _run(self, step_name: str, bin_dir: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        script = self.steps[step_name]["run"]
        return subprocess.run(  # nosec B603 B607 - fixed bash, workflow-owned script
            ["bash", "-c", script],
            cwd=bin_dir,
            capture_output=True,
            text=True,
            env=_child_env(bin_dir, **env_overrides),
            check=False,
            timeout=15,
        )

    def test_job_contract(self) -> None:
        """Main-push only, token via env, later targets still run after a failure."""
        job = _job(self.doc)
        self.assertEqual(job.get("if"), "github.ref == 'refs/heads/main' && github.event_name == 'push'")
        self.assertEqual(job.get("needs"), ["required-checks"])
        expected_env = {"SOURCE_SHA": "${{ github.sha }}"}
        expected_env["TOKEN"] = "${{ secrets.CROSS_REPO_DISPATCH_TOKEN }}"  # nosec B105
        self.assertEqual(job.get("env"), expected_env)
        ordered = [step.get("name") for step in job["steps"]]
        self.assertEqual(ordered, list(TARGETS))
        self.assertIsNone(self.steps["Dispatch to juniper-data"].get("if"))
        for name in ("Dispatch to juniper-cascor", "Dispatch to juniper-canopy"):
            self.assertEqual(self.steps[name].get("if"), "${{ !cancelled() }}")
        for name, url in TARGETS.items():
            script = self.steps[name]["run"]
            self.assertIn("--fail-with-body", script)
            self.assertIn("--show-error", script)
            self.assertIn(url, script)
            self.assertIn('payload="$(jq -nc', script)
            self.assertNotIn("${{", script)
            self.assertNotIn("github.token", script)

    def test_success_posts_the_sha_and_not_the_token(self) -> None:
        """204 carries event_type, source, and sha. The token is only the Authorization header."""
        for name, url in TARGETS.items():
            with self.subTest(step=name):
                with tempfile.TemporaryDirectory() as tmp:
                    bin_dir = Path(tmp)
                    _write_executable(bin_dir / "curl", CURL_STUB)
                    proc = self._run(name, bin_dir, CURL_STATUS="204")
                    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                    calls = _invocations((bin_dir / "curl.log").read_bytes())
                    self.assertEqual(len(calls), 1, msg=calls)
                    args = calls[0]
                    self.assertIn("--fail-with-body", args)
                    self.assertIn(url, args)
                    self.assertIn(f"Authorization: Bearer {TOKEN}", args)
                    payload = json.loads(args[args.index("-d") + 1])
                    self.assertEqual(
                        payload,
                        {
                            "event_type": "data-client-updated",
                            "client_payload": {"source": "juniper-data-client", "sha": SHA},
                        },
                    )
                    self.assertNotIn(TOKEN, payload["client_payload"].values())
                    self.assertNotIn(TOKEN, url)

    def test_http_error_fails_and_prints_the_body(self) -> None:
        """401/403/404/500 fail the step. Without --fail-with-body the stub would exit 0."""
        for name in TARGETS:
            for status in ("401", "403", "404", "500"):
                with self.subTest(step=name, status=status):
                    with tempfile.TemporaryDirectory() as tmp:
                        bin_dir = Path(tmp)
                        _write_executable(bin_dir / "curl", CURL_STUB)
                        proc = self._run(name, bin_dir, CURL_STATUS=status, CURL_BODY=ERROR_BODY)
                        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                        self.assertIn(ERROR_BODY, proc.stdout)
                        calls = _invocations((bin_dir / "curl.log").read_bytes())
                        self.assertEqual(len(calls), 1)

    def test_jq_failure_does_not_call_curl(self) -> None:
        """A payload that never builds must not POST an empty body."""
        for name in TARGETS:
            with self.subTest(step=name):
                with tempfile.TemporaryDirectory() as tmp:
                    bin_dir = Path(tmp)
                    _write_executable(bin_dir / "curl", CURL_STUB)
                    _write_executable(bin_dir / "jq", JQ_FAIL_STUB)
                    proc = self._run(name, bin_dir, CURL_STATUS="401", CURL_BODY=ERROR_BODY)
                    self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                    self.assertIn("jq: forced failure", proc.stderr)
                    self.assertFalse((bin_dir / "curl.log").exists())
                    self.assertNotIn(ERROR_BODY, proc.stdout)
