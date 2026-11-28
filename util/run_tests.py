#!/usr/bin/env python3
"""Run any subset of juniper-data-client's tests locally, with the commands CI runs.

Project: juniper-data-client
Sub-Project: developer tooling (local test runs)
Author: Paul Calnon (authored via Claude Code)
Created: 2026-11-25 (a thin spec over the shared engine since 2026-11-28)
Status: permanent utility
Related: ``juniper_ci_tools.run_tests`` in juniper-ci-tools 0.10 (the engine this declares a spec for; its canonical
         home is juniper-ml's ``juniper-ci-tools/`` tree); .github/workflows/ci.yml and scheduled-tests.yml (the jobs
         it reads); tests/test_run_tests.py (this spec against the real workflows and the docs; the engine's own
         suite is juniper-ml's juniper-ci-tools/tests/test_run_tests.py); docs/REFERENCE.md § Running Tests Locally
         (the operator documentation)

Why this exists
---------------
CI tests this package in the jobs of ``ci.yml``: ``unit-tests`` runs pytest on ``tests/`` with coverage, then a
coverage floor (``coverage report --fail-under=${COVERAGE_FAIL_UNDER}``, set in the workflow's ``env:``) and a
per-file coverage gate; ``integration-tests`` runs ``-m integration`` and accepts pytest's exit 5 (no tests
collected); ``generator-parity`` and ``security`` each re-run one test file with a variable that turns its skip into
a failure. ``scheduled-tests.yml`` runs the integration marker nightly. ``pytest tests/`` alone, AGENTS.md's
command, checks neither the floor nor the per-file gate, and writes CI's reports into the checkout, whose
``reports/`` is not gitignored. This script runs any subset of those jobs as CI runs them, reading the commands
from the workflows on every run, so its lanes cannot drift from CI's.

What is juniper-data-client's own
---------------------------------
Everything that runs is the engine's: ``juniper_ci_tools.run_tests`` (the ``--help`` epilog and docs/REFERENCE.md
say how it selects, runs and reports). This file declares only what the engine cannot read from the workflows, as
its ``RepoSpec``:

- every workflow job that runs pytest is a lane named by its job id (``unit-tests``; ``unit`` works too, the
  ``-tests`` suffix stripped), and the ``python3 -m unittest`` steps of the pull-request workflows would form the
  ``unittest`` lane (none today: every test here is pytest's);
- ``OPT_IN``: the two lanes ``all`` leaves out although a pull request's CI gates on them, and why;
- the ``--help`` text, which tests/test_run_tests.py pins against docs/REFERENCE.md.

The engine ships in juniper-ci-tools 0.10 (``ENGINE_REQUIREMENT``), which the ``[test]`` extra installs: ``pip
install -e ".[test]"``. Without it this script prints the command that installs it and exits 3, the engine's
UNAVAILABLE (nothing failed, and nothing could run).
"""

from __future__ import annotations

import shlex
import sys

# The juniper-ci-tools releases this spec is written for: the [test] extra, and every workflow job that installs
# .[test] and then juniper-ci-tools again, ask for exactly this (tests/test_run_tests.py checks both).
ENGINE_REQUIREMENT = "juniper-ci-tools>=0.10.0,<0.11.0"

try:
    from juniper_ci_tools import run_tests as rt
except ImportError as exc:  # juniper-ci-tools is missing, or older than 0.10 (no run_tests module)
    _install = shlex.join([sys.executable, "-m", "pip", "install", ENGINE_REQUIREMENT])
    sys.stderr.write(f"run_tests.py: the test runner's engine, juniper_ci_tools.run_tests, cannot be imported ({exc})\n")
    sys.stderr.write(f"  install it: {_install}\n")
    sys.stderr.write('  (pip install -e ".[test]" installs it, with everything the lanes need)\n')
    raise SystemExit(3) from None  # the engine's EXIT_UNAVAILABLE

# The lanes a pull request's CI gates on that `all` leaves out, and why. Each re-runs one test file the unit-tests lane
# also runs, with a variable that turns its skip into a failure, and needs a tool this repo's [test] extra does not
# install. A name here that is no lane is a usage error, so this table cannot outlive a renamed job.
OPT_IN = {
    "generator-parity": "needs juniper-data[api]: JUNIPER_DATA_CLIENT_REQUIRE_LIVE_REGISTRY makes its skip a failure",
    "security": "needs the gitleaks binary on PATH (or GITLEAKS_BIN): GITLEAKS_PARITY_REQUIRED makes its skip a failure",
}

_DESCRIPTION = """\
Run any subset of juniper-data-client's tests locally, with the commands CI runs.

The lanes come from the workflows on every run: each job that runs pytest is one (ci.yml's
unit-tests, integration-tests, generator-parity and security; scheduled-tests.yml's
slow-or-integration-tests), with its test steps as CI runs them: its pytest options, its exit
rules and its coverage gates. Every selected job runs (CI stops at the first failure), each step in
its own process group, and the results go to one log per step plus summary.json under the primary
checkout's tmp/."""

_EPILOG = """\
selectors (any number; default: all):
  all                       every lane a pull request's CI gates on that the [test] extra can run:
                            unit-tests and integration-tests
  <lane>                    one CI job, as CI runs it, by its job id or without -tests:
                            unit-tests (unit): pytest on tests/ with coverage, then the coverage
                            floor and the per-file coverage gate; integration-tests (integration):
                            -m integration, where exit 5 (no tests collected) passes, as in CI
  optional                  every lane `all` leaves out, each when named: generator-parity (needs
                            juniper-data[api]) and security (needs the gitleaks binary on PATH or
                            in GITLEAKS_BIN), CI's re-runs of one test file with the variable that
                            makes its skip a failure; slow-or-integration-tests (scheduled-tests.yml,
                            which CI runs nightly, never on a pull request)
  <test file>               a file under tests/, by path, name or stem: tests/test_client.py,
                            test_client.py, test_client or client: unit-tests' pytest on it alone,
                            with the coverage gates off
  '<glob>'                  every test file whose name matches: '*gitleaks*'
  <file>::<Class>[::<test>]  one class or test of a file; tests.test_client.TestX.test_y works too
  tests                     the unit-tests lane, by its test directory
  unittest                  every suite a workflow runs with `python3 -m unittest` (none today)
  --changed adds the lanes and test files a change touches; --failed adds the failures of the last
  run.

examples:
  python3 util/run_tests.py                            everything, -j auto, past failures
  python3 util/run_tests.py unit                       the unit tests with both coverage gates
  python3 util/run_tests.py client -- -x -vv           one test file, pytest stops at a failure
  python3 util/run_tests.py tests/test_client.py::TestHealthEndpoints
  python3 util/run_tests.py unit -k retry              the unit tests matching "retry" (gates off)
  python3 util/run_tests.py --python "$VIRTUAL_ENV/bin/python" all
  python3 util/run_tests.py generator-parity           needs juniper-data[api] installed
  python3 util/run_tests.py --changed                  what the branch touches (vs origin/main)
  python3 util/run_tests.py --failed                   the failures of the last run, again
  python3 util/run_tests.py --list optional            what a selector selects; runs nothing

exit codes:
  0    every selected job passed (a job that failed alongside others and passed alone counts,
       and is listed)
  1    at least one job failed or timed out
  2    usage or configuration error: an unknown selector, a workflow it cannot parse
  3    nothing failed, but at least one selected job could not run (UNAVAILABLE)
  5    no test ran: -k matched nothing in any selected job
  130  interrupted; summary.json holds the jobs that finished

Logs: the primary checkout's tmp/run-tests/<checkout name>/<UTC stamp>/ (one log per step,
summary.json, summary.txt; never pruned). Full documentation: docs/REFERENCE.md,
§ Running Tests Locally."""

SPEC = rt.RepoSpec(
    project="juniper-data-client",
    script=__file__,
    lanes=(rt.unittest_lane(), rt.job_lanes()),
    opt_in=OPT_IN,
    strip_suffix="-tests",
    docs="docs/REFERENCE.md, 'Running Tests Locally'",
    description=_DESCRIPTION,
    epilog=_EPILOG,
)


if __name__ == "__main__":
    raise SystemExit(rt.main(SPEC))
