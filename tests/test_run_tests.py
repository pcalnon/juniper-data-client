#!/usr/bin/env python3
"""Regression tests for util/run_tests.py, the local test runner.

Project:     Juniper
Sub-Project: juniper-data-client
Application: repo tooling tests
Author:      Paul Calnon (authored via Claude Code)
License:     MIT License

util/run_tests.py runs any subset of juniper-data-client's tests with the commands CI runs, reading them from the
workflows on every run. Two kinds of test pin it (ported with it from juniper-recurrence's tests/test_run_tests.py):

- **Against the real workflows** (no test runs): every workflow job that runs pytest is a lane, and only those;
  ``all`` is unit-tests and integration-tests; unit-tests runs pytest, then the coverage floor CI's ``env:`` sets,
  then the per-file gate; both integration lanes accept exit 5, as their workflows do; the opt-in lanes carry the
  variable that turns their skip into a failure. A workflow change the runner cannot read fails here, in the PR that
  makes it (CI's unit-tests job collects this file), instead of on the next person's machine.
- **Against a fake checkout** built under this checkout's gitignored ``tmp/`` (never the system temp directory, which
  the runner refuses as a log root): a ci.yml with the real job ids, a scheduled workflow, and a workflow running
  three unittest suites. Its ``pytest``, ``coverage`` and gate are stubs that record how they were called. These run
  the real script end to end -- selection, env expansion, the exit-5 rule, the report routing, timeouts and
  process-group kills, leftover processes, the parallel re-run, ``-k``, ``--failed``, ``--changed``, the child
  environment, the exit codes -- in well under a minute.

Every option ``--help`` lists, and every lane, must appear in docs/REFERENCE.md's "Running Tests Locally" section.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - runs the runner and git with a fixed argv, in a fake checkout under tmp/
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "util" / "run_tests.py"
REFERENCE = REPO_ROOT / "docs" / "REFERENCE.md"
AGENTS = REPO_ROOT / "AGENTS.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SECTION = "## Running Tests Locally"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


RT = _load_runner()


def _child_env(base: dict | None = None, **overrides: str):
    """A child environment whose repr never shows a value (the runner's own _Env)."""
    env = RT._Env(os.environ if base is None else base)
    env.update(overrides)
    return env


# The real job ids: the runner's OPT_IN names generator-parity and security, and refuses a checkout without them.
CI_YML = """\
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
env:
  PYTHON_TEST_VERSION: "3.99"
  COVERAGE_FAIL_UNDER: "77"
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - name: Run pre-commit hooks
        run: pre-commit run --all-files --show-diff-on-failure
  unit-tests:
    needs: [pre-commit]
    runs-on: ${{ matrix.os }}
    continue-on-error: ${{ matrix.experimental == true }}
    env:
      JOB_LEVEL: "from-${{ env.PYTHON_TEST_VERSION }}"
      FAKE_CI_ONLY: ${{ secrets.FAKE_CI_ONLY }}
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
      - name: Install dependencies
        run: |
          python -m pip install --upgrade "pip>=26.1.1"
          pip install -e ".[test]"
      - name: Create required directories
        run: |
          mkdir -p reports/junit reports/htmlcov
      - name: Run Unit Tests
        run: |
          echo "unit tests"
          python -m pytest \\
            tests/ \\
            --verbose \\
            --junitxml=reports/junit/junit-unit.xml \\
            --cov=fake_client \\
            --cov-report=term-missing:skip-covered \\
            --cov-report=xml:reports/coverage.xml \\
            --cov-report=json:coverage.json \\
            --cov-report=html:reports/htmlcov
      - name: Enforce Coverage Gate
        run: |
          echo "coverage gate"
          python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}
      - name: Enforce Per-File Coverage Gate
        run: |
          pip install "juniper-ci-tools>=0.9.0,<0.10.0"
          python gate.py --coverage-json coverage.json --enforce
      - name: Upload Coverage Artifacts
        if: always()
        uses: actions/upload-artifact@v4
      - name: Report the informational Python 3.15 result
        if: always() && matrix.experimental == true
        env:
          STEPS_JSON: ${{ toJSON(steps) }}
        run: |
          python3 - <<'PY'
          print("${{ toJSON(steps) }}")
          PY
  integration-tests:
    needs: [pre-commit]
    runs-on: ubuntu-latest
    steps:
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run Integration Tests (-m integration)
        run: |
          set +e
          python -m pytest \\
            -m "integration" \\
            tests/ \\
            --junitxml=reports/junit/junit-integration.xml
          rc=$?
          set -e
          if [ "$rc" = "5" ]; then
            echo "::notice::No integration-marked tests collected -- treating as skip."
            exit 0
          fi
          exit "$rc"
  generator-parity:
    needs: [pre-commit]
    runs-on: ubuntu-latest
    steps:
      - name: Install dependencies (including the server)
        run: |
          pip install -e ".[test]"
          pip install "fake-server[api]"
      - name: Confirm the server is importable
        run: python -c "import fake_server; print('ok')"
      - name: Run Generator Parity Cross-Check
        env:
          FAKE_REQUIRE_LIVE: "1"
        run: |
          python -m pytest tests/test_demo.py --verbose -rs
  security:
    needs: [pre-commit]
    runs-on: ubuntu-latest
    steps:
      - name: Install security tools
        run: pip install "bandit[sarif]"
      - name: Run Bandit (SAST)
        run: |
          bandit -r fake_client -f sarif -o reports/security/bandit.sarif --exit-zero
      - name: Run pip-audit (Dependency Vulnerabilities)
        run: |
          pip install -e ".[test]"
          pip list --format=freeze > reports/security/requirements.txt
      - name: Gitleaks pull-request scan self-test (real engine)
        env:
          FAKE_PARITY_REQUIRED: "1"
        run: python -m pytest tests/test_demo.py -v -v
  build:
    needs: [unit-tests]
    runs-on: ubuntu-latest
    steps:
      - name: Build sdist and wheel
        run: python -m build --sdist --wheel
"""

SCHEDULED_YML = """\
name: Scheduled Tests
on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:
env:
  PYTHON_TEST_VERSION: "3.12"
jobs:
  slow-or-integration-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Install dependencies
        run: |
          pip install -e ".[test]" || pip install -e ".[dev]" || pip install -e .
      - name: Run slow / integration tests
        run: |
          mkdir -p reports/junit
          set +e
          python -m pytest \\
            -m integration \\
            --tb=short \\
            --junitxml=reports/junit/junit-slow.xml
          rc=$?
          set -e
          if [ "$rc" -eq 5 ]; then
            echo "::notice::No integration tests collected."
            exit 0
          fi
          exit "$rc"
"""

TOOLING_YML = """\
name: Tooling
on:
  pull_request:
    branches: [main]
jobs:
  tooling:
    runs-on: ubuntu-latest
    steps:
      - name: Install the tooling's dependency
        run: |
          python -m pip install --upgrade "pip>=26.1.1"
          pip install "pyyaml>=6.0.1"
      - name: Run the tooling suites
        run: |
          set -euo pipefail
          # tests/test_ghost.py: a comment names a suite; it is not one
          python3 -m unittest -v tests/test_alpha.py
          python3 -m unittest -v tests/test_beta.py  # a trailing comment
          python3 -m unittest -v tests/test_gamma.py
"""

SCHEDULE_ONLY_YML = """\
name: Scheduled scan
on:
  schedule:
    - cron: "0 3 * * 1"
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - name: A scheduled-only suite
        run: python3 -m unittest -v tests/test_scheduled_only.py
"""

# Stand-ins for pytest and coverage: `python -m X` imports them from the directory it runs in, before any real one.
PYTEST_STUB = """\
    import json
    import os
    import sys
    from pathlib import Path

    import fake_client

    args = sys.argv[1:]
    keys = ("COVERAGE_FAIL_UNDER", "PYTHON_TEST_VERSION", "JOB_LEVEL", "FAKE_CI_ONLY", "FAKE_REQUIRE_LIVE", "FAKE_PARITY_REQUIRED")
    record = {
        "argv": args,
        "cwd": os.getcwd(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "COVERAGE_FILE": os.environ.get("COVERAGE_FILE"),
        "fake_client": fake_client.__file__,
        "env": {key: os.environ.get(key) for key in keys},
    }
    Path(os.environ["FAKE_STATE"], os.environ.get("FAKE_PYTEST_RECORD", "pytest") + ".json").write_text(json.dumps(record))
    for arg in args:  # write every report where it was told to: a path left unrouted lands in the checkout
        if arg.startswith("--junitxml="):
            target = Path(arg.split("=", 1)[1])
        elif arg.startswith("--cov-report=") and ":" in arg and arg.split("=", 1)[1].split(":", 1)[0] in ("xml", "json", "html"):
            target = Path(arg.split(":", 1)[1])
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"totals": {"percent_covered": 100.0}}))
    if any(arg.startswith("--cov=") for arg in args):
        Path(os.environ.get("COVERAGE_FILE", ".coverage")).write_text("fake coverage data")
    rc = int(os.environ.get("FAKE_PYTEST_RC", "5" if "integration" in args else "0"))
    print({0: "1 passed in 0.01s", 5: "12 deselected in 0.01s"}.get(rc, "FAILED tests/test_demo.py::test_demo - boom\\n1 failed in 0.01s"))
    sys.exit(rc)
    """

COVERAGE_STUB = """\
    import json
    import os
    import sys
    from pathlib import Path

    data = os.environ.get("COVERAGE_FILE", ".coverage")
    record = {"argv": sys.argv[1:], "cwd": os.getcwd(), "COVERAGE_FILE": os.environ.get("COVERAGE_FILE"), "data_exists": Path(data).is_file()}
    Path(os.environ["FAKE_STATE"], "coverage-report.json").write_text(json.dumps(record))
    rc = int(os.environ.get("FAKE_COVERAGE_RC", "0"))
    print("Coverage failure: total of 76.50 is less than fail-under=77.00" if rc else "TOTAL  10  0  100.00%")
    sys.exit(rc)
    """

FILES = {
    "pyproject.toml": """\
        [project]
        name = "juniper-data-client"
        version = "0.0.1"
        dependencies = []

        [project.optional-dependencies]
        test = []
        dev = ["black>=23.0.0"]
        """,
    "fake_client/__init__.py": "VALUE = 1\n",
    "tests/__init__.py": "",
    "tests/conftest.py": "",
    "tests/test_demo.py": "def test_demo():\n    assert True\n",
    "tests/test_widget.py": "# The widget reads util/check.py.\ndef test_widget():\n    assert True\n",
    "tests/test_alpha.py": """\
        import unittest


        class AlphaTest(unittest.TestCase):
            def test_cap_one(self):
                self.assertTrue(True)

            def test_two(self):
                self.assertTrue(True)
        """,
    "tests/test_beta.py": """\
        import os
        import unittest
        from pathlib import Path


        class BetaTest(unittest.TestCase):
            def test_fails(self):
                mode = os.environ.get("FAKE_BETA", "fail")
                if mode == "flaky":
                    marker = Path(os.environ["FAKE_STATE"]) / "beta-ran-once"
                    if not marker.exists():
                        marker.write_text("1")
                        self.fail("the first attempt fails")
                    return
                self.assertEqual(mode, "pass", "beta was told to fail")
        """,
    "tests/test_gamma.py": """\
        # Gamma reads what util/check.py prints and uses tests/helper_mod.py (the --changed test needs both names).
        import json
        import os
        import subprocess
        import sys
        import time
        import unittest
        from pathlib import Path

        KEYS = ["TMPDIR", "GIT_CEILING_DIRECTORIES", "DBUS_SESSION_BUS_ADDRESS", "PYTHONPATH", "PYTHONHOME", "DISPLAY",
                "WAYLAND_DISPLAY", "PYTHONUNBUFFERED", "CLAUDE_FAKE", "JUNIPER_FAKE", "WTC_FAKE"]


        class GammaTest(unittest.TestCase):
            def test_environment(self):
                state = Path(os.environ["FAKE_STATE"])
                record = {key: os.environ.get(key) for key in KEYS}
                record["stdin"] = sys.stdin.read()
                (state / "gamma-env.json").write_text(json.dumps(record))
                if os.environ.get("FAKE_GAMMA_CHILD"):
                    child = subprocess.Popen(["sleep", "300"])
                    (state / "gamma-child.pid").write_text(str(child.pid))
                seconds = float(os.environ.get("FAKE_GAMMA_SLEEP", "0"))
                if seconds:
                    (state / "gamma-sleeping").write_text("1")
                    time.sleep(seconds)
        """,
    "tests/test_scheduled_only.py": "import unittest\n",
    "tests/helper_mod.py": "VALUE = 1\n",
    "util/check.py": "print('check ok')\n",
    "gate.py": """\
        import json
        import os
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        report = Path(args[args.index("--coverage-json") + 1])
        Path(os.environ["FAKE_STATE"], "gate.json").write_text(json.dumps({"argv": args, "cwd": os.getcwd(), "report_exists": report.is_file()}))
        sys.exit(0 if report.is_file() else 1)
        """,
    "pytest.py": PYTEST_STUB,
    "coverage.py": COVERAGE_STUB,
}
FAKE_KEYS = ("FAKE_BETA", "FAKE_GAMMA_SLEEP", "FAKE_GAMMA_CHILD", "FAKE_PYTEST_RC", "FAKE_PYTEST_RECORD", "FAKE_COVERAGE_RC", "FAKE_REQUIRE_LIVE", "FAKE_PARITY_REQUIRED", "FAKE_CI_ONLY", "JOB_LEVEL", "NO_COLOR")


def _repo_tmp() -> Path:
    """This checkout's gitignored tmp/: on a CI runner the default temp directory is the system one."""
    path = REPO_ROOT / "tmp"
    path.mkdir(exist_ok=True)
    return path


def _stop(proc: subprocess.Popen) -> None:
    """Cleanup: never leave a runner behind when a test fails before it finishes."""
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=30)


def _pid_gone(pid: int, wait: float = 10.0) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and ") Z " in stat.read_text():
            return True  # a zombie awaiting its reaper: dead
        time.sleep(0.05)
    return False


class FakeCheckout(unittest.TestCase):
    """A disposable checkout: data-client's ci.yml jobs, a scheduled workflow, and three unittest suites."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="run-tests-", dir=_repo_tmp())
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "checkout"
        self.logs = base / "logs"
        self.state = base / "state"
        self.state.mkdir()
        self.workflows = self.root / ".github" / "workflows"
        self.workflows.mkdir(parents=True)
        for name, text in (("ci.yml", CI_YML), ("scheduled-tests.yml", SCHEDULED_YML), ("tooling.yml", TOOLING_YML), ("security-scan.yml", SCHEDULE_ONLY_YML)):
            (self.workflows / name).write_text(text)
        for rel, text in FILES.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text))

    def env(self, **overrides: str):
        env = _child_env(FAKE_STATE=str(self.state), **overrides)
        for key in FAKE_KEYS:
            if key not in overrides:
                env.pop(key, None)
        return env

    def run_tests(self, *args: str, timeout: float = 120, **env: str) -> subprocess.CompletedProcess:
        argv = [sys.executable, str(RUNNER), "--checkout", str(self.root), "--log-dir", str(self.logs), "--color", "never", *args]
        return subprocess.run(argv, cwd=self.root, env=self.env(**env), capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 - this repo's runner, fixed argv

    def run_dir(self) -> Path:
        runs = sorted(path for path in self.logs.iterdir() if path.is_dir() and path.name != "latest")
        self.assertTrue(runs, "no run directory was written")
        return runs[-1]

    def summary(self) -> dict:
        return json.loads((self.run_dir() / "summary.json").read_text())

    def jobs(self) -> dict[str, dict]:
        return {job["id"]: job for job in self.summary()["jobs"]}

    def read_state(self, name: str) -> dict:
        return json.loads((self.state / name).read_text())

    def edit_ci(self, old: str, new: str) -> None:
        path = self.workflows / "ci.yml"
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))

    def assert_checkout_clean(self) -> None:
        for name in ("reports", "coverage.json", ".coverage", "htmlcov"):
            self.assertFalse((self.root / name).exists(), f"{name} landed in the checkout")


# ---------------------------------------------------------------------------------------------------------------
# The real workflows
# ---------------------------------------------------------------------------------------------------------------


class RealWorkflowsTest(unittest.TestCase):
    """The runner reads this checkout's workflows, and reads them the way CI runs them."""

    lanes: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.lanes = RT.discover(REPO_ROOT)

    def test_every_job_that_runs_pytest_is_a_lane_and_only_those(self) -> None:
        import yaml

        expected = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            for job_id, job in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {}).items():
                runs = [str(step.get("run") or "") for step in job.get("steps") or [] if not str(step.get("name") or "").startswith("Report")]
                if any(re.search(r"^\s*(?:pytest|python3? -m pytest)\b", run, re.M) for run in runs):
                    expected.append(job_id)
        self.assertEqual(expected, ["unit-tests", "integration-tests", "generator-parity", "security", "slow-or-integration-tests"])
        self.assertEqual([name for name in self.lanes if name != "unittest"], expected)

    def test_all_is_the_lanes_the_test_extra_can_run(self) -> None:
        self.assertEqual(RT.group_lanes("all", self.lanes), ["unit-tests", "integration-tests"])
        self.assertEqual(RT.group_lanes("optional", self.lanes), ["generator-parity", "security", "slow-or-integration-tests"])
        for name in ("generator-parity", "security"):
            self.assertEqual(self.lanes[name].why, RT.OPT_IN[name])
        self.assertIn("never on a pull request", self.lanes["slow-or-integration-tests"].why)

    def test_unit_tests_runs_pytest_then_the_floor_then_the_per_file_gate(self) -> None:
        import yaml

        lane = self.lanes["unit-tests"]
        steps = lane.jobs[0].steps
        self.assertEqual([step.label for step in steps], ["pytest", "coverage-floor", "coverage-gate"])
        floor = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))["env"]["COVERAGE_FAIL_UNDER"]
        self.assertIn(f"--fail-under={floor}", steps[1].argv, "the floor did not read COVERAGE_FAIL_UNDER from ci.yml's env")
        self.assertTrue(any(hint.startswith("juniper-ci-tools") for hint in steps[2].hints), steps[2].hints)
        self.assertEqual(lane.package_dir, REPO_ROOT / "tests")
        self.assertEqual(lane.pythonpath, [REPO_ROOT])
        self.assertEqual(set(lane.packages), {"juniper_data_client"})
        self.assertTrue(any(arg.startswith("--junitxml=") for arg in steps[0].argv), "CI's junit report moved or vanished: the run directory receives it")

    def test_the_integration_lanes_accept_exit_5_as_their_workflows_do(self) -> None:
        for name in ("integration-tests", "slow-or-integration-tests"):
            with self.subTest(lane=name):
                (step,) = self.lanes[name].jobs[0].steps
                self.assertEqual(step.label, "pytest")
                self.assertEqual(step.tolerated, (5,))
                self.assertIn("integration", step.argv)

    def test_the_opt_in_lanes_set_the_variable_that_makes_a_skip_a_failure(self) -> None:
        (parity,) = self.lanes["generator-parity"].jobs[0].steps
        self.assertEqual(parity.env, {"JUNIPER_DATA_CLIENT_REQUIRE_LIVE_REGISTRY": "1"})
        self.assertIn("tests/test_generator_parity.py", parity.argv)
        self.assertIn("juniper-data", {RT.requirement_name(req) for req in self.lanes["generator-parity"].requirements})
        (security,) = self.lanes["security"].jobs[0].steps
        self.assertEqual(security.env, {"GITLEAKS_PARITY_REQUIRED": "1"})
        self.assertIn("tests/test_gitleaks_pr_scan.py", security.argv)
        self.assertTrue(self.lanes["security"].left_out, "the scans were not left out")
        self.assertEqual(self.lanes["security"].pythonpath, [REPO_ROOT], "pip-audit's editable install was not read")

    def test_the_unittest_lane_holds_what_workflows_run_with_unittest(self) -> None:
        found = sorted({suite for path in WORKFLOWS.glob("*.yml") for suite in re.findall(r"^\s*python3? -m unittest(?: -\w+)* (\S+\.py)", path.read_text(encoding="utf-8"), re.M)})
        self.assertEqual(sorted(job.id for job in self.lanes["unittest"].jobs), found)

    def test_list_names_every_lane_and_runs_nothing(self) -> None:
        proc = subprocess.run([sys.executable, str(RUNNER), "--list", "all", "optional", "--color", "never"], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=120)  # nosec B603 - this repo's runner, fixed argv
        self.assertEqual(proc.returncode, 0, proc.stderr)
        listed = [line.split()[0] for line in proc.stdout.splitlines() if line and not line.startswith((" ", "groups:"))]
        self.assertEqual(listed, [name for name in self.lanes if name != "unittest"])

    def test_dry_run_writes_nothing_and_routes_the_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="run-tests-", dir=_repo_tmp()) as tmp:
            logs = Path(tmp) / "logs"
            proc = subprocess.run([sys.executable, str(RUNNER), "--dry-run", "--log-dir", str(logs), "--color", "never", "all", "optional"], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=300)  # nosec B603 - this repo's runner, fixed argv
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Plan", proc.stdout)
            self.assertIn(f"--junitxml={logs}", proc.stdout)
            self.assertIn(f"--cov-report=json:{logs}", proc.stdout)
            self.assertNotIn("reports/", proc.stdout.split("Plan", 1)[1], "a report path still points into the checkout")
            self.assertFalse(logs.exists(), "a dry run created the log directory")


class DocsTest(unittest.TestCase):
    """docs/REFERENCE.md documents the runner: every option --help lists, and every lane; AGENTS.md points at it."""

    @classmethod
    def setUpClass(cls) -> None:
        text = REFERENCE.read_text(encoding="utf-8")
        start = text.find(SECTION)
        end = text.find("\n## ", start + 1)
        cls.section = "" if start == -1 else text[start : end if end != -1 else len(text)]

    def test_the_section_exists(self) -> None:
        self.assertTrue(self.section, f"docs/REFERENCE.md has no {SECTION!r} section")

    def test_every_option_is_documented(self) -> None:
        options = sorted({option for action in RT.build_parser()._actions for option in action.option_strings if option not in ("-h", "--help")})
        missing = [option for option in options if f"`{option}" not in self.section and f"{option} " not in self.section and f"{option}`" not in self.section]
        self.assertEqual(missing, [], "options --help lists that the reference section does not document")

    def test_every_lane_and_group_is_documented_in_the_reference_and_the_help(self) -> None:
        lanes = RT.discover(REPO_ROOT)
        for name in [*(name for name in lanes if name != "unittest"), *RT.lane_aliases(lanes, REPO_ROOT), *RT.GROUPS, "--changed", "--failed"]:
            with self.subTest(name=name):
                self.assertIn(f"`{name}`", self.section)
        for name in lanes:
            with self.subTest(help=name):
                self.assertIn(name, RT._EPILOG)

    def test_agents_md_points_at_the_runner(self) -> None:
        self.assertIn("util/run_tests.py", AGENTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------------------


class ParsingTest(unittest.TestCase):
    def test_comments_are_stripped_as_bash_reads_them(self) -> None:
        self.assertEqual(RT.strip_comment("pip install wheel  # a comment"), "pip install wheel")
        self.assertEqual(RT.strip_comment('echo "a # b" c'), 'echo "a # b" c')
        self.assertEqual(RT.strip_comment("echo a#b"), "echo a#b")
        self.assertEqual(RT.split_command("python3 -m unittest -v tests/test_x.py  # tests/test_ghost.py", "t"), ["python3", "-m", "unittest", "-v", "tests/test_x.py"])

    def test_continuations_join_and_comment_lines_drop(self) -> None:
        script = "# a comment\npython -m pytest \\\n  --cov=x \\\n  -q\n\necho done\n"
        self.assertEqual(RT.logical_lines(script), ["python -m pytest --cov=x -q", "echo done"])

    def test_commands_that_need_a_shell_are_named(self) -> None:
        for line in ("cd tests && python3 -m unittest t.py", "pytest | tee log", "python $FILE", "export A=1", "FOO=1 pytest", "cmd 2>&1"):
            with self.subTest(line=line):
                self.assertIsNotNone(RT.shell_need(RT.split_command(line, "t")))
        self.assertIsNone(RT.shell_need(RT.split_command('juniper-coverage-gap-map --coverage-json coverage.json --omit "*/__main__.py"', "t")))

    def test_expand_substitutes_what_the_workflow_sets(self) -> None:
        env = {"FLOOR": "80", "SPACED": "a b", "CI_ONLY": None}
        self.assertEqual(RT.expand("python -m coverage report --fail-under=${FLOOR}", env), "python -m coverage report --fail-under=80")
        self.assertEqual(RT.expand("x --fail-under=$FLOOR y", env), "x --fail-under=80 y")
        self.assertEqual(RT.expand("x --fail-under=${{ env.FLOOR }}", env), "x --fail-under=80")
        self.assertEqual(RT.expand('x "$FLOOR"', env), 'x "80"')
        for line in ("echo '$FLOOR'", "x \\$FLOOR", "x $SPACED", "x $CI_ONLY", "x ${{ env.CI_ONLY }}", "x $RUNNER_TEMP", "x ${{ matrix.os }}", "rc=$?"):
            with self.subTest(line=line):
                self.assertEqual(RT.expand(line, env), line, "expanded what bash would not, or what only CI knows")

    def test_env_blocks_layer_and_hide_what_only_ci_knows(self) -> None:
        block = RT.env_block({"A": "1", "B": "${{ env.A }}-x", "C": "${{ secrets.T }}", "D": True, "E": 80, "F": "${{ env.UNSET }}"}, {"A": "0"})
        self.assertEqual(block, {"A": "1", "B": "0-x", "C": None, "D": "true", "E": "80", "F": ""})
        self.assertEqual(RT.env_block({"A": "2"}, {"A": "1", "Z": "z"}), {"A": "2", "Z": "z"})
        self.assertEqual(RT.env_block({"B": "${{ env.C }}"}, {"C": None}), {"C": None, "B": None})
        self.assertEqual(RT.env_block(None, {"A": "1"}), {"A": "1"})

    def test_pip_install_arguments_and_fallbacks(self) -> None:
        args = RT.pip_install_args(RT.split_command('python -m pip install -U "pip>=26" -r req.txt -e ".[test]" --index-url https://example.invalid/simple wheel', "t"))
        self.assertEqual(RT.parse_pip_install(args), (["pip>=26", "wheel"], [".[test]"]))
        self.assertIsNone(RT.pip_install_args(["echo", "pip", "install"]))
        self.assertEqual(RT.install_parts(RT.split_command('pip install -e ".[test]" || pip install -e ".[dev]" || pip install -e .', "t")), [["pip", "install", "-e", ".[test]"]])
        self.assertEqual(len(RT.install_parts(RT.split_command('pip install -U pip && pip install -e ".[test]"', "t"))), 2)

    def test_local_paths_are_not_requirements(self) -> None:
        for spec in (".", "./sub/package[test,extra]", "../x", "sub/", "sub/dir[test]"):
            with self.subTest(spec=spec):
                self.assertTrue(RT.is_local_path(spec))
        for spec in ("juniper-ci-tools>=0.9.0,<0.10.0", "pyyaml", "pkg @ https://example.invalid/pkg.whl", "https://example.invalid/a/b.whl"):
            with self.subTest(spec=spec):
                self.assertFalse(RT.is_local_path(spec))

    def test_editable_targets_and_extras(self) -> None:
        base = Path("/checkout")
        self.assertEqual(RT.editable_target(".[test]", base), (base, ["test"]))
        self.assertEqual(RT.editable_target("./sub[test,bench]", base), (base / "sub", ["test", "bench"]))

    def test_the_rc_capture_rule(self) -> None:
        tail = ["rc=$?", 'if [ "$rc" = "5" ]; then', "exit 0", "fi", 'exit "$rc"']
        self.assertEqual(RT.exit_rule(["python -m pytest -m integration", *tail], "t"), (["python -m pytest -m integration"], (5,)))
        self.assertEqual(RT.exit_rule(["pytest", "rc=$?", 'if [ "$rc" -eq 5 ]; then', "exit 0", "fi", "if [[ $rc == 2 ]]; then", "exit 0", "fi", "exit $rc"], "t"), (["pytest"], (5, 2)))
        self.assertEqual(RT.exit_rule(["python -m pytest"], "t"), (["python -m pytest"], ()))
        for lines in (["pytest", "rc=$?", "exit $rc"], ["pytest", "status=$?", 'if [ "$rc" = "5" ]; then', "exit 0", "fi", 'exit "$rc"'], ["pytest", *tail, "echo after"], ["pytest || rc=$?", 'exit "$rc"']):
            with self.subTest(lines=lines), self.assertRaises(RT.UsageError):
                RT.exit_rule(lines, "t")

    def test_lane_commands(self) -> None:
        for line in ("python -m pytest tests/", "pytest -q", "python3 -m coverage report --fail-under=80", "coverage report", "juniper-coverage-gap-map --coverage-json c.json --enforce"):
            with self.subTest(line=line):
                self.assertTrue(RT._is_lane_command(RT.split_command(line, "t")))
        for line in ("python -m coverage run x.py", "bandit -r x", "python -c 'import pytest'", "python3 -m unittest t.py"):
            with self.subTest(line=line):
                self.assertFalse(RT._is_lane_command(RT.split_command(line, "t")))

    def test_report_paths_move_into_the_run_directory(self) -> None:
        run = Path("/run/unit-tests")
        self.assertEqual(RT._report_path("xml:reports/coverage.xml", run), "xml:/run/unit-tests/coverage.xml")
        self.assertEqual(RT._report_path("html:reports/htmlcov", run), "html:/run/unit-tests/htmlcov")
        for spec in ("term-missing", "term-missing:skip-covered", "term:skip-covered", "json"):
            with self.subTest(spec=spec):
                self.assertEqual(RT._report_path(spec, run), spec)

    def test_events_and_steps_for_other_events(self) -> None:
        self.assertEqual(RT.workflow_events({True: {"pull_request": None, "push": None}}), {"pull_request", "push"})
        self.assertEqual(RT.workflow_events({"on": "push"}), {"push"})
        self.assertEqual(RT.workflow_events({True: ["schedule"]}), {"schedule"})
        self.assertEqual(RT.other_event_only({"if": "github.event_name == 'repository_dispatch'"}), "repository_dispatch")
        for condition in ("github.event_name == 'pull_request' || github.event_name == 'merge_group'", "github.event_name != 'repository_dispatch'", "always() && matrix.experimental == true", "github.ref == 'refs/heads/main' && github.event_name == 'push'", ""):
            with self.subTest(condition=condition):
                self.assertEqual(RT.other_event_only({"if": condition}), "")

    def test_the_changes_job_path_filter(self) -> None:
        job = {"steps": [{"run": "git diff --name-only x | grep -qE '^(juniper_data_client/|\\.github/workflows/ci\\.yml$)'"}]}
        pattern = RT.changes_filter(job)
        self.assertTrue(re.search(pattern, "juniper_data_client/client.py"))
        self.assertFalse(re.search(pattern, "docs/REFERENCE.md"))
        self.assertEqual(RT.changes_filter({"steps": [{"run": "echo no filter"}]}), "")
        self.assertEqual(RT.changes_filter(None), "")

    def test_result_lines_and_floors_are_read(self) -> None:
        counts, failed = RT.unittest_counts("FAIL: test_b (tests.test_x.T.test_b)\nERROR: setUpClass (tests.test_x.U)\nRan 12 tests in 0.5s\n\nFAILED (failures=1, errors=1, skipped=3)\n")
        self.assertEqual(counts, {"ran": 12, "failures": 1, "errors": 1, "skipped": 3})
        self.assertEqual(failed, ["tests.test_x.T.test_b", "tests.test_x.U.setUpClass"])
        self.assertEqual(RT.describe_counts("unittest", counts), "1 failed, 1 errors of 12, 3 skipped")
        counts, failed = RT.pytest_counts("FAILED tests/test_a.py::test_x - boom\n===== 1 failed, 230 passed, 2 skipped in 5.12s =====\n")
        self.assertEqual(counts, {"failed": 1, "passed": 230, "skipped": 2})
        self.assertEqual(failed, ["tests/test_a.py::test_x"])
        _, failed = RT.pytest_counts("SUBFAILED(name='a b', new='x) y') tests/test_a.py::T::test_y - boom\nSUBFAILED tests/test_a.py::T::test_z\nERROR tests/test_b.py\n")
        self.assertEqual(failed, ["tests/test_a.py::T::test_y", "tests/test_a.py::T::test_z", "tests/test_b.py"], "pytest 9's subTest failures are failures too")
        self.assertEqual(RT._COV_REPORT_FLOOR.search("Coverage failure: total of 79.12 is less than fail-under=80.00").group("total", "floor"), ("79.12", "80.00"))
        self.assertEqual(RT._COV_FLOOR.search("FAIL Required test coverage of 80% not reached. Total coverage: 79.50%").group("total", "floor"), ("79.50", "80"))

    def test_origin_verdict(self) -> None:
        here = REPO_ROOT / "juniper_data_client"
        self.assertEqual(RT.origin_verdict("juniper_data_client", str(here / "__init__.py"), here), "")
        self.assertIn("not from this checkout", RT.origin_verdict("juniper_data_client", "/elsewhere/site-packages/juniper_data_client/__init__.py", here))
        self.assertIn("not importable", RT.origin_verdict("juniper_data_client", None, here))


# ---------------------------------------------------------------------------------------------------------------
# Selection, on the fake checkout
# ---------------------------------------------------------------------------------------------------------------


class SelectionTest(FakeCheckout):
    def pick(self, *tokens: str, cwd: Path | None = None) -> dict[str, list[str]]:
        lanes = RT.discover(self.root)
        picker = RT.Picker(lanes)
        for token in tokens:
            RT.resolve_selector(token, lanes, self.root, cwd or self.root, picker)
        return {job.id: job.targets for job in picker.jobs()}

    def test_the_lanes_are_the_jobs_that_run_pytest(self) -> None:
        lanes = RT.discover(self.root)
        self.assertEqual(list(lanes), ["unittest", "unit-tests", "integration-tests", "generator-parity", "security", "slow-or-integration-tests"])
        unit = lanes["unit-tests"]
        self.assertEqual([step.label for step in unit.jobs[0].steps], ["pytest", "coverage-floor", "coverage-gate"])
        self.assertIn("--fail-under=77", unit.jobs[0].steps[1].argv)
        self.assertEqual(unit.env, {"PYTHON_TEST_VERSION": "3.99", "COVERAGE_FAIL_UNDER": "77", "JOB_LEVEL": "from-3.99"}, "a secret reached the env, or an env: block was not read")
        self.assertEqual(unit.left_out, [], "a prep step was listed as left out")
        self.assertEqual(lanes["generator-parity"].left_out, ["Confirm the server is importable"])
        self.assertEqual(lanes["security"].left_out, ["Run Bandit (SAST)", "Run pip-audit (Dependency Vulnerabilities)"])
        self.assertEqual(lanes["security"].pythonpath, [self.root])
        self.assertEqual(lanes["slow-or-integration-tests"].requirements, [], "an install fallback (|| pip install -e .[dev]) was read as an install")
        self.assertEqual(lanes["generator-parity"].requirements, ["fake-server[api]"])

    def test_the_unittest_lane_reads_every_pull_request_workflow_that_runs_a_suite(self) -> None:
        unittest_lane = RT.discover(self.root)["unittest"]
        self.assertEqual([job.id for job in unittest_lane.jobs], ["tests/test_alpha.py", "tests/test_beta.py", "tests/test_gamma.py"])
        self.assertIn(".github/workflows/tooling.yml > tooling > Run the tooling suites", unittest_lane.jobs[0].origin)
        self.assertEqual(unittest_lane.requirements, ["pyyaml>=6.0.1"], "the hosting job's installs, without pip itself")

    def test_groups_lanes_and_aliases(self) -> None:
        suites = ["tests/test_alpha.py", "tests/test_beta.py", "tests/test_gamma.py"]
        self.assertEqual(list(self.pick("all")), [*suites, "unit-tests", "integration-tests"])
        self.assertEqual(list(self.pick("unittest")), suites)
        self.assertEqual(list(self.pick("optional")), ["generator-parity", "security", "slow-or-integration-tests"])
        for token, lane in (("unit", "unit-tests"), ("integration", "integration-tests"), ("slow-or-integration", "slow-or-integration-tests"), ("tests", "unit-tests"), ("tests/", "unit-tests"), ("security", "security")):
            with self.subTest(token=token):
                self.assertEqual(self.pick(token), {lane: []})

    def test_every_spelling_of_a_unittest_suite(self) -> None:
        for token in ("tests/test_alpha.py", "test_alpha.py", "test_alpha", "alpha", "tests.test_alpha"):
            with self.subTest(token=token):
                self.assertEqual(self.pick(token), {"tests/test_alpha.py": []})
        self.assertEqual(self.pick("test_alpha.py", cwd=self.root / "tests"), {"tests/test_alpha.py": []})

    def test_every_spelling_of_a_test_file_narrows_the_unit_lane(self) -> None:
        target = str(self.root / "tests" / "test_demo.py")
        for token in ("tests/test_demo.py", "test_demo", "demo", "tests.test_demo"):
            with self.subTest(token=token):
                self.assertEqual(self.pick(token), {"unit-tests": [target]})
        self.assertEqual(self.pick("test_demo.py", cwd=self.root / "tests"), {"unit-tests": [target]})
        self.assertEqual(self.pick("'*wid*'".strip("'")), {"unit-tests": [str(self.root / "tests" / "test_widget.py")]})
        self.assertEqual(self.pick("tests.test_demo.TestX.test_y"), {"unit-tests": [target + "::TestX::test_y"]})

    def test_a_node_narrows_and_the_whole_lane_wins(self) -> None:
        node = str(self.root / "tests" / "test_demo.py") + "::test_demo"
        self.assertEqual(self.pick("tests/test_demo.py::test_demo"), {"unit-tests": [node]})
        self.assertEqual(self.pick("tests/test_demo.py::test_demo", "unit"), {"unit-tests": []})
        self.assertEqual(self.pick("tests/test_alpha.py::AlphaTest::test_two"), {"tests/test_alpha.py": ["tests.test_alpha.AlphaTest.test_two"]})

    def test_unknown_selectors_are_usage_errors_with_suggestions(self) -> None:
        with self.assertRaises(RT.UsageError) as caught:
            self.pick("alpah")
        self.assertIn("did you mean alpha", str(caught.exception))
        with self.assertRaises(RT.UsageError) as caught:
            self.pick("demoo")
        self.assertIn("demo", str(caught.exception))
        for token in ("no_such_*", "tests/test_nope.py", "util/check.py", "tests.test_nope"):
            with self.subTest(token=token), self.assertRaises(RT.UsageError):
                self.pick(token)

    def test_an_empty_unittest_lane_is_named(self) -> None:
        (self.workflows / "tooling.yml").unlink()
        self.assertEqual(list(self.pick("all")), ["unit-tests", "integration-tests"])
        with self.assertRaises(RT.UsageError) as caught:
            self.pick("unittest")
        self.assertIn("lane is empty", str(caught.exception))

    def test_an_opt_in_name_no_job_defines_is_a_usage_error(self) -> None:
        self.edit_ci("  security:\n", "  security-renamed:\n")
        with self.assertRaises(RT.UsageError) as caught:
            RT.discover(self.root)
        self.assertIn("OPT_IN names security", str(caught.exception))


# ---------------------------------------------------------------------------------------------------------------
# Running, on the fake checkout
# ---------------------------------------------------------------------------------------------------------------


class RunTest(FakeCheckout):
    def test_everything_passes_and_the_record_is_complete(self) -> None:
        proc = self.run_tests("all", "-j", "3", FAKE_BETA="pass")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        jobs = self.jobs()
        self.assertEqual(sorted(jobs), ["integration-tests", "tests/test_alpha.py", "tests/test_beta.py", "tests/test_gamma.py", "unit-tests"])
        self.assertEqual({job["status"] for job in jobs.values()}, {"PASS"}, proc.stdout)
        self.assertEqual(jobs["tests/test_alpha.py"]["counts"]["ran"], 2)
        self.assertEqual(jobs["unit-tests"]["detail"], "1 passed; coverage gates passed")
        self.assertEqual(jobs["integration-tests"]["detail"], "12 deselected; no tests collected: the workflow accepts pytest's exit 5")
        summary = self.summary()
        self.assertEqual(summary["schema"], 1)
        self.assertEqual(summary["exit_code"], 0)
        run_dir = self.run_dir()
        self.assertEqual(os.readlink(self.logs / "latest"), run_dir.name)
        self.assertTrue((run_dir / "summary.txt").is_file())
        self.assertTrue((run_dir / "unittest" / "test_alpha.log").is_file())
        self.assertIn("PASSED", proc.stdout)
        self.assert_checkout_clean()

    def test_the_unit_lane_routes_every_report_and_both_gates_read_them(self) -> None:
        proc = self.run_tests("unit")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lane_dir = self.run_dir() / "unit-tests"
        pytest = self.read_state("pytest.json")
        for arg in (f"--junitxml={lane_dir / 'junit-unit.xml'}", f"--cov-report=xml:{lane_dir / 'coverage.xml'}", f"--cov-report=json:{lane_dir / 'coverage.json'}", f"--cov-report=html:{lane_dir / 'htmlcov'}", "--cov-report=term-missing:skip-covered", "--cov=fake_client"):
            with self.subTest(arg=arg):
                self.assertIn(arg, pytest["argv"])
        self.assertEqual(pytest["COVERAGE_FILE"], str(lane_dir / ".coverage"))
        self.assertEqual(pytest["PYTHONPATH"], str(self.root))
        self.assertEqual(pytest["fake_client"], str(self.root / "fake_client" / "__init__.py"))
        self.assertEqual(pytest["env"], {"COVERAGE_FAIL_UNDER": "77", "PYTHON_TEST_VERSION": "3.99", "JOB_LEVEL": "from-3.99", "FAKE_CI_ONLY": None, "FAKE_REQUIRE_LIVE": None, "FAKE_PARITY_REQUIRED": None})
        floor = self.read_state("coverage-report.json")
        self.assertEqual(floor["argv"], ["report", "--fail-under=77"])
        self.assertEqual(floor["COVERAGE_FILE"], str(lane_dir / ".coverage"))
        self.assertTrue(floor["data_exists"], "the floor did not read the data file pytest wrote")
        gate = self.read_state("gate.json")
        self.assertTrue(gate["report_exists"])
        self.assertEqual(gate["argv"][gate["argv"].index("--coverage-json") + 1], str(lane_dir / "coverage.json"))
        self.assertEqual([step["label"] for step in self.jobs()["unit-tests"]["steps"]], ["pytest", "coverage-floor", "coverage-gate"])
        self.assert_checkout_clean()

    def test_a_floor_failure_names_the_floor_and_the_per_file_gate_does_not_run(self) -> None:
        proc = self.run_tests("unit", FAKE_COVERAGE_RC="2")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        job = self.jobs()["unit-tests"]
        self.assertEqual(job["status"], "FAIL")
        self.assertIn("coverage 76.50% is under the 77.00% floor", job["detail"])
        self.assertEqual([step["status"] for step in job["steps"]], ["PASS", "FAIL", "NOT-RUN"])
        self.assertFalse((self.state / "gate.json").exists())

    def test_a_pytest_failure_is_reported_and_neither_gate_runs(self) -> None:
        proc = self.run_tests("unit", FAKE_PYTEST_RC="1")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        job = self.jobs()["unit-tests"]
        self.assertEqual(job["failed_tests"], ["tests/test_demo.py::test_demo"])
        self.assertEqual([step["status"] for step in job["steps"]], ["FAIL", "NOT-RUN", "NOT-RUN"])
        self.assertFalse((self.state / "coverage-report.json").exists())

    def test_the_integration_lane_passes_on_exit_5_as_ci_does_and_fails_on_any_other(self) -> None:
        proc = self.run_tests("integration")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("no tests collected", self.jobs()["integration-tests"]["detail"])
        self.assertEqual(self.read_state("pytest.json")["argv"], ["-m", "integration", "tests/", f"--junitxml={self.run_dir() / 'integration-tests' / 'junit-integration.xml'}"])
        self.assertEqual(self.run_tests("integration", FAKE_PYTEST_RC="1").returncode, 1)
        self.assertEqual(self.run_tests("integration", FAKE_PYTEST_RC="3").returncode, 1)
        proc = self.run_tests("slow-or-integration-tests")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("no tests collected", self.jobs()["slow-or-integration-tests"]["detail"])
        self.assert_checkout_clean()

    def test_an_opt_in_lane_runs_its_test_step_alone_with_its_variable(self) -> None:
        proc = self.run_tests("generator-parity")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = self.read_state("pytest.json")
        self.assertEqual(record["argv"], ["tests/test_demo.py", "--verbose", "-rs"])
        self.assertEqual(record["env"]["FAKE_REQUIRE_LIVE"], "1")
        self.assertIn("missing: fake-server[api]", proc.stdout)
        self.assertIn("install: ", proc.stdout)
        self.assertEqual([step["label"] for step in self.jobs()["generator-parity"]["steps"]], ["pytest"], "the import probe ran")
        proc = self.run_tests("security")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.read_state("pytest.json")["env"]["FAKE_PARITY_REQUIRED"], "1")
        self.assertEqual([step["label"] for step in self.jobs()["security"]["steps"]], ["pytest"], "a scan ran")
        self.assert_checkout_clean()

    def test_optional_lanes_run_only_when_named(self) -> None:
        listed = self.run_tests("--list")
        self.assertNotIn("generator-parity", listed.stdout.split("groups:", 1)[0])
        listed = self.run_tests("--list", "optional")
        self.assertIn("optional: runs only when named (needs juniper-data[api]", listed.stdout)
        self.assertIn("optional: runs only when named (scheduled-tests.yml: CI runs it on schedule, workflow_dispatch, never on a pull request)", listed.stdout)
        self.assertIn("left out, running no test: Run Bandit (SAST); Run pip-audit (Dependency Vulnerabilities)", listed.stdout)
        self.assertIn("$ FAKE_PARITY_REQUIRED=1 python -m pytest tests/test_demo.py -v -v", listed.stdout)
        self.assertIn("groups: all = unittest unit-tests integration-tests; optional = generator-parity security slow-or-integration-tests", listed.stdout)

    def test_a_narrowed_run_turns_the_coverage_gates_off(self) -> None:
        proc = self.run_tests("demo")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        argv = self.read_state("pytest.json")["argv"]
        self.assertFalse([arg for arg in argv if arg.startswith("--cov")], argv)
        self.assertNotIn("tests/", argv, "the workflow's own test path ran beside the narrowed target")
        self.assertEqual(argv[-1], str(self.root / "tests" / "test_demo.py"))
        self.assertIn(f"--junitxml={self.run_dir() / 'unit-tests' / 'junit-unit.xml'}", argv)
        self.assertFalse((self.state / "coverage-report.json").exists(), "the floor ran on a partial run")
        self.assertFalse((self.state / "gate.json").exists(), "the per-file gate ran on a partial run")
        self.assertIn("coverage gates off", self.jobs()["unit-tests"]["detail"])
        proc = self.run_tests("unit", "--no-coverage")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse([arg for arg in self.read_state("pytest.json")["argv"] if arg.startswith("--cov")])
        self.assert_checkout_clean()

    def test_a_failure_is_reported_and_every_other_job_still_runs(self) -> None:
        proc = self.run_tests("unittest", "-j", "1")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        jobs = self.jobs()
        self.assertEqual(jobs["tests/test_beta.py"]["status"], "FAIL")
        self.assertEqual(jobs["tests/test_beta.py"]["failed_tests"], ["tests.test_beta.BetaTest.test_fails"])
        self.assertEqual(jobs["tests/test_gamma.py"]["status"], "PASS", "a job after the failure did not run")
        self.assertIn("beta was told to fail", proc.stdout, "the failure's output tail was not printed")
        self.assertIn("--failed", proc.stdout)

    def test_fail_fast_starts_nothing_after_the_first_failure(self) -> None:
        proc = self.run_tests("unittest", "-j", "1", "-x")
        self.assertEqual(proc.returncode, 1)
        jobs = self.jobs()
        self.assertEqual(jobs["tests/test_beta.py"]["status"], "FAIL")
        self.assertEqual(jobs["tests/test_gamma.py"]["status"], "NOT-RUN")

    def test_a_job_that_passes_alone_after_failing_in_parallel_is_a_rerun_pass(self) -> None:
        proc = self.run_tests("unittest", "-j", "2", FAKE_BETA="flaky")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        beta = self.jobs()["tests/test_beta.py"]
        self.assertEqual(beta["status"], "RERUN-PASS")
        self.assertEqual(beta["first_pass"]["status"], "FAIL")
        self.assertTrue(beta["steps"][0]["log"].endswith("test_beta.rerun.log"))
        self.assertIn("Passed only when re-run alone", proc.stdout)
        (self.state / "beta-ran-once").unlink()
        proc = self.run_tests("unittest", "-j", "2", "--no-rerun", FAKE_BETA="flaky")
        self.assertEqual(proc.returncode, 1, "--no-rerun still re-ran the failure")

    def test_keywords_filter_tests_and_a_filter_that_matches_nothing_exits_5(self) -> None:
        proc = self.run_tests("unittest", "-k", "cap", FAKE_BETA="pass")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        jobs = self.jobs()
        self.assertEqual(jobs["tests/test_alpha.py"]["counts"]["ran"], 1)
        self.assertEqual(jobs["tests/test_beta.py"]["status"], "NO-TESTS")
        proc = self.run_tests("unittest", "-k", "no_such_test")
        self.assertEqual(proc.returncode, 5, proc.stdout + proc.stderr)
        proc = self.run_tests("unit", "-k", "nothing", FAKE_PYTEST_RC="5")
        self.assertEqual(proc.returncode, 5, "a -k that matches nothing in a pytest lane is not NO-TESTS")
        self.assertIn("-k", self.read_state("pytest.json")["argv"])

    def test_failed_reruns_the_failures_of_the_last_run(self) -> None:
        self.assertEqual(self.run_tests("unittest", "-j", "1").returncode, 1)
        listed = self.run_tests("--failed", "--list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        stamp = sorted(p.name for p in self.logs.iterdir() if p.name != "latest")[-1]
        self.assertEqual([line.strip() for line in listed.stdout.splitlines() if line.startswith("  tests/")], [f"tests/test_beta.py  (failed in {stamp})"])
        proc = self.run_tests("--failed", FAKE_BETA="pass")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(list(self.jobs()), ["tests/test_beta.py"])
        self.assertEqual(self.run_tests("demo", FAKE_PYTEST_RC="1").returncode, 1)
        listed = self.run_tests("--failed", "--list")
        self.assertIn("only: " + str(self.root / "tests" / "test_demo.py"), listed.stdout, "--failed lost a narrowed lane's target")

    def test_a_timeout_kills_the_whole_process_group(self) -> None:
        # 4 s, not less: on a loaded CI runner the suite must still reach its sleep (and write the pid) first.
        proc = self.run_tests("gamma", "--timeout", "4", FAKE_GAMMA_SLEEP="60", FAKE_GAMMA_CHILD="1")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(self.jobs()["tests/test_gamma.py"]["status"], "TIMEOUT")
        self.assertTrue((self.state / "gamma-child.pid").is_file(), "the suite timed out before it started its child: the test proved nothing")
        self.assertTrue(_pid_gone(int((self.state / "gamma-child.pid").read_text())), "the step's grandchild survived the timeout")

    def test_processes_a_step_leaves_behind_are_killed(self) -> None:
        proc = self.run_tests("gamma", FAKE_GAMMA_CHILD="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        job = self.jobs()["tests/test_gamma.py"]
        self.assertTrue(job["steps"][0]["leftovers"])
        self.assertIn("killed processes it left behind", job["detail"])
        self.assertTrue(_pid_gone(int((self.state / "gamma-child.pid").read_text())))

    def test_sigterm_ends_the_run_like_ctrl_c_and_keeps_the_summary(self) -> None:
        argv = [sys.executable, str(RUNNER), "--checkout", str(self.root), "--log-dir", str(self.logs), "--color", "never", "gamma", "--timeout", "0"]
        runner = subprocess.Popen(argv, cwd=self.root, env=self.env(FAKE_GAMMA_SLEEP="60", FAKE_GAMMA_CHILD="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # nosec B603 - this repo's runner, fixed argv
        self.addCleanup(_stop, runner)
        deadline = time.monotonic() + 30
        while not (self.state / "gamma-sleeping").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue((self.state / "gamma-sleeping").exists(), "the suite never started")
        runner.send_signal(signal.SIGTERM)
        out, err = runner.communicate(timeout=60)
        self.assertEqual(runner.returncode, 130, out + err)
        summary = self.summary()
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["jobs"][0]["status"], "NOT-RUN")
        self.assertTrue(_pid_gone(int((self.state / "gamma-child.pid").read_text())))

    def test_ctrl_c_in_a_parallel_run_kills_every_running_step(self) -> None:
        argv = [sys.executable, str(RUNNER), "--checkout", str(self.root), "--log-dir", str(self.logs), "--color", "never", "unittest", "-j", "2", "--timeout", "0"]
        runner = subprocess.Popen(argv, cwd=self.root, env=self.env(FAKE_BETA="pass", FAKE_GAMMA_SLEEP="60", FAKE_GAMMA_CHILD="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # nosec B603 - this repo's runner, fixed argv
        self.addCleanup(_stop, runner)
        deadline = time.monotonic() + 30
        while not (self.state / "gamma-sleeping").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue((self.state / "gamma-sleeping").exists(), "the suite never started")
        runner.send_signal(signal.SIGINT)
        out, err = runner.communicate(timeout=60)
        self.assertEqual(runner.returncode, 130, out + err)
        self.assertIn("INTERRUPTED", out)
        self.assertEqual(self.jobs()["tests/test_gamma.py"]["status"], "NOT-RUN")
        self.assertTrue(self.summary()["interrupted"])
        self.assertTrue(_pid_gone(int((self.state / "gamma-child.pid").read_text())), "Ctrl-C left the step's grandchild running")

    def test_the_child_environment(self) -> None:
        tmpdir = self.root.parent / "children-tmp"
        tmpdir.mkdir()
        proc = self.run_tests("gamma", TMPDIR=str(tmpdir), CLAUDE_FAKE="1", JUNIPER_FAKE="1", WTC_FAKE="1", DISPLAY=":99", PYTHONPATH="/nonexistent", WAYLAND_DISPLAY="w")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = self.read_state("gamma-env.json")
        self.assertEqual(record["DBUS_SESSION_BUS_ADDRESS"], RT.BUS_GUARD)
        self.assertIsNone(record["DISPLAY"])
        self.assertIsNone(record["WAYLAND_DISPLAY"])
        self.assertIsNone(record["PYTHONPATH"])
        self.assertEqual(record["PYTHONUNBUFFERED"], "1")
        self.assertEqual(record["TMPDIR"], str(tmpdir))
        self.assertEqual(record["GIT_CEILING_DIRECTORIES"].split(os.pathsep)[0], str(tmpdir))
        self.assertEqual(record["stdin"], "", "a child's stdin is not /dev/null")
        self.assertEqual((record["CLAUDE_FAKE"], record["JUNIPER_FAKE"]), ("1", "1"))
        proc = self.run_tests("gamma", "--clean-env", TMPDIR=str(tmpdir), CLAUDE_FAKE="1", JUNIPER_FAKE="1", WTC_FAKE="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = self.read_state("gamma-env.json")
        self.assertEqual((record["CLAUDE_FAKE"], record["JUNIPER_FAKE"], record["WTC_FAKE"]), (None, None, None))

    def test_a_tmpdir_under_the_system_temp_directory_is_replaced(self) -> None:
        system_tmp = "/tmp"  # tmp-path-literal: allow -- the value under test; the runner must refuse it, so nothing is written there
        proc = self.run_tests("gamma", TMPDIR=os.path.join(system_tmp, "run-tests-never-created"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.read_state("gamma-env.json")["TMPDIR"], str(self.root / "tmp"))
        self.assertIn("TMPDIR was under the system temp directory", proc.stdout)
        refused = self.run_tests("gamma", "--tmpdir", os.path.join(system_tmp, "x"))
        self.assertEqual(refused.returncode, 2)
        self.assertIn("system temp directory", refused.stderr)

    def test_a_missing_gate_tool_makes_the_lane_unavailable(self) -> None:
        self.edit_ci('pip install "juniper-ci-tools>=0.9.0,<0.10.0"\n          python gate.py', 'pip install "juniper-no-such-tools>=1"\n          juniper-no-such-tool')
        proc = self.run_tests("unit")
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        job = self.jobs()["unit-tests"]
        self.assertEqual(job["status"], "UNAVAILABLE")
        self.assertIn("juniper-no-such-tool is not installed", job["detail"])
        self.assertIn("juniper-no-such-tools>=1", job["detail"])

    def test_usage_errors_exit_2(self) -> None:
        for args, needle in ((("alpah",), "did you mean alpha"), (("alpha", "unit", "--", "-x"), "one runner"), (("--stream", "--quiet"), "do not go together")):
            with self.subTest(args=args):
                proc = self.run_tests(*args)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(needle, proc.stderr)

    def test_a_test_step_the_runner_cannot_read_is_refused_by_name(self) -> None:
        for old, new, needle in (
            ("            --cov-report=html:reports/htmlcov\n", "            --cov-report=html:reports/htmlcov | tee unit.log\n", "needs a shell"),
            ("          python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}\n", "          python -m coverage report --fail-under=${COVERAGE_FAIL_UNDER}\n          python util/check.py\n", "shares a step with a test command"),
            ('          exit "$rc"\n  generator-parity:', '          echo "$rc"\n  generator-parity:', "captures an exit code"),
            ('          FAKE_REQUIRE_LIVE: "1"\n', "          FAKE_REQUIRE_LIVE: ${{ secrets.LIVE }}\n", "only CI can evaluate"),
            ("--cov=fake_client \\\n", "--cov=${{ matrix.package }} \\\n", "only CI can evaluate"),
            ("--fail-under=${COVERAGE_FAIL_UNDER}", "--fail-under=${NOT_SET_ANYWHERE}", "shell expansion"),
        ):
            with self.subTest(needle=needle, new=new):
                (self.workflows / "ci.yml").write_text(CI_YML)
                self.edit_ci(old, new)
                proc = self.run_tests("--list")
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(needle, proc.stderr)
                self.assertIn("ci.yml > ", proc.stderr, "the error does not name the step")

    def test_a_unittest_step_with_another_command_is_refused_by_name(self) -> None:
        workflow = self.workflows / "tooling.yml"
        workflow.write_text(TOOLING_YML.replace("python3 -m unittest -v tests/test_gamma.py", "python3 util/check.py verify"))
        proc = self.run_tests("--list")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("python3 util/check.py verify", proc.stderr)
        self.assertIn("shares a step with a unittest command", proc.stderr)

    def test_not_a_data_client_checkout_is_a_usage_error(self) -> None:
        pyproject = self.root / "pyproject.toml"
        pyproject.write_text(pyproject.read_text().replace('name = "juniper-data-client"', 'name = "another-project"'))
        proc = self.run_tests("--list")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("is not a juniper-data-client checkout", proc.stderr)

    def test_dry_run_prints_the_commands_and_writes_nothing(self) -> None:
        proc = self.run_tests("--dry-run", "all", "optional")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("-m unittest -v tests/test_alpha.py", proc.stdout)
        self.assertIn("PYTHONPATH=", proc.stdout)
        self.assertIn("FAKE_PARITY_REQUIRED=1", proc.stdout)
        self.assertIn("--fail-under=77", proc.stdout)
        self.assertFalse(self.logs.exists())
        self.assertFalse((self.state / "gamma-env.json").exists(), "a dry run ran a suite")
        self.assertFalse((self.state / "pytest.json").exists(), "a dry run ran pytest")


class ChangedTest(FakeCheckout):
    """--changed: the lanes by what they install and test, the test files and suites by what they name."""

    def git(self, *args: str) -> str:
        config = self.root.parent / "gitconfig"
        config.touch()
        env = _child_env(
            GIT_CONFIG_GLOBAL=str(config),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="run-tests",
            GIT_AUTHOR_EMAIL="run-tests@example.invalid",
            GIT_COMMITTER_NAME="run-tests",
            GIT_COMMITTER_EMAIL="run-tests@example.invalid",
            GIT_CEILING_DIRECTORIES=str(self.root.parent),
        )
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(key, None)
        return subprocess.run(["git", "-C", str(self.root), *args], env=env, capture_output=True, text=True, check=True).stdout  # nosec B603 B607 - fixed git argv in the fake checkout

    def changed(self, base: str) -> str:
        proc = self.run_tests("--changed", "--base", base, "--list")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc.stdout

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def test_changed_selects_what_a_change_touches(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        self.git("init", "-q", "-b", "main")
        self.commit("base")
        (self.root / "util" / "check.py").write_text("print('changed')\n")
        out = self.changed("main")
        self.assertIn("tests/test_gamma.py  (mentions util/check.py)", out)
        tests = self.root / "tests"
        self.assertIn(f"  only: {tests / 'test_gamma.py'}, {tests / 'test_widget.py'}  (mentions util/check.py)", out, "the unit lane was not narrowed to the two files that name util/check.py")
        self.assertNotIn("integration-tests", out.split("groups:", 1)[0], "a file outside the lanes chose a whole lane")
        self.commit("next")
        (self.root / "fake_client" / "extra.py").write_text("X = 1\n")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "notes.md").write_text("# notes\n")
        out = self.changed("HEAD")
        self.assertIn("unit-tests  1 job", out)
        self.assertIn("integration-tests  1 job", out)
        self.assertIn("(changed fake_client/extra.py)", out)
        for optional in ("generator-parity", "security", "slow-or-integration-tests"):
            self.assertNotIn(f"{optional}  1 job", out, "--changed chose an optional lane")
        self.commit("third")
        (self.root / "docs" / "notes.md").write_text("# more notes\n")
        self.assertIn("0 job(s) touch them", self.changed("HEAD"))
        (self.root / "tests" / "helper_mod.py").write_text("VALUE = 2\n")
        out = self.changed("HEAD")
        self.assertIn("tests/test_gamma.py  (mentions tests/helper_mod.py)", out)
        self.assertIn("unit-tests  1 job", out)
        proc = self.run_tests("--changed", "--base", "no-such-ref", "--list")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no merge base", proc.stderr)


if __name__ == "__main__":
    unittest.main()
