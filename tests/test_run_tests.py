#!/usr/bin/env python3
"""Regression tests for util/run_tests.py, the local test runner: juniper-data-client's spec, against the real workflows.

Project:     Juniper
Sub-Project: juniper-data-client
Application: repo tooling tests
Author:      Paul Calnon (authored via Claude Code)
License:     MIT License

util/run_tests.py is a thin spec over the shared engine, ``juniper_ci_tools.run_tests`` (juniper-ci-tools 0.10, which
the ``[test]`` extra installs). The engine's own suite -- selection, running, the exit codes, ``--changed``, on fake
checkouts of this repo's shape and two others -- is juniper-ml's ``juniper-ci-tools/tests/test_run_tests.py``. This
file pins what is juniper-data-client's own, against the real workflows (no test runs):

- every workflow job that runs pytest is a lane, and only those; ``all`` is unit-tests and integration-tests;
  unit-tests runs pytest, then the coverage floor CI's ``env:`` sets, then the per-file gate; both integration lanes
  accept exit 5, as their workflows do; the opt-in lanes carry the variable that turns their skip into a failure;
- the engine is the one CI installs: the ``[test]`` extra asks for ``ENGINE_REQUIREMENT``, so does every later
  ``pip install`` of juniper-ci-tools in a job that installs ``.[test]`` (another range would replace the engine the
  job's tests ran with), and the installed juniper-ci-tools is in that range;
- every option ``--help`` lists, and every lane and its other names, appears in docs/REFERENCE.md's "Running Tests
  Locally" section.

It uses only the engine's public API (``__all__``: ``discover``, the ``Lane`` / ``Job`` / ``Step`` records,
``RepoSpec``, ``build_parser``, and the command line), so a patch release of the engine, which may change anything
else, cannot break it. A workflow change the runner cannot read fails here, in the PR that makes it (CI's unit-tests
job collects this file), instead of on the next person's machine.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import subprocess  # nosec B404 - runs this repo's runner with a fixed argv
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "util" / "run_tests.py"
REFERENCE = REPO_ROOT / "docs" / "REFERENCE.md"
AGENTS = REPO_ROOT / "AGENTS.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SECTION = "## Running Tests Locally"
ENGINE_PIN = re.compile(r"juniper-ci-tools[^\"'\s]*")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


RT = _load_runner()


def _repo_tmp() -> Path:
    """This checkout's gitignored tmp/: on a CI runner the default temp directory is the system one."""
    path = REPO_ROOT / "tmp"
    path.mkdir(exist_ok=True)
    return path


def _run(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    argv = [sys.executable, str(RUNNER), "--color", "never", *args]
    return subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=timeout)  # nosec B603 - this repo's runner, fixed argv


def _listed_lanes(stdout: str) -> list[str]:
    """The lanes a ``--list`` names: its lines that start in the first column, but the groups line."""
    return [line.split()[0] for line in stdout.splitlines() if line and not line.startswith((" ", "groups:"))]


def _requirement_name(text: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", text)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else text


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _aliases(lanes: dict) -> dict[str, str]:
    """The other names a lane answers to, by the rule the engine documents: its name without the spec's suffix, and a
    gating test lane's directory. test_every_alias_names_its_lane holds this to what the command line accepts."""
    suffix = RT.SPEC.strip_suffix
    aliases = {name[: -len(suffix)]: name for name in lanes if suffix and name.endswith(suffix) and len(name) > len(suffix)}
    for name, lane in lanes.items():
        if lane.is_package and not lane.optional and lane.package_dir is not None:
            aliases.setdefault(lane.package_dir.relative_to(REPO_ROOT).as_posix(), name)
    return {alias: name for alias, name in aliases.items() if alias not in lanes and alias not in RT.SPEC.groups}


# ---------------------------------------------------------------------------------------------------------------
# The real workflows
# ---------------------------------------------------------------------------------------------------------------


class RealWorkflowsTest(unittest.TestCase):
    """The spec reads this checkout's workflows, and reads them the way CI runs them."""

    lanes: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.lanes = RT.rt.discover(REPO_ROOT, RT.SPEC)

    def test_every_job_that_runs_pytest_is_a_lane_and_only_those(self) -> None:
        expected = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            for job_id, job in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {}).items():
                runs = [str(step.get("run") or "") for step in job.get("steps") or [] if not str(step.get("name") or "").startswith("Report")]
                if any(re.search(r"^\s*(?:pytest|python3? -m pytest)\b", run, re.M) for run in runs):
                    expected.append(job_id)
        self.assertEqual(expected, ["unit-tests", "integration-tests", "generator-parity", "security", "slow-or-integration-tests"])
        self.assertEqual([name for name in self.lanes if name != "unittest"], expected)

    def test_all_is_the_lanes_the_test_extra_can_run(self) -> None:
        self.assertEqual([name for name, lane in self.lanes.items() if lane.jobs and not lane.optional], ["unit-tests", "integration-tests"])
        self.assertEqual([name for name, lane in self.lanes.items() if lane.optional], ["generator-parity", "security", "slow-or-integration-tests"])
        for name in ("generator-parity", "security"):
            self.assertEqual(self.lanes[name].why, RT.OPT_IN[name])
        self.assertIn("never on a pull request", self.lanes["slow-or-integration-tests"].why)
        proc = _run("--list", "all", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_listed_lanes(proc.stdout), ["unit-tests", "integration-tests"])

    def test_unit_tests_runs_pytest_then_the_floor_then_the_per_file_gate(self) -> None:
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
        self.assertIn("juniper-data", {_requirement_name(req) for req in self.lanes["generator-parity"].requirements})
        (security,) = self.lanes["security"].jobs[0].steps
        self.assertEqual(security.env, {"GITLEAKS_PARITY_REQUIRED": "1"})
        self.assertIn("tests/test_gitleaks_pr_scan.py", security.argv)
        self.assertTrue(self.lanes["security"].left_out, "the scans were not left out")
        self.assertEqual(self.lanes["security"].pythonpath, [REPO_ROOT], "pip-audit's editable install was not read")

    def test_the_unittest_lane_holds_what_workflows_run_with_unittest(self) -> None:
        found = sorted({suite for path in WORKFLOWS.glob("*.yml") for suite in re.findall(r"^\s*python3? -m unittest(?: -\w+)* (\S+\.py)", path.read_text(encoding="utf-8"), re.M)})
        self.assertEqual(sorted(job.id for job in self.lanes["unittest"].jobs), found)

    def test_every_alias_names_its_lane(self) -> None:
        aliases = _aliases(self.lanes)
        self.assertEqual(aliases, {"unit": "unit-tests", "integration": "integration-tests", "slow-or-integration": "slow-or-integration-tests", "tests": "unit-tests"})
        for alias, name in aliases.items():
            with self.subTest(alias=alias):
                proc = _run("--list", alias, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(_listed_lanes(proc.stdout), [name])

    def test_list_names_every_lane_and_runs_nothing(self) -> None:
        proc = _run("--list", "all", "optional", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_listed_lanes(proc.stdout), [name for name in self.lanes if name != "unittest"])

    def test_dry_run_writes_nothing_and_routes_the_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="run-tests-", dir=_repo_tmp()) as tmp:
            logs = Path(tmp) / "logs"
            proc = _run("--dry-run", "--log-dir", str(logs), "all", "optional")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Plan", proc.stdout)
            self.assertIn(f"--junitxml={logs}", proc.stdout)
            self.assertIn(f"--cov-report=json:{logs}", proc.stdout)
            self.assertNotIn("reports/", proc.stdout.split("Plan", 1)[1], "a report path still points into the checkout")
            self.assertFalse(logs.exists(), "a dry run created the log directory")


# ---------------------------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------------------------


class EngineTest(unittest.TestCase):
    """The spec runs the engine CI installs, and every install asks for the same range."""

    def test_the_spec_is_one_for_the_installed_engine(self) -> None:
        self.assertEqual(RT.rt.__name__, "juniper_ci_tools.run_tests")
        self.assertIsInstance(RT.SPEC, RT.rt.RepoSpec)
        self.assertEqual(Path(RT.SPEC.script).resolve(), RUNNER.resolve())

    def test_the_installed_engine_is_in_the_specs_range(self) -> None:
        match = re.fullmatch(r"juniper-ci-tools>=([\d.]+),<([\d.]+)", RT.ENGINE_REQUIREMENT)
        self.assertIsNotNone(match, RT.ENGINE_REQUIREMENT)
        have = importlib.metadata.version("juniper-ci-tools")
        self.assertTrue(_release(match.group(1)) <= _release(have) < _release(match.group(2)), f"juniper-ci-tools {have} is outside {RT.ENGINE_REQUIREMENT}")

    def test_the_test_extra_asks_for_the_engine(self) -> None:
        extra = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["test"]
        self.assertIn(RT.ENGINE_REQUIREMENT, extra, "every job that collects tests/ installs .[test], and this file loads the runner")

    def test_no_job_that_installs_the_test_extra_installs_another_engine(self) -> None:
        checked = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            for job_id, job in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {}).items():
                lines = [line.strip() for step in job.get("steps") or [] for line in str(step.get("run") or "").splitlines()]
                if not any(".[test]" in line and not line.startswith("#") for line in lines):
                    continue
                for line in lines:
                    if line.startswith("#") or not re.search(r"\bpip3? install\b", line) or "juniper-ci-tools" not in line:
                        continue
                    checked.append(f"{path.name} > {job_id}")
                    with self.subTest(job=checked[-1], line=line):
                        self.assertEqual(ENGINE_PIN.findall(line), [RT.ENGINE_REQUIREMENT])
        self.assertEqual(checked, ["ci.yml > unit-tests", "ci.yml > dependency-docs"], "the per-file gate's and the dependency docs' installs moved: check this test still reads them")


# ---------------------------------------------------------------------------------------------------------------
# The documentation
# ---------------------------------------------------------------------------------------------------------------


class DocsTest(unittest.TestCase):
    """docs/REFERENCE.md documents the runner: every option --help lists, every lane and its other names; AGENTS.md
    points at it."""

    section: str

    @classmethod
    def setUpClass(cls) -> None:
        text = REFERENCE.read_text(encoding="utf-8")
        start = text.find(SECTION)
        end = text.find("\n## ", start + 1)
        cls.section = "" if start == -1 else text[start : end if end != -1 else len(text)]

    def test_the_section_exists(self) -> None:
        self.assertTrue(self.section, f"docs/REFERENCE.md has no {SECTION!r} section")

    def test_every_option_is_documented(self) -> None:
        options = sorted({option for action in RT.rt.build_parser(RT.SPEC)._actions for option in action.option_strings if option not in ("-h", "--help")})
        missing = [option for option in options if f"`{option}" not in self.section and f"{option} " not in self.section and f"{option}`" not in self.section]
        self.assertEqual(missing, [], "options --help lists that the reference section does not document")

    def test_every_lane_and_group_is_documented_in_the_reference_and_the_help(self) -> None:
        lanes = RT.rt.discover(REPO_ROOT, RT.SPEC)
        for name in [*(name for name in lanes if name != "unittest"), *_aliases(lanes), *RT.SPEC.groups, "--changed", "--failed"]:
            with self.subTest(name=name):
                self.assertIn(f"`{name}`", self.section)
        for name in lanes:
            with self.subTest(help=name):
                self.assertIn(name, RT._EPILOG)

    def test_the_section_names_the_engine_and_its_install(self) -> None:
        for text in ("juniper_ci_tools.run_tests", 'pip install -e ".[test]"', "juniper-ci-tools/tests/test_run_tests.py"):
            with self.subTest(text=text):
                self.assertIn(text, self.section)

    def test_agents_md_points_at_the_runner(self) -> None:
        self.assertIn("util/run_tests.py", AGENTS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
