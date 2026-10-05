"""
W1.4 downstream census probe: run this checkout's validator on what each consumer actually feeds it.

Project: juniper-data-client
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-10-05
Status: ad-hoc — investigation
Retire when: W1.4 has merged and W3.2 makes juniper-recurrence's app tests exercise the real validator, so CI measures what this probe measures.
Related: juniper-ml notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md (W1.4, findings F-P3 / F-S3, ruling R2)

The grep half of the census (who calls ``validate_npz_contract``, and which tests stub it) is
recorded in the PR body. This is the measured half. It runs THIS checkout's validator against:

1. every juniper-data generator that runs offline, called directly with its default params;
2. juniper-data's own test that feeds the real validator through the API route
   (``test_e2e_artifact_passes_client_contract_validator``, all five synthetic cases);
3. juniper-canopy's fake-vs-real agreement test (``test_the_fake_agrees_with_the_real_helper``);
4. juniper-recurrence's ``synthetic_npz_arrays`` and crossval ``full_npz_arrays`` fixtures, which
   are stubbed away from the validator today and which W3.2 un-stubs.

Read-only toward the other repos: their files are loaded by path (or, for fixtures whose module
imports a package this env lacks, extracted from source by AST) and called in-process, so no pytest
cache or coverage file lands in their trees. Run with PYTHONDONTWRITEBYTECODE=1 so no bytecode does.

Run from the repo root, in an env where juniper-data is importable (e.g. JuniperData):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python util/ad-hoc/2026-10-05_w1_4_downstream_census_probe.py

Exit 0 when every row matched its expectation, 1 when any did not, 2 when the probe could not
measure (the validator resolved outside this checkout, or a consumer file is missing).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Network-bound generators: equities / equities_seq fetch yfinance + SEC, arc_agi / mnist fetch from
# Hugging Face. Reported, not run; their float32 casts are cited by file:line in the PR body.
NETWORK_BOUND = ("equities", "equities_seq", "arc_agi", "mnist")

JD_E2E = "juniper-data/juniper_data/tests/integration/test_e2e_synthetic_regression.py"
CANOPY_ADVISORY = "juniper-canopy/src/tests/unit/test_npz_contract_advisory.py"
REC_CONFTEST = "juniper-recurrence/juniper-recurrence/tests/conftest.py"
REC_CROSSVAL = "juniper-recurrence/juniper-recurrence/tests/test_crossval_routes.py"


class Row(NamedTuple):
    repo: str
    subject: str
    expected: str
    observed: str

    @property
    def ok(self) -> bool:
        return self.observed == self.expected


class ProbeBlind(RuntimeError):
    """The probe cannot measure what it claims to (exit 2), as distinct from a measured failure."""


def _observe(fn: Callable[[], Any]) -> str:
    """Run ``fn`` and render its outcome; every exception is data here, not a crash."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 -- the census records every failure mode rather than filtering them
        return f"{type(exc).__name__}: {exc}"
    return "passed" if result is None else str(result)


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise ProbeBlind(f"consumer file missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeBlind(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_from_source(path: Path, name: str) -> Callable[[], Any]:
    """Compile one numpy-only fixture function out of ``path``, decorators stripped.

    For a test module whose top-level imports need a package this env lacks: only the function
    itself is executed, against numpy alone.
    """
    if not path.is_file():
        raise ProbeBlind(f"consumer file missing: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []
            node.returns = None
            namespace: dict[str, Any] = {"np": np, "__builtins__": builtins}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # nosec B102 -- a repo-local fixture, compiled from source on disk
            return namespace[name]  # type: ignore[no-any-return]
    raise ProbeBlind(f"no top-level function {name!r} in {path}")


def _ecosystem_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for parent in REPO_ROOT.parents:
        if (parent / "juniper-data").is_dir() and (parent / "juniper-canopy").is_dir():
            return parent
    raise ProbeBlind("cannot find the ecosystem root above this checkout; pass --ecosystem-root")


def _dtype_inventory(arrays: dict[str, np.ndarray]) -> str:
    """``stem:dtype`` for the keys W1.4 checks, plus which optional sequence keys are present."""
    seen: dict[str, set[str]] = {}
    for key, value in arrays.items():
        for stem in ("X", "y_reg", "y", "dt", "target_dt", "seq_lengths"):
            if key.startswith(f"{stem}_") and key.rsplit("_", 1)[-1] in ("train", "val", "test"):
                if stem == "y" and key.startswith("y_reg_"):
                    continue
                seen.setdefault(stem, set()).add(str(value.dtype))
                break
    return " ".join(f"{stem}:{'/'.join(sorted(dtypes))}" for stem, dtypes in sorted(seen.items()))


def juniper_data_generator_rows(validate: Callable[..., str]) -> list[Row]:
    from juniper_data.api.routes.generators import GENERATOR_REGISTRY

    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as import_dir:
        os.environ["JUNIPER_DATA_IMPORT_DIR"] = import_dir  # read once by the lru_cached settings
        Path(import_dir, "probe.csv").write_text("a,b,label\n" + "".join(f"{i % 7}.5,{i % 5}.25,{i % 2}\n" for i in range(40)), encoding="utf-8")
        for name in sorted(GENERATOR_REGISTRY):
            entry = GENERATOR_REGISTRY[name]
            if name in NETWORK_BOUND:
                rows.append(Row("juniper-data", f"generator {name} (default params)", "not run: network-bound", "not run: network-bound"))
                continue
            params = entry["params_class"](file_path="probe.csv", label_column="label") if name == "csv_import" else entry["params_class"]()
            arrays = {key: value for key, value in entry["generator"].generate(params).items() if isinstance(value, np.ndarray)}
            expected = "sequence" if arrays["X_train"].ndim == 3 else "tabular"
            rows.append(Row("juniper-data", f"generator {name} (default params) [{_dtype_inventory(arrays)}]", expected, _observe(lambda arrays=arrays: validate(arrays))))
    return rows


def juniper_data_e2e_rows(root: Path) -> list[Row]:
    from fastapi.testclient import TestClient
    from juniper_data.api.app import create_app
    from juniper_data.api.routes import datasets
    from juniper_data.api.settings import Settings
    from juniper_data.storage.memory import InMemoryDatasetStore

    module = _load_module(root / JD_E2E, "w14_probe_jd_e2e")
    test = module.test_e2e_artifact_passes_client_contract_validator
    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as storage:
        for generator, params in module.SYNTHETIC_CASES:
            # The test module's ``client`` fixture, rebuilt: a fresh app over an in-memory store.
            app = create_app(settings=Settings(storage_path=storage))
            datasets.set_store(InMemoryDatasetStore())
            client = TestClient(app)
            rows.append(Row("juniper-data", f"test_e2e_artifact_passes_client_contract_validator[{generator}]", "passed", _observe(lambda c=client, g=generator, p=dict(params): test(c, g, p))))
    return rows


def canopy_rows(root: Path, validate: Callable[..., str]) -> list[Row]:
    module = _load_module(root / CANOPY_ADVISORY, "w14_probe_canopy_advisory")
    if module._real_helper() is not validate:
        raise ProbeBlind("canopy's _real_helper() does not resolve to this checkout's validator")
    return [Row("juniper-canopy", "test_the_fake_agrees_with_the_real_helper", "passed", _observe(module.test_the_fake_agrees_with_the_real_helper))]


def recurrence_rows(root: Path, validate: Callable[..., str]) -> list[Row]:
    synthetic = _fixture_from_source(root / REC_CONFTEST, "synthetic_npz_arrays")()
    full_only = _fixture_from_source(root / REC_CROSSVAL, "full_npz_arrays")()
    return [
        Row("juniper-recurrence", "conftest synthetic_npz_arrays (stubbed today; W3.2 un-stubs)", "sequence", _observe(lambda: validate(synthetic))),
        # Carries only *_full keys, which decision 11 retired: the X_train dispatch (data-client 0.5.0) already refuses it.
        Row("juniper-recurrence", "test_crossval_routes full_npz_arrays (stubbed today; W3.2 un-stubs)", "KeyError: 'X_train'", _observe(lambda: validate(full_only))),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ecosystem-root", help="directory holding juniper-data, juniper-canopy and juniper-recurrence (default: found above this checkout)")
    args = parser.parse_args()

    try:
        import juniper_data_client.contract as contract

        resolved = Path(contract.__file__).resolve()
        print(f"validator under test: {resolved}")
        if not resolved.is_relative_to(REPO_ROOT):
            raise ProbeBlind(f"juniper_data_client resolved outside this checkout ({REPO_ROOT}); set PYTHONPATH=. from the repo root")
        root = _ecosystem_root(args.ecosystem_root)
        print(f"ecosystem root:       {root}\n")
        validate = contract.validate_npz_contract
        rows = juniper_data_generator_rows(validate) + juniper_data_e2e_rows(root) + canopy_rows(root, validate) + recurrence_rows(root, validate)
    except ProbeBlind as exc:
        print(f"PROBE BLIND: {exc}", file=sys.stderr)
        return 2

    for row in rows:
        print(f"[{'ok' if row.ok else 'SURPRISE'}] {row.repo:<18} {row.subject}\n{'':>6}expected {row.expected!r}; observed {row.observed!r}")
    surprises = [row for row in rows if not row.ok]
    print(f"\n{len(rows)} rows, {len(rows) - len(surprises)} as expected, {len(surprises)} surprise(s)")
    return 1 if surprises else 0


if __name__ == "__main__":
    sys.exit(main())
