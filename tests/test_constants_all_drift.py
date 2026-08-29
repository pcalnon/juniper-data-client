"""Drift gate: ``constants.__all__`` must stay complete.

The module's ``__all__`` covered 30 of 141 public constants until it was
completed in one pass (the unfiled finding recorded with the
``APD-DCLIENT-005``/``-006`` register close). Two failure classes this gate
pins:

- a **new public constant** added without an ``__all__`` entry silently
  narrows ``from juniper_data_client.constants import *`` and — on any module
  WITH ``__all__`` — makes CodeQL's ``py/unused-global-variable`` fire on the
  next PR that touches the line, blocking that unrelated merge;
- a **phantom export** (an ``__all__`` entry with no assignment) breaks
  star-imports outright.

Assignments are read from the AST (the source of truth for what the file
declares); ``__all__`` is read from the imported module (the surface consumers
actually see). Grouping and order inside ``__all__`` stay free — the contract
is set equality.
"""

import ast
import pathlib

import juniper_data_client.constants as constants_module


def _public_assignments():
    source = pathlib.Path(constants_module.__file__).read_text(encoding="utf-8")
    names = []
    for node in ast.parse(source).body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        for name in targets:
            if name != "__all__" and not name.startswith("_"):
                names.append(name)
    return names


def test_every_public_constant_is_exported():
    missing = sorted(set(_public_assignments()) - set(constants_module.__all__))
    assert not missing, f"public constants missing from __all__: {missing}"


def test_every_export_is_a_real_assignment():
    phantom = sorted(set(constants_module.__all__) - set(_public_assignments()))
    assert not phantom, f"__all__ names with no module assignment: {phantom}"


def test_all_has_no_duplicates():
    dupes = sorted({n for n in constants_module.__all__ if constants_module.__all__.count(n) > 1})
    assert not dupes, f"duplicate __all__ entries: {dupes}"


def test_every_export_resolves_at_runtime():
    unresolvable = [n for n in constants_module.__all__ if not hasattr(constants_module, n)]
    assert not unresolvable, f"__all__ names that do not resolve on the module: {unresolvable}"
