# juniper-data-client - Agent Development Guide

**Project**: juniper-data-client — HTTP Client for JuniperData
**Repository**: pcalnon/juniper-data-client
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.4.2
**Last Updated**: 2026-09-01

---

## Hazards (resident — do not relocate)

Directives whose **non-application destroys work**. Everything else in this file may be demoted to
`docs/REFERENCE.md` under the memory budget; these may not, because a pointer only helps an agent
that already knows to look. Adding a new hazard here is legitimate — ratchet space out of a
reference section in the same PR rather than waiving the budget gate.

- **The `JuniperDataError` constructor contract must not be broken.** Three constraints, each of
  which fails silently rather than loudly: the extra parameters are **keyword-only**, so making any
  of them positional breaks every downstream caller; **`detail` keeps the server's structure** (the
  message renders a 422 list via `_render_error_detail`, but the list itself stays on the attribute
  — interpolating it was the original defect and produced an unparseable repr); and **`__reduce__`
  must stay**, because `BaseException.__reduce__` rebuilds from `args`, which holds only the
  message, so without it a pickle/copy round-trip returns an exception that **looks right and has
  silently lost the context**. `status_code` is the only thing separating a 400 from a 422. Full
  rationale: § Exception Hierarchy.
- **`/tmp/` is prohibited** as the home for any script that produces, modifies or analyzes
  repository content — it is reaped when sessions, sandboxes or containers end, and the scripts are
  irrecoverable. Scratch *data* there is fine; source files are not. Permanent utilities live in
  `util/`, single-use ones in `util/ad-hoc/`. Full rule: § Script Placement.

## Quick Reference

### Essential Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run all tests via script
bash util/run_all_tests.bash

# Run unit tests only
pytest tests/ -m unit -v

# Run with coverage
pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80

# Type checking (strict mode)
mypy juniper_data_client --strict

# Linting
flake8 juniper_data_client --max-line-length=512
black --check --diff juniper_data_client
isort --check-only --diff juniper_data_client

# Validate documentation links
python scripts/check_doc_links.py

# Generate dependency docs
bash scripts/generate_dep_docs.sh
```

### Coverage

Reproduce the CI coverage gate locally (full suite):

```bash
make coverage                 # convenience wrapper
bash util/run_coverage.bash   # source of truth (mirrors .github/workflows/ci.yml)
```

Gate: 80% aggregate (override with `COVERAGE_FAIL_UNDER=<n>`). The script runs the full suite by design so the percentage matches CI; for a narrower run use plain `pytest`.

---

## Project Overview

What the package is, its two consumers, the NPZ data contract and the env vars they use. Moved to [`docs/REFERENCE.md` § Project Overview Reference](docs/REFERENCE.md#project-overview-reference) — read it when working on this area.

## Directory Structure

The annotated source tree, with the purpose of every package and key module. Moved to [`docs/REFERENCE.md` § Directory Structure Reference](docs/REFERENCE.md#directory-structure-reference) — read it when working on this area.

## Script Placement

**Permanent utilities** live in `util/`. **Single-use / temporary / unfinished scripts** go in `util/ad-hoc/` (create on first use). See [`util/ad-hoc/README.md`](util/ad-hoc/README.md) for the per-script header / lifecycle conventions.

`/tmp/` is **prohibited** as the home for any script that produces, modifies, or analyzes repository content. `/tmp/` is reaped when sessions / sandboxes / containers end, and scripts placed there are lost (irrecoverable). `/tmp/` remains fine as a scratch *workspace* for intermediate artifacts the script itself creates and reads — the prohibition is on script *source files*.

This is an ecosystem-wide rule restated in the parent `Juniper/AGENTS.md` "Cross-Project Conventions" section. Motivating incident: irrecoverable loss of `phase4_consolidate.py` and `v2_citation_validate.py` from the juniper-ml requirements-snapshot effort.

---

## Key Files

Per-file reference for the modules a change is most likely to touch. Moved to [`docs/REFERENCE.md` § Key Files Reference](docs/REFERENCE.md#key-files-reference) — read it when working on this area.

## Public API

Every public entry point, its signature, and the exception it raises. Moved to [`docs/REFERENCE.md` § Public API Reference](docs/REFERENCE.md#public-api-reference) — read it when working on this area.

## Exception Hierarchy

```bash
JuniperDataClientError (base)
├── JuniperDataConnectionError   — Connection failures
├── JuniperDataTimeoutError      — Request timeouts
├── JuniperDataNotFoundError     — HTTP 404
├── JuniperDataValidationError   — HTTP 400/422
└── JuniperDataConfigurationError — Invalid client configuration
```

| HTTP Status | Exception |
|-------------|-----------|
| 400, 422 | `JuniperDataValidationError` |
| 404 | `JuniperDataNotFoundError` |
| Connection failure | `JuniperDataConnectionError` |
| Timeout | `JuniperDataTimeoutError` |

### Exception context (do not remove)

Every exception in the hierarchy carries four attributes, set by the base
`__init__`:

| Attribute | Meaning |
|-----------|---------|
| `message` | The human-readable summary; also what `str(exc)` returns. |
| `status_code` | HTTP status of the originating response, or `None` when the error was raised without one (configuration, connection, timeout, retry-exhausted). |
| `detail` | The server's `detail` payload **exactly as decoded** — a `str` for most handlers, a `list[dict]` for FastAPI's 422. Never stringified. |
| `response` | The originating `requests.Response`, when there was one. |

`status_code` is the **only** thing separating a 400 from a 422 — both raise
`JuniperDataValidationError`. Before it existed, telling them apart meant
substring-matching the message.

Three constraints a refactor must not break:

- **The extra parameters are keyword-only**, so existing single-positional-message
  call sites keep working. Making any of them positional is a breaking change for
  every downstream caller.
- **`detail` keeps the server's structure.** The message renders a 422 list as
  `body.seed: Field required` via `client._render_error_detail`, but the list
  itself stays on the attribute. Interpolating it into the message was the whole
  defect — the result was an unparseable Python repr.
- **`__reduce__` must stay.** `BaseException.__reduce__` rebuilds from `args`,
  which holds only the message, so without it a pickle/copy round-trip returns an
  exception that looks right and has silently lost the context. That is what
  flake8-bugbear's `B042` warns about; the `noqa` on `__init__` is paired with
  `__reduce__`, not a dismissal.

`FakeDataClient` populates `status_code` on every error it raises, mirroring the
real service (400 for an unknown generator, 422 for a `ttl_seconds` violation,
404 for a missing resource). It is documented as a drop-in replacement — a double
that raised the right type with `status_code=None` would let a consumer's test
pass against behaviour production does not have.

---

## Testing Utilities

FakeDataClient usage and the synthetic-generator table for consumer tests without a live service. Moved to [`docs/REFERENCE.md` § Testing Utilities Reference](docs/REFERENCE.md#testing-utilities-reference) — read it when working on this area.

## Architecture & Design Patterns

The client's layering, retry/backoff design, and the patterns a new method must follow. Moved to [`docs/REFERENCE.md` § Architecture and Design Patterns Reference](docs/REFERENCE.md#architecture-and-design-patterns-reference) — read it when working on this area.

## Constants

Every exported constant, its default, and the failure each one guards against. Moved to [`docs/REFERENCE.md` § Constants Reference](docs/REFERENCE.md#constants-reference) — read it when working on this area.

## CI/CD

Per-workflow reference for `.github/workflows/`, including the contract each job must not break. Moved to [`docs/REFERENCE.md` § CI/CD Reference](docs/REFERENCE.md#cicd-reference) — read it when working on this area.

## Worktree Procedures (Mandatory — Task Isolation)

> **OPERATING INSTRUCTION**: All feature, bugfix, and task work SHOULD use git worktrees for isolation. Worktrees keep the main working directory on the default branch while task work proceeds in a separate checkout.

### What This Is

Git worktrees allow multiple branches of a repository to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using a standardized naming convention.

The full setup and cleanup procedures are defined in:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** — Creating a worktree for a new task
- **`notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`** — Merging, removing, and pushing after task completion (V2 — fixes CWD-trap bug)

Read the appropriate file when starting or completing a task.

### Worktree Directory Naming

Format: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

Example: `juniper-data-client--feature--add-retry--20260225-1430--73294fc1`

- Slashes in branch names are replaced with `--`
- All worktrees reside in `/home/pcalnon/Development/python/Juniper/worktrees/`

### When to Use Worktrees

| Scenario | Use Worktree? |
| -------- | ------------- |
| Feature development (new feature branch) | **Yes** |
| Bug fix requiring a dedicated branch | **Yes** |
| Quick single-file documentation fix on main | No |
| Exploratory work that may be discarded | **Yes** |
| Hotfix requiring immediate merge | **Yes** |

### Quick Reference

**Setup** (full procedure in `notes/WORKTREE_SETUP_PROCEDURE.md`):

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data-client
git fetch origin && git checkout main && git pull origin main
BRANCH_NAME="feature/my-task"
git branch "$BRANCH_NAME" main
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
WORKTREE_DIR="/home/pcalnon/Development/python/Juniper/worktrees/${REPO_NAME}--${SAFE_BRANCH}--$(date +%Y%m%d-%H%M)--$(git rev-parse --short=8 HEAD)"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
cd "$WORKTREE_DIR"
```

**Cleanup** (full procedure in `notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`):

```bash
# Phase 1: Push current work
cd "$OLD_WORKTREE_DIR" && git push origin "$OLD_BRANCH"
# Phase 2: Create new worktree BEFORE removing old (prevents CWD-trap)
git fetch origin
git worktree add "$NEW_WORKTREE_DIR" -b "$NEW_BRANCH" origin/main
cd "$NEW_WORKTREE_DIR"
# Phase 3: Create PR (do NOT merge directly to main)
gh pr create --base main --head "$OLD_BRANCH" --title "<title>" --body "<body>"
# Phase 4: Cleanup
git worktree remove "$OLD_WORKTREE_DIR"
git branch -d "$OLD_BRANCH"
git worktree prune
```

### Rules

- **Centralized location**: All worktrees go in `/home/pcalnon/Development/python/Juniper/worktrees/`. Never create worktrees inside the repo directory.
- **Clean before you start**: Ensure the main working directory is clean before creating a worktree.
- **Push before you merge**: Always push the working branch to remote before merging (backup).
- **Prune after cleanup**: Run `git worktree prune` after removing a worktree to clean metadata.
- **Do not leave stale worktrees**: Clean up worktrees promptly after merging.

---

## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of thread compaction. This section defines mandatory behavior that overrides Claude Code's default compaction process.

### What This Is

When a long-running conversation approaches context limits, Claude Code normally performs **thread compaction** — summarizing prior context to free token capacity. This introduces information loss. Instead, Claude Code instances working on this project MUST perform a **proactive thread handoff**: transferring a curated, high-signal summary to a fresh thread with full context capacity.

The full handoff protocol is defined in **`notes/THREAD_HANDOFF_PROCEDURE.md`**. Read that file when a handoff is triggered.

### When to Trigger a Handoff

**Automatic trigger (pre-compaction threshold):** Initiate a thread handoff when token utilization reaches **95% to 99%** of the level at which thread compaction would normally be triggered. This means the handoff fires when you are within **1% to 5%** of the compaction threshold, ensuring the handoff completes before compaction would occur.

Concretely:

- If compaction would trigger at N% context utilization, begin handoff at (N − 5)% to (N − 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work, assess whether you are approaching the compaction threshold. If you estimate you are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression notification, treat this as a signal that handoff should have already occurred — immediately initiate one.

**Additional triggers** (from `notes/THREAD_HANDOFF_PROCEDURE.md`):

| Condition                   | Indicator                                                            |
| --------------------------- | -------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files               |
| **Phase boundary**          | A logical phase of work is complete                                  |
| **Degraded recall**         | Re-reading a file already read, or re-asking a resolved question     |
| **Multi-module transition** | Moving between major components                                      |
| **User request**            | User says "hand off", "new thread", or similar                       |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

### How to Execute a Handoff

1. **Checkpoint**: Inventory what was done, what remains, what was discovered, and what files are in play
2. **Compose the handoff goal**: Write a concise, actionable summary (see templates in `notes/THREAD_HANDOFF_PROCEDURE.md`)
3. **Present to user**: Output the handoff goal to the user and recommend starting a new thread with that goal as the initial prompt
4. **Include verification commands**: Always specify how the new thread should verify its starting state (test commands, file checks)
5. **State git status**: Mention branch, staged files, and any uncommitted work

### Rules

- **This is not optional.** Every Claude Code instance on this project must follow these rules.
- **Handoff early, not late.** A handoff at 70% context usage is better than compaction at 95%.
- **Do not duplicate CLAUDE.md content** in the handoff goal — the new thread reads CLAUDE.md automatically.
- **Be specific** in the handoff goal: include file paths, decisions made, and test status.
