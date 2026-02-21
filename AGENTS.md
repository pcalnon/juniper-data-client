# juniper-data-client - Agent Development Guide

**Project**: juniper-data-client — HTTP Client for JuniperData
**Version**: 0.3.0
**License**: MIT License
**Author**: Paul Calnon
**Last Updated**: 2026-02-20

---

## Quick Reference

### Essential Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/ -m unit -v

# Run with coverage
pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80

# Type checking (strict mode)
mypy juniper_data_client --strict

# Linting
flake8 juniper_data_client --max-line-length=120
black --check --diff juniper_data_client
isort --check-only --diff juniper_data_client
```

---

## Project Overview

`juniper-data-client` is the official Python client library for the JuniperData dataset generation service. It is a shared dependency used by both **JuniperCascor** (neural network backend) and **JuniperCanopy** (web dashboard).

### Key Files

| File | Purpose |
|------|---------|
| `juniper_data_client/client.py` | `JuniperDataClient` class — all API methods |
| `juniper_data_client/exceptions.py` | `JuniperDataError` exception hierarchy |
| `juniper_data_client/__init__.py` | Public API exports |
| `tests/` | Test suite (pytest) |
| `pyproject.toml` | Package config, dependencies, tool settings |

### Public API

```python
from juniper_data_client import JuniperDataClient

client = JuniperDataClient("http://localhost:8100")
client.health_check()
client.create_spiral_dataset(n_spirals=2, n_points_per_spiral=100, noise=0.1, seed=42)
client.download_artifact_npz(dataset_id)
```

### Consumers

- **JuniperCascor**: `SpiralDataProvider` uses this client for dataset retrieval
- **JuniperCanopy**: `DemoMode` and `CascorIntegration` use this client

### Data Contract

NPZ artifacts with keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)

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
