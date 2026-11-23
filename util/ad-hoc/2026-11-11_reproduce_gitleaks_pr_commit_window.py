#!/usr/bin/env python3
"""Reproduce gitleaks-action's pull-request commit window, and show ci.yml's replacement step scans the whole range.

Project: juniper-data-client
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-11
Status: ad-hoc — investigation (the reproduction for the port of juniper-data#530; ruling 10 in section 8 of juniper-ml's notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md)
Retire when: gitleaks-action's pull_request path covers a pull request's whole range upstream, or ci.yml stops scanning pull requests with its own step.
Related: .github/workflows/ci.yml ("Run Gitleaks over the pull request"), tests/test_gitleaks_pr_scan.py

gitleaks-action v3.0.0 (e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e) lists a pull request's commits with one
unpaginated ``GET /repos/{owner}/{repo}/pulls/{n}/commits`` (30 per page, oldest first) and scans
``--no-merges --first-parent <page[0]>^..<page[-1]>`` (src/gitleaks.js). No pull request of this repository has
tripped either half (the largest of 257 had 10 commits, and every merge commit in one merged the base branch in), so
there is no run log to point at: this script is the evidence. It builds a THROWAWAY repository with ``git init`` (it
has no remote, so nothing in it can be pushed), plants GitHub-PAT-shaped tokens assembled at runtime (never a
credential), and runs:

    linear-35           35 commits, a token in commit 1 and another in commit 35
      action-window       the action's own argv over the first API page (commits 1-30): finds 1, misses 35
      newest-30-window    the same argv over the newest 30 commits (6-35): finds 35, misses 1
      fixed-step          ci.yml's "Run Gitleaks over the pull request" shell, read from the workflow: finds both
    merge-side-branch   five commits; the token is on a side branch merged into the pull request's branch
      action-window       all five fit on one page, and --first-parent still drops the side-branch commit
      fixed-step          finds it
    clean-35            a token in the BASE commit, then 35 clean commits
      fixed-step          exits 0: the base is outside base..head, and a clean range passes
    update-merges       one commit, then four clean merges of the base branch: this repository's usual shape
      action-window       scans the one commit, as pull request #259's run logged (`1 commits scanned.`)
      fixed-step          logs `Scanning 5 commit(s)`; the engine counts 1, because a clean merge prints no patch
    update-merge-token  the same shape, with a token written into a merge's conflict resolution
      action-window       exits 0: --no-merges drops the merge commit, and neither parent holds the token
      fixed-step          reports the merge commit

The API page is emulated as ``git rev-list --reverse --date-order base..head`` (commit dates are distinct and
ascending), matching the order the endpoint returned for #259. The scratch repository carries whichever of
``.gitleaks.toml`` and ``.gitleaksignore`` this repository's root holds (neither today, so the built-in ruleset).
gitleaks redacts every log and report (``--redact``); the script fails if any token appears in a file it wrote under
--logs. ``--cleanup`` deletes the scratch repository, which holds the planted tokens in its commits.

Usage (with an interpreter that has PyYAML, which the `[test]` extra installs):
    python3 util/ad-hoc/2026-11-11_reproduce_gitleaks_pr_commit_window.py --gitleaks <gitleaks 8.24.3> --work <new empty dir> --logs <dir> [--cleanup]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import string
import subprocess  # nosec B404 - fixed argv: git, gitleaks and bash, in a scratch repository
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
STEP_NAME = "Run Gitleaks over the pull request (Secrets Detection)"
#: gitleaks-action v3.0.0's Scan() argv (src/gitleaks.js), in its order; --report-path and --log-opts are added per run.
ACTION_FLAGS = ("detect", "--redact", "-v", "--exit-code=2", "--report-format=sarif")
ACTION_LOG_LEVEL = "--log-level=debug"
#: GitHub's default per_page for the endpoint, and the action passes none.
PAGE = 30
EPOCH = 1_762_560_000
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SCANNED = re.compile(r"(\d+) commits scanned")
ALPHABET = string.ascii_letters + string.digits


def fabricated_token(step: int) -> str:
    """A GitHub-PAT-SHAPED value from a fixed-step walk over the alphabet: deterministic, never a credential."""
    return "gh" + "p_" + "".join(ALPHABET[(11 + i * step) % len(ALPHABET)] for i in range(36))


def leak_line(token: str) -> str:
    return "GITHUB_" + "TOKEN=" + token + "\n"


class Scratch:
    """git in the scratch repository, isolated from the user's and the system's git config, with dated commits."""

    def __init__(self, repo: Path, home: Path) -> None:
        self.repo = repo
        self.home = home
        self.tick = 0

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(self.home),
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "scratch",
            "GIT_AUTHOR_EMAIL": "scratch@example.invalid",
            "GIT_COMMITTER_NAME": "scratch",
            "GIT_COMMITTER_EMAIL": "scratch@example.invalid",
        }
        env.update(extra)
        return env

    def run_git(self, *args: str, dated: bool = False) -> subprocess.CompletedProcess[str]:
        env = self.env(**self._dated()) if dated else self.env()
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull, *args], cwd=self.repo, env=env, capture_output=True, text=True, check=False)  # nosec B603 B607

    def git(self, *args: str, dated: bool = False, ok: tuple[int, ...] = (0,)) -> str:
        proc = self.run_git(*args, dated=dated)
        if proc.returncode not in ok:
            raise SystemExit(f"git {' '.join(args)} exited {proc.returncode}, expected {ok}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def _dated(self) -> dict[str, str]:
        self.tick += 1
        stamp = f"{EPOCH + 60 * self.tick} +0000"
        return {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}

    def write(self, rel: str, text: str) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, rel: str, text: str, message: str) -> str:
        self.write(rel, text)
        self.git("add", "--", rel)
        self.git("commit", "--no-verify", "-q", "-m", message, dated=True)
        return self.git("rev-parse", "HEAD")

    def merge(self, branch: str, message: str) -> str:
        self.git("merge", "--no-ff", "--no-verify", "-q", "-m", message, branch, dated=True)
        return self.git("rev-parse", "HEAD")

    def branch(self, name: str, start: str) -> None:
        self.git("checkout", "-q", "-B", name, start)

    def api_page(self, base: str, head: str) -> list[str]:
        """The first page the action's unpaginated request returns: the oldest PAGE commits of base..head."""
        return self.git("rev-list", "--reverse", "--date-order", f"{base}..{head}").split()[:PAGE]


def build_linear(sc: Scratch, root: str, name: str, count: int, leaks: dict[int, str]) -> list[str]:
    sc.branch(name, root)
    shas = []
    for i in range(1, count + 1):
        if i in leaks:
            shas.append(sc.commit(f"{name}/leak-in-commit-{i:02d}.txt", leak_line(leaks[i]), f"{name}: commit {i:02d} of {count} (planted token)"))
        else:
            shas.append(sc.commit(f"{name}/filler-{i:02d}.txt", f"{name} filler {i:02d}\n", f"{name}: commit {i:02d} of {count}"))
    return shas


def build_merge(sc: Scratch, root: str, token: str) -> dict[str, str]:
    sc.branch("merge-side-branch", root)
    p1 = sc.commit("merge/p1.txt", "p1\n", "merge: p1")
    sc.branch("merge-side-branch-topic", p1)
    s1 = sc.commit("merge/leak-on-side-branch.txt", leak_line(token), "merge: s1 on the side branch (planted token)")
    sc.git("checkout", "-q", "merge-side-branch")
    p2 = sc.commit("merge/p2.txt", "p2\n", "merge: p2")
    m = sc.merge("merge-side-branch-topic", "merge: merge the side branch")
    p3 = sc.commit("merge/p3.txt", "p3\n", "merge: p3 (head)")
    return {"p1": p1, "s1": s1, "p2": p2, "m": m, "p3": p3}


def build_update_merges(sc: Scratch, root: str, merges: int) -> tuple[str, dict[str, str]]:
    """One commit on the pull request's branch, then ``merges`` clean merges of a base branch that keeps moving."""
    sc.branch("update-base", root)
    sc.branch("update-pr", root)
    names = {"c1": sc.commit("update/c1.txt", "the pull request's one commit\n", "update: c1")}
    base = root
    for k in range(1, merges + 1):
        sc.git("checkout", "-q", "update-base")
        base = sc.commit(f"update-base/b{k}.txt", f"base moves on {k}\n", f"update: the base branch's commit {k}")
        sc.git("checkout", "-q", "update-pr")
        names[f"m{k}"] = sc.merge("update-base", f"update: merge the base branch ({k})")
    return base, names


def build_update_merge_token(sc: Scratch, root: str, token: str) -> tuple[str, dict[str, str]]:
    """The pull request merges its base branch in over a conflict, and the resolution holds a token neither parent has."""
    rel = "conflict/settings.txt"
    sc.branch("conflict-base", root)
    start = sc.commit(rel, "a=1\nb=2\nc=3\n", "conflict: the shared file")
    sc.branch("conflict-pr", start)
    names = {"c1": sc.commit(rel, "a=1\nb=PR\nc=3\n", "conflict: the pull request's side")}
    sc.git("checkout", "-q", "conflict-base")
    base = sc.commit(rel, "a=1\nb=BASE\nc=3\n", "conflict: the base branch's side")
    sc.git("checkout", "-q", "conflict-pr")
    # The merge has to stop on the conflict, or what is committed below is not a conflict resolution.
    sc.git("merge", "--no-ff", "--no-verify", "-q", "-m", "conflict: merge the base branch", "conflict-base", dated=True, ok=(1,))
    sc.write(rel, "a=1\nb=PR\n" + leak_line(token) + "c=3\n")
    sc.git("add", "--", rel)
    sc.git("commit", "--no-verify", "-q", "-m", "conflict: merge the base branch (planted token in the resolution)", dated=True)
    names["m1"] = sc.git("rev-parse", "HEAD")
    names["c2"] = sc.commit("conflict/c2.txt", "c2\n", "conflict: c2 (head)")
    return base, names


def findings(report: Path) -> list[dict[str, str]]:
    if not report.is_file():
        return []
    sarif = json.loads(report.read_text(encoding="utf-8"))
    out = []
    for res in sarif["runs"][0].get("results") or []:
        loc = res["locations"][0]["physicalLocation"]
        out.append({"commit": res["partialFingerprints"]["commitSha"], "file": loc["artifactLocation"]["uri"], "rule": res["ruleId"], "line": loc["region"]["startLine"]})
    return out


def run_action_window(sc: Scratch, gitleaks: Path, logs: Path, label: str, first: str, last: str) -> dict:
    report = logs / f"{label}.sarif.json"
    argv = [str(gitleaks), *ACTION_FLAGS, f"--report-path={report}", ACTION_LOG_LEVEL, f"--log-opts=--no-merges --first-parent {first}^..{last}"]
    proc = subprocess.run(argv, cwd=sc.repo, env=sc.env(), capture_output=True, text=True, check=False)  # nosec B603
    return _record(logs, label, argv, proc, report)


def run_fixed_step(sc: Scratch, gitleaks: Path, logs: Path, label: str, base: str, head: str, pr_commits: int) -> dict:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["security"]["steps"]
    step = next(s for s in steps if s.get("name") == STEP_NAME)
    if "shell" in step:
        raise SystemExit(f"{STEP_NAME!r} names a shell; this rehearsal assumes the default `bash -e`")
    summary = logs / f"{label}.step_summary.txt"
    # GitHub sets RUNNER_TEMP for every step; the scan writes its git filter there.
    runner_temp = logs / f"{label}.runner-temp"
    runner_temp.mkdir(exist_ok=True)
    env = sc.env(PATH=f"{gitleaks.parent}{os.pathsep}{os.environ.get('PATH', os.defpath)}", BASE_SHA=base, HEAD_SHA=head, PR_COMMITS=str(pr_commits), GITHUB_STEP_SUMMARY=str(summary), RUNNER_TEMP=str(runner_temp))
    reports_dir = sc.repo / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    proc = subprocess.run(["bash", "-e", "-c", str(step["run"])], cwd=sc.repo, env=env, capture_output=True, text=True, check=False)  # nosec B603 B607
    report = logs / f"{label}.sarif.json"
    produced = sc.repo / "reports" / "security" / "gitleaks.sarif"
    if produced.is_file():
        shutil.move(str(produced), str(report))
    shutil.rmtree(reports_dir, ignore_errors=True)
    return _record(logs, label, ["bash", "-e", "-c", f"<ci.yml: {STEP_NAME}>"], proc, report)


def _record(logs: Path, label: str, argv: list[str], proc: subprocess.CompletedProcess[str], report: Path) -> dict:
    text = ANSI.sub("", (proc.stdout or "") + (proc.stderr or ""))
    (logs / f"{label}.log").write_text("$ " + " ".join(argv) + "\n" + text + f"\n[exit {proc.returncode}]\n", encoding="utf-8")
    scanned = SCANNED.search(text)
    executed = next((line.split("executing: ", 1)[1] for line in text.splitlines() if "executing: " in line), None)
    return {"label": label, "exit": proc.returncode, "commits_scanned": int(scanned.group(1)) if scanned else None, "git_command": executed, "findings": findings(report), "log": str(logs / f"{label}.log")}


def check(result: dict, names: dict[str, str], exit_code: int, found: set[str], scanned: int | None = None, step_line: str | None = None) -> dict:
    """Compare a run with what is expected of it: its exit code, the commits its findings name, and optionally its counts."""
    commits = {names.get(f["commit"], f["commit"]) for f in result["findings"]}
    result["finding_commits"] = sorted(commits)
    result["expected"] = {"exit": exit_code, "finding_commits": sorted(found), **({"commits_scanned": scanned} if scanned is not None else {}), **({"step_line": step_line} if step_line else {})}
    ok = result["exit"] == exit_code and commits == found
    if scanned is not None:
        ok = ok and result["commits_scanned"] == scanned
    if step_line is not None:
        ok = ok and step_line in Path(result["log"]).read_text(encoding="utf-8")
    result["verdict"] = "AS EXPECTED" if ok else "UNEXPECTED"
    return result


def files_holding_a_token(logs: Path, tokens: list[str]) -> list[str]:
    """Every file this run wrote under --logs that holds a planted token in clear: gitleaks' --redact must leave none."""
    return sorted(str(p.relative_to(logs)) for p in logs.rglob("*") if p.is_file() and any(t in p.read_text(encoding="utf-8", errors="replace") for t in tokens))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gitleaks", type=Path, required=True, help="a gitleaks 8.24.3 binary")
    ap.add_argument("--work", type=Path, required=True, help="a new or empty directory for the scratch repository")
    ap.add_argument("--logs", type=Path, required=True, help="directory for the logs, reports and summary.json")
    ap.add_argument("--cleanup", action="store_true", help="delete the scratch repository afterwards")
    args = ap.parse_args()
    gitleaks, work, logs = args.gitleaks.resolve(), args.work.resolve(), args.logs.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise SystemExit(f"{work} is not empty; give a new directory")
    logs.mkdir(parents=True, exist_ok=True)
    repo = work / "repo"
    repo.mkdir()
    sc = Scratch(repo, work)
    sc.git("init", "-q", "-b", "main")
    sc.write("README.md", "a scratch repository for the pull-request scan reproduction\n")
    tracked = ["README.md"]
    for name in (".gitleaks.toml", ".gitleaksignore"):
        if (REPO / name).is_file():
            shutil.copyfile(REPO / name, repo / name)
            tracked.append(name)
    sc.git("add", "--", *tracked)
    sc.git("commit", "--no-verify", "-q", "-m", "root", dated=True)
    root = sc.git("rev-parse", "HEAD")
    if sc.git("remote"):
        raise SystemExit(f"{repo} has a remote; this script only runs in a repository that cannot be pushed")
    tokens = {"A": fabricated_token(29), "B": fabricated_token(37), "C": fabricated_token(41), "D": fabricated_token(43), "E": fabricated_token(47)}
    names: dict[str, str] = {}
    results = []

    linear = build_linear(sc, root, "linear-35", 35, {1: tokens["A"], 35: tokens["B"]})
    names.update({sha: f"linear-35 commit {i:02d}" for i, sha in enumerate(linear, start=1)})
    page = sc.api_page(root, linear[-1])
    if page != linear[:PAGE]:
        raise SystemExit("the emulated first API page is not commits 1-30 of the linear branch")
    results.append(check(run_action_window(sc, gitleaks, logs, "linear-35.action-window", page[0], page[-1]), names, 2, {"linear-35 commit 01"}, scanned=30))
    results.append(check(run_action_window(sc, gitleaks, logs, "linear-35.newest-30-window", linear[-PAGE], linear[-1]), names, 2, {"linear-35 commit 35"}, scanned=30))
    results.append(check(run_fixed_step(sc, gitleaks, logs, "linear-35.fixed-step", root, linear[-1], 35), names, 1, {"linear-35 commit 01", "linear-35 commit 35"}, scanned=35, step_line="Scanning 35 commit(s)"))

    merge = build_merge(sc, root, tokens["C"])
    names.update({sha: f"merge-side-branch {k}" for k, sha in merge.items()})
    page = sc.api_page(root, merge["p3"])
    r = check(run_action_window(sc, gitleaks, logs, "merge-side-branch.action-window", page[0], page[-1]), names, 0, set(), scanned=3)
    r["api_page"] = [names[s] for s in page]
    results.append(r)
    results.append(check(run_fixed_step(sc, gitleaks, logs, "merge-side-branch.fixed-step", root, merge["p3"], 5), names, 1, {"merge-side-branch s1"}, scanned=4, step_line="Scanning 5 commit(s)"))

    sc.branch("clean-35-base", root)
    pr_base = sc.commit("clean-35/leak-in-the-base.txt", leak_line(tokens["D"]), "clean-35: the BASE commit (planted token, outside the range)")
    names[pr_base] = "clean-35 base"
    clean = build_linear(sc, pr_base, "clean-35", 35, {})
    results.append(check(run_fixed_step(sc, gitleaks, logs, "clean-35.fixed-step", pr_base, clean[-1], 35), names, 0, set(), scanned=35, step_line="Scanning 35 commit(s)"))

    update_base, update = build_update_merges(sc, root, 4)
    names.update({sha: f"update-merges {k}" for k, sha in update.items()})
    page = sc.api_page(update_base, update["m4"])
    r = check(run_action_window(sc, gitleaks, logs, "update-merges.action-window", page[0], page[-1]), names, 0, set(), scanned=1)
    r["api_page"] = [names[s] for s in page]
    results.append(r)
    results.append(check(run_fixed_step(sc, gitleaks, logs, "update-merges.fixed-step", update_base, update["m4"], 5), names, 0, set(), scanned=1, step_line="Scanning 5 commit(s)"))

    conflict_base, conflict = build_update_merge_token(sc, root, tokens["E"])
    names.update({sha: f"update-merge-token {k}" for k, sha in conflict.items()})
    page = sc.api_page(conflict_base, conflict["c2"])
    r = check(run_action_window(sc, gitleaks, logs, "update-merge-token.action-window", page[0], page[-1]), names, 0, set(), scanned=2)
    r["api_page"] = [names[s] for s in page]
    results.append(r)
    results.append(check(run_fixed_step(sc, gitleaks, logs, "update-merge-token.fixed-step", conflict_base, conflict["c2"], 3), names, 1, {"update-merge-token m1"}, scanned=3, step_line="Scanning 3 commit(s)"))

    summary = {"gitleaks": str(gitleaks), "workflow": str(WORKFLOW), "step": STEP_NAME, "results": results}
    if args.cleanup:
        shutil.rmtree(repo)
        summary["scratch_repository_deleted"] = not repo.exists()
    (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    leaked = files_holding_a_token(logs, list(tokens.values()))
    for r in results:
        print(f"{r['verdict']:<11} {r['label']:<34} exit={r['exit']} commits_scanned={r['commits_scanned']} finding_commits={r['finding_commits']}")
    print(f"unredacted token in any file under --logs: {leaked or 'none'}")
    bad = [r["label"] for r in results if r["verdict"] != "AS EXPECTED"]
    return 1 if bad or leaked or (args.cleanup and repo.exists()) else 0


if __name__ == "__main__":
    sys.exit(main())
