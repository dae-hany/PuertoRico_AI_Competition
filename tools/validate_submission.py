"""Check a submission the way the official run will, before you submit it.

    python tools/validate_submission.py my_agent.py:MyAgent
    python tools/validate_submission.py my_agent.py                # one Agent class in the file
    python tools/validate_submission.py my_agent.py:MyAgent --track 2p --games 4

What it checks:

  1. the file loads and the class is an agents.base.Agent with a leaderboard name;
  2. the imports are limited to the standard library, NumPy, PyTorch, and this
     repo (docs/COMPETITION_RULES.md), with no network / process / file access;
  3. for each track, the agent plays real games **inside the official sandbox**
     (tournament/sandbox.py): its own process, the 1 s/move deadline enforced by
     killing the process, no access to the live game. The report shows
     timeouts, illegal moves / exceptions, forfeits, and the per-move
     round-trip time (which includes ~2 ms of sandbox overhead).

Exit status is 0 when nothing failed, 1 otherwise. Warnings do not fail.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPO_PACKAGES = {"puerto_rico", "agents", "tournament", "training"}
ALLOWED_THIRD_PARTY = {"numpy", "torch"}
FORBIDDEN_MODULES = {          # stdlib, but against the rules (process / network / disk)
    "subprocess", "multiprocessing", "socket", "urllib", "http", "ftplib",
    "smtplib", "asyncio", "requests", "ctypes", "shutil", "tempfile",
}
IO_CALLS = {"open", "load", "save", "read_text", "write_text"}

DEFAULT_OPPONENTS = {2: ["ActionValue", "TradeBuilding"],
                     3: ["ShippingRush", "ActionValue"]}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, title, detail=""):
        self.rows.append((status, title, detail))
        line = f"[{status}] {title}"
        if detail:
            line += f": {detail}"
        print(line, flush=True)

    @property
    def failures(self):
        return sum(1 for s, _, _ in self.rows if s == "FAIL")

    @property
    def warnings(self):
        return sum(1 for s, _, _ in self.rows if s == "WARN")


# -- 1. resolve the spec and load the class in-process ------------------------

def _discover_classes(path):
    """Names of module-level classes in ``path`` whose bases mention 'Agent'."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            if any("Agent" in b for b in bases):
                out.append(node.name)
    return out


def resolve_spec(arg):
    """Turn ``file.py`` into ``file.py:Class`` when the file has one agent."""
    if arg.endswith(".py") and os.path.exists(arg):     # before the ':' test: C:\... paths
        names = _discover_classes(arg)
        if len(names) == 1:
            return f"{arg}:{names[0]}"
        if not names:
            sys.exit(f"{arg}: no class subclassing Agent found - pass file.py:ClassName")
        sys.exit(f"{arg}: several agent classes ({', '.join(names)}) - pass file.py:ClassName")
    if ":" in arg:
        return arg
    sys.exit(f"{arg!r}: expected path/to/file.py:ClassName or module:ClassName")


def check_loads(spec, report):
    from tournament.sandbox import load_agent_class
    try:
        cls = load_agent_class(spec, base_dir=os.getcwd())
    except Exception as e:
        report.add("FAIL", "loads", f"{type(e).__name__}: {e}")
        return None
    name = getattr(cls, "name", "")
    if not name or name in ("Agent", "MyAgent", "ExampleAgent"):
        report.add("WARN", "name", f"class {cls.__name__} has name {name!r} - set a "
                   "distinctive `name`, it is what the leaderboard shows")
    else:
        report.add("PASS", "loads", f"class {cls.__name__}, leaderboard name {name!r}")
    try:
        cls()
    except Exception as e:
        report.add("FAIL", "constructs", f"{cls.__name__}() raised {type(e).__name__}: {e}")
        return None
    return cls


# -- 2. static scan of the source ---------------------------------------------

def check_source(cls, spec, report):
    path = getattr(sys.modules.get(cls.__module__), "__file__", None)
    mod_part = spec.rsplit(":", 1)[0]
    if mod_part.endswith(".py") and os.path.exists(mod_part):
        path = mod_part
    if not path or not os.path.exists(path):
        report.add("WARN", "source scan", "could not locate the source file; skipped")
        return
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    stdlib = getattr(sys, "stdlib_module_names", set())
    own = os.path.splitext(os.path.basename(path))[0]

    imported, bad, forbidden, io_calls = set(), set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if fname in IO_CALLS:
                io_calls.add(fname)
    for m in sorted(imported):
        if m in FORBIDDEN_MODULES:
            forbidden.add(m)
        elif m in stdlib or m in ALLOWED_THIRD_PARTY or m in REPO_PACKAGES or m == own:
            continue
        else:
            bad.add(m)
    if bad:
        report.add("FAIL", "imports", f"{', '.join(sorted(bad))} - not available in the "
                   "official run (allowed: standard library, numpy, torch, this repo)")
    if forbidden:
        report.add("FAIL", "imports", f"{', '.join(sorted(forbidden))} - processes, network "
                   "and disk access are not allowed (docs/COMPETITION_RULES.md)")
    if not bad and not forbidden:
        report.add("PASS", "imports", ", ".join(sorted(imported)) or "(none)")
    if io_calls:
        report.add("WARN", "file access", f"calls to {', '.join(sorted(io_calls))} found - "
                   "the official run forbids disk I/O; if you load model weights, "
                   "check with the organizer how to ship them")


# -- 3. play inside the official sandbox --------------------------------------

class _Timed:
    def __init__(self, agent):
        self.agent, self.times = agent, []
        self.name = agent.name

    def on_game_start(self, forward_model=None):
        self.agent.on_game_start(forward_model)

    def act(self, observation, action_mask):
        t0 = time.perf_counter()
        a = self.agent.act(observation, action_mask)
        self.times.append(time.perf_counter() - t0)
        return a


def check_track(spec, cls, num_players, games, seed, time_limit_s, report):
    from agents.registry import BASELINES
    from puerto_rico import ForwardModel, make_env
    from tournament.match import play_game
    from tournament.sandbox import SandboxError, SandboxedAgent

    label = f"{num_players}p track"
    agent = SandboxedAgent(spec, name=cls.name, time_limit_s=time_limit_s)
    try:
        t0 = time.perf_counter()
        try:
            agent.on_game_start(ForwardModel(make_env(seed=seed, num_players=num_players)))
        except SandboxError as e:
            report.add("FAIL", label, f"the agent could not start in the sandbox - {e}")
            return
        startup = time.perf_counter() - t0
        if startup > 10:
            report.add("WARN", f"{label} startup", f"{startup:.1f} s to start a worker "
                       "(the sandbox allows 30 s; a killed worker pays this again)")

        opps = DEFAULT_OPPONENTS[num_players]
        timed = _Timed(agent)
        to = il = 0
        wins = 0.0
        for g in range(games):
            f_seat = g % num_players
            seats, names = [], []
            k = 0
            for s in range(num_players):
                if s == f_seat:
                    seats.append(timed)
                    names.append(cls.name)
                else:
                    opp = opps[(g + k) % len(opps)]
                    seats.append(BASELINES[opp]())
                    names.append(opp)
                    k += 1
            r = play_game(seats, seed=seed + g, time_limit_s=time_limit_s)
            to += r["timeouts"][f_seat]
            il += r["illegal"][f_seat]
            if f_seat in r["winners"]:
                wins += 1.0 / len(r["winners"])
            if agent.forfeited:
                break

        t = np.asarray(timed.times) * 1000 if timed.times else np.zeros(1)
        summary = (f"{games} sandboxed games vs {', '.join(opps)}: won {wins:g}, "
                   f"{to} timeouts, {il} illegal/raising moves; move time mean "
                   f"{t.mean():.1f} ms / p95 {np.percentile(t, 95):.1f} ms / max "
                   f"{t.max():.0f} ms (budget {1000 * time_limit_s:.0f} ms)")
        if agent.forfeited:
            report.add("FAIL", label, "forfeited a game after repeated timeouts - " + summary)
        elif il or agent.violations:
            report.add("FAIL", label, summary + f"; {agent.violations} sandbox violations "
                       "(an exception or a crash inside act) - every one is played at random")
        elif to:
            report.add("WARN", label, summary + " - each timeout is played at random and "
                       "restarts your process, so the agent loses its state")
        elif t.max() > 800 * time_limit_s:
            report.add("WARN", label, summary + " - the slowest move is close to the "
                       "budget; the official machine may be slower than yours")
        else:
            report.add("PASS", label, summary)
    finally:
        agent.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path/to/file.py:ClassName (or just the file)")
    ap.add_argument("--track", choices=("2p", "3p", "both"), default="both")
    ap.add_argument("--games", type=int, default=2, help="sandboxed games per track")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=1.0)
    args = ap.parse_args()

    spec = resolve_spec(args.spec)
    print(f"Validating {spec}\n")
    report = Report()
    cls = check_loads(spec, report)
    if cls is not None:
        check_source(cls, spec, report)
        tracks = {"2p": [2], "3p": [3], "both": [2, 3]}[args.track]
        for n in tracks:
            check_track(spec, cls, n, args.games, args.seed, args.time_limit, report)

    verdict = "READY" if report.failures == 0 else "NOT READY"
    print(f"\nResult: {verdict} ({report.failures} failures, {report.warnings} warnings)")
    if report.failures == 0:
        print("Submit the single .py file per track you are entering; "
              "see docs/SUBMISSION_GUIDE.md section 7.")
    sys.exit(0 if report.failures == 0 else 1)


if __name__ == "__main__":
    main()
