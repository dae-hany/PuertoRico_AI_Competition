"""Check a submission the way the official run will, before you submit it.

    python tools/validate_submission.py my_agent.py:MyAgent
    python tools/validate_submission.py my_agent.py                # one Agent class in the file
    python tools/validate_submission.py my_agent.py:MyAgent --track 2p --games 4

What it checks:

  1. the file loads and the class is an agents.base.Agent with a leaderboard name;
  2. the imports are limited to the standard library, NumPy, PyTorch, and this
     repo (docs/COMPETITION_RULES.md), with no network / process access;
  3. file access is limited to the one read-only weight file the rules allow:
     it sits next to the agent file, the path is built from ``__file__``, it is
     read at import or in the constructor, it is at most WEIGHT_FILE_MAX_MB,
     and nothing is written. The import and the constructor are watched (an
     audit hook records every file they open), and the source is scanned for
     reads and writes that would only happen later;
  4. for each track, the agent plays real games **inside the official sandbox**
     (tournament/sandbox.py): its own process, the 0.1 s per-decision
     deadline (--time-limit) enforced by killing the process, no access to
     the live game. The report shows
     timeouts, illegal moves / exceptions, forfeits, and the per-move
     round-trip time (which includes ~2 ms of sandbox overhead).

Exit status is 0 when nothing failed, 1 otherwise. Warnings do not fail.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tournament.match import TIME_LIMIT_S  # noqa: E402

REPO_PACKAGES = {"puerto_rico", "agents", "tournament", "training"}
ALLOWED_THIRD_PARTY = {"numpy", "torch"}
FORBIDDEN_MODULES = {          # stdlib, but against the rules (process / network / disk)
    "subprocess", "multiprocessing", "socket", "urllib", "http", "ftplib",
    "smtplib", "asyncio", "requests", "ctypes", "shutil", "tempfile",
}
# One read-only weight file next to the agent (docs/COMPETITION_RULES.md).
WEIGHT_FILE_MAX_MB = 50
READ_CALLS = {"open", "load", "read_text", "read_bytes", "fromfile", "loadtxt",
              "genfromtxt"}
WRITE_CALLS = {"save", "savez", "savez_compressed", "savetxt", "tofile", "dump",
               "write_text", "write_bytes"}
CODE_EXTS = {".py", ".pyc", ".pyd", ".so", ".dll", ".dylib"}

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


def absolute_spec(spec):
    """Make a relative ``file.py`` path absolute.

    The sandboxed games start their process in the repository root, so a path
    that is relative to the caller's directory would not be found there.
    """
    path, sep, cls = spec.rpartition(":")
    if sep and path.endswith(".py") and os.path.exists(path):
        return f"{os.path.abspath(path)}:{cls}"
    return spec


def agent_path(spec, cls=None):
    """The agent's source file, or None when it cannot be located."""
    mod_part = spec.rsplit(":", 1)[0]
    if mod_part.endswith(".py") and os.path.exists(mod_part):
        return os.path.abspath(mod_part)
    path = getattr(sys.modules.get(getattr(cls, "__module__", "")), "__file__", None)
    return os.path.abspath(path) if path and os.path.exists(path) else None


# Files opened while the agent is imported and constructed. The audit hook
# cannot be removed once added, so it stays installed and records only while
# ``_OPENS`` is a list.
_OPENS = None
_HOOKED = False


def _audit(event, args):
    if event == "open" and _OPENS is not None:
        _OPENS.append(args)


@contextlib.contextmanager
def watching_opens():
    """Record ``(path, mode, flags)`` for every file opened inside the block."""
    global _OPENS, _HOOKED
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # a .pyc write is not the agent's
    _OPENS = []
    try:
        yield _OPENS
    finally:
        _OPENS = None
        sys.dont_write_bytecode = saved


def _library_dirs():
    """Where Python, the installed packages and this repo live."""
    import site
    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, ROOT}
    dirs.update(os.path.join(ROOT, p) for p in REPO_PACKAGES)
    try:
        dirs.update(site.getsitepackages())
        dirs.add(site.getusersitepackages())
    except Exception:
        pass
    dirs.update(p for p in sys.path if p and os.path.isdir(p))
    if os.environ.get("SystemRoot"):
        dirs.add(os.environ["SystemRoot"])
    return [os.path.normcase(os.path.abspath(d)) for d in dirs]


def _is_under(path, folder):
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:                    # different drives
        return False


def _is_write(mode, flags):
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    return False


def check_files_opened(opens, path, report):
    """Hold what the import and the constructor opened against the weight-file rule."""
    if path is None:
        return
    here = os.path.normcase(os.path.dirname(path))
    own = os.path.normcase(path)
    libs = _library_dirs()
    weights, nested, outside, relative, written = set(), set(), set(), set(), set()
    for args in opens:
        target, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(target, int):
            continue                      # an already-open descriptor
        try:
            target = os.fsdecode(os.fspath(target))
        except TypeError:
            continue
        name = os.path.basename(target)
        if (os.path.splitext(name)[1].lower() in CODE_EXTS
                or "__pycache__" in target.replace("\\", "/").split("/")):
            continue                      # imports, not data
        if not os.path.isabs(target):
            relative.add(target)
            continue
        full = os.path.normcase(os.path.abspath(target))
        if full == own:
            continue
        beside = os.path.dirname(full) == here
        # A library folder can sit inside the agent's folder (a venv next to
        # it), and the agent's folder inside one (the repo's submissions/):
        # the innermost of the two decides.
        library = full.startswith(("/proc/", "/sys/", "/dev/")) or any(
            _is_under(full, d) and (not _is_under(here, d) or not _is_under(full, here))
            for d in libs)
        if library and not beside:
            continue                      # Python, numpy, torch reading their own files
        if _is_write(mode, flags):
            written.add(target)
        elif beside:
            weights.add(os.path.abspath(target))
        elif _is_under(full, here):
            nested.add(target)
        else:
            outside.add(target)

    if written:
        report.add("FAIL", "file access", f"writes {', '.join(sorted(written))} - writing "
                   "files is not allowed")
    if relative:
        report.add("FAIL", "file access", f"opens {', '.join(sorted(relative))} relative to "
                   "the working directory - the official run starts in another folder; "
                   "build the path from __file__ (docs/COMPETITION_RULES.md, The weight file)")
    if nested or outside:
        report.add("FAIL", "file access", f"reads {', '.join(sorted(nested | outside))} - the "
                   "only file you may read is one weight file next to your agent file")
    if len(weights) > 1:
        names = ", ".join(sorted(os.path.basename(w) for w in weights))
        report.add("FAIL", "weight file", f"reads {len(weights)} files next to the agent "
                   f"({names}) - at most one weight file is allowed")
    for w in sorted(weights):
        mb = os.path.getsize(w) / 1e6 if os.path.exists(w) else 0.0
        if mb > WEIGHT_FILE_MAX_MB:
            report.add("FAIL", "weight file", f"{os.path.basename(w)} is {mb:.1f} MB - the cap "
                       f"is {WEIGHT_FILE_MAX_MB} MB")
        elif len(weights) == 1:
            report.add("PASS", "weight file", f"{os.path.basename(w)}, {mb:.2f} MB (cap "
                       f"{WEIGHT_FILE_MAX_MB} MB), read relative to __file__ while loading")


def check_loads(spec, report):
    from tournament.sandbox import load_agent_class
    cls, constructed = None, False
    with watching_opens() as opens:     # import and construct, as the sandbox does
        try:
            cls = load_agent_class(spec, base_dir=os.getcwd())
            cls()
            constructed = True
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
    if cls is None:
        report.add("FAIL", "loads", failure)
        check_files_opened(opens, agent_path(spec), report)
        return None
    name = getattr(cls, "name", "")
    if not name or name in ("Agent", "MyAgent", "ExampleAgent"):
        report.add("WARN", "name", f"class {cls.__name__} has name {name!r} - set a "
                   "distinctive `name`, it is what the leaderboard shows")
    else:
        report.add("PASS", "loads", f"class {cls.__name__}, leaderboard name {name!r}")
    check_files_opened(opens, agent_path(spec, cls), report)
    if not constructed:
        report.add("FAIL", "constructs", f"{cls.__name__}() raised {failure}")
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

    imported, bad, forbidden = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
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
    scan_file_calls(tree, report)


def _call_name(node):
    fn = node.func
    return getattr(fn, "id", None) or getattr(fn, "attr", None)


def _mentions(expr, names):
    """Does ``expr`` use any of ``names`` (as a variable or an attribute)?"""
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id in names:
            return True
        if isinstance(n, ast.Attribute) and n.attr in names:
            return True
    return False


def _derived_names(tree, seeds):
    """Names whose value is built from one of ``seeds``, followed to a fixpoint.

    Covers ``HERE = os.path.dirname(__file__)``, ``W = os.path.join(HERE, ..)``,
    ``self.path = W``, ``with open(W) as f`` and a function called with ``W``
    (its parameters). Scoping is ignored: a scan, not an interpreter.
    """
    names = set(seeds)
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign) and _mentions(node.value, names):
                targets = node.targets
            elif (isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None
                  and _mentions(node.value, names)):
                targets = [node.target]
            elif (isinstance(node, ast.withitem) and node.optional_vars is not None
                  and _mentions(node.context_expr, names)):
                targets = [node.optional_vars]
            elif isinstance(node, ast.Call) and _call_name(node) in defs and any(
                    _mentions(a, names) for a in [*node.args, *(k.value for k in node.keywords)]):
                for fdef in defs[_call_name(node)]:
                    for a in fdef.args.args + fdef.args.kwonlyargs:
                        if a.arg not in ("self", "cls") and a.arg not in names:
                            names.add(a.arg)
                            changed = True
            for t in targets:
                for n in ast.walk(t):
                    key = n.id if isinstance(n, ast.Name) else (
                        n.attr if isinstance(n, ast.Attribute) else None)
                    if key and key not in names:
                        names.add(key)
                        changed = True
    return names


def _write_mode(node):
    """``open(path, "w")`` and friends: a literal mode that writes."""
    mode = node.args[1] if len(node.args) > 1 else next(
        (k.value for k in node.keywords if k.arg == "mode"), None)
    return (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
            and any(c in mode.value for c in "wax+"))


def scan_file_calls(tree, report):
    """Find file reads and writes the load-time watch cannot see.

    The watch in check_loads sees what the import and the constructor open.
    This scan looks at the rest of the source: a write anywhere, a read in
    ``act`` (every move), and a read whose path is not built from
    ``__file__``. Code under ``if __name__ == "__main__":`` never runs in the
    official run (a training script, say) and is skipped. Name-based, so it
    warns rather than fails.
    """
    skip = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and _mentions(node.test, {"__name__"})):
            for child in node.body:
                skip.update(id(n) for n in ast.walk(child))
    in_act = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "act":
            in_act.update(id(n) for n in ast.walk(node))
    here = _derived_names(tree, {"__file__"})
    memory = _derived_names(tree, {"BytesIO", "StringIO"})

    writes, act_reads, loose_reads = set(), set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in skip:
            continue
        fname = _call_name(node)
        if fname not in READ_CALLS | WRITE_CALLS or _mentions(node, memory):
            continue                      # not file access, or an in-memory buffer
        if fname in WRITE_CALLS or (fname == "open" and _write_mode(node)):
            writes.add(fname)
        else:
            if id(node) in in_act:
                act_reads.add(fname)
            if not _mentions(node, here):
                loose_reads.add(fname)
    if writes:
        report.add("WARN", "file access", f"calls to {', '.join(sorted(writes))} found - "
                   "writing files is not allowed in the official run (training code "
                   "belongs under `if __name__ == \"__main__\":` or in another file)")
    if act_reads:
        report.add("WARN", "file access", f"{', '.join(sorted(act_reads))} called in act() - "
                   "read your weight file once, at import or in the constructor, and keep "
                   "it in a module-level cache")
    if loose_reads:
        report.add("WARN", "file access", f"calls to {', '.join(sorted(loose_reads))} with a "
                   "path not built from __file__ - the only file you may read is one weight "
                   "file next to your agent, located from __file__ (docs/COMPETITION_RULES.md, The weight file)")


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
                       "and build the agent (the sandbox allows 30 s; a worker killed "
                       "after a timeout pays this again - read a weight file once, "
                       "into a module-level cache)")

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
                   f"{t.max():.0f} ms (budget {1000 * time_limit_s:.0f} ms); "
                   f"worker start {startup:.1f} s")
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
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                    help=f"seconds per decision (default {TIME_LIMIT_S:g}, the competition rule)")
    args = ap.parse_args()

    spec = absolute_spec(resolve_spec(args.spec))
    print(f"Validating {spec}\n")
    report = Report()
    cls = check_loads(spec, report)
    if cls is not None:
        check_source(cls, spec, report)
        tracks = {"2p": [2], "3p": [3], "both": [2, 3]}[args.track]
        for n in tracks:
            check_track(spec, cls, n, args.games, args.seed, args.time_limit, report)

    verdict = "READY" if report.failures == 0 else "NOT READY"
    if report.failures == 0:
        print("\nWhat to submit: docs/SUBMISSION_GUIDE.md section 7 (public competition)"
              " or docs/COURSE_CNU_2026.md (CNU course).")
    # The verdict stays the last line: the guides tell entrants to read it there.
    print(f"\nResult: {verdict} ({report.failures} failures, {report.warnings} warnings)")
    sys.exit(0 if report.failures == 0 else 1)


if __name__ == "__main__":
    main()
