"""The participant/organizer tools run end to end, the way they are typed."""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = "submissions/example_agent.py:ExampleAgent"


def run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def test_play_records_games_that_replay(tmp_path):
    r = run("tools/play.py", "--agent", EXAMPLE, "--vs", "Random", "--games", "2",
            "--record", "--record-dir", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "overall" in r.stdout and "Random" in r.stdout
    files = sorted(tmp_path.glob("game_*.json"))
    assert len(files) == 2

    from puerto_rico.records import load_record, replays_exactly
    assert all(replays_exactly(load_record(str(f))) for f in files)


def test_play_accepts_baseline_names_and_the_3p_track():
    r = run("tools/play.py", "--agent", "random", "--players", "3",
            "--vs", "Random,Factory", "--games", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "observation length 293" in r.stdout

    r = run("tools/play.py", "--agent", "no_such_file.py:Nope", "--games", "1")
    assert r.returncode != 0 and "cannot load" in r.stderr


def test_validate_passes_the_example_agent():
    r = run("tools/validate_submission.py", "submissions/example_agent.py",
            "--track", "2p", "--games", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    # PASS normally; a loaded CI runner may book a slow round-trip as a WARN,
    # which must not fail the entry either
    assert "[PASS] 2p track" in r.stdout or "[WARN] 2p track" in r.stdout, r.stdout
    assert "Result: READY" in r.stdout


def test_validate_ends_with_the_verdict_and_runs_from_any_directory(tmp_path):
    """The course guide says the last line reads Result: READY, with --time-limit 0.1."""
    shutil.copy(os.path.join(ROOT, "submissions", "example_agent.py"), tmp_path / "my_agent.py")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"),
                        "my_agent.py", "--track", "2p", "--games", "1", "--time-limit", "0.1"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().splitlines()[-1].startswith("Result: READY"), r.stdout


def test_validate_fails_bad_submissions(tmp_path):
    raising = tmp_path / "raising_agent.py"
    raising.write_text(
        "from agents.base import Agent\n"
        "class Raising(Agent):\n"
        "    name = 'Raising'\n"
        "    def act(self, observation, action_mask):\n"
        "        raise RuntimeError('boom')\n", encoding="utf-8")
    r = run("tools/validate_submission.py", f"{raising}:Raising", "--track", "2p",
            "--games", "1")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FAIL] 2p track" in r.stdout

    spawning = tmp_path / "spawning_agent.py"
    spawning.write_text(
        "import subprocess\nimport numpy as np\nfrom agents.base import Agent\n"
        "class Spawning(Agent):\n"
        "    name = 'Spawning'\n"
        "    def act(self, observation, action_mask):\n"
        "        return int(np.where(action_mask > 0.5)[0][0])\n", encoding="utf-8")
    r = run("tools/validate_submission.py", str(spawning), "--track", "2p", "--games", "1")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FAIL] imports" in r.stdout and "subprocess" in r.stdout



# -- the weight-file rule ------------------------------------------------------

WEIGHT_AGENT = """import os
import numpy as np
from agents.base import Agent

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _weights():
    if "w" not in _CACHE:
        _CACHE["w"] = np.load(os.path.join(HERE, "weights.npy"))
    return _CACHE["w"]


class WeightAgent(Agent):
    name = "WeightAgent"

    def __init__(self):
        super().__init__()
        self.w = _weights()

    def act(self, observation, action_mask):
        return int(np.argmax(np.where(action_mask > 0.5, self.w, -np.inf)))


if __name__ == "__main__":          # training code: never runs in the official run
    np.save(os.path.join(HERE, "weights.npy"), np.random.rand(200))
"""


def _validator():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validate_submission", os.path.join(ROOT, "tools", "validate_submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_accepts_one_weight_file_next_to_the_agent(tmp_path):
    """Loaded from __file__, the file is found from any working directory and in
    the sandbox worker, and the training code under __main__ is not held against it."""
    import numpy as np
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "weight_agent.py").write_text(WEIGHT_AGENT, encoding="utf-8")
    np.save(entry / "weights.npy", np.random.default_rng(0).random(200))
    r = run("tools/validate_submission.py", str(entry / "weight_agent.py"),
            "--track", "2p", "--games", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS] weight file: weights.npy" in r.stdout, r.stdout
    assert "[WARN] file access" not in r.stdout and "[FAIL]" not in r.stdout, r.stdout
    assert "worker start" in r.stdout


def test_validate_fails_a_weight_file_opened_from_the_working_directory(tmp_path):
    """It works on the entrant's machine when run from the agent's folder, and
    nowhere else: the official run starts in another folder."""
    import numpy as np
    src = WEIGHT_AGENT.replace('os.path.join(HERE, "weights.npy")', '"weights.npy"', 1)
    (tmp_path / "weight_agent.py").write_text(src, encoding="utf-8")
    np.save(tmp_path / "weights.npy", np.zeros(200))
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"),
                        "weight_agent.py", "--track", "2p", "--games", "1"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FAIL] file access: opens weights.npy relative to the working directory" in r.stdout
    assert "[WARN] file access" in r.stdout and "not built from __file__" in r.stdout


def test_files_opened_while_loading_are_held_to_the_rule(tmp_path):
    agent = tmp_path / "a.py"
    agent.write_text("", encoding="utf-8")
    one, two = tmp_path / "w1.npz", tmp_path / "w2.npz"
    one.write_bytes(b"x" * 2000)
    two.write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    nested = tmp_path / "sub" / "w.pt"
    nested.write_bytes(b"x")

    venv = tmp_path / ".venv" / "site-packages"

    def statuses(opens, cap=None):
        mod = _validator()
        mod._library_dirs = lambda: [os.path.normcase(str(venv))]
        if cap is not None:
            mod.WEIGHT_FILE_MAX_MB = cap
        report = mod.Report()
        mod.check_files_opened(opens, str(agent), report)
        return [(s, t) for s, t, _ in report.rows], report.rows

    ok, _ = statuses([(str(one), "rb", 0), (str(agent), "r", 0),
                      (str(tmp_path / "__pycache__" / "a.cpython.pyc"), "rb", 0),
                      (str(venv / "torch" / "version.json"), "rb", 0)])   # a venv beside it
    assert ok == [("PASS", "weight file")]

    _, rows = statuses([(str(one), "rb", 0), (str(two), "rb", 0)])
    assert any(s == "FAIL" and "at most one" in d for s, _, d in rows), rows

    _, rows = statuses([(str(one), "rb", 0)], cap=0.001)
    assert any(s == "FAIL" and "the cap is" in d for s, _, d in rows), rows

    for opens, why in [([(str(nested), "rb", 0)], "next to your agent"),
                       ([(str(tmp_path / "log.txt"), "w", 0)], "writing files"),
                       ([("w1.npz", "rb", 0)], "relative to the working directory")]:
        _, rows = statuses(opens)
        assert any(s == "FAIL" and why in d for s, _, d in rows), (why, rows)


def test_source_scan_tells_load_time_reads_from_the_rest():
    import ast
    import textwrap

    def scan(src):
        mod = _validator()
        report = mod.Report()
        mod.scan_file_calls(ast.parse(textwrap.dedent(src)), report)
        return " ".join(d for _, _, d in report.rows)

    assert scan("""
        import io, os, torch
        HERE = os.path.dirname(__file__)
        W = os.path.join(HERE, "w.pt")
        def _load(path):
            return torch.load(path)
        NET = _load(W)
        with open(os.path.join(HERE, "meta.json")) as f:
            META = f.read()
        BLOB = torch.load(io.BytesIO(b""))
        buf = io.BytesIO()
        torch.save(NET, buf)
        if __name__ == "__main__":
            torch.save(NET, "w.pt")
        """) == ""
    assert "writing files" in scan("import numpy as np\nnp.save('x.npy', 0)\n")
    assert "writing files" in scan("f = open('log.txt', 'a')\n")
    assert "in act()" in scan("""
        import os, numpy as np
        HERE = os.path.dirname(__file__)
        class A:
            def act(self, observation, action_mask):
                return np.load(os.path.join(HERE, "w.npy"))
        """)
    assert "not built from __file__" in scan("import numpy as np\nW = np.load('w.npy')\n")


def test_ladder_writes_boards_and_replays(tmp_path):
    entries = tmp_path / "entries"
    entries.mkdir()
    shutil.copy(os.path.join(ROOT, "submissions", "example_agent.py"), entries / "team_a.py")
    out = tmp_path / "board"
    r = run("tools/run_ladder.py", "--track", "2p", "--entries", str(entries),
            "--baselines", "Random", "--games-per-seating", "1", "--out", str(out),
            "--date", "2026-01-01")
    assert r.returncode == 0, r.stdout + r.stderr

    board = (out / "2p" / "2026-01-01" / "leaderboard.md").read_text(encoding="utf-8")
    assert "ExampleAgent" in board and "Random" in board
    games = sorted((out / "2p" / "2026-01-01" / "games").glob("*.json"))
    assert len(games) == 4          # seatings: (A,A), (A,R), (R,A), (R,R)
    assert "ladder 2026-01-01" in (out / "2p" / "latest.md").read_text(encoding="utf-8")

    from puerto_rico.records import load_record, replays_exactly
    assert all(replays_exactly(load_record(str(g))) for g in games)
