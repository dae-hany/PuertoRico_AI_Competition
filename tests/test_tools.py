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
