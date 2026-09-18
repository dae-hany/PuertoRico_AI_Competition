"""The course ladder's discovery, scheduling and scoring (no games are played)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import course_ladder as cl  # noqa: E402


AGENT = """
import numpy as np
from agents.base import Agent

class {cls}(Agent):
    name = "{alias}"

    def act(self, observation, action_mask):
        return int(np.where(action_mask > 0.5)[0][0])
"""


def write(dirpath, fname, cls="MyAgent", alias="MyAgent"):
    path = os.path.join(dirpath, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(AGENT.format(cls=cls, alias=alias))
    return path


def test_discover_reads_alias_class_and_roster(tmp_path):
    d = str(tmp_path)
    write(d, "20201234_hong.py", cls="HongAgent", alias="Coffee Baron")
    entries, problems, clashes = cl.discover(d)
    assert not problems and not clashes
    (entry,) = entries
    assert entry["alias"] == "Coffee Baron"
    assert entry["cls"] == "HongAgent"
    assert (entry["student_id"], entry["student_name"]) == ("20201234", "hong")
    assert entry["spec"].endswith("20201234_hong.py:HongAgent")


def test_discover_reports_unusable_files(tmp_path):
    d = str(tmp_path)
    write(d, "20200001_a.py", cls="A", alias="A")
    with open(os.path.join(d, "20200002_b.py"), "w", encoding="utf-8") as f:
        f.write("class Broken(Agent)\n    pass\n")          # syntax error
    with open(os.path.join(d, "20200003_c.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")                                   # no Agent subclass
    with open(os.path.join(d, "20200004_d.py"), "w", encoding="utf-8") as f:
        f.write(AGENT.format(cls="One", alias="One")
                + AGENT.format(cls="Two", alias="Two"))      # two classes
    entries, problems, _ = cl.discover(d)
    assert [e["alias"] for e in entries] == ["A"]
    assert len(problems) == 3
    assert any("syntax error" in why for _, why in problems)
    assert any("no Agent subclass" in why for _, why in problems)
    assert any("submit exactly one" in why for _, why in problems)


def test_alias_clash_is_fatal_not_silently_suffixed(tmp_path):
    """Two rows keyed by one alias would merge two students; the run must stop."""
    d = str(tmp_path)
    write(d, "20200001_a.py", cls="A", alias="Same Name")
    write(d, "20200002_b.py", cls="B", alias="Same Name")
    entries, problems, clashes = cl.discover(d)
    assert len(entries) == 2
    assert clashes and "Same Name" in clashes[0][1]


def test_alias_may_not_impersonate_a_baseline(tmp_path):
    d = str(tmp_path)
    write(d, "20200001_a.py", cls="A", alias="search")        # case-insensitive
    _, _, clashes = cl.discover(d)
    assert clashes and "baseline" in clashes[0][1]


def test_discover_reads_a_folder_submission_and_takes_the_roster_from_it(tmp_path):
    """--zips unpacks one folder per student; the folder name carries the roster."""
    folder = tmp_path / "20201234_hong"
    (folder / "nested").mkdir(parents=True)
    write(str(folder / "nested"), "agent.py", cls="HongAgent", alias="Coffee Baron")
    (folder / "README.md").write_text("hi", encoding="utf-8")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "agent.py").write_text("garbage(", encoding="utf-8")
    entries, problems, clashes = cl.discover(str(tmp_path))
    assert not problems and not clashes
    (entry,) = entries
    assert entry["alias"] == "Coffee Baron"
    assert (entry["student_id"], entry["student_name"]) == ("20201234", "hong")
    assert entry["spec"].endswith("agent.py:HongAgent")


def test_folder_with_two_agent_files_is_rejected_and_helpers_warn(tmp_path):
    two = tmp_path / "20200001_a"
    two.mkdir()
    write(str(two), "one.py", cls="One", alias="One")
    write(str(two), "two.py", cls="Two", alias="Two")

    helper = tmp_path / "20200002_b"
    helper.mkdir()
    write(str(helper), "agent.py", cls="B", alias="B")
    (helper / "features.py").write_text("VALUE = 1\n", encoding="utf-8")

    entries, problems, _ = cl.discover(str(tmp_path))
    assert [e["alias"] for e in entries] == ["B"]
    assert any("submit exactly one" in why for _, why in problems)
    assert entries[0]["warnings"] and "features.py" in entries[0]["warnings"][0]


def test_safe_members_refuses_paths_that_escape_the_folder(tmp_path):
    import zipfile

    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("agent.py", "ok")
        z.writestr("../escape.py", "no")
        z.writestr("__MACOSX/._agent.py", "no")
    with zipfile.ZipFile(path) as z:
        keep, skipped = cl.safe_members(z)
    assert [rel for _, rel in keep] == ["agent.py"]
    assert {why for _, why in skipped} == {"unsafe path", "mac metadata"}


def test_intake_unpacks_one_folder_per_student_and_lists_the_contents(tmp_path):
    import zipfile

    zips, entries_dir = tmp_path / "zips", tmp_path / "entries"
    zips.mkdir()
    entries_dir.mkdir()
    with zipfile.ZipFile(zips / "20201234_hong.zip", "w") as z:
        z.writestr("my_agent.py", AGENT.format(cls="H", alias="H"))
        z.writestr("weights.npy", "binary")
        z.writestr("README.md", "run it")
    with zipfile.ZipFile(zips / "20205678_kim.zip", "w") as z:
        z.writestr("agent.py", AGENT.format(cls="K", alias="K"))

    manifest = cl.intake_zips(str(zips), str(entries_dir))
    rows = {r["student"]: r for r in manifest}
    assert rows["20201234_hong"]["weights"] == ["weights.npy"]
    assert rows["20201234_hong"]["docs"] == ["README.md"]
    assert rows["20205678_kim"]["docs"] == []            # flags the missing write-up
    assert (entries_dir / "20201234_hong" / "my_agent.py").exists()

    entries, problems, _ = cl.discover(str(entries_dir))
    assert not problems
    assert {e["student_id"] for e in entries} == {"20201234", "20205678"}


def test_build_tasks_pairs_seatings_and_skips_baseline_pairs():
    players = [{"alias": "S1", "kind": "student"}, {"alias": "S2", "kind": "student"},
               {"alias": "B1", "kind": "baseline"}, {"alias": "B2", "kind": "baseline"}]
    tasks = cl.build_tasks(players, games_per_pair=4, seed=0, baseline_vs_baseline=False)
    pairs = {(a, b) for a, b, _ in tasks}
    assert ("B1", "B2") not in pairs                      # anchors, not opponents
    assert ("S1", "S2") in pairs and ("S1", "B1") in pairs
    assert len(tasks) == 5
    for _, _, games in tasks:
        assert len(games) == 4
        assert sum(1 for _, swapped in games if swapped) == 2      # seat-balanced
        assert len({seed for seed, _ in games}) == 4               # distinct seeds

    with_baselines = cl.build_tasks(players, 2, 0, baseline_vs_baseline=True)
    assert len(with_baselines) == 6


def record(a, b, winners, timeouts=(0, 0), illegal=(0, 0)):
    return {"agents": [a, b],
            "result": {"winners": list(winners), "timeouts": list(timeouts),
                       "illegal": list(illegal), "tampered": [0, 0], "elapsed_s": 1.0}}


def entry(alias, sid="20200000", name="x"):
    return cl.Entry(alias=alias, file=f"{sid}_{name}.py", path="", student_id=sid,
                    student_name=name, cls="C", spec="")


def test_board_ranks_on_student_games_only():
    entries = [entry("A", "1", "a"), entry("B", "2", "b")]
    records = [
        record("A", "B", [0]),            # A beats B
        record("B", "A", [0]),            # B beats A (seats swapped)
        record("A", "B", [0, 1]),         # a draw between them
        record("A", "Random", [0]),       # baseline games: reference only
        record("A", "Random", [0]),
        record("Random", "B", [0]),
    ]
    rows = {r["alias"]: r for r in cl.build_boards(records, entries, ["Random"])}
    a, b = rows["A"], rows["B"]
    assert (a["wins"], a["losses"], a["draws"], a["games"]) == (1, 1, 1, 3)
    assert a["win_rate"] == pytest.approx(0.5)             # a draw is half a win
    assert b["win_rate"] == pytest.approx(0.5)
    assert a["baseline_games"] == 2 and a["baseline_win_rate"] == pytest.approx(1.0)
    assert b["baseline_games"] == 1 and b["baseline_win_rate"] == pytest.approx(0.0)
    assert a["elo"] is not None                            # organizer-only column


def test_board_counts_timeouts_and_violations_per_seat():
    entries = [entry("A", "1", "a"), entry("B", "2", "b")]
    records = [record("A", "B", [1], timeouts=(3, 0), illegal=(1, 0))]
    rows = {r["alias"]: r for r in cl.build_boards(records, entries, [])}
    assert (rows["A"]["timeouts"], rows["A"]["illegal"]) == (3, 1)
    assert (rows["B"]["timeouts"], rows["B"]["illegal"]) == (0, 0)


def test_students_markdown_hides_the_roster_and_elo():
    entries = [entry("Coffee Baron", "20201234", "hong")]
    rows = cl.build_boards([record("Coffee Baron", "Random", [0])], entries, ["Random"])
    md = cl.students_markdown(rows, {"date": "2026-01-01", "entries": 1, "games": 1,
                                     "games_per_pair": 2, "time_limit": 1.0})
    assert "Coffee Baron" in md
    assert "20201234" not in md and "hong" not in md and "Elo" not in md
