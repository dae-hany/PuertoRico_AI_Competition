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
    assert rows["20201234_hong"]["readme"] is True
    assert rows["20205678_kim"]["readme"] is False       # flags the missing README
    assert (entries_dir / "20201234_hong" / "my_agent.py").exists()

    entries, problems, _ = cl.discover(str(entries_dir))
    assert not problems
    assert {e["student_id"] for e in entries} == {"20201234", "20205678"}


def test_build_tasks_plays_every_seed_in_both_seatings_and_skips_baseline_pairs():
    players = [{"alias": "S1", "kind": "student"}, {"alias": "S2", "kind": "student"},
               {"alias": "B1", "kind": "baseline"}, {"alias": "B2", "kind": "baseline"}]
    tasks = cl.build_tasks(players, seeds=[3, 7], baseline_vs_baseline=False)
    pairs = {(a, b) for a, b, _ in tasks}
    assert ("B1", "B2") not in pairs                      # anchors, not opponents
    assert ("S1", "S2") in pairs and ("S1", "B1") in pairs
    assert len(tasks) == 5
    for _, _, games in tasks:
        assert sorted(games) == [(3, False), (3, True), (7, False), (7, True)]

    with_baselines = cl.build_tasks(players, [1], baseline_vs_baseline=True)
    assert len(with_baselines) == 6


def test_course_schedule_sizes():
    """Ladder: seeds 1-25 x 2 seatings = 50 games; final: 500 seeds = 1000 games."""
    players = [{"alias": "S1", "kind": "student"}, {"alias": "S2", "kind": "student"}]
    ((_, _, ladder),) = cl.build_tasks(players, cl.parse_seeds(cl.DEFAULT_SEEDS), False)
    assert len(ladder) == 50
    final_seeds = cl.parse_seeds("1-25") + list(range(10_000, 10_475))
    ((_, _, final),) = cl.build_tasks(players, final_seeds, False)
    assert len(final) == 1000
    assert cl.DEFAULT_TIME_LIMIT == 0.1


def test_parse_seeds_ranges_lists_and_files(tmp_path):
    assert cl.parse_seeds("1-3, 7 9-10") == [1, 2, 3, 7, 9, 10]
    assert cl.parse_seeds("5,5,4-6") == [5, 4, 6]               # repeats dropped
    assert cl.parse_seeds("") == []
    for bad in ("3-1", "x", "-4", "1.5"):
        with pytest.raises(ValueError):
            cl.parse_seeds(bad)
    path = tmp_path / "private.txt"
    path.write_text("# private\n101\n202 303  # trailing note\n\n404,505\n", encoding="utf-8")
    assert cl.read_seeds_file(str(path)) == [101, 202, 303, 404, 505]
    assert cl.seeds_digest([1, 2]) != cl.seeds_digest([2, 1])


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
    assert a["points"] == pytest.approx(1.5)               # a draw is half a win
    assert a["win_rate"] == pytest.approx(0.5)
    assert b["win_rate"] == pytest.approx(0.5)
    assert a["baseline_games"] == 2 and a["baseline_win_rate"] == pytest.approx(1.0)
    assert b["baseline_games"] == 1 and b["baseline_win_rate"] == pytest.approx(0.0)
    assert a["elo"] is not None                            # organizer-only column
    assert a["rank"] == b["rank"] == 1                     # equal wins share a rank

    table = cl.vs_baselines(records, entries, ["Random"])
    assert table["A"]["Random"] == (2.0, 2) and table["B"]["Random"] == (0.0, 1)


def test_ranks_follow_wins_and_skip_after_a_tie():
    entries = [entry(n, str(i), n.lower()) for i, n in enumerate("ABCD")]
    records = [record("A", "B", [0]), record("A", "C", [0]), record("A", "D", [0]),
               record("B", "C", [0, 1]), record("B", "D", [0]), record("C", "D", [0])]
    rows = cl.build_boards(records, entries, [])
    assert [(r["alias"], r["points"], r["rank"]) for r in rows] == [
        ("A", 3, 1), ("B", 1.5, 2), ("C", 1.5, 2), ("D", 0, 4)]


def test_board_counts_timeouts_and_violations_per_seat():
    entries = [entry("A", "1", "a"), entry("B", "2", "b")]
    records = [record("A", "B", [1], timeouts=(3, 0), illegal=(1, 0))]
    rows = {r["alias"]: r for r in cl.build_boards(records, entries, [])}
    assert (rows["A"]["timeouts"], rows["A"]["illegal"]) == (3, 1)
    assert (rows["B"]["timeouts"], rows["B"]["illegal"]) == (0, 0)


META = {"label": "ladder 1", "date": "2026-11-19", "entries": 2, "games": 50,
        "seeds": 25, "games_per_pair": 50, "time_limit": 0.1}


def test_students_markdown_shows_only_rank_and_wins():
    entries = [entry("Coffee Baron", "20201234", "hong"), entry("Sugar King", "20205678", "kim")]
    records = [record("Coffee Baron", "Sugar King", [0]), record("Sugar King", "Coffee Baron", [0, 1]),
               record("Coffee Baron", "Random", [0])]
    rows = cl.build_boards(records, entries, ["Random"])
    md = cl.students_markdown(rows, META)
    assert "| Rank | Agent | Wins |" in md
    assert "| 1 | Coffee Baron | 1.5 |" in md and "| 2 | Sugar King | 0.5 |" in md
    assert "20201234" not in md and "hong" not in md and "Elo" not in md
    assert "Win%" not in md and "games/" not in md         # no losses, rates or replays

    ref = cl.baselines_markdown(rows, cl.vs_baselines(records, entries, ["Random"]),
                                ["Random"], META)
    assert "| Coffee Baron | 1 / 1 |" in ref and "| Sugar King | - |" in ref


def test_private_seeds_are_distinct_avoid_the_public_ones_and_stay_out_of_the_repo(tmp_path):
    import subprocess

    import make_private_seeds as mps

    seeds = mps.draw(200, exclude=range(1, 26), low=1, high=300)
    assert len(seeds) == len(set(seeds)) == 200
    assert not set(seeds) & set(range(1, 26))
    with pytest.raises(ValueError):
        mps.draw(300, exclude=range(1, 26), low=1, high=300)   # only 275 candidates

    script = os.path.join(ROOT, "tools", "make_private_seeds.py")
    inside = os.path.join(ROOT, "private_seeds_test.txt")
    r = subprocess.run([sys.executable, script, "--count", "3", "--out", inside],
                       capture_output=True, text=True)
    assert r.returncode != 0 and not os.path.exists(inside)

    outside = tmp_path / "private.txt"
    r = subprocess.run([sys.executable, script, "--count", "475", "--out", str(outside)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = cl.read_seeds_file(str(outside))
    assert len(got) == 475 and not set(got) & set(range(1, 26))
    assert "sha256" in r.stdout
