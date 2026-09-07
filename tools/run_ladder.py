"""Run one round of the public ladder: every entry vs every entry and the baselines.

Organizer tool. During the competition window the ladder is run periodically
(weekly) and the boards are published from ``leaderboard/``:

    python tools/run_ladder.py --track 2p --games-per-seating 2
    python tools/run_ladder.py --track both --entries submissions --out leaderboard

Entries are discovered in ``--entries`` (default ``submissions/``): every
``*.py`` file, every class in it that subclasses ``Agent`` (found by parsing
the file, so no entrant code runs in this process). Entries play inside the
official sandbox (tournament/sandbox.py); the bundled baselines play in-process.

Writes, per track:

    <out>/<track>/<date>/leaderboard.{md,csv,json}   the standings
    <out>/<track>/<date>/games/*.json                every game, replayable with
                                                     tools/replay_game.py
    <out>/<track>/latest.md                          a copy of the newest board

A round-robin grows quadratically with the field: with E entries and B
baselines the 2p track plays (E+B)(E+B+1) games per games-per-seating (every
ordered pair, including self-play seatings), each up to ~300 decisions at up
to 1 s each. Print-only estimate first with --dry-run.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.registry import BASELINES, baseline_names  # noqa: E402
from puerto_rico.records import record_from_result, save_record  # noqa: E402
from tournament.leaderboard import save  # noqa: E402
from tournament.runner import run_tournament  # noqa: E402
from tournament.sandbox import sandboxed_pool  # noqa: E402


def discover_entries(entries_dir, include_example=False):
    """``{leaderboard_name: "path.py:Class"}`` for every agent class found."""
    entries = {}
    if not os.path.isdir(entries_dir):
        return entries
    for fname in sorted(os.listdir(entries_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        if fname == "example_agent.py" and not include_example:
            continue
        path = os.path.join(entries_dir, fname)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        except SyntaxError as e:
            print(f"  skipping {fname}: syntax error ({e})")
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            if not any("Agent" in b for b in bases):
                continue
            label = node.name
            for stmt in node.body:                      # name = "..." in the class body
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and getattr(stmt.targets[0], "id", None) == "name"
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)):
                    label = stmt.value.value
            try:
                rel = os.path.relpath(path, ROOT)
            except ValueError:                      # another drive on Windows
                rel = os.path.abspath(path)
            if rel.startswith(".."):
                rel = os.path.abspath(path)
            spec = f"{rel.replace(os.sep, '/')}:{node.name}"
            while label in entries or label in BASELINES:
                label = f"{label}@{fname[:-3]}"
            entries[label] = spec
    return entries


def games_in_round_robin(n_agents, n_seats, games_per_seating):
    """Games run_round_robin plays: every multiset of seats, every distinct order."""
    total = 0
    for group in itertools.combinations_with_replacement(range(n_agents), n_seats):
        total += len(set(itertools.permutations(group)))
    return total * games_per_seating


def run_track(n_seats, entries, baselines, args):
    track = f"{n_seats}p"
    pool = {name: BASELINES[name] for name in baselines}
    n_games = games_in_round_robin(len(pool) + len(entries), n_seats, args.games_per_seating)
    print(f"\n===== {track} track: {len(entries)} entries + {len(baselines)} baselines, "
          f"~{n_games} games ({args.games_per_seating} per seating) =====")
    for name, spec in entries.items():
        print(f"  entry    {name:<24} {spec}")
    print(f"  baselines: {', '.join(baselines)}")
    if args.dry_run:
        return

    close = lambda: None  # noqa: E731
    if entries:
        if args.no_sandbox:
            from agents.registry import resolve
            for name, spec in entries.items():
                pool[name] = resolve(spec, base_dir=ROOT)[1]
        else:
            sandboxed, close = sandboxed_pool(entries, n_seats=n_seats,
                                              time_limit_s=args.time_limit)
            pool.update(sandboxed)
    try:
        result = run_tournament(pool, games_per_seating=args.games_per_seating,
                                seed=args.seed, time_limit_s=args.time_limit,
                                verbose=True, n_seats=n_seats)
    finally:
        close()

    date = args.date or dt.date.today().isoformat()
    out_dir = os.path.join(args.out, track, date)
    saved = save(result, out_dir=out_dir)
    if not args.no_games:
        games_dir = os.path.join(out_dir, "games")
        for i, rec in enumerate(result["records"]):
            r = rec["result"]
            if r.get("truncated") or "actions" not in r:
                continue
            record = record_from_result(r, n_seats, player_types=rec["agents"],
                                        player_labels=rec["agents"])
            save_record(record, games_dir, filename=f"game_{i:05d}_{r['seed']}.json")
    header = (f"# {track} track - ladder {date}\n\n"
              f"{len(entries)} entries, {len(baselines)} baselines, "
              f"{len(result['records'])} games ({args.games_per_seating} per seating), "
              f"{args.time_limit:g} s/move, seed {args.seed}. Official metric: "
              f"{result['official_metric']}. Replays: `{track}/{date}/games/`.\n\n")
    with open(os.path.join(args.out, track, "latest.md"), "w", encoding="utf-8") as f:
        f.write(header + saved["markdown"] + "\n")
    print("\n" + saved["markdown"])
    print(f"\nSaved to {out_dir}/ and {args.out}/{track}/latest.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=("2p", "3p", "both"), default="both")
    ap.add_argument("--entries", default="submissions", help="directory of entry .py files")
    ap.add_argument("--include-example", action="store_true",
                    help="also rank submissions/example_agent.py")
    ap.add_argument("--baselines", default="", help="comma-separated names; default = "
                    "every baseline built for the track")
    ap.add_argument("--games-per-seating", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=1.0)
    ap.add_argument("--out", default="leaderboard")
    ap.add_argument("--date", default=None, help="folder name (default: today)")
    ap.add_argument("--no-games", action="store_true", help="do not save replay records")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="run entries in-process (faster; NOT the official guarantees)")
    ap.add_argument("--dry-run", action="store_true", help="list the field and stop")
    args = ap.parse_args()

    entries = discover_entries(args.entries, include_example=args.include_example)
    if not entries:
        print(f"(no entries found in {args.entries}/ - ranking the baselines only)")
    tracks = {"2p": [2], "3p": [3], "both": [2, 3]}[args.track]
    for n in tracks:
        if args.baselines:
            names = [b.strip() for b in args.baselines.split(",") if b.strip()]
            unknown = [b for b in names if b not in BASELINES]
            if unknown:
                sys.exit(f"unknown baselines: {', '.join(unknown)}; known: {', '.join(BASELINES)}")
        else:
            names = baseline_names(n)
        run_track(n, entries, names, args)


if __name__ == "__main__":
    main()
