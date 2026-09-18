"""Run one ladder round for a course cohort: validate, play in parallel, publish two boards.

Organizer tool for running the 2p track as a university final project, where
the grade is a **win rate among the students' own games** rather than the
competition's Elo board. It differs from ``tools/run_ladder.py`` in four ways:

1. every entry is **validated first** (the official sandbox check) and alias
   clashes are reported, so a broken file is caught before the games start;
2. games are played **in parallel** (one process per concurrent game), because
   a cohort round-robin is quadratic and a single process is too slow;
3. it writes **two boards** -- a student-facing one with wins/losses/win rate
   only, and an organizer one that adds the roster, Elo, timeouts and
   violations;
4. the ranking uses **student-vs-student games only**. Baselines are played as
   anchors and reported in a separate column.

Submissions are discovered in ``--entries``: either one ``*.py`` per student
named ``<student id>_<name>.py``, or one folder per student with the same name
-- which is what ``--zips`` unpacks from the ``<student id>_<name>.zip`` files
a submission form collects. Each submission holds exactly one ``Agent``
subclass; a weight file may sit beside it (the agent locates it from
``__file__``).

    python tools/course_ladder.py --entries C:/course_2p/submissions \
        --zips C:/course_2p/zips --zip-only          # unpack + checklist
    python tools/course_ladder.py --entries C:/course_2p/submissions --dry-run
    python tools/course_ladder.py --entries C:/course_2p/submissions \
        --out C:/course_2p/out --games-per-pair 4 --workers 8

Writes, per run:

    <out>/<date>/board_students.md    publish this (aliases only)
    <out>/<date>/board_students.csv
    <out>/<date>/board_organizer.csv  keep this (roster, Elo, timeouts)
    <out>/<date>/validation.txt       per-entry PASS / WARN / FAIL
    <out>/<date>/intake.csv           what each zip contained (with --zips)
    <out>/<date>/summary.json
    <out>/<date>/games/*.json         every replay, for tools/replay_game.py

Student code never belongs in this repository: point ``--entries`` and
``--out`` at a working directory outside it.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.registry import BASELINES, baseline_names, lookup_baseline  # noqa: E402
from puerto_rico.records import record_from_result, save_record  # noqa: E402
from tournament.match import play_game                          # noqa: E402
from tournament.rankers.elo import rank_elo                     # noqa: E402
from tournament.rankers.win_rate import wilson_ci               # noqa: E402

N_SEATS = 2                                   # this tool is the 1-vs-1 track
FILENAME_RE = re.compile(r"^(?P<sid>[0-9]{4,12})_(?P<name>.+)$")
DEFAULT_BASELINES = "random,factory,shippingrush,tradebuilding,actionvalue,searchlite"

WEIGHT_EXTS = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".bin", ".safetensors",
               ".pkl", ".h5", ".onnx"}
DOC_EXTS = {".md", ".txt", ".pdf", ".docx", ".doc", ".hwp", ".hwpx", ".ipynb"}


# ----------------------------------------------------------------- discovery
class Entry(dict):
    """One student's submission: alias, file, roster fields and agent spec."""


def parse_filename(stem):
    """``20201234_hong`` -> ``("20201234", "hong")``; anything else -> ("", stem)."""
    m = FILENAME_RE.match(stem)
    if not m:
        return "", stem
    return m.group("sid"), m.group("name")


def agent_classes(path):
    """``[(class_name, alias)]`` for every Agent subclass defined in ``path``."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    found = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        if not any("Agent" in b for b in bases):
            continue
        alias = node.name
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and getattr(stmt.targets[0], "id", None) == "name"
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                alias = stmt.value.value
        found.append((node.name, alias))
    return found


def scan_folder(folder):
    """``(hits, others)`` for the ``*.py`` under ``folder``.

    ``hits`` are ``(path, classes)`` for files that define an Agent subclass;
    ``others`` are ``(path, why)`` for the rest -- a broken file, or a helper
    module. Walks the tree, because a student's zip often nests one more
    folder inside itself.
    """
    hits, others = [], []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames
                       if d != "__MACOSX" and not d.startswith((".", "_"))]
        for fname in sorted(filenames):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                classes = agent_classes(path)
            except SyntaxError as e:
                others.append((path, f"syntax error: {e}"))
                continue
            if classes:
                hits.append((path, classes))
            else:
                others.append((path, "no Agent subclass"))
    return hits, others


def discover(entries_dir):
    """``(entries, problems, clashes)`` for the submissions in ``entries_dir``.

    A submission is either a ``<student id>_<name>.py`` file at the top level
    or a ``<student id>_<name>/`` folder (what ``--zips`` unpacks); the roster
    fields come from that name either way.

    An entry is skipped (and listed in ``problems``) when it does not parse,
    defines no Agent subclass, or defines more than one: the rules ask for
    exactly one class, and guessing which one to seat would put the wrong
    agent on the board. ``clashes`` are the subset the caller must treat as
    fatal -- two entries sharing a leaderboard alias, or one taking a
    baseline's name -- because the board is keyed by alias and would silently
    merge two students into one row.
    """
    entries, problems = [], []
    if not os.path.isdir(entries_dir):
        return entries, [("(none)", f"no such directory: {entries_dir}")], []

    def add(label, path, classes, warnings=()):
        if len(classes) > 1:
            names = ", ".join(c for c, _ in classes)
            problems.append((label, f"{len(classes)} Agent subclasses ({names});"
                                    " submit exactly one"))
            return
        cls_name, alias = classes[0]
        sid, student = parse_filename(label[:-3] if label.endswith(".py") else label)
        entries.append(Entry(alias=alias, file=label, path=os.path.abspath(path),
                             student_id=sid, student_name=student, cls=cls_name,
                             warnings=list(warnings),
                             spec=f"{os.path.abspath(path)}:{cls_name}"))

    for name in sorted(os.listdir(entries_dir)):
        if name.startswith((".", "_")):
            continue
        path = os.path.join(entries_dir, name)

        if os.path.isdir(path):
            hits, others = scan_folder(path)
            broken = [(p, why) for p, why in others if why.startswith("syntax error")]
            if not hits:
                why = broken[0][1] if broken else "no Agent subclass anywhere in the folder"
                problems.append((name + "/", why))
                continue
            if len(hits) > 1:
                files = ", ".join(os.path.basename(p) for p, _ in hits)
                problems.append((name + "/", f"{len(hits)} files define an Agent "
                                             f"subclass ({files}); submit exactly one"))
                continue
            extra = [os.path.basename(p) for p, _ in others]
            warn = ([f"extra .py files are ignored and cannot be imported "
                     f"({', '.join(extra)}); the entry must be self-contained"]
                    if extra else [])
            add(name, hits[0][0], hits[0][1], warn)
            continue

        if not name.endswith(".py"):
            continue
        try:
            classes = agent_classes(path)
        except SyntaxError as e:
            problems.append((name, f"syntax error: {e}"))
            continue
        if not classes:
            problems.append((name, "no Agent subclass found"))
            continue
        add(name, path, classes)

    seen = {}
    for e in entries:
        seen.setdefault(e["alias"], []).append(e["file"])
    clashes = []
    for alias, files in sorted(seen.items()):
        if len(files) > 1:
            clashes.append((", ".join(files),
                            f"alias {alias!r} is used by {len(files)} entries;"
                            " ask them to pick distinct names"))
        if lookup_baseline(alias):
            clashes.append((", ".join(files),
                            f"alias {alias!r} collides with a bundled baseline"))
    return entries, problems + clashes, clashes


# -------------------------------------------------------------- zip intake
def safe_members(zf):
    """The members worth extracting, and the ones refused, from ``zf``.

    Refuses absolute paths and ``..`` segments (a zip can otherwise write
    outside the folder it is unpacked into) as well as the ``__MACOSX``
    bookkeeping a Mac adds to every archive.
    """
    keep, skipped = [], []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir():
            continue
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if not parts or name.startswith("/") or ".." in parts or ":" in parts[0]:
            skipped.append((name, "unsafe path"))
            continue
        if parts[0] == "__MACOSX" or parts[-1].startswith("._"):
            skipped.append((name, "mac metadata"))
            continue
        keep.append((info, "/".join(parts)))
    return keep, skipped


def intake_zips(zip_dir, entries_dir):
    """Unpack every ``*.zip`` in ``zip_dir`` into ``<entries_dir>/<zip stem>/``.

    Each student keeps their own file names inside their own folder, so two
    students who both named their agent ``my_agent.py`` do not collide and an
    agent that loads a weight file next to itself still finds it.

    Returns a manifest -- one row per zip, listing the code, weight and
    document files found -- so the organizer can see at a glance who is
    missing the README or the write-up.
    """
    import zipfile

    manifest = []
    zips = sorted(f for f in os.listdir(zip_dir) if f.lower().endswith(".zip"))
    for fname in zips:
        stem = fname[:-4]
        dest = os.path.join(entries_dir, stem)
        os.makedirs(dest, exist_ok=True)
        row = {"zip": fname, "student": stem, "code": [], "weights": [],
               "docs": [], "other": [], "skipped": []}
        try:
            with zipfile.ZipFile(os.path.join(zip_dir, fname)) as zf:
                keep, skipped = safe_members(zf)
                row["skipped"] = [f"{n} ({why})" for n, why in skipped]
                for info, rel in keep:
                    target = os.path.join(dest, *rel.split("/"))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:
                        out.write(src.read())
                    ext = os.path.splitext(rel)[1].lower()
                    bucket = ("code" if ext == ".py" else
                              "weights" if ext in WEIGHT_EXTS else
                              "docs" if ext in DOC_EXTS else "other")
                    row[bucket].append(rel)
        except zipfile.BadZipFile as e:
            row["skipped"].append(f"unreadable zip: {e}")
        manifest.append(row)
    return manifest


def print_manifest(manifest):
    print(f"\nunpacked {len(manifest)} zips")
    print(f"  {'student':<28} {'py':>3} {'weights':>8} {'docs':>5}  {'missing':<18} notes")
    for row in manifest:
        missing = []
        if not row["code"]:
            missing.append("AGENT")
        if not row["docs"]:
            missing.append("README/write-up")
        notes = f"{len(row['skipped'])} files skipped" if row["skipped"] else ""
        print(f"  {row['student']:<28} {len(row['code']):>3} {len(row['weights']):>8} "
              f"{len(row['docs']):>5}  {', '.join(missing) if missing else '-':<18} {notes}")


# ---------------------------------------------------------------- validation
def validate_entries(entries, games, time_limit_s, workers):
    """Run the official sandbox check on each entry; ``{alias: (status, text)}``."""
    def one(entry):
        cmd = [sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"),
               entry["spec"], "--track", "2p", "--games", str(games),
               "--time-limit", str(time_limit_s)]
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=600, encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            status = "READY" if p.returncode == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            out, status = "validator timed out after 600 s", "FAIL"
        return entry["alias"], status, out

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for alias, status, out in pool.map(one, entries):
            results[alias] = (status, out)
            print(f"  {status:5s}  {alias}")
    return results


# ------------------------------------------------------------------ schedule
def build_tasks(players, games_per_pair, seed, baseline_vs_baseline):
    """One task per unordered pair: ``(a, b, [(seed, swapped), ...])``.

    Seatings alternate so each side plays first in half its games, which is how
    the seat advantage is balanced out.
    """
    tasks = []
    names = [p["alias"] for p in players]
    kinds = {p["alias"]: p["kind"] for p in players}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not baseline_vs_baseline and kinds[a] == "baseline" and kinds[b] == "baseline":
                continue
            games = []
            for g in range(games_per_pair):
                games.append((seed + 1000 * len(tasks) + g, bool(g % 2)))
            tasks.append((a, b, games))
    return tasks


_SPECS = {}          # alias -> spec | baseline name, set in each worker process


def _init_worker(specs, time_limit_s):
    global _SPECS, _TIME_LIMIT
    _SPECS = specs
    _TIME_LIMIT = time_limit_s


def _factories(aliases):
    """``({alias: factory}, close)`` -- baselines in process, entries sandboxed."""
    from tournament.sandbox import sandboxed_pool
    pool, sandboxed = {}, {}
    for alias in aliases:
        spec = _SPECS[alias]
        if spec["kind"] == "baseline":
            pool[alias] = BASELINES[spec["name"]]
        else:
            sandboxed[alias] = spec["spec"]
    close = lambda: None                                          # noqa: E731
    if sandboxed:
        made, close = sandboxed_pool(sandboxed, n_seats=1, time_limit_s=_TIME_LIMIT)
        pool.update(made)
    return pool, close


def play_pair(task):
    """Play one pair's games in this process; returns per-game records."""
    a, b, games = task
    out = []
    pool, close = _factories([a, b])
    try:
        for seed, swapped in games:
            seats = [b, a] if swapped else [a, b]
            started = time.perf_counter()
            result = play_game([pool[seats[0]](), pool[seats[1]]()], seed=seed,
                               time_limit_s=_TIME_LIMIT)
            result["elapsed_s"] = round(time.perf_counter() - started, 3)
            out.append({"agents": seats, "result": result})
    finally:
        close()
    return out


# -------------------------------------------------------------------- boards
def tally(records, players, kinds):
    """Per-player W/L/D and totals, split into student games and baseline games."""
    blank = lambda: {"w": 0, "l": 0, "d": 0, "bw": 0, "bl": 0, "bd": 0,     # noqa: E731
                     "timeouts": 0, "illegal": 0, "tampered": 0,
                     "elapsed": 0.0, "games": 0}
    t = {p: blank() for p in players}
    for rec in records:
        seats = rec["agents"]
        res = rec["result"]
        winners = res["winners"]
        both_students = all(kinds[s] == "student" for s in seats)
        for i, name in enumerate(seats):
            row = t[name]
            row["games"] += 1
            row["timeouts"] += res["timeouts"][i]
            row["illegal"] += res["illegal"][i]
            row["tampered"] += res["tampered"][i]
            row["elapsed"] += res.get("elapsed_s", 0.0)
            if len(winners) > 1:
                key = "d" if both_students else "bd"
            elif winners[0] == i:
                key = "w" if both_students else "bw"
            else:
                key = "l" if both_students else "bl"
            row[key] += 1
    return t


def score(row, prefix=""):
    """``(wins, losses, draws, games, win_rate)`` -- a draw counts as half a win."""
    w, l, d = row[prefix + "w"], row[prefix + "l"], row[prefix + "d"]
    n = w + l + d
    rate = (w + 0.5 * d) / n if n else 0.0
    return w, l, d, n, rate


def build_boards(records, entries, baselines):
    """Student board rows (student games only) plus organizer extras."""
    kinds = {e["alias"]: "student" for e in entries}
    kinds.update({b: "baseline" for b in baselines})
    players = list(kinds)
    t = tally(records, players, kinds)

    student_records = [r for r in records
                       if all(kinds[s] == "student" for s in r["agents"])]
    aliases = [e["alias"] for e in entries]
    elo = {r["agent"]: r["elo"] for r in rank_elo(student_records, aliases)} if aliases else {}

    rows = []
    for e in entries:
        row = t[e["alias"]]
        w, l, d, n, rate = score(row)
        bw, bl, bd, bn, brate = score(row, prefix="b")
        lo, hi = wilson_ci(w + 0.5 * d, n)
        rows.append({
            "alias": e["alias"], "file": e["file"], "student_id": e["student_id"],
            "student_name": e["student_name"], "wins": w, "losses": l, "draws": d,
            "games": n, "win_rate": rate, "ci_low": lo, "ci_high": hi,
            "baseline_wins": bw, "baseline_losses": bl, "baseline_draws": bd,
            "baseline_games": bn, "baseline_win_rate": brate,
            "elo": elo.get(e["alias"]), "timeouts": row["timeouts"],
            "illegal": row["illegal"], "tampered": row["tampered"],
            "mean_game_s": round(row["elapsed"] / row["games"], 1) if row["games"] else 0.0,
        })
    rows.sort(key=lambda r: (-r["win_rate"], -r["wins"], r["alias"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def students_markdown(rows, meta):
    head = (f"# {meta['date']} ladder -- 2p track\n\n"
            f"{meta['entries']} entries, {meta['games']} games "
            f"({meta['games_per_pair']} per pair), {meta['time_limit']:g} s/move. "
            "Ranked by win rate over games between entries; ties count as half a win. "
            "The baseline column is a reference and does not affect the rank. "
            f"Replays: `{meta['date']}/games/`.\n\n")
    lines = ["| Rank | Agent | W | L | D | Win% | Games | vs baselines |",
             "|-----:|-------|--:|--:|--:|-----:|------:|-------------:|"]
    for r in rows:
        base = (f"{r['baseline_win_rate'] * 100:.1f}%" if r["baseline_games"] else "-")
        lines.append(f"| {r['rank']} | {r['alias']} | {r['wins']} | {r['losses']} | "
                     f"{r['draws']} | {r['win_rate'] * 100:.1f}% | {r['games']} | {base} |")
    return head + "\n".join(lines)


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", required=True,
                    help="directory of submissions (a .py per student, or one "
                         "folder per student as --zips unpacks them)")
    ap.add_argument("--zips", default=None,
                    help="directory of <student id>_<name>.zip files to unpack into "
                         "--entries first (one folder per student)")
    ap.add_argument("--zip-only", action="store_true",
                    help="unpack the zips, print the submission checklist, and stop")
    ap.add_argument("--out", default="course_ladder", help="output directory")
    ap.add_argument("--games-per-pair", type=int, default=2,
                    help="games between every pair (split evenly over both seatings)")
    ap.add_argument("--baselines", default=DEFAULT_BASELINES,
                    help="comma-separated baseline names, or 'none'")
    ap.add_argument("--baseline-vs-baseline", action="store_true",
                    help="also play baselines against each other (off: they are anchors)")
    ap.add_argument("--workers", type=int, default=0,
                    help="concurrent games (default: min(8, cpu-2)); each one runs "
                         "up to 2 agent processes, so watch memory with torch entries")
    ap.add_argument("--time-limit", type=float, default=1.0, help="seconds per move")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default=None, help="output folder name (default: today)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the sandbox check (not recommended)")
    ap.add_argument("--validate-games", type=int, default=2)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-games", action="store_true", help="do not save replays")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    args = ap.parse_args()

    manifest = []
    if args.zips:
        os.makedirs(args.entries, exist_ok=True)
        manifest = intake_zips(args.zips, args.entries)
        print_manifest(manifest)
        if args.zip_only:
            return
    elif args.zip_only:
        sys.exit("--zip-only needs --zips")

    entries, problems, clashes = discover(args.entries)
    print(f"\nentries directory: {args.entries}")
    for e in entries:
        who = f"{e['student_id']} {e['student_name']}".strip()
        print(f"  entry  {e['alias']:<20} {e['file']:<28} class {e['cls']} ({who})")
        for warning in e.get("warnings", []):
            print(f"         warning: {warning}")
    if problems:
        print("\nPROBLEMS (these entries do not run):")
        for fname, why in problems:
            print(f"  {fname}: {why}")
    if not entries:
        sys.exit("\nno usable entries found")
    if clashes:
        sys.exit("\nfix the alias clashes above before running: the board is keyed by "
                 "alias, so two entries sharing one would be merged into a single row")

    raw = ([] if args.baselines.strip().lower() in ("none", "")
           else [b.strip() for b in args.baselines.split(",") if b.strip()])
    unknown = [b for b in raw if lookup_baseline(b) is None]
    if unknown:
        sys.exit(f"unknown baselines: {', '.join(unknown)}; known: {', '.join(sorted(BASELINES))}")
    baselines = [lookup_baseline(b) for b in raw]
    known = set(baseline_names(N_SEATS))
    off_track = [b for b in baselines if b not in known]
    if off_track:
        print(f"\nnote: {', '.join(off_track)} is not a 2p baseline; dropping it")
        baselines = [b for b in baselines if b in known]

    players = ([{"alias": e["alias"], "kind": "student"} for e in entries]
               + [{"alias": b, "kind": "baseline"} for b in baselines])
    tasks = build_tasks(players, args.games_per_pair, args.seed, args.baseline_vs_baseline)
    total_games = sum(len(t[2]) for t in tasks)
    student_pairs = len(entries) * (len(entries) - 1) // 2
    workers = args.workers or max(1, min(8, (os.cpu_count() or 4) - 2))

    print(f"\nplan: {len(entries)} entries + {len(baselines)} baselines, "
          f"{len(tasks)} pairings, {total_games} games "
          f"({student_pairs * args.games_per_pair} of them between entries), "
          f"{workers} concurrent")
    print("      estimated wall time at 1 / 10 / 60 / 300 s per game: "
          + " / ".join(f"{total_games * s / workers / 3600:.1f} h" for s in (1, 10, 60, 300)))
    if args.dry_run:
        return

    date = args.date or dt.date.today().isoformat()
    out_dir = os.path.join(args.out, date)
    os.makedirs(out_dir, exist_ok=True)

    if manifest:
        write_csv(os.path.join(out_dir, "intake.csv"),
                  [{"student": r["student"], "zip": r["zip"],
                    "code": " ".join(r["code"]), "weights": " ".join(r["weights"]),
                    "docs": " ".join(r["docs"]), "other": " ".join(r["other"]),
                    "skipped": " ".join(r["skipped"])} for r in manifest],
                  ["student", "zip", "code", "weights", "docs", "other", "skipped"])

    if not args.no_validate:
        print(f"\nvalidating {len(entries)} entries in the official sandbox...")
        report = validate_entries(entries, args.validate_games, args.time_limit, workers)
        with open(os.path.join(out_dir, "validation.txt"), "w", encoding="utf-8") as f:
            for e in entries:
                status, text = report[e["alias"]]
                f.write(f"===== {e['file']} ({e['alias']}): {status} =====\n{text}\n")
        failed = [e for e in entries if report[e["alias"]][0] == "FAIL"]
        if failed:
            print("\nFAIL (excluded from this round; see validation.txt):")
            for e in failed:
                print(f"  {e['file']} ({e['alias']})")
            drop = {e["alias"] for e in failed}
            entries = [e for e in entries if e["alias"] not in drop]
            if not entries:
                sys.exit("every entry failed validation")
            players = ([{"alias": e["alias"], "kind": "student"} for e in entries]
                       + [{"alias": b, "kind": "baseline"} for b in baselines])
            tasks = build_tasks(players, args.games_per_pair, args.seed,
                                args.baseline_vs_baseline)
            total_games = sum(len(t[2]) for t in tasks)
    if args.validate_only:
        print(f"\nvalidation report: {out_dir}/validation.txt")
        return

    specs = {e["alias"]: {"kind": "student", "spec": e["spec"]} for e in entries}
    specs.update({b: {"kind": "baseline", "name": b} for b in baselines})

    print(f"\nplaying {total_games} games with {workers} concurrent...")
    started = time.perf_counter()
    records, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(specs, args.time_limit)) as pool:
        futures = {pool.submit(play_pair, t): t for t in tasks}
        for fut in as_completed(futures):
            a, b, games = futures[fut]
            try:
                records.extend(fut.result())
            except Exception as exc:                  # one pairing must not sink the round
                print(f"  ERROR {a} vs {b}: {type(exc).__name__}: {exc}")
            done += len(games)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0
            left = (total_games - done) / rate / 60 if rate else 0
            print(f"  {done}/{total_games} games, {elapsed / 60:.1f} min elapsed, "
                  f"~{left:.1f} min left", flush=True)

    kinds = {e["alias"]: "student" for e in entries}
    kinds.update({b: "baseline" for b in baselines})
    rows = build_boards(records, entries, baselines)
    meta = {"date": date, "entries": len(entries), "baselines": baselines,
            "games": len(records), "games_per_pair": args.games_per_pair,
            "time_limit": args.time_limit, "seed": args.seed,
            "wall_minutes": round((time.perf_counter() - started) / 60, 1)}

    md = students_markdown(rows, meta)
    with open(os.path.join(out_dir, "board_students.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    write_csv(os.path.join(out_dir, "board_students.csv"), rows,
              ["rank", "alias", "wins", "losses", "draws", "games", "win_rate",
               "baseline_win_rate"])
    write_csv(os.path.join(out_dir, "board_organizer.csv"), rows,
              ["rank", "student_id", "student_name", "file", "alias", "wins",
               "losses", "draws", "games", "win_rate", "ci_low", "ci_high", "elo",
               "baseline_wins", "baseline_losses", "baseline_games",
               "baseline_win_rate", "timeouts", "illegal", "tampered", "mean_game_s"])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "board": rows}, f, indent=2, ensure_ascii=False)

    if not args.no_games:
        games_dir = os.path.join(out_dir, "games")
        for i, rec in enumerate(records):
            r = rec["result"]
            if r.get("truncated") or "actions" not in r:
                continue
            record = record_from_result(r, N_SEATS, player_types=rec["agents"],
                                        player_labels=rec["agents"])
            save_record(record, games_dir, filename=f"game_{i:05d}_{r['seed']}.json")

    flagged = [r for r in rows if r["timeouts"] or r["illegal"] or r["tampered"]]
    print("\n" + md)
    if flagged:
        print("\nentries with timeouts / illegal moves / tampering:")
        for r in flagged:
            print(f"  {r['alias']:<20} timeouts {r['timeouts']}, illegal "
                  f"{r['illegal']}, tampered {r['tampered']}")
    print(f"\npublish: {out_dir}/board_students.md and {out_dir}/games/")
    print(f"keep:    {out_dir}/board_organizer.csv (roster, Elo) and validation.txt")
    print(f"took {meta['wall_minutes']} min")


if __name__ == "__main__":
    main()
