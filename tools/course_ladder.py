"""Run one ladder round for a course cohort: validate, play in parallel, publish the boards.

Organizer tool for running the 2p track as a university final project, where
the grade uses the **wins and the rank among the students' own games** rather
than the competition's Elo board (docs/COURSE_CNU_2026.md). It differs from
``tools/run_ladder.py`` in five ways:

1. every entry is **validated first** (the official sandbox check) and alias
   clashes are reported, so a broken file is caught before the games start;
2. every pair plays a **fixed list of seeds, each once in either seating**
   (``--seeds``, ``--seeds-file``), so every entry meets every opponent on the
   same starts from both sides;
3. games are played **in parallel** (one process per concurrent game), because
   a cohort round-robin is quadratic and a single process is too slow;
4. the published board shows **only the rank and the wins** (a draw is half a
   win); the organizer board adds the roster, W/L/D, Elo, timeouts and
   violations, and the replays stay with the organizer;
5. the ranking uses **student-vs-student games only**. Baselines are played on
   the same seeds as a reference and reported on a separate board.

Seeds. ``--seeds`` takes ranges and lists (``1-25``, ``1-25,40,90-99``) and
defaults to the public ladder seeds ``1-25``: 25 seeds x 2 seatings = 50 games
per pair. ``--seeds-file`` adds the seeds in a file (one or more per line,
``#`` comments allowed) -- this is how the private seeds of the final run come
in without ever being typed into this repository; draw them with
``tools/make_private_seeds.py``. The course's final run is the 25 public seeds
plus 475 private ones: 500 seeds x 2 seatings = 1000 games per pair.

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
        --out C:/course_2p/out --workers 8           # a ladder round: seeds 1-25
    python tools/course_ladder.py --entries C:/course_2p/final \
        --out C:/course_2p/out --label final --seeds 1-25 \
        --seeds-file C:/course_2p/private_seeds.txt --workers 8

The time limit defaults to **0.1 s per decision** (the competition rule; the
bundled baselines keep their own 1 s budget). The code zips
and the report zips are both named ``<student id>_<name>.zip``; keep them in
separate folders and give ``--zips`` the code zips only.

Writes, per run:

    <out>/<date>/board_students.md    publish this: rank and wins, aliases only
    <out>/<date>/board_students.csv   the same as a table
    <out>/<date>/board_baselines.md   publish this too: results against the
                                      baselines, for reference only
    <out>/<date>/board_organizer.csv  keep this (roster, W/L/D, Elo, timeouts)
    <out>/<date>/validation.txt       keep: per-entry PASS / WARN / FAIL
    <out>/<date>/intake.csv           keep: what each zip contained (with --zips)
    <out>/<date>/summary.json         keep
    <out>/<date>/games/*.json         keep: every replay, for tools/replay_game.py;
                                      the course does not publish replays

Student code and the private seeds never belong in this repository: point
``--entries``, ``--out`` and ``--seeds-file`` at a working directory outside it.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
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
from tournament.match import TIME_LIMIT_S, play_game            # noqa: E402
from tournament.rankers.elo import rank_elo                     # noqa: E402
from tournament.rankers.win_rate import wilson_ci               # noqa: E402

N_SEATS = 2                                   # this tool is the 1-vs-1 track
FILENAME_RE = re.compile(r"^(?P<sid>[0-9]{4,12})_(?P<name>.+)$")
DEFAULT_BASELINES = "random,factory,shippingrush,tradebuilding,actionvalue,searchlite"
DEFAULT_SEEDS = "1-25"                        # the course's public ladder seeds
DEFAULT_TIME_LIMIT = TIME_LIMIT_S             # 0.1 s per decision, as in the competition

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


# --------------------------------------------------------------------- seeds
def parse_seeds(text):
    """``"1-3, 7 9-10"`` -> ``[1, 2, 3, 7, 9, 10]`` (order kept, repeats dropped).

    Commas and whitespace both separate items; ``a-b`` is an inclusive range.
    Seeds must be non-negative integers.
    """
    seeds = []
    for item in re.split(r"[,\s]+", text.strip()):
        if not item:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not m:
            raise ValueError(f"not a seed or a range of seeds: {item!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ValueError(f"empty range: {item!r}")
        seeds.extend(range(lo, hi + 1))
    return unique(seeds)


def read_seeds_file(path):
    """The seeds in ``path``: ``parse_seeds`` syntax, one or more per line, ``#`` comments."""
    with open(path, encoding="utf-8") as f:
        text = " ".join(line.split("#", 1)[0] for line in f)
    return parse_seeds(text)


def unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def seeds_digest(seeds):
    """A SHA-256 of the seed list, to identify it without revealing it."""
    return hashlib.sha256(",".join(map(str, seeds)).encode("ascii")).hexdigest()


def inside_repo(path):
    """Whether ``path`` lies inside this repository (False on another drive)."""
    root = os.path.realpath(ROOT)
    try:
        return os.path.commonpath([os.path.realpath(path), root]) == root
    except ValueError:                       # Windows: different drives
        return False


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
    missing the agent or the README. (The write-up and the slides come in a
    separate zip and are not read here.)
    """
    import zipfile

    manifest = []
    zips = sorted(f for f in os.listdir(zip_dir) if f.lower().endswith(".zip"))
    for fname in zips:
        stem = fname[:-4]
        dest = os.path.join(entries_dir, stem)
        os.makedirs(dest, exist_ok=True)
        row = {"zip": fname, "student": stem, "code": [], "weights": [],
               "docs": [], "other": [], "skipped": [], "readme": False}
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
                    if os.path.basename(rel).lower().startswith("readme"):
                        row["readme"] = True
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
        if not row["readme"]:
            missing.append("README")
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
def build_tasks(players, seeds, baseline_vs_baseline):
    """One task per unordered pair: ``(a, b, [(seed, swapped), ...])``.

    Every pair plays every seed twice, once in each seating. A seed fixes the
    game's random setup, so the swap gives both sides the same start, which is
    how the seat advantage is balanced out.
    """
    tasks = []
    names = [p["alias"] for p in players]
    kinds = {p["alias"]: p["kind"] for p in players}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not baseline_vs_baseline and kinds[a] == "baseline" and kinds[b] == "baseline":
                continue
            games = [(s, swapped) for s in seeds for swapped in (False, True)]
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


def vs_baselines(records, entries, baselines):
    """``{alias: {baseline: (wins, games)}}`` -- a draw counts as half a win."""
    aliases = {e["alias"] for e in entries}
    out = {a: {b: [0.0, 0] for b in baselines} for a in aliases}
    for rec in records:
        seats = rec["agents"]
        winners = rec["result"]["winners"]
        for i, name in enumerate(seats):
            other = seats[1 - i]
            if name not in aliases or other not in out[name]:
                continue
            cell = out[name][other]
            cell[1] += 1
            if i in winners:
                cell[0] += 1.0 / len(winners)
    return {a: {b: (w, n) for b, (w, n) in row.items()} for a, row in out.items()}


def build_boards(records, entries, baselines):
    """Student board rows (student games only) plus organizer extras.

    ``points`` is the published "wins": wins plus half the draws. Entries with
    the same points share a rank (1, 2, 2, 4, ...).
    """
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
            "points": w + 0.5 * d,
            "games": n, "win_rate": rate, "ci_low": lo, "ci_high": hi,
            "baseline_wins": bw, "baseline_losses": bl, "baseline_draws": bd,
            "baseline_games": bn, "baseline_win_rate": brate,
            "elo": elo.get(e["alias"]), "timeouts": row["timeouts"],
            "illegal": row["illegal"], "tampered": row["tampered"],
            "mean_game_s": round(row["elapsed"] / row["games"], 1) if row["games"] else 0.0,
        })
    rows.sort(key=lambda r: (-r["points"], r["alias"]))
    for i, r in enumerate(rows):
        r["rank"] = (rows[i - 1]["rank"] if i and r["points"] == rows[i - 1]["points"]
                     else i + 1)
    return rows


def fmt_wins(x):
    return f"{x:g}"


def students_markdown(rows, meta):
    """The published board: rank and wins only, keyed by alias."""
    head = (f"# {meta['label']} -- {meta['date']} -- 2p track\n\n"
            f"{meta['entries']} entries. Every pair played {meta['seeds']} seeds x 2 "
            f"seatings = {meta['games_per_pair']} games, {meta['time_limit']:g} s per "
            "decision. Only games between entries count; a draw counts as half a win. "
            "Results against the baselines are a separate reference "
            "(`board_baselines.md`).\n\n")
    lines = ["| Rank | Agent | Wins |",
             "|-----:|-------|-----:|"]
    for r in rows:
        lines.append(f"| {r['rank']} | {r['alias']} | {fmt_wins(r['points'])} |")
    return head + "\n".join(lines)


def baselines_markdown(rows, table, baselines, meta):
    """The reference board: each entry's wins against each baseline."""
    head = (f"# {meta['label']} -- {meta['date']} -- results against the baselines\n\n"
            "For reference only: these games do not affect the rank. Each cell is "
            "wins / games (a draw counts as half a win), "
            f"{meta['seeds']} seeds x 2 seatings per baseline.\n\n")
    lines = ["| Agent | " + " | ".join(baselines) + " |",
             "|-------|" + "|".join("-----:" for _ in baselines) + "|"]
    for r in rows:
        cells = [f"{fmt_wins(w)} / {n}" if n else "-"
                 for w, n in (table[r["alias"]][b] for b in baselines)]
        lines.append(f"| {r['alias']} | " + " | ".join(cells) + " |")
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
    ap.add_argument("--seeds", default=None,
                    help=f"seeds every pair plays, once in each seating: ranges and lists "
                         f"such as 1-25 or 1-25,40 (default {DEFAULT_SEEDS} when "
                         f"--seeds-file is not given)")
    ap.add_argument("--seeds-file", default=None,
                    help="add the seeds in this file (keep it outside the repository; "
                         "see tools/make_private_seeds.py)")
    ap.add_argument("--label", default="ladder",
                    help="title of the published board, e.g. 'ladder 2' or 'final'")
    ap.add_argument("--baselines", default=DEFAULT_BASELINES,
                    help="comma-separated baseline names, or 'none'")
    ap.add_argument("--baseline-vs-baseline", action="store_true",
                    help="also play baselines against each other (off: they are anchors)")
    ap.add_argument("--workers", type=int, default=0,
                    help="concurrent games (default: min(8, cpu-2)); each one runs "
                         "up to 2 agent processes, so watch memory with torch entries")
    ap.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                    help=f"seconds per decision for an entry (default {DEFAULT_TIME_LIMIT:g})")
    ap.add_argument("--date", default=None, help="output folder name (default: today)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the sandbox check (not recommended)")
    ap.add_argument("--validate-games", type=int, default=2)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-games", action="store_true",
                    help="do not save replays (they are organizer-only either way)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    args = ap.parse_args()

    try:
        seeds = parse_seeds(args.seeds if args.seeds is not None
                            else ("" if args.seeds_file else DEFAULT_SEEDS))
        if args.seeds_file:
            if inside_repo(args.seeds_file):
                print(f"WARNING: {args.seeds_file} is inside the repository; a private "
                      "seed list must never be committed")
            extra = read_seeds_file(args.seeds_file)
            dup = len(set(seeds) & set(extra))
            if dup:
                print(f"note: {dup} seeds in {args.seeds_file} are already in --seeds; "
                      "each is played once")
            seeds = unique(seeds + extra)
    except (OSError, ValueError) as e:
        sys.exit(f"bad seeds: {e}")
    if not seeds:
        sys.exit("no seeds to play")

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
    tasks = build_tasks(players, seeds, args.baseline_vs_baseline)
    total_games = sum(len(t[2]) for t in tasks)
    games_per_pair = 2 * len(seeds)
    student_pairs = len(entries) * (len(entries) - 1) // 2
    workers = args.workers or max(1, min(8, (os.cpu_count() or 4) - 2))

    print(f"\nseeds: {len(seeds)} x 2 seatings = {games_per_pair} games per pair, "
          f"{args.time_limit:g} s per decision (seed list sha256 {seeds_digest(seeds)[:16]}...)")
    print(f"plan: {len(entries)} entries + {len(baselines)} baselines, "
          f"{len(tasks)} pairings, {total_games} games "
          f"({student_pairs * games_per_pair} of them between entries), "
          f"{workers} concurrent")
    print("      estimated wall time at 5 / 15 / 30 / 60 s per game: "
          + " / ".join(f"{total_games * s / workers / 3600:.1f} h" for s in (5, 15, 30, 60)))
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
            tasks = build_tasks(players, seeds, args.baseline_vs_baseline)
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
    # The seed list itself is not written anywhere: the private seeds stay in
    # the organizer's file. The digest identifies which list was played.
    meta = {"label": args.label, "date": date, "entries": len(entries),
            "baselines": baselines, "games": len(records), "seeds": len(seeds),
            "games_per_pair": games_per_pair, "seed_list_sha256": seeds_digest(seeds),
            "time_limit": args.time_limit,
            "wall_minutes": round((time.perf_counter() - started) / 60, 1)}

    md = students_markdown(rows, meta)
    with open(os.path.join(out_dir, "board_students.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    write_csv(os.path.join(out_dir, "board_students.csv"),
              [{"rank": r["rank"], "alias": r["alias"], "wins": fmt_wins(r["points"])}
               for r in rows], ["rank", "alias", "wins"])
    if baselines:
        table = vs_baselines(records, entries, baselines)
        with open(os.path.join(out_dir, "board_baselines.md"), "w", encoding="utf-8") as f:
            f.write(baselines_markdown(rows, table, baselines, meta) + "\n")
    write_csv(os.path.join(out_dir, "board_organizer.csv"), rows,
              ["rank", "student_id", "student_name", "file", "alias", "points", "wins",
               "losses", "draws", "games", "win_rate", "ci_low", "ci_high", "elo",
               "baseline_wins", "baseline_losses", "baseline_draws", "baseline_games",
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
    published = "board_students.md" + (" and board_baselines.md" if baselines else "")
    print(f"\npublish: {out_dir}/{published} (rank and wins only)")
    print(f"keep:    {out_dir}/board_organizer.csv (roster, W/L/D, Elo), validation.txt, "
          f"summary.json and games/ (replays are not published)")
    print(f"took {meta['wall_minutes']} min")


if __name__ == "__main__":
    main()
