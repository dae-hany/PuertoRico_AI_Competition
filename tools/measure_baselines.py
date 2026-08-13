"""tools/measure_baselines.py — regenerate the baseline numbers quoted in the docs.

The README and the submission guide make strength claims ("strongest in 2p",
"beats every other baseline"). Those claims go stale the moment a baseline is
improved, so this script re-measures them instead of anyone re-guessing:

* a **seat-balanced round-robin** over the fast heuristic pool, per track,
  scored with the track's official metric (Elo in 2p, TrueSkill in 3p);
* **seat-balanced duels** for the expensive planners (MCTS, Search, PPO), which
  cannot sit in a full round-robin at a useful sample size.

Search-based strengths are only defined **at a stated compute budget** — an
agent that searches twice as long is a different opponent — so the budget is
part of every planner's name in the output.

Usage (from the repo root)::

    python tools/measure_baselines.py                    # everything, 12 workers
    python tools/measure_baselines.py --workers 4 --quick # smaller samples
    python tools/measure_baselines.py --skip-planners     # heuristics only (fast)

Runs games in parallel processes; each worker is pinned to a single BLAS thread
so a large run stays polite on a shared machine.
"""
import argparse
import itertools
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPO_CHECKPOINT = os.path.join(ROOT, "training", "checkpoints", "ppo_baseline.pt")

# Stated compute budgets for the planning baselines. These are part of their
# identity: quoting "MCTS beats X" without the budget is meaningless.
MCTS_SIMS = 60
MCTS_ROLLOUT_DEPTH = 40

HEURISTICS = ["Random", "Factory", "TradeBuilding", "ShippingRush", "ActionValue"]


def make_agent(name):
    """Build one agent by short name (workers call this, so it must be top-level)."""
    from agents import (ActionValueAgent, FactoryAgent, MctsAgent, RandomAgent,
                        SearchAgent, SearchLiteAgent, ShippingRushAgent,
                        TradeBuildingAgent)
    if name == "Random":
        return RandomAgent(seed=0)
    if name == "Factory":
        return FactoryAgent()
    if name == "TradeBuilding":
        return TradeBuildingAgent()
    if name == "ShippingRush":
        return ShippingRushAgent()
    if name == "ActionValue":
        return ActionValueAgent()
    if name == f"MCTS{MCTS_SIMS}":
        return MctsAgent(num_simulations=MCTS_SIMS,
                         max_rollout_depth=MCTS_ROLLOUT_DEPTH)
    if name == "SearchLite250":
        return SearchLiteAgent()
    if name == "Search1500":
        return SearchAgent()
    if name == "PPO":
        from agents.ppo_agent import PpoAgent
        return PpoAgent(PPO_CHECKPOINT)
    raise KeyError(name)


# ── unit jobs (run in worker processes) ──────────────────────────────────────

def _play_group(args):
    """Every seating of one round-robin group."""
    from tournament.match import play_game
    group, n_seats, games_per_seating, base_seed = args

    records = []
    counter = 0
    for seating in sorted(set(itertools.permutations(group))):
        for _ in range(games_per_seating):
            agents = [make_agent(n) for n in seating]
            seed = base_seed + counter
            result = play_game(agents, seed=seed, time_limit_s=120.0)
            records.append({"agents": list(seating), "result": result, "seed": seed})
            counter += 1
    return records


def _duel(args):
    """Seat-balanced head-to-head: focal vs a field of one opponent.

    Stops early if the cell runs past ``budget_s`` (after a minimum sample), and
    reports how many games it actually played. A search agent costs ~75 s a game,
    so without a bound one cell decides how long the whole report takes — and a
    silently truncated cell would be worse than a small one honestly labelled.
    """
    from tournament.match import play_game
    focal, opponent, n_seats, n_games, base_seed, budget_s = args

    min_games = min(n_games, 2 * n_seats)        # keep the seat rotation balanced
    deadline = time.perf_counter() + budget_s if budget_s else None

    wins = 0.0
    margins = []
    for g in range(n_games):
        if (deadline is not None and len(margins) >= min_games
                and time.perf_counter() > deadline):
            break
        seating = [opponent] * n_seats
        seat = g % n_seats                       # rotate the focal seat
        seating[seat] = focal
        agents = [make_agent(n) for n in seating]
        result = play_game(agents, seed=base_seed + g, time_limit_s=120.0)
        if seat in result["winners"]:
            wins += 1.0 / len(result["winners"])
        others = [v for i, v in enumerate(result["scores"]) if i != seat]
        margins.append(result["scores"][seat] - max(others))

    played = len(margins)
    return {"focal": focal, "opponent": opponent, "n_seats": n_seats,
            "games": played, "win_rate": wins / played,
            "mean_margin": sum(margins) / played}


# ── reporting ────────────────────────────────────────────────────────────────

def round_robin_table(records, names, n_seats):
    from tournament.leaderboard import build_leaderboard, to_markdown
    from tournament.rankers.elo import rank_elo
    from tournament.rankers.trueskill_ranker import rank_trueskill
    from tournament.rankers.win_rate import rank_win_rate

    records = sorted(records, key=lambda r: r["seed"])      # order-independent output
    result = {"agents": names, "records": records, "n_seats": n_seats,
              "win_rate": rank_win_rate(records, names),
              "trueskill": rank_trueskill(records, names)}
    if n_seats == 2:
        result["elo"] = rank_elo(records, names)
        result["official_metric"] = "elo"
    else:
        result["official_metric"] = "trueskill"
    return to_markdown(build_leaderboard(result))


def duel_table(rows, n_seats):
    """One row per planner; each cell is ``win% (games)`` — the sample size is
    part of the number, since the slow cells play fewer games."""
    opponents = HEURISTICS
    by_focal = {}
    for row in rows:
        by_focal.setdefault(row["focal"], {})[row["opponent"]] = row

    lines = ["| Agent | " + " | ".join(f"vs {o}" for o in opponents) + " |",
             "|-------|" + "|".join(["------:"] * len(opponents)) + "|"]
    for focal, cells in by_focal.items():
        rates = [f"{cells[o]['win_rate'] * 100:.0f}% ({cells[o]['games']})"
                 if o in cells else "—" for o in opponents]
        lines.append(f"| `{focal}` | " + " | ".join(rates) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--quick", action="store_true",
                        help="smaller samples; for a smoke test, not for the docs")
    parser.add_argument("--skip-planners", action="store_true",
                        help="heuristic round-robins only (a couple of minutes)")
    parser.add_argument("--cell-budget", type=float, default=600.0,
                        help="wall-clock cap per duel cell; bounds the whole run")
    # Tracked, not written into the gitignored results/ dir: the numbers the
    # README and the submission guide quote should be versioned next to them.
    parser.add_argument("--out", default="docs/BASELINES.md")
    args = parser.parse_args()

    rr_2p = 4 if args.quick else 20          # games per seating
    rr_3p = 2 if args.quick else 8
    duel_n = 6 if args.quick else 24         # games per duel cell

    started = time.time()
    jobs_rr, jobs_duel = [], []

    for n_seats, per_seating, base in ((2, rr_2p, 100_000), (3, rr_3p, 300_000)):
        for i, group in enumerate(
                itertools.combinations_with_replacement(HEURISTICS, n_seats)):
            jobs_rr.append((group, n_seats, per_seating, base + i * 1_000))

    if not args.skip_planners:
        planners = {
            2: [f"MCTS{MCTS_SIMS}", "SearchLite250", "Search1500"],
            3: [f"MCTS{MCTS_SIMS}", "PPO"],
        }
        # The slow cells get fewer games; the table prints the count per cell.
        cheap = {"PPO", "SearchLite250"}
        for n_seats, focals in planners.items():
            for focal in focals:
                if focal == "PPO" and not os.path.exists(PPO_CHECKPOINT):
                    continue
                n = duel_n if focal in cheap else max(6, duel_n // 3)
                for j, opponent in enumerate(HEURISTICS):
                    # A stable seed: Python's hash() of a str is salted per
                    # process, so using it here would silently reseed the whole
                    # table on every run.
                    tag = zlib.crc32(f"{focal}|{opponent}|{n_seats}".encode())
                    jobs_duel.append((focal, opponent, n_seats, n,
                                      500_000 + tag % 10_000, args.cell_budget))

    print(f"{len(jobs_rr)} round-robin groups + {len(jobs_duel)} duel cells "
          f"on {args.workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Longest jobs first: a duel cell costs minutes and a round-robin group
        # costs seconds, so submitting the duels last leaves them as a tail that
        # runs almost serially while the rest of the pool sits idle.
        duel_futures = [(job[2], pool.submit(_duel, job)) for job in jobs_duel]
        rr_futures = [(job[1], pool.submit(_play_group, job)) for job in jobs_rr]

        records = {2: [], 3: []}
        for n_seats, future in rr_futures:
            records[n_seats].extend(future.result())
        duels = {2: [], 3: []}
        for n_seats, future in duel_futures:
            duels[n_seats].append(future.result())

    out = [f"# Baseline strengths (generated by `tools/measure_baselines.py`)",
           "",
           f"Seat-balanced. Ties split the win. Planner budgets: "
           f"MCTS = {MCTS_SIMS} simulations / rollout depth {MCTS_ROLLOUT_DEPTH}, "
           f"SearchLite = 250 nodes, Search = 1500 nodes.", ""]
    for n_seats in (2, 3):
        label = "2p (1-vs-1)" if n_seats == 2 else "3p"
        out += [f"## {label} — heuristic round-robin "
                f"({len(records[n_seats])} games)", "",
                round_robin_table(records[n_seats], HEURISTICS, n_seats), ""]
        if duels[n_seats]:
            out += [f"### {label} — planners vs the heuristics (win rate)", "",
                    duel_table(duels[n_seats], n_seats), ""]

    text = "\n".join(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)
    print(f"\nWritten to {args.out} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
