"""Run a small round-robin tournament for each track and print the boards.

Run from the repo root:  python examples/run_tournament.py

The competition has two independent tracks, each with its own leaderboard:

  * 2p track — 1-vs-1 Puerto Rico  (``n_seats=2``)
  * 3p track — 3-player Puerto Rico (``n_seats=3``)

Each track is ranked by its **official** metric — **Elo** in the 2p track,
**TrueSkill** in the 3p track — with **win rate** shown alongside. Both boards are
printed and written to results/2p/leaderboard.{md,csv,json} and
results/3p/leaderboard.{md,csv,json}. Add your own agent to `pool` to see how it
stacks up. (α-Rank is opt-in analysis — pass ``compute_alpha_rank=True`` if you
want it; it is not part of the standings.)
"""
import os
import sys

# Make the repo root importable when run as a script from the examples/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (ActionValueAgent, FactoryAgent, MctsAgent, RandomAgent,
                    ShippingRushAgent, TradeBuildingAgent)
from agents.search_agent import SearchAgent, SearchLiteAgent  # noqa: F401 (used if pool lines below are uncommented)
from tournament.leaderboard import save
from tournament.runner import run_tournament


def make_pool():
    # A fresh factory per game keeps every match reproducible. The same pool
    # plays both tracks here; for the real competition each track has its own
    # submissions. (PpoAgent is omitted: its bundled checkpoint is 3p-only.)
    return {
        "Random":        lambda: RandomAgent(),
        "ActionValue":   lambda: ActionValueAgent(),
        "ShippingRush":  lambda: ShippingRushAgent(),
        "TradeBuilding": lambda: TradeBuildingAgent(),
        "Factory":       lambda: FactoryAgent(),
        # MCTS is strong but slow — uncomment to include it (the round-robin
        # will take noticeably longer):
        # "MCTS":        lambda: MctsAgent(num_simulations=30, max_rollout_depth=40),
        # 2p-track alpha-beta baselines (deterministic; slow in a round-robin —
        # in a 3p game they fall back to a reactive policy). Uncomment to rank
        # against them; see docs/SEARCH_BASELINE_2P.md.
        # "SearchLite":  lambda: SearchLiteAgent(),
        # "Search":      lambda: SearchAgent(),
    }


def run_track(n_seats: int):
    track = f"{n_seats}p"
    print(f"\n===== {track} track ({n_seats}-player) =====")
    result = run_tournament(
        make_pool(),
        games_per_seating=1,        # raise for tighter confidence intervals
        seed=0,
        time_limit_s=1.0,
        verbose=True,
        n_seats=n_seats,
    )
    out = save(result, out_dir=os.path.join("results", track))
    print("\n" + out["markdown"])
    print(f"\nSaved to results/{track}/leaderboard.{{md,csv,json}}")


def main():
    run_track(2)        # 1-vs-1 track
    run_track(3)        # 3-player track


if __name__ == "__main__":
    main()
