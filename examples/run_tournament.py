"""Run a small round-robin tournament between the baselines and print the board.

Run from the repo root:  python examples/run_tournament.py

The leaderboard (Win% official, plus TrueSkill and α-Rank) is printed and also
written to results/leaderboard.{md,csv,json}. Add your own agent to `pool` to
see how it stacks up. MCTS is included at a small simulation budget so the demo
stays fast; raise `num_simulations` for a stronger (slower) opponent.
"""
import os
import sys

# Make the repo root importable when run as a script from the examples/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (ActionValueAgent, FactoryAgent, MctsAgent, RandomAgent,
                    ShippingRushAgent, TradeBuildingAgent)
from tournament.leaderboard import save
from tournament.runner import run_tournament


def main():
    pool = {
        "Random":        lambda: RandomAgent(),
        "ActionValue":   lambda: ActionValueAgent(),
        "ShippingRush":  lambda: ShippingRushAgent(),
        "TradeBuilding": lambda: TradeBuildingAgent(),
        "Factory":       lambda: FactoryAgent(),
        # MCTS is strong but slow — uncomment to include it (the round-robin
        # will take noticeably longer):
        # "MCTS":        lambda: MctsAgent(num_simulations=30, max_rollout_depth=40),
        # "PPO":         lambda: PpoAgent("training/checkpoints/ppo_baseline.pt"),
    }

    result = run_tournament(
        pool,
        games_per_seating=1,        # raise for tighter confidence intervals
        seed=0,
        compute_alpha_rank=True,
        alpha_games_per_pair=4,
        time_limit_s=1.0,
        verbose=True,
    )

    out = save(result, out_dir="results")
    print("\n" + out["markdown"])
    print("\nSaved to results/leaderboard.{md,csv,json}")


if __name__ == "__main__":
    main()
