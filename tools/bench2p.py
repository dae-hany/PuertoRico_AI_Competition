"""tools/bench2p.py — fast head-to-head benchmarking for the 2p (1-vs-1) track.

A development tool (not part of the competition). It plays seat-balanced duels
between two agents and reports, per matchup:

    win% (ties count as 0.5), mean final VP margin, per-move time (mean / p95 /
    max), and any timeouts / illegal moves.

Move timing matters: a search agent must stay under the 1 s/move budget, and
this surfaces the real wall-clock cost so you catch overruns before they become
random-fallback timeouts on the leaderboard.

Usage (from repo root, with the venv python):

    python tools/bench2p.py                      # default agent vs the baseline pool
    python tools/bench2p.py --agent Search --n 20
    python tools/bench2p.py --agent Search --opponents TradeBuilding,ActionValue --n 30

For the official Elo board, use examples/run_tournament.py (n_seats=2) instead.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (ActionValueAgent, FactoryAgent, RandomAgent,  # noqa: E402
                    ShippingRushAgent, TradeBuildingAgent)
from tournament.match import play_game  # noqa: E402

# name -> zero-arg factory. New agents (e.g. the search agent) register here.
REGISTRY = {
    "Random": lambda: RandomAgent(),
    "Factory": lambda: FactoryAgent(),
    "ShippingRush": lambda: ShippingRushAgent(),
    "TradeBuilding": lambda: TradeBuildingAgent(),
    "ActionValue": lambda: ActionValueAgent(),
}

try:  # the search agents are added as they are built; keep the tool usable before then
    from agents.search_agent import SearchAgent, SearchLiteAgent
    REGISTRY["Search"] = lambda: SearchAgent()
    REGISTRY["SearchLite"] = lambda: SearchLiteAgent()
except Exception:
    pass


class _Timed:
    """Wraps an agent to record its per-move act() wall-clock times."""

    def __init__(self, agent):
        self.agent = agent
        self.name = getattr(agent, "name", agent.__class__.__name__)
        self.times = []

    def on_game_start(self, forward_model=None):
        self.agent.on_game_start(forward_model)

    def act(self, observation, action_mask):
        t0 = time.perf_counter()
        a = self.agent.act(observation, action_mask)
        self.times.append(time.perf_counter() - t0)
        return a


def duel(a_name, b_name, n, seed0=1000, time_limit_s=1.0):
    wa = wb = tie = 0
    margins, a_times, to, il = [], [], 0, 0
    for g in range(n):
        swap = g % 2 == 1                      # seat-balance: alternate seats
        fa, fb = REGISTRY[a_name], REGISTRY[b_name]
        agents = [_Timed(fb()), _Timed(fa())] if swap else [_Timed(fa()), _Timed(fb())]
        ai, bi = (1, 0) if swap else (0, 1)
        r = play_game(agents, seed=seed0 + g, time_limit_s=time_limit_s)
        to += r["timeouts"][ai]
        il += r["illegal"][ai]
        a_times.extend(agents[ai].times)
        margins.append(r["scores"][ai] - r["scores"][bi])
        w = set(r["winners"])
        if ai in w and bi in w:
            tie += 1
        elif ai in w:
            wa += 1
        else:
            wb += 1
    t = np.asarray(a_times) if a_times else np.zeros(1)
    return {
        "win": wa, "loss": wb, "tie": tie,
        "winpct": 100 * (wa + 0.5 * tie) / max(1, n),
        "margin": float(np.mean(margins)),
        "t_mean": t.mean() * 1000, "t_p95": np.percentile(t, 95) * 1000,
        "t_max": t.max() * 1000, "timeouts": to, "illegal": il,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="ActionValue", help="focal agent name (REGISTRY key)")
    ap.add_argument("--opponents", default="", help="comma-separated; default = all others")
    ap.add_argument("--n", type=int, default=20, help="games per matchup (seat-balanced)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=1.0)
    args = ap.parse_args()

    if args.agent not in REGISTRY:
        sys.exit(f"unknown agent {args.agent!r}; known: {', '.join(REGISTRY)}")
    opps = ([o for o in args.opponents.split(",") if o]
            if args.opponents else [k for k in REGISTRY if k != args.agent])

    print(f"Focal: {args.agent}   games/matchup: {args.n} (seat-balanced)\n")
    print(f"{'opponent':<16}{'win%':>6}{'W-L-T':>10}{'VPd':>7}"
          f"{'t_mean':>9}{'t_p95':>8}{'t_max':>8}{'to/il':>8}")
    for opp in opps:
        if opp not in REGISTRY:
            print(f"{opp:<16}  (unknown, skipped)")
            continue
        d = duel(args.agent, opp, args.n, seed0=args.seed, time_limit_s=args.time_limit)
        print(f"{opp:<16}{d['winpct']:>5.0f}%{d['win']:>4}-{d['loss']}-{d['tie']:<3}"
              f"{d['margin']:>7.1f}{d['t_mean']:>8.1f}m{d['t_p95']:>7.0f}m"
              f"{d['t_max']:>7.0f}m{str(d['timeouts'])+'/'+str(d['illegal']):>8}")


if __name__ == "__main__":
    main()
