"""Play your agent against the baselines with one command.

    python tools/play.py --agent my_agent.py:MyAgent
    python tools/play.py --agent my_agent.py:MyAgent --vs Search --games 20
    python tools/play.py --agent my_agent.py:MyAgent --players 3 --vs ShippingRush,ActionValue
    python tools/play.py --agent my_agent.py:MyAgent --vs all --record

--agent   your agent as  path/to/file.py:ClassName  (or module:ClassName),
          or the name of a bundled baseline (Random, Factory, ShippingRush,
          TradeBuilding, ActionValue, MCTS, SearchLite, Search, PPO)
--vs      opponents, comma-separated, same forms as --agent; "all" = every
          baseline for the track. Default: ActionValue,TradeBuilding (2p) or
          ShippingRush,ActionValue (3p). In the 3p track two opponents fill
          the other seats (one name is used twice).
--games   games per opponent; seats rotate so every seat is played equally.
--record  save every game under results/games/ so you can replay one with
          python tools/replay_game.py <file> --moves

Games are played in-process under the competition rules (1 s per move,
illegal moves and exceptions replaced by a random legal move). The per-move
times printed are your agent's real wall-clock cost, so you can see how close
to the budget you are before you submit. For the official sandbox (your agent
in its own process, killed at the deadline) run tools/validate_submission.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.registry import DEFAULT_OPPONENTS, baseline_names, resolve  # noqa: E402
from puerto_rico.observation import GLOBAL_DIM, PER_PLAYER_DIM  # noqa: E402
from puerto_rico.records import record_from_result, save_record  # noqa: E402
from tournament.match import play_game  # noqa: E402


class Timed:
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


def play_matchup(focal, opponents, num_players, games, seed0=1000,
                 time_limit_s=1.0, record_dir=None):
    """Seat-rotated games of ``focal`` against ``opponents`` (label, factory).

    Returns a stats dict for the focal agent: wins/losses/ties, win%, mean VP
    margin over the best opponent, move-time stats, timeouts/illegal/tampered
    counts, and the saved record paths (when ``record_dir`` is given).
    """
    f_label, f_factory = focal
    wins = losses = ties = 0
    margins, times, to, il, tp, saved = [], [], 0, 0, 0, []
    for g in range(games):
        f_seat = g % num_players
        seats, labels, specs = [], [], []
        opp_iter = iter(opponents[i % len(opponents)] for i in range(num_players - 1))
        for s in range(num_players):
            if s == f_seat:
                label, factory = f_label, f_factory
            else:
                label, factory = next(opp_iter)
            seats.append(Timed(factory()))
            labels.append(label)
            specs.append(label)
        r = play_game(seats, seed=seed0 + g, time_limit_s=time_limit_s)

        to += r["timeouts"][f_seat]
        il += r["illegal"][f_seat]
        tp += r["tampered"][f_seat]
        times.extend(seats[f_seat].times)
        others = [r["scores"][s] for s in range(num_players) if s != f_seat]
        margins.append(r["scores"][f_seat] - max(others))
        w = set(r["winners"])
        if f_seat in w and len(w) > 1:
            ties += 1
        elif f_seat in w:
            wins += 1
        else:
            losses += 1
        if record_dir and not r["truncated"]:
            rec = record_from_result(r, num_players, player_types=specs,
                                     player_labels=labels)
            saved.append(save_record(rec, record_dir))
    t = np.asarray(times) if times else np.zeros(1)
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "winpct": 100 * (wins + 0.5 * ties) / max(1, games),
        "margin": float(np.mean(margins)) if margins else 0.0,
        "t_mean_ms": 1000 * t.mean(), "t_p95_ms": 1000 * np.percentile(t, 95),
        "t_max_ms": 1000 * t.max(),
        "timeouts": to, "illegal": il, "tampered": tp, "records": saved,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="file.py:Class, module:Class, or a baseline name")
    ap.add_argument("--vs", default="", help="comma-separated opponents, or 'all'")
    ap.add_argument("--players", type=int, default=2, choices=(2, 3), help="track: 2 or 3")
    ap.add_argument("--games", type=int, default=10, help="games per opponent")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=1.0, help="seconds per move")
    ap.add_argument("--record", action="store_true", help="save game records for replay")
    ap.add_argument("--record-dir", default=os.path.join("results", "games"))
    args = ap.parse_args()

    n = args.players
    try:
        focal = resolve(args.agent, base_dir=os.getcwd())
    except Exception as e:
        sys.exit(f"cannot load --agent {args.agent!r}: {e}")
    if args.vs.strip().lower() == "all":
        opp_specs = [b for b in baseline_names(n) if b.lower() != focal[0].lower()]
    else:
        opp_specs = [o for o in args.vs.split(",") if o.strip()] or DEFAULT_OPPONENTS[n]
    try:
        opponents = [resolve(o, base_dir=os.getcwd()) for o in opp_specs]
    except Exception as e:
        sys.exit(f"cannot load an opponent: {e}")

    obs_len = GLOBAL_DIM + PER_PLAYER_DIM * n
    print(f"Agent : {focal[0]}  ({args.agent})")
    print(f"Track : {n}p (observation length {obs_len}), {args.games} games per "
          f"opponent, seats rotated, {args.time_limit:g} s/move\n")
    print(f"{'opponent':<18}{'win%':>6}{'W-L-T':>10}{'VP diff':>9}"
          f"{'ms/move':>9}{'p95':>8}{'max':>8}{'to/il/tp':>10}")

    totals = {"wins": 0, "losses": 0, "ties": 0, "games": 0, "timeouts": 0,
              "illegal": 0, "tampered": 0}
    records = []
    record_dir = args.record_dir if args.record else None
    for i, opp in enumerate(opponents):
        if n == 2:
            opps = [opp]
        else:                                    # 3p: the other seats
            opps = [opp, opponents[(i + 1) % len(opponents)]] if len(opponents) > 1 else [opp, opp]
        d = play_matchup(focal, opps, n, args.games, seed0=args.seed + 1000 * i,
                         time_limit_s=args.time_limit, record_dir=record_dir)
        name = opp[0] if n == 2 else " + ".join(o[0] for o in opps)
        print(f"{name:<18}{d['winpct']:>5.0f}%{d['wins']:>4}-{d['losses']}-{d['ties']:<3}"
              f"{d['margin']:>+9.1f}{d['t_mean_ms']:>9.1f}{d['t_p95_ms']:>8.1f}"
              f"{d['t_max_ms']:>8.0f}{str(d['timeouts'])+'/'+str(d['illegal'])+'/'+str(d['tampered']):>10}")
        for k in ("wins", "losses", "ties", "timeouts", "illegal", "tampered"):
            totals[k] += d[k]
        totals["games"] += args.games
        records.extend(d["records"])

    g = max(1, totals["games"])
    print(f"{'overall':<18}{100 * (totals['wins'] + 0.5 * totals['ties']) / g:>5.0f}%"
          f"{totals['wins']:>4}-{totals['losses']}-{totals['ties']:<3}")
    if totals["timeouts"] or totals["illegal"] or totals["tampered"]:
        print(f"\nWARNING: {totals['timeouts']} timeouts, {totals['illegal']} illegal/"
              f"raising moves, {totals['tampered']} tampering breaches were replaced by "
              f"random moves - each one costs you in the real tournament.")
    if records:
        print(f"\n{len(records)} game records saved under {args.record_dir}/ - replay one with\n"
              f"    python tools/replay_game.py {records[-1]} --moves")


if __name__ == "__main__":
    main()
