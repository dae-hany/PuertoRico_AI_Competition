"""Engine micro/macro benchmarks for a search / MCTS workload (2p track).

Measures the operations that set the self-play/search budget:

  clone_live   ForwardModel.clone() of the live model (deepcopy + determinize)
  clone_child  clone() of a clone (deepcopy only) — the in-tree operation
  fm_mask      ForwardModel.action_mask()  (the mask-only fast path)
  env_mask     env.unwrapped.valid_action_mask()  (pure mask computation)
  fm_obs       ForwardModel.observation()  (observe + flatten)
  step         ForwardModel.step() through random games (steps/s, games/s)
  mcts_proxy   clone root + walk D random steps querying the mask each ply
               (sims/s — the number that becomes your search budget)

Run before and after any engine performance work:

    python tools/bench_engine.py --label baseline
    python tools/bench_engine.py --label after --json_out results/bench_history.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from puerto_rico import make_env
from puerto_rico.forward_model import ForwardModel


def _mid_game_model(seed: int, warm_steps: int = 60) -> ForwardModel:
    """A live model advanced into the mid-game (richer state than move 0)."""
    env = make_env(seed=seed, num_players=2)
    fm = ForwardModel(env)
    rng = np.random.default_rng(seed)
    for _ in range(warm_steps):
        if fm.is_terminal():
            break
        legal = fm.legal_actions()
        # Live model must not be mutated by agents; the bench owns it, so
        # stepping it directly here is fine (this is the organizer-side use).
        fm.step(int(rng.choice(legal)))
    return fm


def _time_op(fn, n: int) -> dict:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    total = sum(times)
    return {
        "n": n,
        "mean_ms": 1000 * total / n,
        "p95_ms": 1000 * sorted(times)[int(0.95 * (n - 1))],
        "ops_per_s": n / total if total > 0 else float("inf"),
    }


def bench_clone(fm: ForwardModel, n: int) -> tuple[dict, dict]:
    live = _time_op(fm.clone, n)
    root = fm.clone()
    child = _time_op(root.clone, n)
    return live, child


def bench_masks(fm: ForwardModel, n: int) -> tuple[dict, dict, dict]:
    root = fm.clone()
    env = root.env  # on a clone this is the real underlying env
    fm_mask = _time_op(root.action_mask, n)
    env_mask = _time_op(env.unwrapped.valid_action_mask, n)
    fm_obs = _time_op(root.observation, n)
    return fm_mask, env_mask, fm_obs


def bench_random_games(seed: int, n_games: int) -> dict:
    rng = np.random.default_rng(seed)
    steps = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        fm = ForwardModel(make_env(seed=seed + g, num_players=2))
        while not fm.is_terminal():
            legal = fm.legal_actions()
            fm.step(int(rng.choice(legal)))
            steps += 1
            if steps > 100_000:  # safety net
                break
    dt = time.perf_counter() - t0
    return {
        "games": n_games,
        "steps": steps,
        "steps_per_s": steps / dt,
        "games_per_s": n_games / dt,
        "s_per_game": dt / n_games,
    }


def bench_mcts_proxy(fm: ForwardModel, n_sims: int, depth: int, seed: int) -> dict:
    """One 'simulation' = clone the root, then walk `depth` random plies,
    reading the action mask at every ply (what tree descent + expansion pay)."""
    rng = np.random.default_rng(seed)
    root = fm.clone()
    t0 = time.perf_counter()
    done = 0
    for _ in range(n_sims):
        sim = root.clone()
        for _ in range(depth):
            if sim.is_terminal():
                break
            legal = sim.legal_actions()
            sim.step(int(rng.choice(legal)))
        done += 1
    dt = time.perf_counter() - t0
    return {
        "sims": done,
        "depth": depth,
        "sims_per_s": done / dt,
        "ms_per_sim": 1000 * dt / done,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300, help="reps for micro benches")
    ap.add_argument("--games", type=int, default=10, help="random games for macro bench")
    ap.add_argument("--sims", type=int, default=200, help="sims for the MCTS proxy")
    ap.add_argument("--depth", type=int, default=10, help="plies per proxy sim")
    ap.add_argument("--label", type=str, default="run", help="tag stored in the JSON line")
    ap.add_argument("--json_out", type=str, default=None,
                    help="append a JSON line with all numbers to this file")
    args = ap.parse_args()

    fm = _mid_game_model(args.seed)

    clone_live, clone_child = bench_clone(fm, args.n)
    fm_mask, env_mask, fm_obs = bench_masks(fm, args.n)
    games = bench_random_games(args.seed, args.games)
    proxy = bench_mcts_proxy(fm, args.sims, args.depth, args.seed)

    rows = [
        ("clone_live", clone_live), ("clone_child", clone_child),
        ("fm_mask", fm_mask), ("env_mask", env_mask), ("fm_obs", fm_obs),
    ]
    print(f"\n== bench_engine [{args.label}] (2p, seed={args.seed}) ==")
    print(f"{'op':<12}{'mean ms':>10}{'p95 ms':>10}{'ops/s':>12}")
    for name, r in rows:
        print(f"{name:<12}{r['mean_ms']:>10.3f}{r['p95_ms']:>10.3f}{r['ops_per_s']:>12.0f}")
    print(f"\nrandom games : {games['steps_per_s']:.0f} steps/s, "
          f"{games['games_per_s']:.2f} games/s ({games['s_per_game']:.2f} s/game)")
    print(f"mcts proxy   : {proxy['sims_per_s']:.1f} sims/s "
          f"({proxy['ms_per_sim']:.2f} ms/sim @ depth {proxy['depth']})")

    if args.json_out:
        record = {
            "label": args.label, "seed": args.seed,
            "clone_live": clone_live, "clone_child": clone_child,
            "fm_mask": fm_mask, "env_mask": env_mask, "fm_obs": fm_obs,
            "random_games": games, "mcts_proxy": proxy,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
