"""
tournament/runner.py — schedule games and aggregate the rankings.

The pool is a dict ``{name: factory}`` where ``factory()`` returns a fresh
:class:`~agents.base.Agent` (a factory, not a shared instance, so every game
starts from a clean state and is reproducible).

`run_round_robin` plays a seat-balanced round-robin and feeds the win-rate
(official) and TrueSkill rankers. `estimate_payoff_matrix` builds the monomorphic
payoff matrix used by α-Rank. `run_tournament` does both and returns a combined
result for `tournament.leaderboard`.
"""
import itertools

from tournament.match import play_game
from tournament.rankers.alpha_rank import rank_alpha
from tournament.rankers.trueskill_ranker import rank_trueskill
from tournament.rankers.win_rate import rank_win_rate

N_SEATS = 3


def run_round_robin(pool, games_per_seating: int = 1, seed: int = 0,
                    allow_repeats: bool = True, time_limit_s: float = 1.0,
                    verbose: bool = False):
    """Play every group of 3 agents over all distinct seatings.

    Args:
        pool: ``{name: factory}``.
        games_per_seating: games per distinct seating (different seed each).
        seed: base seed; each game gets a unique derived seed.
        allow_repeats: include groups where an agent appears more than once
            (e.g. ``[A, A, B]``); set False for distinct triples only.
        time_limit_s: per-move budget passed to :func:`play_game`.

    Returns:
        list of ``{"agents": [...], "result": {...}}`` match records.
    """
    names = list(pool)
    groups = (itertools.combinations_with_replacement(names, N_SEATS)
              if allow_repeats else itertools.combinations(names, N_SEATS))

    records = []
    counter = 0
    for group in groups:
        for seating in sorted(set(itertools.permutations(group))):
            for _ in range(games_per_seating):
                agents = [pool[name]() for name in seating]
                result = play_game(agents, seed=seed + counter,
                                   time_limit_s=time_limit_s)
                records.append({"agents": list(seating), "result": result})
                counter += 1
        if verbose:
            print(f"  played group {group}: {counter} games so far")
    return records


def estimate_payoff_matrix(pool, games_per_pair: int = 4, seed: int = 10_000,
                           time_limit_s: float = 1.0):
    """Estimate ``M[s][r]`` = win rate of focal ``s`` against a field of ``r``.

    For each ordered pair (s, r) we play ``[s, r, r]`` and record how often the
    focal seat 0 (strategy s) wins. Used by α-Rank.

    Returns:
        ``(names, M)`` with ``names`` a list and ``M`` a ``K x K`` numpy array.
    """
    import numpy as np

    names = list(pool)
    K = len(names)
    M = np.zeros((K, K), dtype=np.float64)
    counter = 0
    for si, s in enumerate(names):
        for ri, r in enumerate(names):
            wins = 0.0
            for _ in range(games_per_pair):
                agents = [pool[s](), pool[r](), pool[r]()]
                result = play_game(agents, seed=seed + counter,
                                   time_limit_s=time_limit_s)
                if 0 in result["winners"]:
                    wins += 1.0 / len(result["winners"])
                counter += 1
            M[si, ri] = wins / games_per_pair
    return names, M


def run_tournament(pool, games_per_seating: int = 1, seed: int = 0,
                   compute_alpha_rank: bool = True, alpha: float = 1.0,
                   alpha_m: int = 50, alpha_games_per_pair: int = 4,
                   time_limit_s: float = 1.0, verbose: bool = False):
    """Run the full tournament and compute all three rankings.

    Returns:
        dict with ``win_rate``, ``trueskill``, and (optionally) ``alpha_rank``
        ranking row-lists, plus the raw ``records``.
    """
    names = list(pool)
    if verbose:
        print(f"Round-robin over {len(names)} agents...")
    records = run_round_robin(pool, games_per_seating=games_per_seating,
                              seed=seed, time_limit_s=time_limit_s, verbose=verbose)

    out = {
        "agents": names,
        "records": records,
        "win_rate": rank_win_rate(records, names),
        "trueskill": rank_trueskill(records, names),
    }

    if compute_alpha_rank:
        if verbose:
            print("Estimating payoff matrix for α-Rank...")
        a_names, M = estimate_payoff_matrix(
            pool, games_per_pair=alpha_games_per_pair, seed=seed + 1_000_000,
            time_limit_s=time_limit_s)
        out["alpha_rank"] = rank_alpha(a_names, M, alpha=alpha, m=alpha_m)
        out["payoff_matrix"] = M.tolist()

    return out
