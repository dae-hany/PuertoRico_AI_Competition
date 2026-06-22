"""
tournament/runner.py — schedule games and aggregate the rankings.

The pool is a dict ``{name: factory}`` where ``factory()`` returns a fresh
:class:`~agents.base.Agent` (a factory, not a shared instance, so every game
starts from a clean state and is reproducible).

`run_round_robin` plays a seat-balanced round-robin and feeds the win-rate
(official) and TrueSkill rankers. `estimate_payoff_matrix` builds the monomorphic
payoff matrix used by α-Rank. `run_tournament` does both and returns a combined
result for `tournament.leaderboard`.

The competition has **two independent tracks** with separate leaderboards:
``n_seats=2`` (the 2p / 1-vs-1 track) and ``n_seats=3`` (the 3p track). Every
function below takes ``n_seats`` so the same machinery runs either track; it
defaults to 3 to keep the historical 3-player behaviour.
"""
import itertools

from tournament.match import play_game
from tournament.rankers.alpha_rank import rank_alpha
from tournament.rankers.elo import rank_elo
from tournament.rankers.trueskill_ranker import rank_trueskill
from tournament.rankers.win_rate import rank_win_rate

N_SEATS = 3          # default seats per game (3p track); the 2p track uses 2


def run_round_robin(pool, games_per_seating: int = 1, seed: int = 0,
                    allow_repeats: bool = True, time_limit_s: float = 1.0,
                    verbose: bool = False, n_seats: int = N_SEATS):
    """Play every group of ``n_seats`` agents over all distinct seatings.

    Args:
        pool: ``{name: factory}``.
        games_per_seating: games per distinct seating (different seed each).
        seed: base seed; each game gets a unique derived seed.
        allow_repeats: include groups where an agent appears more than once
            (e.g. ``[A, A, B]``); set False for distinct groups only.
        time_limit_s: per-move budget passed to :func:`play_game`.
        n_seats: seats per game — ``2`` for the 1-vs-1 track, ``3`` for the 3p track.

    Returns:
        list of ``{"agents": [...], "result": {...}}`` match records.
    """
    names = list(pool)
    groups = (itertools.combinations_with_replacement(names, n_seats)
              if allow_repeats else itertools.combinations(names, n_seats))

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
                           time_limit_s: float = 1.0, n_seats: int = N_SEATS):
    """Estimate ``M[s][r]`` = win rate of focal ``s`` against a field of ``r``.

    For each ordered pair (s, r) we play a game whose seat 0 is the focal
    strategy ``s`` and whose remaining ``n_seats - 1`` seats are all ``r``
    (``[s, r, r]`` in the 3p track, the head-to-head ``[s, r]`` in the 2p track),
    and record how often seat 0 wins. Used by α-Rank.

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
                agents = [pool[s]()] + [pool[r]() for _ in range(n_seats - 1)]
                result = play_game(agents, seed=seed + counter,
                                   time_limit_s=time_limit_s)
                if 0 in result["winners"]:
                    wins += 1.0 / len(result["winners"])
                counter += 1
            M[si, ri] = wins / games_per_pair
    return names, M


def run_tournament(pool, games_per_seating: int = 1, seed: int = 0,
                   compute_alpha_rank: bool = False, alpha: float = 1.0,
                   alpha_m: int = 50, alpha_games_per_pair: int = 4,
                   time_limit_s: float = 1.0, verbose: bool = False,
                   n_seats: int = N_SEATS):
    """Run the full tournament for one track and compute its rankings.

    The **official** metric depends on the track: **Elo** for the 2p (1-vs-1)
    track, **TrueSkill** for the 3p track. **Win rate** is always computed too.
    **α-Rank** is *opt-in* analysis (``compute_alpha_rank=True``) — it needs an
    extra round of payoff-matrix games and is not part of the leaderboard.

    Args:
        n_seats: seats per game — ``2`` for the 1-vs-1 track, ``3`` for the 3p
            track. Each track is ranked independently.
        compute_alpha_rank: also estimate the α-Rank analysis metric (off by
            default).

    Returns:
        dict with ``official_metric`` (``"elo"`` or ``"trueskill"``),
        ``win_rate``, ``trueskill``, ``elo`` (2p only), the raw ``records``, and
        — only when requested — ``alpha_rank`` / ``payoff_matrix``.
    """
    names = list(pool)
    if verbose:
        print(f"Round-robin over {len(names)} agents ({n_seats}p track)...")
    records = run_round_robin(pool, games_per_seating=games_per_seating,
                              seed=seed, time_limit_s=time_limit_s,
                              verbose=verbose, n_seats=n_seats)

    out = {
        "agents": names,
        "records": records,
        "n_seats": n_seats,
        "win_rate": rank_win_rate(records, names),
        "trueskill": rank_trueskill(records, names),
    }

    if n_seats == 2:
        out["elo"] = rank_elo(records, names)
        out["official_metric"] = "elo"
    else:
        out["official_metric"] = "trueskill"

    if compute_alpha_rank:
        if verbose:
            print("Estimating payoff matrix for α-Rank (analysis)...")
        a_names, M = estimate_payoff_matrix(
            pool, games_per_pair=alpha_games_per_pair, seed=seed + 1_000_000,
            time_limit_s=time_limit_s, n_seats=n_seats)
        out["alpha_rank"] = rank_alpha(a_names, M, alpha=alpha, m=alpha_m)
        out["payoff_matrix"] = M.tolist()

    return out
