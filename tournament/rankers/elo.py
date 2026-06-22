"""
tournament/rankers/elo.py — Elo rating, the OFFICIAL metric of the 2p (1-vs-1) track.

Elo is the classic 1-vs-1 rating: the gap between two ratings predicts the
win probability through the logistic curve
``P(a beats b) = 1 / (1 + 10**(-(R_a - R_b) / 400))``.

A naive sequential Elo pass depends on game order and the K-factor, which is bad
for a competition that must be reproducible. We instead report the **converged**
Elo — the Bradley–Terry maximum-likelihood rating that the same logistic model
implies — computed order-independently from the round-robin win counts with the
MM algorithm (Hunter, 2004). A light prior (``prior`` virtual even games against
an average-strength reference) keeps ratings finite when an agent wins or loses
every game, and shrinks small-sample extremes toward the field mean (1500).

This ranker expects **pairwise** records (two seats); records with another seat
count are ignored.
"""
import math


def rank_elo(records, agent_names, base: float = 1500.0, prior: float = 1.0,
             iters: int = 1000, tol: float = 1e-9):
    """Rank agents by converged (Bradley–Terry) Elo over the 1-vs-1 ``records``.

    Args:
        records: list of ``{"agents": [name_seat0, name_seat1], "result": {...}}``
            where ``result["winners"]`` holds the winning seat indices (a tie
            lists both, and splits the game as half a win each).
        agent_names: every agent that should appear on the board.
        base: the rating the field averages to (conventionally 1500).
        prior: strength of the regularizing virtual games (keeps Elo finite).
        iters, tol: MM iteration budget and convergence tolerance.

    Returns:
        list of row dicts sorted best-first, each with ``agent``, ``elo``,
        ``games``, ``wins`` and ``win_rate``.
    """
    names = list(agent_names)
    idx = {a: i for i, a in enumerate(names)}
    K = len(names)

    wins = [0.0] * K                       # fractional wins
    games = [0] * K
    n = [[0.0] * K for _ in range(K)]      # games played between i and j

    for rec in records:
        seats = rec["agents"]
        if len(seats) != 2:
            continue                       # Elo is a 1-vs-1 metric
        a, b = idx[seats[0]], idx[seats[1]]
        winners = rec["result"]["winners"]
        share0 = (1.0 / len(winners)) if 0 in winners else 0.0
        wins[a] += share0
        wins[b] += 1.0 - share0
        games[a] += 1
        games[b] += 1
        n[a][b] += 1
        n[b][a] += 1

    # MM fixed-point iteration for the Bradley–Terry MLE (Hunter, 2004).
    p = [1.0] * K
    for _ in range(iters):
        new = [0.0] * K
        for i in range(K):
            denom = prior / (p[i] + 1.0)   # virtual games vs a strength-1 ref
            for j in range(K):
                if n[i][j] > 0:
                    denom += n[i][j] / (p[i] + p[j])
            num = wins[i] + prior / 2.0
            new[i] = (num / denom) if denom > 0 else p[i]
        # Normalize to geometric mean 1 so the scale (and the strength-1 ref) is stable.
        gm = math.exp(sum(math.log(x) for x in new) / K)
        new = [x / gm for x in new]
        max_rel = max(abs(new[i] - p[i]) / (p[i] + 1e-12) for i in range(K))
        p = new
        if max_rel < tol:
            break

    scale = 400.0 / math.log(10.0)         # logit -> Elo points
    log_p = [math.log(x) for x in p]
    mean_log = sum(log_p) / K
    elo = [base + scale * (lp - mean_log) for lp in log_p]

    rows = [{"agent": names[i], "elo": elo[i], "games": games[i],
             "wins": round(wins[i], 2),
             "win_rate": (wins[i] / games[i] if games[i] else 0.0)}
            for i in range(K)]
    rows.sort(key=lambda r: r["elo"], reverse=True)
    return rows
