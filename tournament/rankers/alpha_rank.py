"""
tournament/rankers/alpha_rank.py — α-Rank evolutionary ranking (analysis metric).

α-Rank (Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution",
Scientific Reports 2019) ranks strategies by an evolutionary process rather than
a single scalar skill. Unlike win-rate or TrueSkill it does **not** assume a
transitive skill order, so it can surface rock-paper-scissors style cycles in the
agent pool. We ship it as a secondary *analysis* metric, not the live
leaderboard: it needs the full payoff matrix and is sensitive to the
ranking-intensity ``alpha``.

Model (single population, finite size ``m``): the population is monomorphic and a
lone mutant occasionally appears; it fixates with the Moran fixation probability
under selection intensity ``alpha``. The induced Markov chain over strategies has
a stationary distribution whose mass ranks the agents.

Input is a payoff matrix ``M`` where ``M[s][r]`` is the expected payoff of a focal
agent playing strategy ``s`` while every opponent plays strategy ``r`` (for
3-player Puerto Rico: the win rate of ``s`` in games ``[s, r, r]``).
"""
import numpy as np


def _fixation_prob(f_mutant: float, f_resident: float, alpha: float, m: int) -> float:
    """Moran fixation probability of one mutant invading a resident population."""
    diff = f_mutant - f_resident
    if abs(diff) < 1e-12:
        return 1.0 / m
    num = 1.0 - np.exp(-alpha * diff)
    den = 1.0 - np.exp(-alpha * m * diff)
    return num / den


def alpha_rank(payoff_matrix, alpha: float = 1.0, m: int = 50) -> np.ndarray:
    """Stationary distribution over strategies under the single-population model.

    Args:
        payoff_matrix: ``K x K`` array; ``M[s][r]`` = payoff of ``s`` vs an all-``r`` field.
        alpha: ranking intensity (selection strength); higher = sharper.
        m: evolutionary population size.

    Returns:
        ``np.ndarray`` of shape ``(K,)`` summing to 1 — mass per strategy.
    """
    M = np.asarray(payoff_matrix, dtype=np.float64)
    K = M.shape[0]
    if K == 1:
        return np.array([1.0])

    # Transition matrix over monomorphic populations.
    C = np.zeros((K, K), dtype=np.float64)
    eta = 1.0 / (K - 1)            # uniform proposal of the next candidate mutant
    for r in range(K):            # resident
        for s in range(K):        # mutant
            if s == r:
                continue
            C[r, s] = eta * _fixation_prob(M[s, r], M[r, r], alpha, m)
        C[r, r] = 1.0 - C[r].sum()

    # Stationary distribution = left eigenvector of C for eigenvalue 1.
    vals, vecs = np.linalg.eig(C.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, idx]))
    total = pi.sum()
    return pi / total if total > 0 else np.full(K, 1.0 / K)


def rank_alpha(agent_names, payoff_matrix, alpha: float = 1.0, m: int = 50):
    """Rank agents by α-Rank stationary mass (higher = more dominant)."""
    pi = alpha_rank(payoff_matrix, alpha=alpha, m=m)
    rows = [{"agent": agent_names[i], "mass": float(pi[i])}
            for i in range(len(agent_names))]
    rows.sort(key=lambda r: r["mass"], reverse=True)
    return rows
