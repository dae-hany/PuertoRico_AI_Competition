"""
tournament/rankers/win_rate.py — win rate, reported alongside both tracks.

Agents are ranked by their **win rate** across a seat-balanced round-robin:
the fraction of games each agent wins (ties split the win equally). A Wilson
95% confidence interval is reported so close standings can be read with their
uncertainty. Simple and transparent — "who wins the most games".

This is the *companion* view, not the standings: the official metric is Elo in
the 2p track and TrueSkill in the 3p track, both of which weight a result by the
strength of the opponent beaten. Win rate is shown next to them as an
assumption-free check. See ``docs/RANKING.md``.
"""
import math


def wilson_ci(wins: float, n: int, z: float = 1.96):
    """Wilson score 95% CI for a win rate ``wins/n`` (wins may be fractional)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def rank_win_rate(records, agent_names):
    """Rank agents by win rate over the match ``records``.

    Args:
        records: list of ``{"agents": [name_seat0, name_seat1, name_seat2],
                            "result": <play_game result>}``.
        agent_names: every agent that should appear on the board.

    Returns:
        list of row dicts sorted best-first, each with ``agent``, ``win_rate``,
        ``games``, ``wins``, and ``ci`` (low, high).
    """
    games = {a: 0 for a in agent_names}
    wins = {a: 0.0 for a in agent_names}

    for rec in records:
        seats = rec["agents"]
        result = rec["result"]
        for name in seats:
            games[name] += 1
        share = 1.0 / len(result["winners"])
        for w in result["winners"]:
            wins[seats[w]] += share

    rows = []
    for a in agent_names:
        n = games[a]
        rate = wins[a] / n if n else 0.0
        lo, hi = wilson_ci(wins[a], n)
        rows.append({"agent": a, "win_rate": rate, "games": n,
                     "wins": round(wins[a], 2), "ci": (lo, hi)})

    rows.sort(key=lambda r: r["win_rate"], reverse=True)
    return rows
