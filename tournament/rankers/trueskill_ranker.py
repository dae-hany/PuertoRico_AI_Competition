"""
tournament/rankers/trueskill_ranker.py — TrueSkill rating (secondary metric).

TrueSkill (Herbrich et al., 2006) is a Bayesian skill-rating system designed for
free-for-all games with more than two players — a natural fit for 3-player
Puerto Rico. Each agent has a rating ``N(mu, sigma^2)``; after every game the
ratings are updated from the finishing order. The leaderboard value is the
conservative ``mu - 3*sigma`` (skill we are 99% sure the agent exceeds), so
agents with few games (high sigma) are not flattered.

Requires the ``trueskill`` package.
"""
import trueskill


def rank_trueskill(records, agent_names, draw_probability: float = 0.0):
    """Rate agents with TrueSkill over the match ``records``.

    Args:
        records: list of ``{"agents": [...3 names...], "result": {...}}`` where
            ``result["ranks"]`` gives each seat's finishing place (0 = best).
        agent_names: every agent to rate.
        draw_probability: TrueSkill draw prior (Puerto Rico ties are rare).

    Returns:
        list of row dicts sorted best-first, each with ``agent``, ``mu``,
        ``sigma`` and ``score`` (= ``mu - 3*sigma``).
    """
    env = trueskill.TrueSkill(draw_probability=draw_probability)
    ratings = {a: env.create_rating() for a in agent_names}

    for rec in records:
        seats = rec["agents"]
        ranks = rec["result"]["ranks"]
        groups = [(ratings[seats[i]],) for i in range(len(seats))]
        updated = env.rate(groups, ranks=ranks)
        for i in range(len(seats)):
            ratings[seats[i]] = updated[i][0]

    rows = [{"agent": a,
             "mu": ratings[a].mu,
             "sigma": ratings[a].sigma,
             "score": ratings[a].mu - 3 * ratings[a].sigma}
            for a in agent_names]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows
