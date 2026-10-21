"""
tournament/rankers/trueskill_ranker.py — the OFFICIAL metric of the 3p track.

TrueSkill (Herbrich et al., 2006) is a Bayesian skill-rating system designed for
free-for-all games with more than two players — a natural fit for 3-player
Puerto Rico. Each agent has a rating ``N(mu, sigma^2)``; after every game the
ratings are updated from the finishing order. The leaderboard value is the
conservative ``mu - 3*sigma`` (skill we are 99% sure the agent exceeds), so
agents with few games (high sigma) are not flattered.

TrueSkill models every seat of a game as a different player, so a game in
which one agent holds more than one seat (the round-robin plays groups such as
``[A, A, B]``) is not rated: updating each of that agent's seats in turn kept
only the last seat's update and let the whole scale drift (over the same
heuristic round-robin on two engines, the field's mean ``mu`` ended at 35 and
at 19, against 27 and 28 without those games). Those games still count toward
the win rate.

Requires the ``trueskill`` package.
"""
import trueskill


def rank_trueskill(records, agent_names, draw_probability: float = 0.0):
    """Rate agents with TrueSkill over the match ``records``.

    Args:
        records: list of ``{"agents": [...3 names...], "result": {...}}`` where
            ``result["ranks"]`` gives each seat's finishing place (0 = best).
            A record that seats one agent more than once is skipped.
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
        if len(set(seats)) < len(seats):
            continue                    # an agent against itself: not ratable
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
