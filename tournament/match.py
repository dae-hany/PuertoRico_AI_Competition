"""
tournament/match.py — play a single 3-player game between Agent instances.

Each move is given a wall-clock budget (``time_limit_s``, default 1 s). If an
agent overruns the budget, raises an exception, or returns an illegal action,
the harness substitutes a uniformly random legal action and records the event.
See ``docs/COMPETITION_RULES.md`` for the exact rules.

The function returns a per-game record (finishing ranks, winners, scores, and
rule-violation counts) that the rankers in ``tournament/rankers`` consume.
"""
import time

import numpy as np

from puerto_rico import ForwardModel, flatten_observation, make_env

PASS_ACTION = 15


def _random_legal(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.where(np.asarray(mask) > 0.5)[0]
    return int(rng.choice(legal)) if len(legal) else PASS_ACTION


def _finishing_ranks(vp, tiebreak):
    """Standard competition ranking (0 = best); tied players share a rank."""
    n = len(vp)
    order = sorted(range(n), key=lambda i: (vp[i], tiebreak[i]), reverse=True)
    ranks = [0] * n
    pos = 0
    for idx, i in enumerate(order):
        prev = order[idx - 1]
        if idx > 0 and (vp[i], tiebreak[i]) != (vp[prev], tiebreak[prev]):
            pos = idx
        ranks[i] = pos
    return ranks


def play_game(agents, seed: int = None, time_limit_s: float = 1.0) -> dict:
    """Play one game between ``agents`` (a list of 3 :class:`Agent` instances).

    Args:
        agents: the seated players, in seat order (index = player id).
        seed: game seed; the same seed reproduces the same initial setup.
        time_limit_s: per-move wall-clock budget; overruns play a random legal move.

    Returns:
        dict with ``scores``, ``tiebreak``, ``ranks`` (0 = best), ``winners``,
        ``steps``, ``timeouts``, ``illegal`` (per-player counts), and ``seed``.
    """
    n = len(agents)
    env = make_env(seed=seed, num_players=n)
    model = ForwardModel(env)               # live, clone-able view for planners
    for agent in agents:
        try:
            agent.on_game_start(model)
        except Exception:
            pass                            # on_game_start is optional / best-effort

    rng = np.random.default_rng(seed)       # reproducible fallback moves
    timeouts = [0] * n
    illegal = [0] * n
    steps = 0

    while env.agents:
        name = env.agent_selection
        if env.terminations.get(name, False) or env.truncations.get(name, False):
            env.step(None)
            continue

        p = env.unwrapped.agent_name_mapping[name]
        raw = env.observe(name)
        obs = flatten_observation(raw["observation"])
        mask = np.asarray(raw["action_mask"], dtype=np.int8)

        start = time.perf_counter()
        try:
            action = int(agents[p].act(obs, mask))
        except Exception:
            action = _random_legal(mask, rng)
            illegal[p] += 1
        elapsed = time.perf_counter() - start

        if elapsed > time_limit_s:
            action = _random_legal(mask, rng)   # overran the budget -> random legal
            timeouts[p] += 1
        elif not (0 <= action < 200) or mask[action] < 0.5:
            action = _random_legal(mask, rng)   # illegal move -> random legal
            illegal[p] += 1

        env.step(action)
        steps += 1

    scores = env.unwrapped.game.get_scores()
    vp = [int(s[0]) for s in scores]
    tiebreak = [int(s[1]) for s in scores]
    ranks = _finishing_ranks(vp, tiebreak)

    best_vp = max(vp)
    best_tb = max(tiebreak[i] for i in range(n) if vp[i] == best_vp)
    winners = [i for i in range(n) if vp[i] == best_vp and tiebreak[i] == best_tb]

    return {
        "seed": seed,
        "scores": vp,
        "tiebreak": tiebreak,
        "ranks": ranks,
        "winners": winners,
        "steps": steps,
        "timeouts": timeouts,
        "illegal": illegal,
    }
