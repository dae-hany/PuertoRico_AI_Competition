"""
tournament/match.py — play a single game between Agent instances.

The number of seats is simply ``len(agents)``: pass 2 agents for a 1-vs-1 (2p
track) game or 3 for a 3p-track game; the engine sets up the matching variant.

Each move is given a wall-clock budget (``time_limit_s``, default 1 s). If an
agent overruns the budget, raises an exception, or returns an illegal action,
the harness substitutes a uniformly random legal action and records the event.
See ``docs/COMPETITION_RULES.md`` for the exact rules.

The function returns a per-game record (finishing ranks, winners, scores, and
rule-violation counts) that the rankers in ``tournament/rankers`` consume.

Agents run **in this process** here, which is right for the course setting and
for local testing but leaves two rules on the honour system: the per-move budget
is measured *after* the fact (a hung agent stalls the run), and the forward model
an agent is handed is a live object it could in principle mutate. ``tampered``
in the result closes the second gap by *checking* rather than trusting — see
``_state_fingerprint``. For the public run, use ``tournament.sandbox`` instead,
which enforces both by construction.
"""
import time

import numpy as np

from puerto_rico import ForwardModel, flatten_observation, make_env

PASS_ACTION = 15


def _state_fingerprint(env, obs, mask):
    """Cheap fingerprint of everything an agent must not be able to change.

    ``act()`` is a pure decision: it may read and simulate, but the real game
    must be exactly where it was when the agent was asked. Comparing this before
    and after each call turns "agents must not mutate the live model" from an
    honour-system rule into one the harness notices being broken. The hidden
    plantation order is included because the observation deliberately omits it.
    """
    game = env.unwrapped.game
    return (
        obs.tobytes(),
        mask.tobytes(),
        tuple(game.plantation_stack),
        tuple(game.plantation_discard),
        game.current_player_idx,
        game.round_number,
        game.vp_chips,
        game.colonists_supply,
        game.colonists_ship,
        game.quarry_stack,
        tuple(game.trading_house),
        tuple(sorted(game.building_supply.items())),
    )


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


MAX_STEPS = 4000            # ~8x the longest game observed between baselines


def play_game(agents, seed: int = None, time_limit_s: float = 1.0,
              check_tampering: bool = True, max_steps: int = MAX_STEPS) -> dict:
    """Play one game between ``agents`` (a list of :class:`Agent` instances).

    The seat count is ``len(agents)`` — 2 for the 1-vs-1 track, 3 for the 3p
    track.

    Args:
        agents: the seated players, in seat order (index = player id).
        seed: game seed; the same seed reproduces the same initial setup.
        time_limit_s: per-move wall-clock budget; overruns play a random legal move.
        check_tampering: verify after every move that the agent left the real
            game untouched, and count the breaches in ``tampered``. Cheap (one
            extra observation per move); turn it off only for profiling.
        max_steps: hard cap on decisions. Puerto Rico ends on its own, but only
            if *somebody* moves the game along, and agents choose the roles. A
            table that never picks the Mayor never spends a colonist, so the
            colonist supply — one of the three end triggers — never drains, and
            the game runs for ever. That is a legal, if useless, way to play, so
            the harness bounds it rather than trusting the field: on reaching
            the cap the game is scored where it stands and flagged ``truncated``.

    Returns:
        dict with ``scores``, ``tiebreak``, ``ranks`` (0 = best), ``winners``,
        ``steps``, ``timeouts``, ``illegal``, ``tampered`` (per-player counts),
        ``truncated``, and ``seed``.
    """
    n = len(agents)
    env = make_env(seed=seed, num_players=n)
    for agent in agents:
        try:
            # One model per seat: a single shared instance let any agent perturb
            # the object every other agent was planning with (its determinization
            # RNG, its cached snapshot).
            agent.on_game_start(ForwardModel(env))
        except Exception:
            pass                            # on_game_start is optional / best-effort

    rng = np.random.default_rng(seed)       # reproducible fallback moves
    timeouts = [0] * n
    illegal = [0] * n
    tampered = [0] * n
    steps = 0

    while env.agents and steps < max_steps:
        name = env.agent_selection
        if env.terminations.get(name, False) or env.truncations.get(name, False):
            env.step(None)
            continue

        p = env.unwrapped.agent_name_mapping[name]
        raw = env.observe(name)
        obs = flatten_observation(raw["observation"])
        mask = np.asarray(raw["action_mask"], dtype=np.int8)

        before = _state_fingerprint(env, obs, mask) if check_tampering else None

        start = time.perf_counter()
        try:
            action = int(agents[p].act(obs, mask))
        except Exception:
            action = _random_legal(mask, rng)
            illegal[p] += 1
        elapsed = time.perf_counter() - start

        if before is not None:
            raw_after = env.observe(name)
            after = _state_fingerprint(
                env, flatten_observation(raw_after["observation"]),
                np.asarray(raw_after["action_mask"], dtype=np.int8))
            if after != before:
                tampered[p] += 1

        if elapsed > time_limit_s:
            action = _random_legal(mask, rng)   # overran the budget -> random legal
            timeouts[p] += 1
        elif not (0 <= action < 200) or mask[action] < 0.5:
            action = _random_legal(mask, rng)   # illegal move -> random legal
            illegal[p] += 1

        env.step(action)
        steps += 1

    truncated = bool(env.agents)          # hit the cap instead of ending naturally

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
        "tampered": tampered,
        "truncated": truncated,
    }
