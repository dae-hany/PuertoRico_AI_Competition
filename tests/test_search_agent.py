"""Tests for the 2p fast-clone optimization, the evaluator, and the SearchAgent.

These guard the pieces the 1-vs-1 search agent relies on:

* ``PuertoRicoEnv.__deepcopy__`` must produce an *independent* clone (mutating the
  clone never touches the original) while *sharing* the immutable gym spaces (the
  whole point of the speedup).
* ``eval2p`` terminal/margin values must have the right sign.
* ``SearchAgent`` must always return a legal action and stay within budget.
"""
import time

import numpy as np

from puerto_rico import ForwardModel, flatten_observation, make_env
from agents.search_agent import SearchAgent, SearchLiteAgent
from agents import ActionValueAgent


def _play_random_to_midgame(model, rng, n=40):
    for _ in range(n):
        if model.is_terminal():
            break
        legal = model.legal_actions()
        model.step(int(rng.choice(legal)) if legal else 15)


def test_clone_is_independent_but_shares_spaces():
    env = make_env(seed=3, num_players=2)
    model = ForwardModel(env)
    rng = np.random.default_rng(3)
    _play_random_to_midgame(model, rng, 30)

    base_obs = model.observation().copy()
    base_legal = model.legal_actions()

    clone = model.clone()
    # Spaces are shared (not deep-copied) — that is the optimization.
    assert clone.env._observation_spaces is env._observation_spaces
    assert clone.env._action_spaces is env._action_spaces
    # But the mutable game state is a genuine independent copy.
    assert clone.env.unwrapped.game is not env.unwrapped.game

    # Mutating the clone to the end must not disturb the original.
    crng = np.random.default_rng(99)
    while not clone.is_terminal():
        legal = clone.legal_actions()
        clone.step(int(crng.choice(legal)) if legal else 15)
    assert model.legal_actions() == base_legal
    assert np.array_equal(model.observation(), base_obs)


def test_searchlite_is_a_lower_tier():
    # SearchLite must use a smaller node budget than the full agent (the ladder),
    # and still return legal moves within budget.
    lite, full = SearchLiteAgent(), SearchAgent()
    assert lite.node_budget < full.node_budget
    env = make_env(seed=8, num_players=2)
    model = ForwardModel(env)
    lite.on_game_start(model)
    moves = 0
    while not model.is_terminal() and moves < 25:
        name = env.agent_selection
        pidx = env.unwrapped.agent_name_mapping[name]
        raw = env.observe(name)
        mask = np.asarray(raw["action_mask"])
        if pidx == 0:
            a = lite.act(flatten_observation(raw["observation"]), mask)
            assert mask[a] > 0.5
        else:
            a = int(np.where(mask > 0.5)[0][0])
        model.step(a)
        moves += 1


def test_node_budget_play_is_deterministic():
    # A node-capped search must pick the same move from the same position no
    # matter the hardware/timing — two fresh agents (same seed) must agree.
    env = make_env(seed=12, num_players=2)
    model = ForwardModel(env)
    _play_random_to_midgame(model, np.random.default_rng(12), 20)
    raw = env.observe(env.agent_selection)
    obs, mask = flatten_observation(raw["observation"]), np.asarray(raw["action_mask"])

    def move():
        ag = SearchAgent(seed=0)
        ag.on_game_start(model)
        return ag.act(obs, mask)

    assert move() == move()


def test_search_agent_uses_2p_calibrated_eval():
    # The 2p track agent must default to the recalibrated heuristic, not the
    # 3p base (guards against the "dead code" regression where the subclass
    # exists but is never wired in).
    agent = SearchAgent()
    assert agent.h._EXPECTED_ROLE_USES_BASE > 9.0


def test_eval_terminal_and_margin_signs():
    from agents.eval2p import terminal_value, evaluate_state, WIN_BASE
    # me=0 wins on VP.
    scores = [(30, 5, 0, 0, 0), (20, 9, 0, 0, 0)]
    assert terminal_value(scores, 0, 1) > WIN_BASE - 1
    assert terminal_value(scores, 1, 0) < -WIN_BASE + 1
    # A symmetric start should evaluate near zero (no one is ahead).
    env = make_env(seed=1, num_players=2)
    v = evaluate_state(env.unwrapped.game, 0, 1)
    assert abs(v) < 5.0


def test_search_agent_returns_legal_and_in_budget():
    env = make_env(seed=5, num_players=2)
    model = ForwardModel(env)
    agent = SearchAgent(time_limit_s=0.3, seed=1)
    agent.on_game_start(model)
    opp = ActionValueAgent()
    opp.on_game_start(model)

    moves = 0
    while not model.is_terminal() and moves < 40:
        name = env.agent_selection
        pidx = env.unwrapped.agent_name_mapping[name]
        raw = env.observe(name)
        obs = flatten_observation(raw["observation"])
        mask = np.asarray(raw["action_mask"])
        if pidx == 0:
            t0 = time.perf_counter()
            a = agent.act(obs, mask)
            assert time.perf_counter() - t0 < 1.0, "search exceeded the hard 1s limit"
            assert mask[a] > 0.5, "search returned an illegal action"
        else:
            a = opp.act(obs, mask)
        model.step(a)
        moves += 1
