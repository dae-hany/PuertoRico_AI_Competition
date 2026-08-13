"""Isolation tests: the guarantees ``tournament.sandbox`` exists to provide.

In-process, three competition rules rest on the honour system — the per-move
deadline, "don't mutate the live model", and "you never learn the face-down
draw order". Each test below breaks one of them on purpose and checks that the
sandbox holds.

These spawn real processes, so they are the slowest tests in the suite (a few
seconds each). That is the point: what is being tested is process isolation.
"""
import os
import time

import numpy as np
import pytest

from agents import RandomAgent
from puerto_rico import ForwardModel, flatten_observation, make_env
from tournament.match import play_game
from tournament.sandbox import (SandboxedAgent, TIMEOUT_ACTION, SandboxError,
                                load_agent_class, sandboxed_pool)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_agents.py")


def spec(class_name: str) -> str:
    return f"{FIXTURES}:{class_name}"


@pytest.fixture
def sandboxed():
    """Hand out SandboxedAgents and make sure their processes are reaped."""
    created = []

    def make(class_name, **kwargs):
        agent = SandboxedAgent(spec(class_name), **kwargs)
        created.append(agent)
        return agent

    yield make
    for agent in created:
        agent.close()


def _one_move(agent, num_players=2, seed=0):
    """Set up a game, ask ``agent`` for the opening move, return (env, action)."""
    env = make_env(seed=seed, num_players=num_players)
    agent.on_game_start(ForwardModel(env))
    raw = env.observe(env.agent_selection)
    obs = flatten_observation(raw["observation"])
    mask = np.asarray(raw["action_mask"], dtype=np.int8)
    return env, obs, mask


# ── it works at all ──────────────────────────────────────────────────────────

def test_reactive_agent_plays_a_full_isolated_game(sandboxed):
    agent = sandboxed("WellBehavedAgent", time_limit_s=1.0)
    result = play_game([agent, RandomAgent(seed=1)], seed=0)

    assert result["timeouts"] == [0, 0]
    assert result["illegal"] == [0, 0]
    assert result["tampered"] == [0, 0]
    assert result["steps"] > 50


def test_planning_agent_still_gets_a_usable_forward_model(sandboxed):
    """A planner must be able to clone and simulate from inside the sandbox."""
    agent = sandboxed("PlannerAgent", time_limit_s=1.0)
    result = play_game([agent, RandomAgent(seed=1)], seed=0)

    assert agent.worker.needs_model, "an agent overriding on_game_start needs a snapshot"
    assert result["timeouts"] == [0, 0]
    assert result["illegal"] == [0, 0]


# ── the deadline is enforced, not measured ───────────────────────────────────

def test_hung_agent_is_killed_at_the_deadline(sandboxed):
    """In-process this agent never returns and the tournament stops for ever."""
    agent = sandboxed("AlwaysSlowAgent", time_limit_s=0.3)
    env, obs, mask = _one_move(agent)

    start = time.perf_counter()
    action = agent.act(obs, mask)
    elapsed = time.perf_counter() - start

    assert action == TIMEOUT_ACTION
    assert agent.timeouts == 1
    assert elapsed < 5.0, "the parent must not wait on a hung agent"


def test_worker_is_rebuilt_and_keeps_playing_after_a_kill(sandboxed):
    """One overrun costs one move, not the rest of the match."""
    agent = sandboxed("WellBehavedAgent", time_limit_s=1.0)
    env, obs, mask = _one_move(agent)

    agent.time_limit_s = -1.0                    # force the next move to time out
    assert agent.act(obs, mask) == TIMEOUT_ACTION
    assert agent.timeouts == 1

    agent.time_limit_s = 1.0                     # and now it should be playing again
    action = agent.act(obs, mask)
    assert mask[action] == 1
    assert agent.violations == 0


def test_persistently_slow_agent_forfeits_instead_of_restarting_forever(sandboxed):
    """Restarting a process per move is expensive; an agent that overruns every
    move must not be able to hold the schedule hostage."""
    agent = sandboxed("AlwaysSlowAgent", time_limit_s=0.2,
                      max_consecutive_timeouts=2)

    start = time.perf_counter()
    result = play_game([agent, RandomAgent(seed=1)], seed=0)
    elapsed = time.perf_counter() - start

    assert agent.forfeited
    assert agent.timeouts == 3          # the cap, plus the one that trips it
    assert result["steps"] > 50, "the game still finishes"
    assert elapsed < 60.0, "and it finishes quickly"


def test_crashing_agent_is_booked_as_violations_and_the_game_finishes(sandboxed):
    agent = sandboxed("CrashingAgent", time_limit_s=1.0)
    result = play_game([agent, RandomAgent(seed=1)], seed=0)

    assert result["illegal"][0] > 0
    assert result["timeouts"] == [0, 0]
    assert result["steps"] > 50


# ── the game and its secrets stay in the organizer's process ─────────────────

def test_tampering_cannot_touch_the_real_game(sandboxed):
    agent = sandboxed("TamperingAgent", time_limit_s=1.0)
    env, obs, mask = _one_move(agent, num_players=2)
    game = env.unwrapped.game
    before = (game.players[0].doubloons, game.vp_chips, game.colonists_supply)

    agent.act(obs, mask)

    assert (game.players[0].doubloons, game.vp_chips,
            game.colonists_supply) == before


def test_tampering_agent_scores_no_advantage_over_a_whole_game(sandboxed):
    agent = sandboxed("TamperingAgent", time_limit_s=1.0)
    result = play_game([agent, RandomAgent(seed=1)], seed=0)
    assert result["tampered"] == [0, 0], "nothing reached the real game to detect"


def test_hidden_draw_order_never_enters_the_worker(sandboxed):
    """The worker may know *which* tiles are face down — that is deducible — but
    never the order they will come out in."""
    agent = sandboxed("DeckReportingAgent", time_limit_s=5.0)
    env, obs, mask = _one_move(agent, num_players=3, seed=13)
    true_order = [int(t) for t in env.unwrapped.game.plantation_stack]
    assert len(true_order) > 5, "need a hidden pile for this to mean anything"

    with pytest.raises(RuntimeError) as excinfo:
        agent.act(obs, mask)
    seen = [int(t) for t in str(excinfo.value).split("DECK:")[1].split(",")]

    assert sorted(seen) == sorted(true_order), "the multiset is public"
    assert seen != true_order, "the order is not"
    assert [int(t) for t in env.unwrapped.game.plantation_stack] == true_order


# ── plumbing ─────────────────────────────────────────────────────────────────

def test_loader_rejects_things_that_are_not_agents():
    with pytest.raises(ValueError):
        load_agent_class("no_colon_here")
    with pytest.raises(TypeError):
        load_agent_class(spec("NotAnAgent"))
    assert load_agent_class(spec("WellBehavedAgent")).name == "WellBehaved"


def test_a_failing_spec_is_reported_rather_than_hanging():
    agent = SandboxedAgent(spec("NoSuchAgent"), time_limit_s=1.0)
    try:
        with pytest.raises(SandboxError):
            agent.on_game_start(None)
    finally:
        agent.close()


def test_pool_gives_every_seat_of_a_self_match_its_own_process():
    """The round-robin plays groups like [A, A, B]; two seats of the same
    submission must not end up sharing one agent instance."""
    pool, close = sandboxed_pool({"Bot": spec("WellBehavedAgent")}, n_seats=3,
                                 time_limit_s=1.0)
    try:
        seats = [pool["Bot"](), pool["Bot"](), pool["Bot"]()]
        for seat in seats:
            seat.on_game_start(None)
        pids = {seat.worker._proc.pid for seat in seats}
        assert len(pids) == 3

        result = play_game(seats, seed=0)
        assert result["illegal"] == [0, 0, 0]
        assert all(a.name == "Bot" for a in seats)      # leaderboard name wins
    finally:
        close()
