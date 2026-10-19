"""The per-decision budget: 0.1 s for every entrant, and the bundled baselines' own 1 s."""
import time

import pytest

from agents import RandomAgent, SearchAgent, SearchLiteAgent
from agents.registry import is_bundled_baseline
from tournament.match import BASELINE_TIME_LIMIT_S, TIME_LIMIT_S, play_game


class Entrant(RandomAgent):
    """A subclass of a baseline is an entrant, not a baseline."""
    name = "Entrant"


def slow(agent, delay):
    inner = agent.act

    def act(observation, action_mask):
        time.sleep(delay)
        return inner(observation, action_mask)

    agent.act = act
    return agent


def test_the_limits():
    assert TIME_LIMIT_S == 0.1
    assert BASELINE_TIME_LIMIT_S == 1.0


def test_bundled_baselines_are_recognised_by_exact_class():
    assert is_bundled_baseline(RandomAgent())
    assert is_bundled_baseline(SearchAgent()) and is_bundled_baseline(SearchLiteAgent())
    assert not is_bundled_baseline(Entrant())
    assert not is_bundled_baseline(object())


def test_an_entrant_times_out_at_0_1_s_and_a_baseline_does_not():
    r = play_game([slow(Entrant(seed=0), 0.15), slow(RandomAgent(seed=1), 0.15)],
                  seed=3, max_steps=16)
    assert r["timeouts"][0] > 0
    assert r["timeouts"][1] == 0


def test_baselines_can_be_held_to_the_entrant_limit_or_seats_set_explicitly():
    r = play_game([slow(RandomAgent(seed=0), 0.15), RandomAgent(seed=1)],
                  seed=3, max_steps=16, baseline_time_limit_s=None)
    assert r["timeouts"][0] > 0

    r = play_game([slow(Entrant(seed=0), 0.15), RandomAgent(seed=1)],
                  seed=3, max_steps=16, time_limit_s=[1.0, 0.1])
    assert r["timeouts"][0] == 0

    from tournament.match import seat_time_limits
    assert seat_time_limits([Entrant(), RandomAgent()]) == [0.1, 1.0]
    assert seat_time_limits([Entrant(), RandomAgent()], 5.0) == [5.0, 5.0]   # never cut
    assert seat_time_limits([Entrant(), RandomAgent()], 0.1, None) == [0.1, 0.1]

    with pytest.raises(ValueError):
        play_game([RandomAgent(), RandomAgent()], time_limit_s=[0.1], max_steps=1)
