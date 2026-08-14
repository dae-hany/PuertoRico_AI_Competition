"""The competition promises seeded, reproducible games (docs/COMPETITION_RULES.md).

That only holds if the engine's randomness is its own. When the engine drew from
the process-global ``random`` module and ``reset()`` re-seeded the global
``random`` / ``np.random`` states, the promise leaked in both directions: an
entrant's agent could shift the deal it was about to be dealt, and the env
silently clobbered the agent's own RNG stream.
"""
import random

import numpy as np

from agents import ActionValueAgent, FactoryAgent, RandomAgent, ShippingRushAgent
from puerto_rico import make_env
from tournament.match import play_game


def test_env_never_touches_the_global_rng_states():
    np.random.seed(0)
    expected_np = np.random.rand()
    np.random.seed(0)
    make_env(seed=99, num_players=3)
    assert np.random.rand() == expected_np

    random.seed(0)
    expected_py = random.random()
    random.seed(0)
    make_env(seed=99, num_players=3)
    assert random.random() == expected_py


def test_same_seed_reproduces_the_same_deal():
    def deck(seed, n):
        return list(make_env(seed=seed, num_players=n).unwrapped.game.plantation_stack)

    for n in (2, 3):
        assert deck(4242, n) == deck(4242, n)
    assert deck(1, 3) != deck(2, 3)


def test_unseeded_games_are_actually_fresh():
    """``make_env(seed=None)`` promises a fresh random game.

    It used to inherit whatever the last seeded reset left in the global RNG, so
    two "random" games created after a seeded one were identical.
    """
    make_env(seed=7, num_players=3)
    first = list(make_env(seed=None, num_players=3).unwrapped.game.plantation_stack)
    make_env(seed=7, num_players=3)
    second = list(make_env(seed=None, num_players=3).unwrapped.game.plantation_stack)
    assert first != second


class _GlobalRngUser(ActionValueAgent):
    """A legal agent that happens to use (and even re-seed) the global RNGs."""

    name = "GlobalRngUser"

    def act(self, observation, action_mask):
        random.random()
        random.seed(1234)
        np.random.seed(7)
        np.random.rand(3)
        return super().act(observation, action_mask)


def test_agent_rng_use_cannot_change_the_game():
    for n_seats in (2, 3):
        fillers = n_seats - 1
        for seed in (5, 23):
            plain = play_game([ActionValueAgent()] + [ActionValueAgent() for _ in range(fillers)],
                              seed=seed)
            noisy = play_game([_GlobalRngUser()] + [ActionValueAgent() for _ in range(fillers)],
                              seed=seed)
            assert plain["scores"] == noisy["scores"]
            assert plain["steps"] == noisy["steps"]


def test_matches_with_the_baseline_pool_replay_identically():
    """The heuristics jitter their action scores; that must come from their own
    RNG, not from the global state the env used to re-seed for them."""
    def match(seed):
        r = play_game([ShippingRushAgent(), FactoryAgent(), RandomAgent(seed=1)],
                      seed=seed)
        return r["scores"], r["steps"]

    for seed in (11, 12):
        assert match(seed) == match(seed)
