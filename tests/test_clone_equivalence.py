"""Equivalence tests for the engine performance work.

The hand-rolled ``__deepcopy__`` implementations (PuertoRicoGame, Player, the
component dataclasses) and the ForwardModel.action_mask() fast path are pure
performance refactors — they must be *behaviorally invisible*. These tests pin
that down:

  * the fast mask equals ``observe(agent_selection)["action_mask"]`` at every
    decision of seeded random games (2p and 3p),
  * a hand-rolled clone is value-identical to a generic ``copy.deepcopy`` of
    the same env (custom hooks disabled),
  * a clone shares no mutable state with the source (play the clone to the
    end; the source must not move),
  * hand-rolled and generic clones behave identically under the same action
    sequence,
  * a mid-game env still pickles (the sandbox ships snapshots across
    processes; ``slots=True`` dataclasses must not break that).
"""
import copy
import pickle
import random
from contextlib import contextmanager
from enum import Enum

import numpy as np
import pytest

from puerto_rico import make_env
from puerto_rico.components import CargoShip, CityBuilding, IslandTile
from puerto_rico.engine import PuertoRicoGame
from puerto_rico.forward_model import ForwardModel
from puerto_rico.player import Player


# ── deep value-signature of an env (aliasing-free comparison) ───────────────

def _sig(x):
    if isinstance(x, random.Random):
        return ("Random", x.getstate())
    if isinstance(x, Enum):
        return (type(x).__name__, x.value)
    if isinstance(x, dict):
        return ("dict", tuple(sorted(((_sig(k), _sig(v)) for k, v in x.items()),
                                     key=repr)))
    if isinstance(x, (list, tuple)):
        return ("seq", tuple(_sig(v) for v in x))
    if isinstance(x, (set, frozenset)):
        return ("set", tuple(sorted((_sig(v) for v in x), key=repr)))
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, (IslandTile, CityBuilding, CargoShip)):
        fields = tuple(_sig(getattr(x, s)) for s in x.__slots__)
        return (type(x).__name__, fields)
    if isinstance(x, (Player, PuertoRicoGame)):
        return (type(x).__name__, _sig(x.__dict__))
    if x is None or isinstance(x, (int, float, str, bool, bytes)):
        return x
    raise TypeError(f"unhandled type in signature: {type(x)!r}")


def env_signature(env):
    u = env.unwrapped
    return (
        _sig(u.game.__dict__),
        tuple(u.agents),
        u.agent_selection,
        _sig(u.terminations),
        _sig(u.truncations),
        u._game_step_count,
    )


@contextmanager
def generic_deepcopy_hooks():
    """Temporarily remove the hand-rolled __deepcopy__ hooks so copy.deepcopy
    falls back to its generic (slow, reference) implementation."""
    classes = [PuertoRicoGame, Player, IslandTile, CityBuilding, CargoShip]
    saved = [(c, c.__dict__.get("__deepcopy__")) for c in classes]
    for c, hook in saved:
        if hook is not None:
            delattr(c, "__deepcopy__")
    try:
        yield
    finally:
        for c, hook in saved:
            if hook is not None:
                setattr(c, "__deepcopy__", hook)


def _random_playout(fm, rng, max_steps):
    steps = 0
    while not fm.is_terminal() and steps < max_steps:
        legal = fm.legal_actions()
        assert legal, "non-terminal state must have a legal action"
        fm.step(int(rng.choice(legal)))
        steps += 1
    return steps


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num_players,seed", [(2, 0), (2, 7), (3, 3)])
def test_mask_fast_path_matches_observe(num_players, seed):
    env = make_env(seed=seed, num_players=num_players)
    fm = ForwardModel(env)
    rng = random.Random(seed)
    for _ in range(400):
        if fm.is_terminal():
            break
        fast = fm.action_mask()
        slow = np.asarray(env.observe(env.agent_selection)["action_mask"],
                          dtype=np.int8)
        assert np.array_equal(fast, slow)
        legal = [int(a) for a in np.flatnonzero(slow)]
        fm.step(rng.choice(legal))


@pytest.mark.parametrize("num_players,seed", [(2, 1), (2, 11), (3, 5)])
def test_handrolled_clone_matches_generic_deepcopy(num_players, seed):
    env = make_env(seed=seed, num_players=num_players)
    fm = ForwardModel(env)
    rng = random.Random(seed + 100)
    # advance into the mid-game so the state is rich
    _random_playout(fm, rng, 80)

    fast_clone = copy.deepcopy(env)
    with generic_deepcopy_hooks():
        ref_clone = copy.deepcopy(env)

    src = env_signature(env)
    assert env_signature(fast_clone) == src
    assert env_signature(ref_clone) == src


@pytest.mark.parametrize("seed", [2, 13])
def test_clone_does_not_alias_live_state(seed):
    env = make_env(seed=seed, num_players=2)
    fm = ForwardModel(env)
    rng = random.Random(seed)
    _random_playout(fm, rng, 60)

    before = env_signature(env)
    clone = fm.clone()
    _random_playout(clone, rng, 4000)
    assert clone.is_terminal()
    assert env_signature(env) == before


@pytest.mark.parametrize("seed", [4, 21])
def test_clone_behaves_identically_to_generic_copy(seed):
    env = make_env(seed=seed, num_players=2)
    fm = ForwardModel(env)
    rng = random.Random(seed)
    _random_playout(fm, rng, 70)

    fast = ForwardModel(copy.deepcopy(env), _live=False)
    with generic_deepcopy_hooks():
        ref = ForwardModel(copy.deepcopy(env), _live=False)

    steps = random.Random(seed + 1)
    for _ in range(300):
        if fast.is_terminal():
            assert ref.is_terminal()
            break
        m1, m2 = fast.action_mask(), ref.action_mask()
        assert np.array_equal(m1, m2)
        assert fast.current_player() == ref.current_player()
        action = steps.choice([int(a) for a in np.flatnonzero(m1)])
        fast.step(action)
        ref.step(action)
    assert fast.scores() == ref.scores()
    assert env_signature(fast._env) == env_signature(ref._env)


def test_midgame_env_still_pickles(seed=6):
    env = make_env(seed=seed, num_players=2)
    fm = ForwardModel(env)
    _random_playout(fm, random.Random(seed), 50)

    restored = pickle.loads(pickle.dumps(env))
    assert env_signature(restored) == env_signature(env)
