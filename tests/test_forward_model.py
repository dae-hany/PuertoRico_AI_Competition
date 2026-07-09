from collections import Counter

import numpy as np

from puerto_rico import ForwardModel, make_env


def test_clone_determinizes_plantation_order():
    """Cloning hides the face-down draw order but keeps the (public) multiset.

    A search agent clones the live model to simulate, so this is the path that
    must not leak the secret plantation order.
    """
    fm = ForwardModel(make_env(seed=7))
    true_stack = list(fm._env.unwrapped.game.plantation_stack)
    assert len(true_stack) > 1, "need a hidden order for this test to be meaningful"

    orders = []
    for _ in range(8):
        clone_stack = fm.clone().game.plantation_stack
        # Only the order is randomized — which tiles remain is deducible, so kept.
        assert Counter(clone_stack) == Counter(true_stack)
        orders.append(list(clone_stack))

    # The true order is hidden: clones do not simply replay it.
    assert any(o != true_stack for o in orders)
    # Cloning never disturbs the real game.
    assert list(fm._env.unwrapped.game.plantation_stack) == true_stack


def test_live_game_does_not_expose_true_order():
    """Reading the live ``.game`` directly (no clone) must not reveal the order."""
    fm = ForwardModel(make_env(seed=11))
    true_stack = list(fm._env.unwrapped.game.plantation_stack)

    redacted = fm.game.plantation_stack
    # Same deducible multiset...
    assert Counter(redacted) == Counter(true_stack)
    # ...but the accessor hands out a determinized snapshot, not the real engine.
    assert fm.game is not fm._env.unwrapped.game
    # Repeated reads within one decision point are stable (cached snapshot).
    assert fm.game is fm.game


def test_clone_of_clone_keeps_order_for_consistent_search():
    """Cloning a clone preserves its order so a search tree stays self-consistent."""
    fm = ForwardModel(make_env(seed=5))
    root = fm.clone()
    root_order = list(root.game.plantation_stack)
    for _ in range(5):
        assert list(root.clone().game.plantation_stack) == root_order


def test_clone_is_independent():
    fm = ForwardModel(make_env(seed=0))
    start_player = fm.current_player()

    sim = fm.clone()
    for _ in range(20):
        if sim.is_terminal():
            break
        sim.step(sim.legal_actions()[0])

    # Mutating the clone must not touch the original.
    assert fm.current_player() == start_player
    assert not fm.is_terminal()


def test_legal_actions_match_mask():
    fm = ForwardModel(make_env(seed=2))
    mask = fm.action_mask()
    assert set(fm.legal_actions()) == set(np.where(mask > 0.5)[0].tolist())


def test_step_reaches_terminal_with_winner():
    fm = ForwardModel(make_env(seed=3))
    steps = 0
    while not fm.is_terminal() and steps < 5000:
        fm.step(fm.legal_actions()[0])
        steps += 1
    assert fm.is_terminal()
    assert len(fm.winners()) >= 1
