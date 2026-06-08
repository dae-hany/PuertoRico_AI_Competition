import numpy as np

from puerto_rico import ForwardModel, make_env


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
