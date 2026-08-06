import numpy as np

from puerto_rico import ACTION_DIM, OBS_DIM, flatten_observation, make_env


def test_obs_and_mask_shapes():
    env = make_env(seed=0)
    obs = env.observe(env.agent_selection)
    flat = flatten_observation(obs["observation"])
    assert flat.shape == (OBS_DIM,)             # 3p track: 74 + 73*3 = 293
    assert flat.dtype == np.float32
    assert np.asarray(obs["action_mask"]).shape == (ACTION_DIM,)


def test_obs_shape_2p_track():
    # The 2p (1-vs-1) track drops one per-player block: 74 + 73*2 = 220.
    # The action space is unchanged at 200.
    env = make_env(seed=0, num_players=2)
    obs = env.observe(env.agent_selection)
    flat = flatten_observation(obs["observation"])
    assert flat.shape == (220,)
    assert np.asarray(obs["action_mask"]).shape == (ACTION_DIM,)


def test_random_game_completes_with_valid_scores():
    env = make_env(seed=1)
    rng = np.random.default_rng(0)
    steps = 0
    while env.agents and steps < 5000:
        name = env.agent_selection
        if env.terminations.get(name, False) or env.truncations.get(name, False):
            env.step(None)
            continue
        mask = np.asarray(env.observe(name)["action_mask"])
        legal = np.where(mask > 0.5)[0]
        env.step(int(rng.choice(legal)))
        steps += 1
    assert not env.agents                       # the game terminated
    scores = env.unwrapped.game.get_scores()
    assert len(scores) == 3
    assert all(s[0] >= 0 for s in scores)


def test_random_2p_game_completes_with_valid_scores():
    env = make_env(seed=1, num_players=2)
    rng = np.random.default_rng(0)
    steps = 0
    while env.agents and steps < 5000:
        name = env.agent_selection
        if env.terminations.get(name, False) or env.truncations.get(name, False):
            env.step(None)
            continue
        mask = np.asarray(env.observe(name)["action_mask"])
        legal = np.where(mask > 0.5)[0]
        env.step(int(rng.choice(legal)))
        steps += 1
    assert not env.agents                       # the 1-vs-1 game terminated
    scores = env.unwrapped.game.get_scores()
    assert len(scores) == 2
    assert all(s[0] >= 0 for s in scores)


def test_same_seed_is_reproducible():
    def first_obs(seed):
        env = make_env(seed=seed)
        return flatten_observation(env.observe(env.agent_selection)["observation"])

    assert np.allclose(first_obs(123), first_obs(123))


def _advance_to_mayor(env, rng, limit=4000):
    """Play legal random moves until a Mayor-phase decision is on the table."""
    from puerto_rico.constants import Phase
    for _ in range(limit):
        name = env.agent_selection
        if env.terminations.get(name, False) or env.truncations.get(name, False):
            env.step(None)
            continue
        if env.unwrapped.game.current_phase == Phase.MAYOR:
            return True
        mask = np.asarray(env.observe(name)["action_mask"])
        legal = np.where(mask > 0.5)[0]
        env.step(int(rng.choice(legal)))
    return False


def test_mayor_turn_always_has_a_legal_move():
    """Mayor is the only phase without a Pass, so an empty mask would stall it.

    Two engine invariants prevent that: every seat is recalled at the start of
    its Mayor turn (freeing all of its slots) and no seat can reach a Mayor turn
    owning zero colonists or zero tiles. Assert them where they matter.
    """
    from puerto_rico.constants import BUILDING_DATA, BuildingType, Phase, TileType

    for seed in range(12):
        env = make_env(seed=seed)
        rng = np.random.default_rng(seed)
        seen = 0
        while env.agents and seen < 400:
            name = env.agent_selection
            if env.terminations.get(name, False) or env.truncations.get(name, False):
                env.step(None)
                continue
            game = env.unwrapped.game
            mask = np.asarray(env.observe(name)["action_mask"])
            legal = np.where(mask > 0.5)[0]
            if game.current_phase == Phase.MAYOR:
                p = game.players[game.current_player_idx]
                assert legal.size > 0, "Mayor phase handed out an empty action mask"
                assert p.unplaced_colonists > 0
                seen += 1
            env.step(int(rng.choice(legal)))


def test_mayor_auto_advances_when_nothing_can_be_placed():
    """A seat with no colonist to place must be auto-advanced, not stalled.

    The state is unreachable through normal play, so it is injected directly:
    the guard exists so that a future engine change cannot deadlock the phase.
    """
    from puerto_rico.constants import Phase

    env = make_env(seed=3)
    rng = np.random.default_rng(3)
    assert _advance_to_mayor(env, rng), "never reached the Mayor phase"

    base = env.unwrapped
    game = base.game
    p = game.players[game.current_player_idx]

    # Nothing left to place, and Mayor has no Pass action -> empty mask.
    p.unplaced_colonists = 0
    assert not np.asarray(base.valid_action_mask()).any()

    before = (game.current_player_idx, game.players_taken_action, game.current_phase)
    assert base._execute_auto_actions() is True, "Mayor phase stalled on an empty mask"
    after = (game.current_player_idx, game.players_taken_action, game.current_phase)
    assert after != before, "auto-action reported progress but nothing moved"


def test_mayor_auto_advances_when_no_slots_remain():
    """Same guard, reached from the other side: colonists left but nowhere to put them."""
    from puerto_rico.constants import Phase, TileType

    env = make_env(seed=5)
    rng = np.random.default_rng(5)
    assert _advance_to_mayor(env, rng), "never reached the Mayor phase"

    base = env.unwrapped
    game = base.game
    p = game.players[game.current_player_idx]

    # Colonists in hand, but every island tile and city slot is gone.
    p.unplaced_colonists = 3
    for t in p.island_board:
        t.tile_type = TileType.EMPTY
        t.is_occupied = False
    p.city_board.clear()
    assert base._mayor_empty_slots(p) == []
    assert not np.asarray(base.valid_action_mask()).any()

    before = (game.current_player_idx, game.players_taken_action, game.current_phase)
    assert base._execute_auto_actions() is True, "Mayor phase stalled on an empty mask"
    assert (game.current_player_idx, game.players_taken_action, game.current_phase) != before
