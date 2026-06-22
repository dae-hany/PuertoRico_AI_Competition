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
