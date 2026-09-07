"""A record (seed + actions) replays to exactly the game that was played."""
import json

import numpy as np
import pytest

from puerto_rico import make_env
from puerto_rico.describe import describe_action
from puerto_rico.records import (ReplayError, build_record, game_over,
                                 load_record, replay_record, replays_exactly,
                                 save_record)


def _play_random_game(seed: int, num_players: int = 2):
    env = make_env(seed=seed, num_players=num_players)
    rng = np.random.default_rng(seed)
    actions = []
    for _ in range(10_000):
        if game_over(env):
            break
        name = env.agent_selection
        legal = np.where(np.asarray(env.observe(name)["action_mask"]) > 0.5)[0]
        action = int(rng.choice(legal))
        actions.append((env.agent_name_mapping[name], action))
        env.step(action)
    assert game_over(env), "the random game did not finish"
    return env, actions


@pytest.mark.parametrize("num_players", [2, 3])
def test_record_replays_to_the_same_scores(tmp_path, num_players):
    env, actions = _play_random_game(seed=1234, num_players=num_players)
    record = build_record(env, seed=1234, actions=actions,
                          player_types=["human", "random", "random"][:num_players])

    path = save_record(record, str(tmp_path))
    loaded = load_record(path)
    assert loaded == json.loads(json.dumps(record))      # JSON round trip

    replayed = replay_record(loaded)
    assert [list(map(int, s)) for s in replayed.game.get_scores()] == record["scores"]
    assert replays_exactly(loaded)
    assert record["human_seats"] == [0]
    assert record["human_won"] == (0 in record["winners"])


def test_tampered_record_is_rejected():
    env, actions = _play_random_game(seed=99)
    record = build_record(env, seed=99, actions=actions)

    masks = {}
    replay_record(record, on_step=lambda i, seat, a, e: masks.__setitem__(
        i, np.asarray(e.observe(e.agent_selection)["action_mask"])), stop_after=6)
    illegal = int(np.where(masks[5] == 0)[0][0])
    record["actions"][5][1] = illegal
    with pytest.raises(ReplayError, match="decision 5"):
        replay_record(record)
    assert not replays_exactly(record)

    wrong_seat = build_record(env, seed=99, actions=actions)
    wrong_seat["actions"][0][0] = 1 - wrong_seat["actions"][0][0]
    with pytest.raises(ReplayError, match="decision 0"):
        replay_record(wrong_seat)


def test_stop_after_leaves_the_next_player_to_move():
    env, actions = _play_random_game(seed=7)
    record = build_record(env, seed=7, actions=actions)
    partial = replay_record(record, stop_after=10)
    assert not game_over(partial)
    assert partial.agent_name_mapping[partial.agent_selection] == record["actions"][10][0]

    with pytest.raises(ReplayError, match="already ended"):
        replay_record({**record, "actions": record["actions"] + [record["actions"][-1]]})


def test_describe_action_names_the_role():
    assert describe_action(0, 2) == "Player 0 selected role BUILDER."
    assert "Unknown phase" in describe_action(1, 15)      # pass, no env given
