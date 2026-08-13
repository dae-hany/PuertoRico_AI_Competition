import numpy as np
import pytest

from agents import (ActionValueAgent, FactoryAgent, MctsAgent, PpoAgent,
                    RandomAgent, ShippingRushAgent, TradeBuildingAgent)
from tournament.match import play_game

AGENT_FACTORIES = [
    ("Random", RandomAgent),
    ("ActionValue", ActionValueAgent),
    ("ShippingRush", ShippingRushAgent),
    ("TradeBuilding", TradeBuildingAgent),
    ("Factory", FactoryAgent),
    ("MCTS", lambda: MctsAgent(num_simulations=8, max_rollout_depth=20)),
    ("PPO", PpoAgent),                      # untrained network, but must play legally
]


# Every baseline except PPO plays both tracks; the bundled PPO checkpoint is
# 293-dim, so it is 3p-only by construction.
TWO_PLAYER_FACTORIES = [(n, m) for n, m in AGENT_FACTORIES if n != "PPO"]


@pytest.mark.parametrize("name,make", AGENT_FACTORIES)
def test_agent_plays_a_legal_game(name, make):
    agents = [make(), RandomAgent(seed=1), RandomAgent(seed=2)]
    result = play_game(agents, seed=0, time_limit_s=5.0)
    assert sum(result["illegal"]) == 0           # nobody emitted an illegal action
    assert len(result["winners"]) >= 1
    assert sum(result["scores"]) >= 0


@pytest.mark.parametrize("name,make", TWO_PLAYER_FACTORIES)
def test_agent_plays_a_legal_2p_game(name, make):
    """The 2p track had no agent coverage at all, which is how MctsAgent could
    ship with hard-coded 3-player value vectors: every one of its moves raised,
    the harness quietly substituted a random legal action, and the "baseline"
    was really a random agent."""
    result = play_game([make(), RandomAgent(seed=1)], seed=0, time_limit_s=5.0)
    assert sum(result["illegal"]) == 0
    assert len(result["winners"]) >= 1


def test_actionvalue_scores_wharf_loads_at_their_real_ids():
    # Regression: the Wharf valuation logic used to sit at action IDs 74-78,
    # which the env never emits — real Wharf loads (59-63) fell into the
    # cargo-ship branch as an out-of-range ship index and scored 0.0, so the
    # greedy policy (and any search ordering reusing it) ranked a Wharf load
    # dead last. It must be scored by the barrels it ships, above pass.
    from puerto_rico import make_env
    from puerto_rico.constants import Good

    env = make_env(seed=4, num_players=2)
    game = env.unwrapped.game
    game.players[0].goods[Good.COFFEE] = 3

    h = ActionValueAgent()
    assert h._estimate_action_value(game, 0, 59, 0.0) == 3.0
    assert (h._estimate_action_value(game, 0, 59, 0.0)
            > h._estimate_action_value(game, 0, 15, 0.0))


def test_shippingrush_scores_wharf_loads_at_their_real_ids():
    # Same class of bug as the ActionValue regression above: the Wharf branch
    # scored actions 74-78, which the env never emits, so it could never fire —
    # while the agent's building priority list buys the Wharf *first*.
    agent = ShippingRushAgent()
    mask = np.zeros(200, dtype=np.int8)
    mask[59:64] = 1                     # only Wharf loads are legal right now
    goods = np.array([3, 0, 0, 0, 0])   # three Coffee

    action, score = agent._get_best_shipping_action(
        mask, goods,
        cargo_ships_good_onehot=np.zeros(18), cargo_ships_load=np.zeros(3),
        cargo_ships_space=np.zeros(3),  # every ship full -> Wharf is the only option
        has_harbor=False, has_wharf=True, wharf_used=False)

    assert action == 59, "Wharf load of Coffee is action 59"
    assert score > 0


def test_factory_settler_uses_tile_type_ids_and_takes_quarry_at_13():
    from puerto_rico.constants import TileType

    agent = FactoryAgent()
    face_up = np.array([int(TileType.COFFEE_PLANTATION)])

    mask = np.zeros(200, dtype=np.int8)
    mask[8] = 1                                     # 8 + TileType.COFFEE
    assert agent._settler_action(mask, face_up, {}) == 8

    mask[13] = 1                                    # Quarry is 13, never 14
    assert agent._settler_action(mask, face_up, {}) == 13


def test_factory_reads_real_ship_capacities():
    # The 2p track has two ships (4 and 6). A hard-coded [4, 5, 6] invented a
    # third ship and gave ship 1 one slot too many.
    agent = FactoryAgent()
    mask = np.zeros(200, dtype=np.int8)
    mask[44 + 5 + 0] = 1                            # ship 1, Coffee
    goods = np.array([6, 0, 0, 0, 0])

    action, score = agent._get_best_shipping_action(
        mask, goods,
        cargo_ships_good=np.array([5, 5]), cargo_ships_load=np.array([0, 0]),
        cargo_ships_capacity=np.array([4, 6]), has_harbor=False)

    assert action == 49
    assert score == 6 * 10 + 6, "ship 1 holds 6 in the 2p track, not 5"


def test_actionvalue_beats_random():
    wins = 0.0
    n = 8
    for g in range(n):
        result = play_game(
            [ActionValueAgent(), RandomAgent(seed=g), RandomAgent(seed=g + 9)],
            seed=g,
        )
        if 0 in result["winners"]:
            wins += 1.0 / len(result["winners"])
    assert wins / n > 0.5        # a strong heuristic clearly beats random
