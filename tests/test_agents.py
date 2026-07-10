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


@pytest.mark.parametrize("name,make", AGENT_FACTORIES)
def test_agent_plays_a_legal_game(name, make):
    agents = [make(), RandomAgent(seed=1), RandomAgent(seed=2)]
    result = play_game(agents, seed=0, time_limit_s=5.0)
    assert sum(result["illegal"]) == 0           # nobody emitted an illegal action
    assert len(result["winners"]) >= 1
    assert sum(result["scores"]) >= 0


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
