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
