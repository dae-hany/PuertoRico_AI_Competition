"""Competition agents: the base interface, baselines, and the RL example.

Baseline opponents (strong → weak):

    PpoAgent            — PPO self-play RL (3p-only; beats every heuristic below)
    MctsAgent           — Max^N UCT planning (strong, slow)
    ActionValueAgent    — greedy heuristic action scorer (strong, fast)
    ShippingRushAgent   — shipping-focused heuristic
    TradeBuildingAgent  — trade → building heuristic
    FactoryAgent        — Factory-engine heuristic
    RandomAgent         — uniform random over legal actions (weakest)
"""
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.action_value_agent import ActionValueAgent
from agents.shipping_rush_agent import ShippingRushAgent
from agents.trade_building_agent import TradeBuildingAgent
from agents.factory_agent import FactoryAgent
from agents.mcts_agent import MctsAgent


def __getattr__(name):
    # PpoAgent is imported lazily so the package stays importable without the
    # optional `torch` dependency (see requirements.txt — torch is optional and
    # only needed for the PPO baseline / training).
    if name == "PpoAgent":
        from agents.ppo_agent import PpoAgent
        return PpoAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "RandomAgent",
    "ActionValueAgent",
    "ShippingRushAgent",
    "TradeBuildingAgent",
    "FactoryAgent",
    "MctsAgent",
    "PpoAgent",
]
