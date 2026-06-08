"""Competition agents: the base interface, baselines, and the RL example.

Baseline opponents (strong → weak):

    MctsAgent           — Max^N UCT planning (strong, slow)
    ActionValueAgent    — one-step lookahead heuristic (strong, fast)
    ShippingRushAgent   — shipping-focused heuristic
    TradeBuildingAgent  — trade → building heuristic
    FactoryAgent        — Factory-engine heuristic
    PpoAgent            — self-play RL example (currently ~random; a starting point)
    RandomAgent         — uniform random over legal actions (weakest)
"""
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.action_value_agent import ActionValueAgent
from agents.shipping_rush_agent import ShippingRushAgent
from agents.trade_building_agent import TradeBuildingAgent
from agents.factory_agent import FactoryAgent
from agents.mcts_agent import MctsAgent
from agents.ppo_agent import PpoAgent

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
