"""agents/registry.py - the bundled baselines by name, and a loader for anything else.

    from agents.registry import make_agent

    make_agent("TradeBuilding")                     # a bundled baseline, by name
    make_agent("submissions/my_agent.py:MyAgent")   # your own file
    make_agent("my_package.bots:StrongBot")         # an importable module

Names are case-insensitive. The tools in ``tools/`` (play, validate_submission,
run_ladder) and the docs all use these names, so a participant can say
``--vs Search`` instead of editing a registry.
"""
import os

from agents import (ActionValueAgent, FactoryAgent, MctsAgent, RandomAgent,
                    SearchAgent, SearchLiteAgent, ShippingRushAgent,
                    TradeBuildingAgent)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPO_CHECKPOINT = os.path.join(REPO_ROOT, "training", "checkpoints", "ppo_baseline.pt")


def _ppo():
    from agents import PpoAgent          # lazy: needs torch
    return PpoAgent(PPO_CHECKPOINT if os.path.exists(PPO_CHECKPOINT) else None)


#: name -> zero-argument factory, in rough order of strength.
BASELINES = {
    "Random": RandomAgent,
    "Factory": FactoryAgent,
    "ShippingRush": ShippingRushAgent,
    "TradeBuilding": TradeBuildingAgent,
    "ActionValue": ActionValueAgent,
    "MCTS": MctsAgent,                   # 60 simulations, both tracks (slow)
    "SearchLite": SearchLiteAgent,       # alpha-beta, 250 nodes (2p; reactive in 3p)
    "Search": SearchAgent,               # alpha-beta, 1500 nodes (2p; reactive in 3p)
    "PPO": _ppo,                         # RL baseline, 3p only, needs torch
}

#: which tracks (player counts) a baseline is meant for; absent = both.
BASELINE_TRACKS = {"SearchLite": (2,), "Search": (2,), "PPO": (3,)}

#: the fast opponents the tools play against when none are named
DEFAULT_OPPONENTS = {2: ["ActionValue", "TradeBuilding"],
                     3: ["ShippingRush", "ActionValue"]}


def baseline_names(num_players: int = None, include_ppo: bool = None):
    """Baseline names for a track (all of them when ``num_players`` is None).

    PPO is listed for the 3p track only when torch imports.
    """
    names = []
    for name in BASELINES:
        tracks = BASELINE_TRACKS.get(name)
        if num_players is not None and tracks and num_players not in tracks:
            continue
        if name == "PPO":
            ok = include_ppo
            if ok is None:
                try:
                    import torch  # noqa: F401
                    ok = True
                except Exception:
                    ok = False
            if not ok:
                continue
        names.append(name)
    return names


def lookup_baseline(name: str):
    """The canonical baseline name for ``name`` (case-insensitive), or None."""
    low = name.strip().lower()
    for key in BASELINES:
        if key.lower() == low:
            return key
    return None


def resolve(spec: str, base_dir: str = None):
    """``(label, factory)`` for a baseline name or a ``file.py:Class`` /
    ``module:Class`` spec. Raises ValueError/TypeError for a bad spec."""
    key = lookup_baseline(spec)
    if key is not None:
        return key, BASELINES[key]
    from tournament.sandbox import load_agent_class
    cls = load_agent_class(spec, base_dir=base_dir)
    return getattr(cls, "name", None) or cls.__name__, cls


def make_agent(spec: str, base_dir: str = None):
    """Instantiate a baseline (by name) or any agent given as a spec."""
    return resolve(spec, base_dir)[1]()
