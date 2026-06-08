"""Puerto Rico — core game engine, environment, and forward model.

Public API::

    from puerto_rico import make_env, PuertoRicoEnv, ForwardModel
    from puerto_rico import flatten_observation, OBS_DIM, ACTION_DIM, constants
"""
from puerto_rico import constants
from puerto_rico.env import PuertoRicoEnv
from puerto_rico.forward_model import ForwardModel
from puerto_rico.observation import ACTION_DIM, OBS_DIM, flatten_observation


def make_env(seed=None, num_players: int = 3) -> PuertoRicoEnv:
    """Create and reset a Puerto Rico environment, ready for the first move.

    Args:
        seed: if given, the game setup is seeded for reproducibility; if None,
            a fresh random game is created.
        num_players: number of players (the competition uses 3).
    """
    env = PuertoRicoEnv(num_players=num_players, random_seed_mode=True, fixed_seed=42)
    env.reset(seed=seed)
    return env


__all__ = [
    "make_env",
    "PuertoRicoEnv",
    "ForwardModel",
    "flatten_observation",
    "OBS_DIM",
    "ACTION_DIM",
    "constants",
]
