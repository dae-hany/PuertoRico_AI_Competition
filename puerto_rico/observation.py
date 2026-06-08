"""
puerto_rico/observation.py — flat observation encoding for competition agents.

The environment produces a *nested-dict* observation (global state + one block
per player). Competition agents instead receive the **flattened 293-dimensional
float vector** returned by ``flatten_observation()``, together with a 200-dim
binary action mask (1 = legal, 0 = illegal).

Flat layout (3 players)::

    [ global(74) | player_0(73) | player_1(73) | player_2(73) ]  ->  293

Within each block the sub-features are concatenated in *sorted key order*.
This exact ordering is the one the bundled PPO baseline was trained on, so it
must not change. See ``docs/OBSERVATION_AND_ACTIONS.md`` for the full field map.
"""
import numpy as np

OBS_DIM = 293          # length of the flattened observation vector
ACTION_DIM = 200       # number of discrete actions
GLOBAL_DIM = 74        # global-state features
PER_PLAYER_DIM = 73    # per-player features


def flatten_observation(obs_dict: dict) -> np.ndarray:
    """Flatten the nested env observation into a 1-D ``float32`` vector.

    Args:
        obs_dict: the ``"observation"`` field returned by ``env.observe(agent)``,
            i.e. ``{"global_state": {...}, "players": {...}}``.

    Returns:
        ``np.ndarray`` of shape ``(293,)``, dtype ``float32``.
    """
    flat = []

    # Global state, sub-features in sorted key order.
    global_state = obs_dict["global_state"]
    for key in sorted(global_state.keys()):
        flat.append(global_state[key].flatten())

    # Per-player state, players in sorted order then sub-features in sorted order.
    players = obs_dict["players"]
    for p_key in sorted(players.keys()):
        p_state = players[p_key]
        for key in sorted(p_state.keys()):
            flat.append(p_state[key].flatten())

    return np.concatenate(flat).astype(np.float32)
