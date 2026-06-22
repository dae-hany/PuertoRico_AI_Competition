"""
puerto_rico/observation.py — flat observation encoding for competition agents.

The environment produces a *nested-dict* observation (global state + one block
per player). Competition agents instead receive the **flattened float vector**
returned by ``flatten_observation()``, together with a 200-dim binary action
mask (1 = legal, 0 = illegal).

Its length is ``GLOBAL_DIM + PER_PLAYER_DIM * num_players``, so it depends on the
track::

    2p (1-vs-1):  74 + 73*2 = 220
    3p:           74 + 73*3 = 293   (== OBS_DIM)

Flat layout::

    [ global(74) | player_0(73) | player_1(73) | ... ]

Within each block the sub-features are concatenated in *sorted key order*.
This exact ordering is the one the bundled (3p) PPO baseline was trained on, so
it must not change. The action space is a fixed ``Discrete(200)`` in both tracks.
See ``docs/OBSERVATION_AND_ACTIONS.md`` for the full field map.
"""
import numpy as np

GLOBAL_DIM = 74        # global-state features
PER_PLAYER_DIM = 73    # per-player features
OBS_DIM = 293          # flattened observation length for the 3p track (74 + 73*3)
ACTION_DIM = 200       # number of discrete actions (same in every track)


def flatten_observation(obs_dict: dict) -> np.ndarray:
    """Flatten the nested env observation into a 1-D ``float32`` vector.

    Args:
        obs_dict: the ``"observation"`` field returned by ``env.observe(agent)``,
            i.e. ``{"global_state": {...}, "players": {...}}``.

    Returns:
        ``np.ndarray`` of shape ``(GLOBAL_DIM + PER_PLAYER_DIM * num_players,)``
        — ``(220,)`` in the 2p track, ``(293,)`` in the 3p track — dtype ``float32``.
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
