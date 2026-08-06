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

# Offsets of the two absolute-player-indexed fields inside the global block.
# Both are derived from the sorted-key ordering documented above; the module-level
# self-check at the bottom of this file asserts they stay in sync with the env.
CURRENT_PLAYER_OFFSET = 36   # global_state["current_player"]
GOVERNOR_IDX_OFFSET   = 49   # global_state["governor_idx"]


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


def egocentric_view(flat: np.ndarray, self_idx: int = None,
                    num_players: int = None) -> np.ndarray:
    """Rotate a flat observation into the *acting player's* frame of reference.

    ``flatten_observation`` lays the player blocks out in absolute seat order, so
    the same network sees "my stuff" at a different offset depending on which seat
    it occupies. A parameter-sharing self-play policy then has to learn a gating
    rule ("if current_player == 1, read block 1") from a single raw scalar — which
    an MLP does badly. Rotating the view removes the problem entirely: **block 0 is
    always you**, block 1 the player to your left, and so on in turn order.

    The two absolute-player-indexed global fields (``current_player``,
    ``governor_idx``) are rewritten as offsets *relative to you*, so a rotated
    observation is fully self-consistent.

    This is a view transform on top of the canonical encoding — the competition's
    flat layout is unchanged, and any agent may opt in or ignore it.

    Args:
        flat: 1-D vector from :func:`flatten_observation`.
        self_idx: seat to centre on. ``None`` (default) reads it from the
            ``current_player`` field, which is correct whenever the observation
            belongs to the player about to move.
        num_players: inferred from ``len(flat)`` when omitted.

    Returns:
        A new ``float32`` vector of the same length, centred on ``self_idx``.
    """
    flat = np.asarray(flat, dtype=np.float32)
    if num_players is None:
        num_players = (flat.shape[-1] - GLOBAL_DIM) // PER_PLAYER_DIM
    if self_idx is None:
        self_idx = int(round(float(flat[CURRENT_PLAYER_OFFSET])))
    self_idx %= num_players

    out = flat.copy()
    if self_idx == 0:
        return out                      # already centred; nothing to rewrite

    for dst in range(num_players):
        src = (self_idx + dst) % num_players
        d = GLOBAL_DIM + dst * PER_PLAYER_DIM
        s = GLOBAL_DIM + src * PER_PLAYER_DIM
        out[d:d + PER_PLAYER_DIM] = flat[s:s + PER_PLAYER_DIM]

    for off in (CURRENT_PLAYER_OFFSET, GOVERNOR_IDX_OFFSET):
        out[off] = (float(flat[off]) - self_idx) % num_players

    return out
