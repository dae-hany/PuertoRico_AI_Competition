"""Human-readable descriptions of actions, for logs, replays, and debugging.

    >>> from puerto_rico.describe import describe_action
    >>> describe_action(0, 2)
    'Player 0 selected role BUILDER.'

The action index layout is documented in docs/OBSERVATION_AND_ACTIONS.md.
"""
from puerto_rico.constants import BuildingType, Good, Role, TileType


def describe_action(player_idx: int, action_idx: int, env=None) -> str:
    """One English sentence for ``action_idx`` played by seat ``player_idx``.

    ``env`` is optional: it is only used to name the current phase when the
    action is a pass (15).
    """
    try:
        if 0 <= action_idx <= 7:
            return f"Player {player_idx} selected role {Role(action_idx).name}."
        elif 8 <= action_idx <= 12:
            return f"Player {player_idx} drafted plantation {TileType(action_idx - 8).name}."
        elif action_idx == 13:
            return f"Player {player_idx} drafted a Quarry tile."
        elif action_idx == 15:
            game = getattr(env, "game", None)
            phase = getattr(game, "current_phase", None)
            return f"Player {player_idx} passed in {phase.name if phase else 'Unknown'} phase."
        elif 16 <= action_idx <= 38:
            return f"Player {player_idx} built {BuildingType(action_idx - 16).name}."
        elif 39 <= action_idx <= 43:
            return f"Player {player_idx} sold {Good(action_idx - 39).name} to Trading House."
        elif 44 <= action_idx <= 58:
            idx = action_idx - 44
            return f"Player {player_idx} loaded {Good(idx % 5).name} onto Cargo Ship {idx // 5 + 1}."
        elif 59 <= action_idx <= 63:
            return f"Player {player_idx} loaded {Good(action_idx - 59).name} via Wharf."
        elif action_idx == 105:
            return f"Player {player_idx} used Hacienda to draw an extra plantation."
        elif 64 <= action_idx <= 68:
            return f"Player {player_idx} stored {Good(action_idx - 64).name} on Windrose."
        elif 106 <= action_idx <= 110:
            return f"Player {player_idx} stored {Good(action_idx - 106).name} in Warehouse."
        elif 120 <= action_idx <= 125:
            return f"Player {player_idx} placed a colonist on {TileType(action_idx - 120).name}."
        elif 140 <= action_idx <= 162:
            return f"Player {player_idx} placed a colonist on {BuildingType(action_idx - 140).name}."
        elif 93 <= action_idx <= 97:
            return f"Player {player_idx} chose {Good(action_idx - 93).name} as Craftsman privilege."
        return f"Player {player_idx} executed action {action_idx}."
    except Exception as e:  # never let a description break a log line
        return f"Player {player_idx} executed action {action_idx} (desc error: {e})."
