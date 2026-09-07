"""Game records: ``(seed, actions)`` is enough to replay a game exactly.

Because ``make_env(seed=...)`` is deterministic and every decision is an
integer action, a finished game can be stored as a small JSON file and
replayed move by move — to look at the position where your agent went wrong,
to reproduce a bug report, or to keep a library of games.

A record is a plain JSON-serialisable dict::

    {
      "seed": 12345,                    # make_env(seed=..., num_players=...)
      "num_players": 2,
      "actions": [[seat, action], ...], # every decision, in play order
      "scores": [[vp, tiebreak, ...], ...],   # game.get_scores() at the end
      "winners": [0],
      "player_types": ["human", "submissions/my_agent.py:MyAgent"],   # optional
      "player_labels": ["You", "MyAgent"],                            # optional
      "human_seats": [0], "human_won": true, "time": "2026-09-07 10:00:00"
    }

The web UI (``webui/server.py``) writes one of these for every finished game
under ``results/webui_games/``; ``tools/replay_game.py`` replays them.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from puerto_rico.env import PuertoRicoEnv


class ReplayError(RuntimeError):
    """The recorded actions do not replay legally from the recorded seed."""


def game_over(env: PuertoRicoEnv) -> bool:
    """True once the game has ended.

    (PettingZoo keeps ``env.agents`` populated until each finished agent has
    been stepped with ``None``, so ``env.agents`` alone is not the signal.)
    """
    return not env.agents or bool(env.game.check_game_end())


def build_record(env: PuertoRicoEnv, seed: int, actions, player_types=None,
                 player_labels=None) -> dict:
    """Describe a finished game as a record dict (see the module docstring)."""
    scores = env.game.get_scores()
    vp = [int(s[0]) for s in scores]
    tb = [int(s[1]) for s in scores]
    best_vp = max(vp)
    best_tb = max(tb[i] for i in range(len(vp)) if vp[i] == best_vp)
    winners = [i for i in range(len(vp)) if vp[i] == best_vp and tb[i] == best_tb]
    player_types = list(player_types or [])
    human_seats = [i for i, t in enumerate(player_types) if t == "human"]
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(seed),
        "num_players": int(env.num_players),
        "player_types": player_types,
        "player_labels": list(player_labels or []),
        "actions": [[int(s), int(a)] for s, a in actions],
        "scores": [list(map(int, s)) for s in scores],
        "winners": winners,
        "human_seats": human_seats,
        "human_won": bool(set(human_seats) & set(winners)),
    }


def save_record(record: dict, directory: str) -> str:
    """Write ``record`` as JSON into ``directory``; returns the file path."""
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"game_{stamp}_{record['seed'] % 100000}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return path


def load_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def replay_record(record: dict,
                  on_step: Optional[Callable[[int, int, int, PuertoRicoEnv], None]] = None,
                  stop_after: Optional[int] = None) -> PuertoRicoEnv:
    """Replay a record from its seed and return the resulting environment.

    ``on_step(i, seat, action, env)`` is called before decision ``i`` is
    applied (so ``env`` shows the position the player saw). ``stop_after=n``
    applies only the first ``n`` decisions. Raises :class:`ReplayError` if a
    decision is out of turn, illegal, or played after the game ended.
    """
    from puerto_rico import make_env  # local import: puerto_rico/__init__ imports us

    env = make_env(seed=record["seed"], num_players=record["num_players"])
    actions = record["actions"]
    if stop_after is not None:
        actions = actions[:stop_after]
    for i, (seat, action) in enumerate(actions):
        if game_over(env):
            raise ReplayError(f"decision {i}: the game had already ended")
        name = env.agent_selection
        if env.agent_name_mapping[name] != seat:
            raise ReplayError(f"decision {i}: seat {seat} recorded, but "
                              f"seat {env.agent_name_mapping[name]} is to move")
        if env.observe(name)["action_mask"][action] == 0:
            raise ReplayError(f"decision {i}: action {action} is illegal for seat {seat}")
        if on_step is not None:
            on_step(i, seat, action, env)
        env.step(action)
    return env


def replays_exactly(record: dict) -> bool:
    """True if the record replays legally and ends with the recorded scores."""
    try:
        env = replay_record(record)
    except ReplayError:
        return False
    if "scores" not in record:
        return True
    return [list(map(int, s)) for s in env.game.get_scores()] == record["scores"]
