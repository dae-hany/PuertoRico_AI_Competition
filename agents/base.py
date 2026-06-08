"""
agents/base.py — the interface every competition agent implements.

A competition agent is any object that, given the current observation and the
set of legal actions, returns one legal action. Subclass :class:`Agent` and
implement :meth:`act`; that is all a basic agent needs.

    import numpy as np
    from agents.base import Agent

    class MyAgent(Agent):
        name = "MyAgent"

        def act(self, observation, action_mask):
            legal = np.where(action_mask > 0.5)[0]
            return int(legal[0])          # always pick the first legal action

Planning agents (lookahead / MCTS) additionally override :meth:`on_game_start`
to capture a :class:`~puerto_rico.forward_model.ForwardModel`, which lets them
simulate future states. See ``agents/mcts_agent.py`` for a worked example.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:                       # avoid importing the env at module load
    from puerto_rico.forward_model import ForwardModel


class Agent:
    """Base class for all Puerto Rico competition agents."""

    #: Human-readable name shown on the leaderboard. Override in subclasses.
    name: str = "Agent"

    def on_game_start(self, forward_model: "Optional[ForwardModel]" = None) -> None:
        """Called once at the start of every game (optional to override).

        Args:
            forward_model: a live, clone-able view of the game. Reactive agents
                may ignore it; planning agents should store it and call
                ``forward_model.clone()`` inside :meth:`act` to simulate.
        """

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Choose one legal action for the current state.

        Args:
            observation: float32 vector of shape (293,) — the game state.
            action_mask: int vector of shape (200,); ``mask[a] == 1`` iff action
                ``a`` is legal. You must return an action with ``mask[a] == 1``.

        Returns:
            The chosen action index (an ``int`` in ``[0, 200)``).
        """
        raise NotImplementedError
