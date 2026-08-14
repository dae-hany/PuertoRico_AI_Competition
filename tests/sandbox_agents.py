"""Agents used by ``tests/test_sandbox.py`` — well-behaved and adversarial.

They are loaded by *path* (``tests/sandbox_agents.py:WellBehavedAgent``), the
same way a real submission is, so the tests exercise the loader too.
"""
import time

import numpy as np

from agents.base import Agent


def _first_legal(action_mask) -> int:
    return int(np.where(np.asarray(action_mask) > 0.5)[0][0])


class WellBehavedAgent(Agent):
    """A reactive agent: never touches the forward model."""

    name = "WellBehaved"

    def act(self, observation, action_mask):
        return _first_legal(action_mask)


class PlannerAgent(Agent):
    """Overrides ``on_game_start``, so the sandbox must ship it a snapshot."""

    name = "Planner"

    def on_game_start(self, forward_model=None):
        self.model = forward_model

    def act(self, observation, action_mask):
        legal = [int(a) for a in np.where(np.asarray(action_mask) > 0.5)[0]]
        me = self.model.current_player()
        best, best_value = legal[0], -1e18
        for action in legal[:4]:                  # a shallow, cheap lookahead
            child = self.model.clone()
            child.step(action)
            value = child.scores()[me][0]
            if value > best_value:
                best, best_value = action, value
        return best


class AlwaysSlowAgent(Agent):
    """Never answers inside the budget. In-process this stalls the run for ever."""

    name = "AlwaysSlow"

    def act(self, observation, action_mask):
        while True:
            time.sleep(0.05)


class CrashingAgent(Agent):
    name = "Crashing"

    def act(self, observation, action_mask):
        raise RuntimeError("boom")


class TamperingAgent(Agent):
    """Reaches past the documented API to edit the real game."""

    name = "Tampering"

    def on_game_start(self, forward_model=None):
        self.model = forward_model

    def act(self, observation, action_mask):
        try:
            game = self.model._env.unwrapped.game
            game.players[0].doubloons += 500
            game.vp_chips = 0
            game.colonists_supply = 0
        except Exception:
            pass
        return _first_legal(action_mask)


class DeckReportingAgent(Agent):
    """Raises with the face-down plantation order it can see.

    Raising is how a test in the parent process gets to read something the
    isolated agent computed — the exception message travels back over the pipe.
    """

    name = "DeckReporting"

    def on_game_start(self, forward_model=None):
        self.model = forward_model

    def act(self, observation, action_mask):
        stack = self.model._env.unwrapped.game.plantation_stack
        raise RuntimeError("DECK:" + ",".join(str(int(t)) for t in stack))


class NotAnAgent:
    """Deliberately not an Agent subclass — the loader must reject it."""
