"""
submissions/example_agent.py — a tiny example agent for the web UI.

Drop your own agent .py file in this folder; the web UI (webui/server.py)
auto-discovers every Agent subclass and lists it in the per-seat dropdown.
Copy this file as a starting point, or use submission_template/my_agent.py.
"""
import numpy as np

from agents.base import Agent


class ExampleAgent(Agent):
    """A simple, readable heuristic: prefer to act, ship, and build.

    It never plans ahead — it just inspects which actions are legal right now
    and applies a fixed preference order. It beats Random a bit, and shows how
    to read ``action_mask``. Replace ``act`` with your real strategy.
    """

    name = "ExampleAgent"

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        legal = set(int(a) for a in np.where(action_mask > 0.5)[0])

        # Preference order over action *groups* (see docs/OBSERVATION_AND_ACTIONS.md):
        #   Captain ship-loading (44-63) -> earns victory points by shipping
        #   Building (16-38)             -> permanent VP and engine pieces
        #   pick the Captain role (5)    -> enables shipping this round
        groups = [range(44, 64), range(16, 39), [5]]
        for group in groups:
            choices = [a for a in group if a in legal]
            if choices:
                return choices[0]

        # Otherwise take any legal action except passing, falling back to pass.
        non_pass = [a for a in legal if a != 15]
        return non_pass[0] if non_pass else 15
