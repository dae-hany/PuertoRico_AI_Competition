"""agents/random_agent.py — the simplest baseline: uniform random legal moves."""
import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Picks uniformly at random among the legal actions.

    The weakest baseline; useful as a sanity check (any competent agent should
    beat it comfortably). Pass a ``seed`` for reproducible behaviour.
    """

    name = "Random"

    def __init__(self, seed: int = None):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        legal = np.where(np.asarray(action_mask) > 0.5)[0]
        if len(legal) == 0:
            return 15  # "pass" — should not happen, the env always offers a move
        return int(self._rng.choice(legal))
