"""
Competition submission template.

Copy this file, rename the class, and implement `act`. That is the only method
you must write. See docs/SUBMISSION_GUIDE.md for the full walkthrough and
docs/OBSERVATION_AND_ACTIONS.md for what the observation/action numbers mean.
"""
import numpy as np

from agents.base import Agent


class MyAgent(Agent):
    # Shown on the leaderboard — change it to your team/agent name.
    name = "MyAgent"

    def __init__(self):
        super().__init__()
        self._rng = np.random.default_rng()

    def on_game_start(self, forward_model=None):
        """Optional. Called once at the start of every game.

        Planning agents (lookahead / MCTS) keep `forward_model` and call
        `forward_model.clone()` inside `act` to simulate. Reactive agents
        (rule-based or neural) can ignore it.
        """
        self.model = forward_model

    def act(self, observation, action_mask):
        """Choose ONE legal action.

        observation : float32 vector of shape (293,) — the game state.
        action_mask : int vector of shape (200,); action_mask[a] == 1 iff
                      action a is legal. You MUST return an `a` with mask[a] == 1.

        Returns: the chosen action index (int in [0, 200)).

        The body below plays a random legal move — replace it with your strategy.
        """
        legal = np.where(action_mask > 0.5)[0]
        return int(self._rng.choice(legal))
