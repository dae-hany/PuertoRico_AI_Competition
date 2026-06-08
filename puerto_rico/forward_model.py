"""
puerto_rico/forward_model.py — a clone-able view of the game for planning agents.

Most agents only need the observation vector and the action mask passed to
``Agent.act()``. **Planning** agents (lookahead, Monte-Carlo Tree Search, ...)
also need to *simulate* the consequences of actions. The ``ForwardModel`` gives
them exactly that, without exposing the full environment internals:

    model = ...                      # received in Agent.on_game_start(model)
    sim = model.clone()              # an independent copy you may freely mutate
    for a in sim.legal_actions():
        child = sim.clone()
        child.step(a)
        ... evaluate child.observation() / child.scores() ...

The live model always reflects the *current* decision point, so calling
``model.clone()`` inside ``act()`` snapshots the game exactly as the acting
player sees it. The bundled MCTS agent is built entirely on this API.
"""
import copy

import numpy as np

from puerto_rico.observation import flatten_observation


class ForwardModel:
    """A thin, clone-able facade over a Puerto Rico environment.

    Never mutate the live model you receive in ``on_game_start`` — clone it
    first. Each ``clone()`` is a fully independent deep copy.
    """

    def __init__(self, env):
        self._env = env

    # ── cloning ──────────────────────────────────────────────────────────────
    def clone(self) -> "ForwardModel":
        """Return an independent deep copy that can be mutated freely."""
        return ForwardModel(copy.deepcopy(self._env))

    # ── advanced access ──────────────────────────────────────────────────────
    @property
    def env(self):
        """The underlying PettingZoo environment (advanced use)."""
        return self._env

    @property
    def game(self):
        """The underlying engine state (``PuertoRicoGame``).

        Read-only: do not mutate it directly — ``clone()`` first. Heuristic
        agents read fields such as ``game.current_player_idx`` and
        ``game.players[i]`` from here.
        """
        return self._env.unwrapped.game

    # ── queries ──────────────────────────────────────────────────────────────
    def is_terminal(self) -> bool:
        """True once the game is over (no agents left to act)."""
        return not self._env.agents

    def current_player(self) -> int:
        """Index (0..num_players-1) of the player to move, or -1 if terminal."""
        if self.is_terminal():
            return -1
        return self._env.unwrapped.agent_name_mapping[self._env.agent_selection]

    def action_mask(self) -> np.ndarray:
        """Binary mask of shape (200,): 1 = legal action, 0 = illegal."""
        if self.is_terminal():
            return np.zeros(200, dtype=np.int8)
        obs = self._env.observe(self._env.agent_selection)
        return np.asarray(obs["action_mask"], dtype=np.int8)

    def legal_actions(self) -> list:
        """List of legal action indices for the player to move."""
        return [int(a) for a in np.where(self.action_mask() > 0.5)[0]]

    def observation(self) -> np.ndarray:
        """Flattened 293-dim observation for the player to move."""
        obs = self._env.observe(self._env.agent_selection)
        return flatten_observation(obs["observation"])

    def observation_dict(self) -> dict:
        """Raw *nested* observation (``global_state`` + per-player ``players``).

        Advanced: most agents use the flat ``observation`` argument of
        ``act()``. Some heuristic baselines read structured fields from here.
        """
        if self.is_terminal():
            return {}
        return self._env.observe(self._env.agent_selection)["observation"]

    def scores(self) -> list:
        """Per-player score tuples ``(total_vp, tiebreak, ship_vp, bldg_vp, bonus_vp)``."""
        return self._env.unwrapped.game.get_scores()

    def winners(self) -> list:
        """Indices of the winning player(s) (more than one only on an exact tie)."""
        scores = self.scores()
        best_vp = max(s[0] for s in scores)
        best_tb = max(s[1] for s in scores if s[0] == best_vp)
        return [i for i, s in enumerate(scores)
                if s[0] == best_vp and s[1] == best_tb]

    # ── transition ───────────────────────────────────────────────────────────
    def step(self, action: int) -> None:
        """Apply ``action`` for the current player and advance to the next
        decision point, skipping any players that have already finished."""
        self._env.step(int(action))
        self._skip_finished()

    def _skip_finished(self) -> None:
        env = self._env
        while env.agents:
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
            else:
                break
