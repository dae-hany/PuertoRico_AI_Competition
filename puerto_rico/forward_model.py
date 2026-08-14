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

Hidden information — the plantation draw order
----------------------------------------------
The face-down plantation pile is the *only* secret in Puerto Rico. Players
legitimately know *which* tiles remain (the multiset is deducible by counting
the public tiles), but never the *order* they will be drawn in. The forward
model enforces this for every agent, search-based or not:

* ``clone()`` **determinizes** the hidden pile — the copy's face-down order is
  randomly permuted (multiset preserved). Simulating a clone therefore plays
  out a random *consistent* world, never the true future draws. Cloning a clone
  keeps its order, so a search tree stays self-consistent within one move.
* The raw ``game``/``env`` accessors on the *live* model return a determinized
  snapshot too, so the real draw order cannot be read directly without cloning.

The randomisation is independent of the engine's own RNG (it never perturbs the
real game) yet reproducible when the match is seeded, so seeded matches still
replay identically.
"""
import copy
import random

import numpy as np

from puerto_rico.observation import flatten_observation


def _determinize_plantations(game, rng) -> None:
    """Randomly permute the face-down plantation order in place (multiset kept).

    Hides the only secret in the game — the draw order of the face-down
    plantation tiles — while preserving everything a player may legitimately
    deduce (which tiles remain). The discard pile is shuffled too; it is itself
    reshuffled by the engine before it is ever drawn, but randomising it here
    means a peek at its order reveals nothing either.
    """
    stack = getattr(game, "plantation_stack", None)
    if stack and len(stack) > 1:
        rng.shuffle(stack)
    discard = getattr(game, "plantation_discard", None)
    if discard and len(discard) > 1:
        rng.shuffle(discard)


class ForwardModel:
    """A thin, clone-able facade over a Puerto Rico environment.

    Never mutate the live model you receive in ``on_game_start`` — clone it
    first. Each ``clone()`` is a fully independent deep copy whose hidden
    plantation order is freshly determinized.
    """

    def __init__(self, env, *, _live=True, _rng=None):
        self._env = env
        self._live = _live
        if _rng is not None:
            self._rng = _rng
        elif _live:
            # An RNG independent of the engine's own stream (so determinization
            # never perturbs the real game) but derived from the match seed when
            # present (so seeded matches replay identically). The +1 offset keeps
            # it decorrelated from the seed that produced the *true* order.
            seed = getattr(env.unwrapped, "_seed", None)
            self._rng = random.Random() if seed is None else random.Random(seed + 1)
        else:
            self._rng = None
        # Cache for the redacted live view, rebuilt only when the game advances.
        self._redacted = None
        self._redacted_step = object()  # sentinel: forces first build

    # ── cloning ──────────────────────────────────────────────────────────────
    def clone(self) -> "ForwardModel":
        """Return an independent deep copy that can be mutated freely.

        Cloning the *live* model determinizes the hidden plantation order, so no
        amount of search can recover the true draw order. Cloning a clone keeps
        its order, so a search tree stays self-consistent within one move.
        """
        new_env = copy.deepcopy(self._env)
        if self._live:
            _determinize_plantations(new_env.unwrapped.game, self._rng)
        return ForwardModel(new_env, _live=False)

    # ── advanced access ──────────────────────────────────────────────────────
    @property
    def env(self):
        """The underlying PettingZoo environment (advanced use).

        On the live model this is a determinized snapshot — the real plantation
        draw order is never exposed. Clone first if you need to simulate.
        """
        return self._redacted_view() if self._live else self._env

    @property
    def game(self):
        """The underlying engine state (``PuertoRicoGame``).

        Read-only: do not mutate it directly — ``clone()`` first. Heuristic
        agents read fields such as ``game.current_player_idx`` and
        ``game.players[i]`` from here.

        On the live model this is a determinized snapshot: every public field is
        the true current value, but ``plantation_stack`` is randomly permuted so
        the secret draw order cannot be read off it.
        """
        view = self._redacted_view() if self._live else self._env
        return view.unwrapped.game

    def redacted_snapshot(self):
        """A determinized copy of the live env, safe to hand outside this process.

        Every public field holds its true current value, but the face-down
        plantation order is randomly permuted, so the snapshot carries no
        information the player is not entitled to. This is what the subprocess
        sandbox ships to an isolated planning agent: the agent gets a forward
        model to simulate with, while the real game — and the real draw order —
        stay in the organizer's process.
        """
        if not self._live:
            raise ValueError("redacted_snapshot() is only meaningful on the live model")
        return self._redacted_view()

    def _redacted_view(self):
        """A determinized snapshot of the live env, rebuilt when the game moves.

        Repeated reads within one decision point reuse the cached snapshot, so
        ``game``/``env`` stay stable and cheap during a single ``act()``.
        """
        step = getattr(self._env.unwrapped, "_game_step_count", None)
        if self._redacted is None or self._redacted_step != step:
            snap = copy.deepcopy(self._env)
            _determinize_plantations(snap.unwrapped.game, self._rng)
            self._redacted = snap
            self._redacted_step = step
        return self._redacted

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
        """Flattened observation for the player to move (220-dim in 2p, 293-dim in 3p)."""
        obs = self._env.observe(self._env.agent_selection)
        return flatten_observation(obs["observation"])

    def observation_dict(self) -> dict:
        """Raw *nested* observation (``global_state`` + per-player ``players``).

        Advanced: most agents use the flat ``observation`` argument of
        ``act()``. Some heuristic baselines read structured fields from here.
        The observation never encodes the plantation draw order.
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
