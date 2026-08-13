"""agents/search_agent.py — alpha-beta search agent for the 2p (1-vs-1) track.

Puerto Rico 1-vs-1 is a two-player, winner-takes-all (so effectively zero-sum),
**near-perfect-information** game: the only hidden state is the face-down
plantation draw order. That structure is exactly what adversarial search is good
at, so this agent plays by lookahead rather than by a fixed policy.

How it works
------------
* **Alpha-beta over the forward model.** At each decision node the *acting*
  player (``model.current_player()``) chooses — not strictly alternating, because
  one player can make several decisions in a row within a phase. So the node type
  (maximise vs minimise the root player's margin) is decided by *who is to move*,
  not by ply parity. Values are zero-sum margins from the root player's view
  (see ``agents/eval2p.py``).
* **Iterative deepening** with a wall-clock cap: search depth 1, 2, 3, … and keep
  the best move from the deepest *completed* depth. The per-move budget is 1 s
  (competition rule); we stop early and return the best move so far.
* **Move ordering** reuses ``ActionValueAgent``'s static action score so good
  moves are tried first, which makes alpha-beta prune far more.
* **Determinization** of the hidden plantation order: at the start of each move we
  reshuffle the face-down stack in our *cloned* search state, so the search never
  reads the true draw order (that would be both cheating and brittle).

Fallback: with no forward model, or in a non-2p game, it plays the reactive
``ActionValueAgent`` policy (best action by static score).
"""
import random
import time

import numpy as np

from agents.base import Agent
from agents.eval2p import (_HEURISTIC, evaluate_state, terminal_value,
                           WIN_BASE)

PASS_ACTION = 15


class _TimeUp(Exception):
    """Raised to unwind the search when the per-move budget is exhausted."""


class SearchAgent(Agent):
    """Iterative-deepening alpha-beta agent for 1-vs-1 Puerto Rico.

    Args:
        time_limit_s: hard per-move wall-clock safety ceiling (< the 1 s
            competition limit). A move never overruns this.
        max_depth: hard cap on iterative-deepening depth (plies).
        determinize: reshuffle the hidden plantation order before searching.
        seed: RNG seed for determinization (fixed → reproducible play).
        node_budget: stop after this many search nodes (default 1500). This
            fixes strength reproducibly across hardware — the agent plays the
            same move on any machine, only the wall time differs. Set to None
            for max-strength play that uses the whole time budget instead
            (stronger, but hardware-dependent). See :class:`SearchLiteAgent`
            for a lighter, beginner-friendly tier.
    """

    name = "Search"

    def __init__(self, time_limit_s: float = 0.9, max_depth: int = 64,
                 determinize: bool = True, seed: int = 0, heuristic=None,
                 node_budget=1500):
        super().__init__()
        # node_budget makes strength reproducible across hardware: the search
        # stops after exactly this many nodes, so it plays the SAME move on a fast
        # or slow machine (only the wall time differs). A fixed search *depth*
        # cannot do this — high-branching positions blow past 1 s before the depth
        # completes, so the time cap (hardware-dependent) would bind instead.
        # Set node_budget=None for max-strength play that uses the full time
        # budget (stronger, but hardware-dependent). time_limit_s is always a hard
        # safety ceiling so a move never overruns the 1 s competition limit.
        self.time_limit_s = time_limit_s
        self.max_depth = max_depth
        self.determinize = determinize
        self.node_budget = node_budget
        self.h = heuristic or _HEURISTIC      # leaf eval + move-ordering heuristic
        self._rng = random.Random(seed)
        self._model = None
        self._deadline = 0.0
        self._nodes = 0
        # diagnostics from the last act() (handy for benchmarking/tuning)
        self.last_depth = 0
        self.last_nodes = 0
        self.last_value = 0.0

    def on_game_start(self, forward_model=None):
        self._model = forward_model

    # ── policy entry point ────────────────────────────────────────────────────
    def act(self, observation, action_mask) -> int:
        mask = np.asarray(action_mask)
        legal = [int(a) for a in np.where(mask > 0.5)[0]]
        if not legal:
            return PASS_ACTION
        if len(legal) == 1:
            return legal[0]

        # Fallback: no clone-able model, or not a 2p game -> reactive policy.
        if self._model is None or len(self._model.game.players) != 2:
            return self._reactive_best(self._model.game, legal) \
                if self._model is not None else legal[0]

        # Set the deadline FIRST so clone + determinization + ordering are all
        # charged against the per-move budget (they are sub-ms, so this costs ~no
        # depth but removes any unbounded setup tail before the time checks begin).
        self._deadline = time.perf_counter() + self.time_limit_s
        self._nodes = 0
        me = self._model.current_player()
        root = self._model.clone()
        if self.determinize:
            self._reshuffle_hidden(root.game)
        best = self._reactive_order(root.game, legal, me)[0]  # sensible default

        # Iterative deepening: each completed depth refines `best`.
        prev_best = best
        for depth in range(1, self.max_depth + 1):
            try:
                value, move = self._alphabeta(root, depth, -np.inf, np.inf,
                                              me, root_hint=prev_best)
            except _TimeUp:
                break
            if move is not None:
                best = move
                prev_best = move
            self.last_depth = depth
            self.last_value = value
            # Stop if the result is already decided (proven win or loss).
            if abs(value) >= WIN_BASE - 1e3:
                break

        self.last_nodes = self._nodes
        return best

    # ── alpha-beta ────────────────────────────────────────────────────────────
    def _alphabeta(self, model, depth, alpha, beta, me, root_hint=None):
        self._nodes += 1
        # Node budget: checked every node (exact) so play is deterministic.
        if self.node_budget is not None and self._nodes >= self.node_budget:
            raise _TimeUp
        # Time cap: a coarse (every-16-nodes) hard safety net against overrun.
        if (self._nodes & 15) == 0 and time.perf_counter() > self._deadline:
            raise _TimeUp

        if model.is_terminal():
            return terminal_value(model.scores(), me, 1 - me), None
        if depth == 0:
            return evaluate_state(model.game, me, 1 - me, self.h), None

        legal = model.legal_actions()
        if not legal:
            # No real choice: advance and keep searching at the same depth.
            child = model.clone()
            child.step(PASS_ACTION)
            v, _ = self._alphabeta(child, depth, alpha, beta, me)
            return v, None

        cur = model.current_player()
        moves = self._reactive_order(model.game, legal, cur)
        if root_hint is not None and root_hint in moves:
            moves.remove(root_hint)
            moves.insert(0, root_hint)

        best_move = moves[0]
        if cur == me:                                   # MAX node
            value = -np.inf
            for a in moves:
                child = model.clone()
                child.step(a)
                v, _ = self._alphabeta(child, depth - 1, alpha, beta, me)
                if v > value:
                    value, best_move = v, a
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            return value, best_move
        else:                                           # MIN node (opponent)
            value = np.inf
            for a in moves:
                child = model.clone()
                child.step(a)
                v, _ = self._alphabeta(child, depth - 1, alpha, beta, me)
                if v < value:
                    value, best_move = v, a
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break
            return value, best_move

    # ── helpers ───────────────────────────────────────────────────────────────
    def _reactive_order(self, game, legal, player_idx):
        """Order actions best-first for `player_idx` using the static score.

        ``_estimate_action_value`` returns ``base + per-action bonus``; the base
        term is identical for every sibling, so it never changes the sort order.
        We pass ``0.0`` to skip the expensive O(board) ``_compute_heuristic`` pass
        at every interior node — same ordering, more nodes/depth for the budget.
        """
        scored = [(self.h._estimate_action_value(game, player_idx, a, 0.0), a)
                  for a in legal]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [a for _, a in scored]

    def _reactive_best(self, game, legal):
        """Reactive fallback: the single best action by static score."""
        me = game.current_player_idx
        return self._reactive_order(game, legal, me)[0]

    def _reshuffle_hidden(self, game):
        """Determinize: randomize the face-down plantation stack order.

        Keeps the search honest about the only hidden information in the game —
        the draw order of the face-down plantation tiles.
        """
        stack = game.plantation_stack
        if len(stack) > 1:
            self._rng.shuffle(stack)


class SearchLiteAgent(SearchAgent):
    """A lighter, beginner-friendly tier of :class:`SearchAgent`.

    Same alpha-beta engine, but a small node budget so it searches only a few
    plies — fast (~0.04 s/move) and, like its parent, identical in strength on
    any hardware.

    "Lighter" is relative to :class:`SearchAgent`, not to the field: measured, it
    still beats every bundled heuristic (88-100%, see ``docs/BASELINES.md``). It
    is the second rung of a ladder whose first rung is ``ActionValueAgent``.
    """

    name = "SearchLite"

    def __init__(self, time_limit_s: float = 0.9, seed: int = 0,
                 heuristic=None, node_budget: int = 250):
        super().__init__(time_limit_s=time_limit_s, seed=seed,
                         heuristic=heuristic, node_budget=node_budget)
