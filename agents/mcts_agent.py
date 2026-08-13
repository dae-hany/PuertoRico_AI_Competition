"""
agents/mcts_agent.py — Max^N UCT Monte Carlo Tree Search agent.

A planning baseline. At each move it runs ``num_simulations`` cycles of
select -> expand -> rollout -> backpropagate on clones of the
:class:`~puerto_rico.forward_model.ForwardModel`, then plays the most-visited
action at the root.

Because Puerto Rico is a general-sum game, the tree uses **Max^N** value
vectors: every node stores one value estimate per player, and each player
greedily maximises its own component during selection. The vector length is
taken from the game being searched, so the same agent plays both competition
tracks (2p and 3p).

The hidden plantation order is determinized **once per move**: the agent clones
the live model once and simulates from that clone, so every simulation plays out
the same consistent world and the actions stored in the tree stay legal in it.

Strong but slow, and it has **no internal time cap** — cost is
``num_simulations`` x rollout length regardless of the clock. Tune
``num_simulations`` / ``max_rollout_depth`` to your per-move time budget (the
competition allows 1 s/move — see ``docs/COMPETITION_RULES.md``); the default of
200 does *not* fit that budget on a typical machine.
"""
import math

import numpy as np

from agents.base import Agent

_INF = float("inf")


class _Node:
    """A node in the Max^N MCTS tree."""

    __slots__ = ("parent", "action_taken", "acting_player", "n_players",
                 "children", "untried_actions", "visit_count", "value_sum")

    def __init__(self, parent=None, action_taken=None, acting_player=0,
                 n_players: int = 3):
        self.parent = parent
        self.action_taken = action_taken      # action that led here from parent
        self.acting_player = acting_player    # player to move at this node
        self.n_players = n_players            # length of the Max^N value vector
        self.children = {}                    # action -> _Node
        self.untried_actions = None           # set on first visit
        self.visit_count = 0
        self.value_sum = np.zeros(n_players, dtype=np.float32)

    @property
    def q(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.n_players, dtype=np.float32)
        return self.value_sum / self.visit_count

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def uct_child(self, c_uct: float) -> "_Node":
        """Child maximising UCT for *this node's* acting player."""
        p = self.acting_player
        log_n = math.log(self.visit_count)
        best_score, best_child = -_INF, None
        for child in self.children.values():
            if child.visit_count == 0:
                score = _INF
            else:
                score = child.q[p] + c_uct * math.sqrt(log_n / child.visit_count)
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def best_action(self) -> int:
        """Most-visited child's action (robust child selection)."""
        return max(self.children, key=lambda a: self.children[a].visit_count)


def _terminal_rewards(model) -> np.ndarray:
    """Reward vector at a finished game: +1 winner, -1 losers (shared on ties)."""
    rewards = np.full(len(model.scores()), -1.0, dtype=np.float32)
    winners = model.winners()
    share = 1.0 / len(winners)
    for w in winners:
        rewards[w] = share
    return rewards


def _heuristic_rewards(model) -> np.ndarray:
    """Proxy reward for a truncated rollout: rank by current VP (top = +1)."""
    vps = [s[0] for s in model.scores()]
    best = max(vps)
    rewards = np.where(np.array(vps) == best, 1.0, -1.0).astype(np.float32)
    if all(v == best for v in vps):   # all tied -> neutral, avoid bias
        rewards[:] = 0.0
    return rewards


def _random_rollout(model, max_depth, rng) -> np.ndarray:
    """Play ``model`` to the end (or ``max_depth`` steps) with a random policy."""
    steps = 0
    while not model.is_terminal():
        if max_depth is not None and steps >= max_depth:
            return _heuristic_rewards(model)
        legal = model.legal_actions()
        model.step(int(rng.choice(legal)) if legal else 15)
        steps += 1
    return _terminal_rewards(model)


class MctsAgent(Agent):
    """Max^N UCT MCTS planning agent (general-sum, any player count).

    The Max^N value vector is sized from the game it is handed, so one instance
    plays both the 2p and the 3p track.

    Args:
        num_simulations: rollouts per move (higher = stronger and slower). The
            default is chosen to fit the competition's 1 s/move budget with room
            to spare — measured ~0.3 s/move, where the previous default of 200
            sat right on the limit and would time out on a slower machine, i.e.
            the baseline would have been breaking the rule entrants must keep.
        c_uct: UCT exploration constant (default ``sqrt(2)``).
        max_rollout_depth: cap rollout length; ``None`` plays to the end.
        seed: RNG seed for reproducibility.
    """

    name = "MCTS"

    def __init__(self, num_simulations: int = 60, c_uct: float = math.sqrt(2),
                 max_rollout_depth: int = 40, seed: int = 42):
        super().__init__()
        self.num_simulations = num_simulations
        self.c_uct = c_uct
        self.max_rollout_depth = max_rollout_depth
        self._rng = np.random.default_rng(seed)
        self._model = None

    def on_game_start(self, forward_model=None):
        self._model = forward_model

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self._model is None:
            legal = np.where(np.asarray(action_mask) > 0.5)[0]
            return int(self._rng.choice(legal)) if len(legal) else 15

        # Seat count of the game actually being played (2 in the 1-vs-1 track,
        # 3 in the 3p track) — every value vector in the tree uses this length.
        n_players = len(self._model.scores())

        # Determinize ONCE per move, then simulate from that world.
        #
        # Cloning the *live* model reshuffles the face-down plantations every
        # time, so cloning it per simulation would put every simulation in a
        # different world while the tree still stores actions chosen in an
        # earlier one — "take the face-up Sugar" is not a legal move in a world
        # that dealt Coffee there. Cloning a clone preserves the order (see
        # ForwardModel.clone), which is exactly the self-consistency a search
        # tree needs within one move.
        root_model = self._model.clone()

        root = _Node(acting_player=root_model.current_player(),
                     n_players=n_players)
        root.untried_actions = [int(a) for a in np.where(np.asarray(action_mask) > 0.5)[0]]
        if not root.untried_actions:
            return 15

        for _ in range(self.num_simulations):
            sim = root_model.clone()
            node = root

            # 1. SELECTION — descend the tree by UCT until a not-fully-expanded node
            while node.is_fully_expanded() and node.children:
                child = node.uct_child(self.c_uct)
                sim.step(child.action_taken)
                node = child
                if sim.is_terminal():
                    break

            # 2. EXPANSION — try one untried action (unless terminal)
            if not sim.is_terminal() and node.untried_actions:
                action = int(self._rng.choice(node.untried_actions))
                node.untried_actions.remove(action)
                sim.step(action)
                if not sim.is_terminal():
                    child = _Node(parent=node, action_taken=action,
                                  acting_player=sim.current_player(),
                                  n_players=n_players)
                    child.untried_actions = sim.legal_actions()
                else:
                    child = _Node(parent=node, action_taken=action,
                                  acting_player=node.acting_player,
                                  n_players=n_players)
                    child.untried_actions = []
                node.children[action] = child
                node = child

            # 3. ROLLOUT
            if not sim.is_terminal():
                rewards = _random_rollout(sim, self.max_rollout_depth, self._rng)
            else:
                rewards = _terminal_rewards(sim)

            # 4. BACKPROPAGATION
            n = node
            while n is not None:
                n.visit_count += 1
                n.value_sum += rewards
                n = n.parent

        if not root.children:
            return root.untried_actions[0] if root.untried_actions else 15
        return int(root.best_action())
