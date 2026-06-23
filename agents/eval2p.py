"""agents/eval2p.py — leaf evaluation for the 2p (1-vs-1) search agent.

The search agent (``agents/search_agent.py``) needs a scalar value for a game
state from the root player's perspective. Because the 2p track is a winner-takes-
all (zero-sum) game, we evaluate a **margin**:

    value(state, me) = h(me) - h(opp)

where ``h`` is a per-player "how much victory-point potential does this player
hold" heuristic. We reuse :meth:`ActionValueAgent._compute_heuristic`, which
already estimates realized VP (chips, buildings, activated large-building
bonuses) plus potential VP (goods, doubloons, production capacity, infrastructure)
for a single player. Taking the difference turns it into a zero-sum score, and
the difference also cancels much of the heuristic's absolute miscalibration.

At a finished game we ignore the heuristic and return the true outcome, scaled so
that a decided win always dominates any heuristic leaf estimate.

The weights live in :class:`ActionValueAgent`; Phase 3 tuning swaps in a
recalibrated, parameterised evaluator behind this same interface.
"""
from agents.action_value_agent import ActionValueAgent


class TwoPlayerActionValue(ActionValueAgent):
    """ActionValue heuristic with the role-frequency constant recalibrated for 2p.

    The base heuristic assumes the 3-player cadence (17 rounds × 3 picks over 6
    roles → a given role is picked ~8.5 times per game). Measured 2p games run
    ~11 rounds with 6 of 7 roles picked each round, so a role is used ~6·11/7 ≈
    9.5 times. That sets how much future "engine" payoff (production / commercial
    abilities) is worth. Everything else in the heuristic already reads the live
    game state, which adapts to the 2p setup on its own.
    """
    _NUM_ROLES = 7.0                          # 2p has 6 standard roles + Prospector
    _TOTAL_ROLE_SELECTIONS = 66.0             # ~11 rounds × 6 picks
    _EXPECTED_ROLE_USES_BASE = _TOTAL_ROLE_SELECTIONS / _NUM_ROLES  # ≈ 9.43


# Default leaf heuristic for the 2p track: the recalibrated subclass above.
# (Measured A/B vs the 3p-calibrated base is strength-neutral — the margin
# h(me)-h(opp) cancels the shared constant — but this is the principled,
# honest default for the 1-vs-1 track. _compute_heuristic / _estimate_action_value
# only read the game they are passed, so one shared instance is safe everywhere.)
_HEURISTIC = TwoPlayerActionValue()

# A decided result must outweigh any heuristic leaf value (heuristic magnitudes
# are on a VP-ish scale, well under a few hundred).
WIN_BASE = 10_000.0


def heuristic_player(game, idx: int, heuristic=None) -> float:
    """Per-player VP-potential heuristic (higher = better for player ``idx``)."""
    return (heuristic or _HEURISTIC)._compute_heuristic(game, idx)


def evaluate_state(game, me_idx: int, opp_idx: int, heuristic=None) -> float:
    """Zero-sum margin value of a non-terminal state for ``me_idx``."""
    h = heuristic or _HEURISTIC
    return h._compute_heuristic(game, me_idx) - h._compute_heuristic(game, opp_idx)


def terminal_value(scores, me_idx: int, opp_idx: int) -> float:
    """True outcome value of a finished game for ``me_idx``.

    ``scores`` is ``game.get_scores()`` — per-player ``(total_vp, tiebreak, ...)``.
    Returns ``±WIN_BASE`` for a win/loss (broken by VP then the tiebreak), plus
    the VP margin so the search prefers winning by more / losing by less.
    """
    my_vp, my_tb = scores[me_idx][0], scores[me_idx][1]
    op_vp, op_tb = scores[opp_idx][0], scores[opp_idx][1]
    margin = float(my_vp - op_vp)
    if (my_vp, my_tb) > (op_vp, op_tb):
        return WIN_BASE + margin
    if (my_vp, my_tb) < (op_vp, op_tb):
        return -WIN_BASE + margin
    return 0.0
