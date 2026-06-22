"""Pluggable rankers.

Official metric per track: **Elo** for the 2p (1-vs-1) track, **TrueSkill** for
the 3p track. **Win rate** is reported alongside in both. **α-Rank** is an opt-in
offline analysis metric, not part of the live leaderboard.
"""
from tournament.rankers.win_rate import rank_win_rate, wilson_ci
from tournament.rankers.elo import rank_elo
from tournament.rankers.trueskill_ranker import rank_trueskill
from tournament.rankers.alpha_rank import rank_alpha, alpha_rank

__all__ = ["rank_win_rate", "wilson_ci", "rank_elo", "rank_trueskill",
           "rank_alpha", "alpha_rank"]
