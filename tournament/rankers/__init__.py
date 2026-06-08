"""Pluggable rankers. Win-rate is the official metric; the others are secondary."""
from tournament.rankers.win_rate import rank_win_rate, wilson_ci
from tournament.rankers.trueskill_ranker import rank_trueskill
from tournament.rankers.alpha_rank import rank_alpha, alpha_rank

__all__ = ["rank_win_rate", "wilson_ci", "rank_trueskill", "rank_alpha", "alpha_rank"]
