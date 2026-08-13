import pytest

from agents import (ActionValueAgent, FactoryAgent, RandomAgent,
                    ShippingRushAgent, TradeBuildingAgent)
from tournament.leaderboard import build_leaderboard
from tournament.match import play_game
from tournament.runner import run_tournament

SELF_MATCH_BASELINES = [
    ("Random", lambda: RandomAgent(seed=0)),
    ("Factory", FactoryAgent),
    ("TradeBuilding", TradeBuildingAgent),
    ("ShippingRush", ShippingRushAgent),
    ("ActionValue", ActionValueAgent),
]


@pytest.mark.parametrize("name,make", SELF_MATCH_BASELINES)
@pytest.mark.parametrize("n_seats", [2, 3])
def test_every_baseline_self_match_finishes(name, make, n_seats):
    """The round-robin plays groups like [A, A, A], so a baseline has to be able
    to finish a game against itself.

    Puerto Rico only ends when somebody spends the resources that trigger the
    end, and the agents choose the roles. A table that never picks the Mayor
    never spends a colonist and plays for ever: an all-Factory 3-player game ran
    734 rounds having chosen the Mayor 3 times, which meant the shipped
    `examples/run_tournament.py` hung.
    """
    result = play_game([make() for _ in range(n_seats)], seed=300_000,
                       time_limit_s=30.0)
    assert not result["truncated"], f"{name} x{n_seats} did not finish on its own"


def test_harness_caps_a_game_that_will_not_end():
    """Whatever the field does, one match must not be able to hang the run."""
    result = play_game([RandomAgent(seed=1), RandomAgent(seed=2)], seed=0,
                       max_steps=25)
    assert result["truncated"]
    assert result["steps"] == 25
    assert len(result["winners"]) >= 1        # still scoreable where it stands


def _small_pool():
    return {
        "Random": lambda: RandomAgent(),
        "ActionValue": lambda: ActionValueAgent(),
        "TradeBuilding": lambda: TradeBuildingAgent(),
    }


def test_3p_track_default_metrics():
    # 3p track: TrueSkill is official, win rate alongside, no α-Rank by default.
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0)
    assert res["official_metric"] == "trueskill"
    assert len(res["win_rate"]) == 3
    assert len(res["trueskill"]) == 3
    assert "alpha_rank" not in res          # opt-in only


def test_2p_track_uses_elo():
    # 2p (1-vs-1) track: Elo is official; every record seats exactly two agents.
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0, n_seats=2)
    assert res["official_metric"] == "elo"
    assert len(res["elo"]) == 3
    assert len(res["win_rate"]) == 3
    assert all(len(rec["agents"]) == 2 for rec in res["records"])


def test_alpha_rank_is_opt_in():
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0,
                         compute_alpha_rank=True, alpha_games_per_pair=2)
    assert len(res["alpha_rank"]) == 3


def test_leaderboard_is_ordered_by_official_metric_3p():
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0)
    board = build_leaderboard(res)
    assert board[0]["rank"] == 1
    scores = [row["trueskill"] for row in board]
    assert scores == sorted(scores, reverse=True)


def test_leaderboard_is_ordered_by_official_metric_2p():
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0, n_seats=2)
    board = build_leaderboard(res)
    assert board[0]["rank"] == 1
    elos = [row["elo"] for row in board]
    assert elos == sorted(elos, reverse=True)


def test_strong_agent_outranks_random_in_both_tracks():
    for n_seats in (2, 3):
        res = run_tournament(_small_pool(), games_per_seating=2, seed=0,
                             n_seats=n_seats)
        order = [row["agent"] for row in build_leaderboard(res)]
        assert order.index("ActionValue") < order.index("Random")
