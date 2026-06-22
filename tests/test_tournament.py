from agents import ActionValueAgent, RandomAgent, TradeBuildingAgent
from tournament.leaderboard import build_leaderboard
from tournament.runner import run_tournament


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
