from agents import ActionValueAgent, RandomAgent, TradeBuildingAgent
from tournament.leaderboard import build_leaderboard
from tournament.runner import run_tournament


def _small_pool():
    return {
        "Random": lambda: RandomAgent(),
        "ActionValue": lambda: ActionValueAgent(),
        "TradeBuilding": lambda: TradeBuildingAgent(),
    }


def test_all_three_rankers_produce_an_order():
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0,
                         compute_alpha_rank=True, alpha_games_per_pair=2)
    assert len(res["win_rate"]) == 3
    assert len(res["trueskill"]) == 3
    assert len(res["alpha_rank"]) == 3


def test_leaderboard_is_ordered_by_official_metric():
    res = run_tournament(_small_pool(), games_per_seating=1, seed=0,
                         compute_alpha_rank=False)
    board = build_leaderboard(res)
    assert board[0]["rank"] == 1
    win_rates = [row["win_rate"] for row in board]
    assert win_rates == sorted(win_rates, reverse=True)


def test_strong_agent_outranks_random():
    res = run_tournament(_small_pool(), games_per_seating=2, seed=0,
                         compute_alpha_rank=False)
    order = [row["agent"] for row in build_leaderboard(res)]
    assert order.index("ActionValue") < order.index("Random")
