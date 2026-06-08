"""
tournament/leaderboard.py — merge the rankings into one table (md / csv / json).

The table is ordered by the **official** metric (win rate). TrueSkill and α-Rank
are shown alongside as secondary columns.
"""
import csv
import json
import os


def build_leaderboard(result: dict):
    """Combine the ranker outputs of :func:`run_tournament` into ordered rows."""
    ts = {r["agent"]: r for r in result.get("trueskill", [])}
    al = {r["agent"]: r for r in result.get("alpha_rank", [])}

    board = []
    for rank, r in enumerate(result["win_rate"], 1):   # official order
        a = r["agent"]
        lo, hi = r["ci"]
        board.append({
            "rank": rank,
            "agent": a,
            "win_rate": r["win_rate"],
            "win_ci_low": lo,
            "win_ci_high": hi,
            "games": r["games"],
            "trueskill": ts.get(a, {}).get("score"),
            "alpha_mass": al.get(a, {}).get("mass"),
        })
    return board


def to_markdown(board) -> str:
    lines = [
        "| Rank | Agent | Win% (official) | 95% CI | Games | TrueSkill | α-Rank |",
        "|-----:|-------|----------------:|:------:|------:|----------:|-------:|",
    ]
    for row in board:
        ci = f"[{row['win_ci_low']:.2f}, {row['win_ci_high']:.2f}]"
        ts = f"{row['trueskill']:.2f}" if row["trueskill"] is not None else "—"
        al = f"{row['alpha_mass']:.3f}" if row["alpha_mass"] is not None else "—"
        lines.append(
            f"| {row['rank']} | {row['agent']} | {row['win_rate']*100:.1f}% | "
            f"{ci} | {row['games']} | {ts} | {al} |"
        )
    return "\n".join(lines)


def to_csv(board, path: str):
    fields = ["rank", "agent", "win_rate", "win_ci_low", "win_ci_high",
              "games", "trueskill", "alpha_mass"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(board)


def save(result: dict, out_dir: str = "results") -> dict:
    """Build the leaderboard and write ``leaderboard.{md,csv,json}`` to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    board = build_leaderboard(result)

    md = to_markdown(board)
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    to_csv(board, os.path.join(out_dir, "leaderboard.csv"))
    with open(os.path.join(out_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump({"leaderboard": board,
                   "payoff_matrix": result.get("payoff_matrix")}, f, indent=2)

    return {"board": board, "markdown": md}
