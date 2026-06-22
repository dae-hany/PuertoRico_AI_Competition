"""
tournament/leaderboard.py — merge the rankings into one table (md / csv / json).

The table is ordered by the track's **official** metric — **Elo** in the 2p
(1-vs-1) track, **TrueSkill** in the 3p track — with **win rate** shown
alongside. **α-Rank** is only included when it was computed (it is opt-in
analysis, not part of the standings).
"""
import csv
import json
import os

_METRIC_TITLE = {
    "elo": "Elo (official)",
    "trueskill": "TrueSkill (official)",
    "win_rate": "Win% (official)",
}


def build_leaderboard(result: dict):
    """Combine the ranker outputs of :func:`run_tournament` into ordered rows.

    Rows are ordered by ``result["official_metric"]`` (``"elo"`` for the 2p
    track, ``"trueskill"`` for the 3p track; falls back to win rate).
    """
    official = result.get("official_metric", "win_rate")
    wr = {r["agent"]: r for r in result.get("win_rate", [])}
    ts = {r["agent"]: r for r in result.get("trueskill", [])}
    el = {r["agent"]: r for r in result.get("elo", [])}
    al = {r["agent"]: r for r in result.get("alpha_rank", [])}

    order = result.get(official) or result.get("win_rate", [])

    board = []
    for rank, r in enumerate(order, 1):       # official order
        a = r["agent"]
        w = wr.get(a, {})
        lo, hi = w.get("ci", (None, None))
        board.append({
            "rank": rank,
            "agent": a,
            "official_metric": official,
            "elo": el.get(a, {}).get("elo"),
            "trueskill": ts.get(a, {}).get("score"),
            "win_rate": w.get("win_rate"),
            "win_ci_low": lo,
            "win_ci_high": hi,
            "games": w.get("games"),
            "alpha_mass": al.get(a, {}).get("mass"),
        })
    return board


def _official_cell(row) -> str:
    official = row.get("official_metric", "win_rate")
    if official == "elo":
        return f"{row['elo']:.0f}" if row.get("elo") is not None else "—"
    if official == "trueskill":
        return f"{row['trueskill']:.2f}" if row.get("trueskill") is not None else "—"
    return f"{row['win_rate']*100:.1f}%" if row.get("win_rate") is not None else "—"


def to_markdown(board) -> str:
    if not board:
        return "(empty leaderboard)"
    official = board[0].get("official_metric", "win_rate")
    show_alpha = any(row.get("alpha_mass") is not None for row in board)

    headers = ["Rank", "Agent", _METRIC_TITLE.get(official, "Official"),
               "Win%", "95% CI", "Games"]
    aligns = ["-----:", "-------", "----------:", "-----:", ":------:", "------:"]
    if show_alpha:
        headers.append("α-Rank")
        aligns.append("-------:")

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(aligns) + "|"]
    for row in board:
        wr = f"{row['win_rate']*100:.1f}%" if row.get("win_rate") is not None else "—"
        ci = (f"[{row['win_ci_low']:.2f}, {row['win_ci_high']:.2f}]"
              if row.get("win_ci_low") is not None else "—")
        games = row.get("games")
        cells = [str(row["rank"]), row["agent"], _official_cell(row), wr, ci,
                 str(games) if games is not None else "—"]
        if show_alpha:
            cells.append(f"{row['alpha_mass']:.3f}" if row.get("alpha_mass") is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(board, path: str):
    fields = ["rank", "agent", "official_metric", "elo", "trueskill",
              "win_rate", "win_ci_low", "win_ci_high", "games", "alpha_mass"]
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
        json.dump({"official_metric": result.get("official_metric"),
                   "n_seats": result.get("n_seats"),
                   "leaderboard": board,
                   "payoff_matrix": result.get("payoff_matrix")}, f, indent=2)

    return {"board": board, "markdown": md}
