# Web UI — play and debug agents

A local browser tool to watch agents play, play a seat yourself, and **debug your
own agent** against the baselines.

```bash
pip install -e ".[webui]"      # or: pip install flask
python webui/server.py         # open http://127.0.0.1:5000
```

![Two baseline agents playing a 2-player game](../docs/img/webui.png)

## What you can do

- **Pick the player count** (2–5). The competition tracks are **2 Players**
  (1‑vs‑1) and **3 Players**; 4–5 are available for experimentation.
- **Configure each seat** independently: a human (you), any baseline
  (Random, Factory, TradeBuilding, ShippingRush, ActionValue, MCTS, or — in 3p —
  PPO), or **your own agent**. The PPO baseline expects the 293-dim 3p
  observation, so use it only with 3 players.
- **Load your agent** two ways: drop a `.py` in [`../submissions/`](../submissions/)
  (auto-listed in the dropdown), or type `module:Class` / `path/to/file.py:Class`
  in a seat's box.
- **Watch bots play** (set every seat to a bot) with auto-run, or **play yourself**
  by taking a seat. Undo, pass, and restart are available.
- **Start from a link.** The setup can be filled in from the URL, so a game is
  shareable: `http://127.0.0.1:5000/?players=2&seats=actionvalue,trade` watches
  two bots, `?players=3&seats=human,mcts,shipping` seats you against two bots.
  Seat tokens are the dropdown values (`human`, `random`, `factory`, `trade`,
  `shipping`, `actionvalue`, `mcts`, `searchlite`, `search`, `ppo`) or a
  `module:Class` / `file.py:Class` spec; add `&auto=0` to step the bots by hand.

## Debugging features

Each agent move is run under the **same rules as the real tournament**:

- the move is **timed**, and the decision time (ms) is shown for the last AI move;
- if a move **exceeds the 1 s budget**, **returns an illegal action**, or **raises**,
  it is replaced by a random legal move — and the substitution (with the reason
  and the action your agent *intended*) is flagged in the log and the
  "Last AI move" line.

This lets you see exactly what the competition harness would do with your agent
before you submit. The full event log narrates every action in English.

## Game records and replay

Every finished game is saved automatically as a small JSON file under
`results/webui_games/` — the random seed plus every decision, which is enough to
replay the game exactly. The event log names the file when the game ends.

```bash
python tools/replay_game.py results/webui_games/game_<stamp>_<seed>.json --moves     # narrate it
python tools/replay_game.py results/webui_games/game_<stamp>_<seed>.json --stop 40   # the position after 40 decisions
```

`--stop N` prints who is to move, the legal actions, and the exact observation
vector your agent would receive — handy for asking "what did my agent see when
it played that move?". The format and helpers live in
[`../puerto_rico/records.py`](../puerto_rico/records.py).

> This is a single-game debug tool with global state — not the competition runner.
> For official, seat-balanced round-robin ranking, use `tournament/`
> (`python examples/run_tournament.py`). See
> [`../docs/COMPETITION_RULES.md`](../docs/COMPETITION_RULES.md).
