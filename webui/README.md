# Web UI — play and debug agents

A local browser tool to watch agents play, play a seat yourself, and **debug your
own agent** against the baselines.

```bash
pip install -e ".[webui]"      # or: pip install flask
python webui/server.py         # open http://127.0.0.1:5000
```

## What you can do

- **Configure each of the 3 seats** independently: a human (you), any baseline
  (Random, Factory, TradeBuilding, ShippingRush, ActionValue, MCTS, PPO), or
  **your own agent**.
- **Load your agent** two ways: drop a `.py` in [`../submissions/`](../submissions/)
  (auto-listed in the dropdown), or type `module:Class` / `path/to/file.py:Class`
  in a seat's box.
- **Watch bots play** (set every seat to a bot) with auto-run, or **play yourself**
  by taking a seat. Undo, pass, and restart are available.

## Debugging features

Each agent move is run under the **same rules as the real tournament**:

- the move is **timed**, and the decision time (ms) is shown for the last AI move;
- if a move **exceeds the 1 s budget**, **returns an illegal action**, or **raises**,
  it is replaced by a random legal move — and the substitution (with the reason
  and the action your agent *intended*) is flagged in the log and the
  "Last AI move" line.

This lets you see exactly what the competition harness would do with your agent
before you submit. The full event log narrates every action in English.

> This is a single-game debug tool with global state — not the competition runner.
> For official, seat-balanced round-robin ranking, use `tournament/`
> (`python examples/run_tournament.py`). See
> [`../docs/COMPETITION_RULES.md`](../docs/COMPETITION_RULES.md).
