# Submission template

Your entry is a single Python class that implements `act`.

## Quick start

1. Copy `my_agent.py` and rename the class and its `name` attribute.
2. Implement `act(self, observation, action_mask) -> int` — return one legal
   action index (`action_mask[a] == 1`). The starter plays random legal moves.
3. Test it locally against the baselines:

   ```python
   from tournament.match import play_game
   from agents import ActionValueAgent, RandomAgent
   from submission_template.my_agent import MyAgent

   result = play_game([MyAgent(), ActionValueAgent(), RandomAgent()], seed=0)
   print(result["scores"], result["winners"])
   ```

   Or add `MyAgent` to the `pool` in `examples/run_tournament.py` and run it to
   see your win rate on the leaderboard.

## Rules in one line

3 players, **1 second per move** (wall-clock), you must return a legal action.
Going over the time limit or returning an illegal action forfeits that move to a
random legal one. Full details: [`../docs/COMPETITION_RULES.md`](../docs/COMPETITION_RULES.md).

## What the inputs mean

- `observation` — a 293-dim float vector describing the full game state.
- `action_mask` — a 200-dim 0/1 vector of which actions are currently legal.

The field-by-field layout is in
[`../docs/OBSERVATION_AND_ACTIONS.md`](../docs/OBSERVATION_AND_ACTIONS.md).

## Want to plan ahead (MCTS / lookahead)?

Override `on_game_start(self, forward_model)` to keep the `forward_model`, then
inside `act` call `forward_model.clone()` to simulate actions without affecting
the real game. See `agents/mcts_agent.py` for a worked example.
