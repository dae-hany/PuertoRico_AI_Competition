# Submission template

Your entry is a single Python class that implements `act`.

## Quick start

1. Copy `my_agent.py` and rename the class and its `name` attribute.
2. Implement `act(self, observation, action_mask) -> int` — return one legal
   action index (`action_mask[a] == 1`). The starter plays random legal moves.
3. Test it locally against the baselines (from the repo root):

   ```bash
   python tools/play.py --agent submission_template/my_agent.py:MyAgent
   python tools/play.py --agent submission_template/my_agent.py:MyAgent --players 3 --vs all
   ```

   or from Python:

   ```python
   from tournament.match import play_game
   from agents import ActionValueAgent
   from submission_template.my_agent import MyAgent      # run from the repo root

   # 2p track — 1 vs 1 (pass 2 agents); 3p track — pass 3 agents
   result = play_game([MyAgent(), ActionValueAgent()], seed=0)
   print(result["scores"], result["winners"])
   ```

4. Before you submit, run the official sandbox check and make sure it ends
   with `Result: READY`:

   ```bash
   python tools/validate_submission.py submission_template/my_agent.py:MyAgent
   ```

## Rules in one line

2 players (2p track) or 3 (3p track), **1 second per move** (wall-clock), you
must return a legal action. Going over the time limit or returning an illegal
action forfeits that move to a random legal one. Full details:
[`../docs/COMPETITION_RULES.md`](../docs/COMPETITION_RULES.md).

## What the inputs mean

- `observation` — a float vector describing the full game state; **220-dim in the
  2p track, 293-dim in the 3p track** (`len(observation)` tells you which).
- `action_mask` — a 200-dim 0/1 vector of which actions are currently legal
  (same in both tracks).

The field-by-field layout is in
[`../docs/OBSERVATION_AND_ACTIONS.md`](../docs/OBSERVATION_AND_ACTIONS.md).

## Want to plan ahead (MCTS / lookahead)?

Override `on_game_start(self, forward_model)` to keep the `forward_model`, then
inside `act` call `forward_model.clone()` to simulate actions without affecting
the real game. See `agents/mcts_agent.py` for a worked example.
