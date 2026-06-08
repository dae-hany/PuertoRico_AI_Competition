# Submission guide

Build an agent that plays 3-player Puerto Rico, test it against the baselines,
and submit it. You only need to write one method: `act`.

## 1. Set up

```bash
git clone <this-repo>
cd PuertoRico_AI_Competition
python -m venv .venv && . .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -e .                                     # deps + importable packages
python examples/play_one_game.py                     # smoke test
```

NumPy is required; PyTorch is only needed for the PPO baseline and training.

## 2. The interface

Your agent subclasses [`agents.base.Agent`](../agents/base.py):

```python
class Agent:
    name = "Agent"
    def on_game_start(self, forward_model=None): ...      # optional
    def act(self, observation, action_mask) -> int: ...    # required
```

- `observation` — `float32` array, shape `(293,)` — the game state.
- `action_mask` — `int` array, shape `(200,)` — `1` = legal. **Return a legal action.**

What the numbers mean: [OBSERVATION_AND_ACTIONS.md](OBSERVATION_AND_ACTIONS.md).
The game itself: [GAME_RULES.md](GAME_RULES.md).

## 3. Write your agent

Copy the template and edit it:

```bash
cp submission_template/my_agent.py my_agent.py
```

```python
import numpy as np
from agents.base import Agent

class MyAgent(Agent):
    name = "TeamName"

    def act(self, observation, action_mask):
        legal = np.where(action_mask > 0.5)[0]
        # ... your strategy here ...
        return int(legal[0])
```

The starter returns a random legal move — replace it with real logic.

## 4. Test against the baselines

```python
from tournament.match import play_game
from agents import ActionValueAgent, ShippingRushAgent
from my_agent import MyAgent

result = play_game([MyAgent(), ActionValueAgent(), ShippingRushAgent()], seed=0)
print(result["scores"], "winner:", result["winners"])
```

Or add `MyAgent` to the `pool` in [`examples/run_tournament.py`](../examples/run_tournament.py)
and run it to see your win rate on the leaderboard.

**Baselines, strongest to weakest:** MCTS ≈ ActionValue ≈ ShippingRush (strong) >
TradeBuilding (moderate) > Factory > PPO ≈ Random (weak). Beating the strong
heuristics and MCTS is the real challenge.

## 5. Going further

- **Planning (lookahead / MCTS):** override `on_game_start(self, forward_model)`,
  keep the `forward_model`, and call `forward_model.clone()` inside `act` to
  simulate actions safely. See [`agents/mcts_agent.py`](../agents/mcts_agent.py).
- **Reinforcement learning:** a PPO self-play trainer and a (weak) starting
  checkpoint are in [`training/`](../training/). Puerto Rico is hard for plain RL —
  improving on it is an open problem.

## 6. Rules you must respect

3 players, **1 second per move**, always return a legal action. Going over the
limit, returning an illegal action, or raising an exception forfeits that move to
a random legal one. Full rules: [COMPETITION_RULES.md](COMPETITION_RULES.md).

## 7. Submit

Submit your single agent file (the one class). Make sure it:

- imports only from the standard library, NumPy, PyTorch, and this repo;
- is self-contained and runs without network or file access;
- sets a distinctive `name` (shown on the leaderboard).

Follow the channel and deadline given by your competition organizer.
