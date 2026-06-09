# Puerto Rico AI Competition

An AI competition built on **Puerto Rico**, a 3-player economic strategy board
game proposed as a reinforcement-learning / game-AI benchmark. You write an agent,
it plays full games against other agents, and a tournament ranks everyone.

This repository is used for the **IEEE CoG 2027 competition** and as the final
project of a university **Game AI** course (identical rules). Everything — the
engine, the agents, the docs — is in English so anyone can take part.

> Why a competition? Puerto Rico is easy to simulate but hard to master, and it is
> *hard for plain reinforcement learning*: hand-written heuristics and tree search
> still beat trained RL agents. Building something that beats them is the challenge.

## Quickstart

```bash
git clone <this-repo>
cd PuertoRico_AI_Competition
python -m venv .venv && . .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -e .                        # installs deps + makes the packages importable

python examples/play_one_game.py        # one game between baseline agents
python examples/run_tournament.py       # a round-robin + leaderboard
python webui/server.py                   # browser UI (needs flask): play / watch / debug
```

`pip install -e .` is recommended (it makes `agents`, `tournament`, `puerto_rico`
importable from anywhere). `pip install -r requirements.txt` also works — the
example scripts add the repo root to the path themselves, but you should then run
them from the repo root.

## Write an agent

Subclass `Agent` and implement one method:

```python
import numpy as np
from agents.base import Agent

class MyAgent(Agent):
    name = "MyAgent"

    def act(self, observation, action_mask):
        # observation: float32[293]  |  action_mask: int[200] (1 = legal)
        legal = np.where(action_mask > 0.5)[0]
        return int(legal[0])            # replace with your strategy
```

Copy [`submission_template/`](submission_template/) to get started, then read the
[Submission guide](docs/SUBMISSION_GUIDE.md).

## Baselines

| Agent | Type | Strength |
|---|---|---|
| `MctsAgent` | Max^N UCT tree search | strong (slow) |
| `ActionValueAgent` | greedy heuristic — scores each legal action, plays the best | strong |
| `ShippingRushAgent` | shipping-focused heuristic | strong |
| `TradeBuildingAgent` | trade → building heuristic | moderate |
| `FactoryAgent` | Factory-engine heuristic | weak |
| `PpoAgent` | PPO self-play (RL) | ~random — a starting point |
| `RandomAgent` | uniform random legal move | weakest |

## Repository layout

```
puerto_rico/        core game engine + environment + forward model
agents/             the Agent interface and all baseline agents
tournament/         single-match harness, round-robin runner, rankers, leaderboard
training/           optional PPO self-play trainer + a weak baseline checkpoint
webui/              browser UI to play, watch, and debug agents
examples/           play_one_game.py, run_tournament.py
submission_template/ copy this to build your competition entry
submissions/        drop an agent here to debug it in the web UI
docs/               rules, observation/action encoding, ranking, submission guide
tests/              pytest suite
```

## How ranking works

The official metric is **win rate** over a seat-balanced round-robin (Wilson 95%
CI). **TrueSkill** and **α-Rank** are also computed and shown. See
[Ranking](docs/RANKING.md). Example leaderboard from a short 4-agent run:

| Rank | Agent | Win% (official) | 95% CI | Games | TrueSkill | α-Rank |
|-----:|-------|----------------:|:------:|------:|----------:|-------:|
| 1 | ActionValue | 50.0% | [0.36, 0.64] | 48 | 27.21 | 0.000 |
| 2 | ShippingRush | 43.8% | [0.31, 0.58] | 48 | 24.44 | 1.000 |
| 3 | TradeBuilding | 33.3% | [0.22, 0.47] | 48 | 23.47 | 0.000 |
| 4 | Random | 6.2% | [0.02, 0.17] | 48 | 15.63 | 0.000 |

_Numbers vary with the agent pool and number of games. Here win-rate and TrueSkill
agree (ActionValue on top), while α-Rank concentrates its mass on the
evolutionarily dominant strategy (ShippingRush) — a reminder that the metrics
measure different things (see [Ranking](docs/RANKING.md))._

## Documentation

- [Game rules](docs/GAME_RULES.md)
- [Observation & action encoding](docs/OBSERVATION_AND_ACTIONS.md)
- [Competition rules](docs/COMPETITION_RULES.md)
- [Submission guide](docs/SUBMISSION_GUIDE.md)
- [Ranking](docs/RANKING.md)

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## License

Code is released under the [MIT License](LICENSE). The board game *Puerto Rico*
(designer Andreas Seyfarth) is the intellectual property of its rights holders;
this is an independent, non-commercial re-implementation for education and
research, with no original artwork. See [LICENSE](LICENSE) for the full notice.
