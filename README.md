# Puerto Rico AI Competition

An AI competition built on **Puerto Rico**, an economic strategy board game
proposed as a reinforcement-learning / game-AI benchmark. You write an agent,
it plays full games against other agents, and a tournament ranks everyone.

The competition runs in **two independent tracks**, each with its own leaderboard:

- **2p track** — 1-vs-1 (2-player) Puerto Rico.
- **3p track** — 3-player Puerto Rico.

You may enter **either track or both**; each track is a **separate submission**
(among other things, the observation length differs — 220 in 2p, 293 in 3p). The
rules, engine, and action space are otherwise shared.

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

python examples/play_one_game.py        # one baseline game per track (2p and 3p)
python examples/run_tournament.py       # a round-robin + leaderboard for each track
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
        # observation: float32[220] in 2p / [293] in 3p  |  action_mask: int[200] (1 = legal)
        legal = np.where(action_mask > 0.5)[0]
        return int(legal[0])            # replace with your strategy
```

The action space is identical in both tracks; only the observation length differs
(`len(observation)` tells you which track you are in). Agents that read the mask
or the `forward_model` — like every baseline below except PPO — work in both.

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

All baselines except `PpoAgent` are player-count-agnostic and play **both
tracks** unchanged. The bundled PPO checkpoint was trained on the 293-dim 3p
observation, so it is a **3p-only** baseline; a 2p PPO agent would need a 220-dim
network and its own training run.

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

**Each track is ranked separately** (its own round-robin and leaderboard). The
official metric is a **skill rating matched to the track** — **Elo** in the 2p
(1‑vs‑1) track, **TrueSkill** in the 3p track — computed over a seat-balanced
round-robin, with **win rate** (Wilson 95% CI) shown alongside. (α‑Rank is
available as opt‑in analysis, not part of the standings.) See
[Ranking](docs/RANKING.md). Example **2p‑track** board (official = **Elo**, win
rate alongside) from a short 5‑agent run:

| Rank | Agent | Elo (official) | Win% | 95% CI | Games |
|-----:|-------|---------------:|-----:|:------:|------:|
| 1 | TradeBuilding | 1766 | 80.0% | [0.49, 0.94] | 10 |
| 2 | ActionValue | 1669 | 70.0% | [0.40, 0.89] | 10 |
| 3 | ShippingRush | 1500 | 50.0% | [0.24, 0.76] | 10 |
| 4 | Random | 1331 | 30.0% | [0.11, 0.60] | 10 |
| 5 | Factory | 1234 | 20.0% | [0.06, 0.51] | 10 |

_Numbers vary with the agent pool and number of games. The **3p track** is ranked
by **TrueSkill** instead (win rate alongside), and orderings can differ between
tracks — here TradeBuilding tops the 1‑vs‑1 board. See [Ranking](docs/RANKING.md)._

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
