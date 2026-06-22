# Puerto Rico AI Competition — Rules

These rules govern the Puerto Rico AI competition. The **same rules** apply both
to the IEEE CoG 2027 competition and to the university "Game AI" course final
project.

## Tracks

The competition runs in **two independent tracks**, each ranked on its **own
leaderboard**:

| Track | Game | Seats | `observation` shape |
|-------|------|------:|--------------------:|
| **2p** | 1-vs-1 Puerto Rico | 2 | **(220,)** |
| **3p** | 3-player Puerto Rico | 3 | **(293,)** |

- A submission is **per track**: you enter the 2p track, the 3p track, or both,
  and may submit a **different agent for each** (you can specialise for 1-vs-1 or
  3-player). The two tracks never play against each other.
- The **rules below apply to both tracks.** Only the seat count and the
  observation length differ; the action space is the same `Discrete(200)`.

## Game format

- A 2p-track match is a **single game of 1-vs-1 Puerto Rico** between 2 agents;
  a 3p-track match is a **single game of 3-player Puerto Rico** among 3 agents.
- A **fresh instance** of your agent is created for each game, so do **not** rely
  on any state persisting across games.

## What you submit

You submit **one Python class** subclassing `agents.base.Agent` that implements
`act`, returning a legal action index:

```python
class MyAgent(Agent):
    def act(self, observation, action_mask) -> int: ...
```

- You may optionally override `on_game_start(self, forward_model=None)`. It is
  called **once at the start of each game**. Planning agents keep the
  `forward_model` for simulation; reactive agents can ignore it.
- Start from `submission_template/` and read `docs/SUBMISSION_GUIDE.md`.

## Inputs

- `observation` — a `float32` NumPy array. Its length is **track-dependent**:
  **(220,)** in the 2p track, **(293,)** in the 3p track (it is
  `74 + 73 × num_players`). Use `len(observation)` if you want one agent class to
  handle both.
- `action_mask` — an `int` array of shape **(200,)** in both tracks, where `1`
  marks a legal action.
- Encoding details are in `docs/OBSERVATION_AND_ACTIONS.md`.

## Per-move time limit

- You have **1 second of wall-clock time per move**.
- If a move exceeds 1 s, the harness **discards your move**, plays a **uniformly
  random legal action** instead, and records a **timeout**.

## Legality and errors

- You must return an action `a` with `action_mask[a] == 1`.
- If you return an **illegal action**, or your code **raises an exception**, the
  harness substitutes a **uniformly random legal action** and records a
  **violation**.
- **Excessive timeouts or violations may lead to disqualification** (at the
  organizer's discretion).

## Determinism and seating

- Games are **seeded and reproducible**.
- The organizer runs a **seat-balanced round-robin**, so turn-order / seating
  advantage is controlled — each agent plays each seating.

## The official run

- Each track is run **separately**. Within a track your agent's opponents
  include the **baseline agents** bundled in this repo (Random, Factory,
  TradeBuilding, ShippingRush, ActionValue, MCTS) plus the other submissions to
  that track. The **PPO** baseline runs in the **3p track only** (its checkpoint
  is 293-dim); the other baselines play both tracks.
- The exact number of games per seating and the random seeds are set by the
  organizer; the tournament is produced by the code in `tournament/` — the same
  runner with `n_seats=2` or `n_seats=3` (see [Ranking](RANKING.md) and
  `examples/run_tournament.py`).

## Ranking

- Metrics are computed **per track** (each track has its own leaderboard).
- The **official metric is a skill rating matched to the track**: **Elo** in the
  2p (1‑vs‑1) track, **TrueSkill** in the 3p track.
- **Win rate** (ties split equally) with a **Wilson 95% confidence interval** is
  reported **alongside** in both tracks.
- **α-Rank** is available as **opt-in analysis** only — it is **not** part of the
  standings.
- Full explanation in `docs/RANKING.md`.

## Allowed code and dependencies

- Agents must be **pure Python**. **NumPy** is always available; **PyTorch** is
  available (optional).
- Your agent must be **self-contained** and must **NOT**:
  - access the network,
  - read or import other entrants' code,
  - spawn processes,
  - read or write files, or do any other disk or network I/O, or
  - tamper with the environment, the engine, or other agents.
- The `forward_model` provided to planning agents is for **read-only
  simulation** — **clone it, never mutate the live game**. Using it to corrupt
  the real game or to peek at hidden information improperly is **cheating**.

## Hidden information

- The **only** hidden information in the game is the **face-down plantation draw
  order**. Do **not** attempt to read it.

## Environment note (for organizers)

- In the **course** setting, agents run **in-process**.
- For the **public IEEE CoG** run, the organizer may additionally **isolate each
  agent in a subprocess** with enforced resource limits.
- Either way, the per-move time limit and all rules above apply.

## Fair play

- Submissions should be **your own work**.
- Be respectful of compute: an agent that reliably finishes its moves **well
  under 1 s** is appreciated.

## See also

- [Submission guide](SUBMISSION_GUIDE.md)
- [Observation and action encoding](OBSERVATION_AND_ACTIONS.md)
- [Ranking](RANKING.md)
- [Game rules](GAME_RULES.md)
