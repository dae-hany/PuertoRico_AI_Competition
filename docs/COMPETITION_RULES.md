# Puerto Rico AI Competition — Rules

These rules govern the Puerto Rico AI competition: the engine, what you submit,
the time limit, legality, and how the official run is executed.

They are also the technical rules of the **course edition** currently running —
the 2p track as the final project of the Reinforcement Learning course at
Chonnam National University. That edition changes a few things on top (it is 2p
only, ranks by wins and rank in games between students rather than by Elo,
and takes a code zip and a separate report and presentation zip): see
**[COURSE_CNU_2026.md](COURSE_CNU_2026.md)**, which takes precedence for
students of that course. Everything not listed there is governed by this page.

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
- A learned agent may ship **one weight file** next to its `.py` file (see
  [The weight file](#the-weight-file) below). An entry is therefore the agent
  file, plus at most one weight file.
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

- You have **0.1 s (100 ms) of wall-clock time per decision**, in both tracks.
- If a decision takes longer, the harness **discards your move**, plays a
  **uniformly random legal action** instead, and records a **timeout**.
- Check your **slowest** positions, not just the average: `tools/play.py` prints
  the p95 and the maximum, and `tools/validate_submission.py` runs your agent
  under the real deadline.
- In the official run this is **enforced, not just measured**: your agent runs in
  its own process and is **killed** when the budget expires, so a move that never
  returns costs you that move rather than hanging the tournament. A killed agent
  is rebuilt for the next move, which means **any state it had accumulated during
  the game is lost** — one more reason to finish well inside the budget.
- An agent that overruns **repeatedly** stops being restarted and **forfeits the
  rest of that game** (its remaining moves are played at random).
- **The bundled baselines are not entrants.** They are reference opponents whose
  strength is fixed by a node or simulation budget, not by the clock, and they
  keep the **1 s** budget they were designed for (`Search` at 1500 nodes and
  `MCTS` need more than 0.1 s per move). The exemption goes by the baseline's
  exact class, so an entry built on a baseline's code is held to 0.1 s: lower its
  node or simulation budget to fit.

## Legality and errors

- You must return an action `a` with `action_mask[a] == 1`.
- If you return an **illegal action**, or your code **raises an exception**, the
  harness substitutes a **uniformly random legal action** and records a
  **violation**.
- **Excessive timeouts or violations may lead to disqualification** (at the
  organizer's discretion).

## Determinism and seating

- Games are **seeded and reproducible**: the same seed and the same agents
  replay the same game, in the same process or a different one.
- The engine keeps its randomness to itself (each game owns a private
  `random.Random`). It never reads or re-seeds the global `random` /
  `np.random` states, so **your agent may use them freely** — you cannot
  perturb the deal, and the environment will not silently reset your RNG.
- The flip side: because nothing re-seeds the global state for you, an agent
  that wants to be reproducible must **seed its own RNG** (e.g. hold a
  `np.random.default_rng(seed)`), as the bundled baselines do.
- The organizer runs a **seat-balanced round-robin**, so turn-order / seating
  advantage is controlled — each agent plays each seating.

## The official run

- Each track is run **separately**. Within a track your agent's opponents
  include the **baseline agents** bundled in this repo (Random, Factory,
  TradeBuilding, ShippingRush, ActionValue, MCTS, and in the 2p track also the
  alpha-beta baselines SearchLite and Search) plus the other submissions to
  that track. The **PPO** baseline runs in the **3p track only** (its checkpoint
  is 293-dim); the heuristics and MCTS play both tracks.
- The runner is `tools/run_ladder.py`: it seats every entry in the official
  sandbox, plays the round-robin, and writes the standings and every game's
  replay. The same command produces the weekly ladder during the competition
  window and the final standings.
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
  - write files, read any file other than its own weight file (below), or do
    any other disk or network I/O, or
  - tamper with the environment, the engine, or other agents.
- The `forward_model` provided to planning agents is for **read-only
  simulation** — **clone it, never mutate the live game**. Using it to corrupt
  the real game or to peek at hidden information improperly is **cheating**.
  This is **checked, not assumed**: in-process the harness compares the real
  game state before and after every one of your moves and records a breach; in
  the official run your process cannot reach the real game at all.

### The weight file

A learned agent needs its trained parameters, so an entry may include **one
read-only weight file** (e.g. `weights.pt` or `weights.npz`) besides its `.py`
file. The rule:

- **At most one file**, placed **next to the agent file** (the same folder, not
  a subfolder), and **at most 50 MB**.
- Locate it **relative to your own file**, from `__file__`, never relative to
  the working directory: the official run starts in another folder.
- Read it **only at import or while the agent is constructed**, never inside
  `act`. The harness builds a **new agent instance every game**, so keep the
  loaded parameters in a **module-level cache** and read the file once per
  process:

  ```python
  import os
  import torch

  HERE = os.path.dirname(os.path.abspath(__file__))
  _NET = None

  def _net():
      global _NET
      if _NET is None:
          _NET = torch.load(os.path.join(HERE, "weights.pt"), map_location="cpu")
      return _NET

  class MyAgent(Agent):
      def __init__(self):
          super().__init__()
          self.net = _net()
  ```

- **Nothing else**: no second file, no files elsewhere on disk, and no writing,
  not even next to your agent.
- Loading counts toward the worker's start-up, not toward any move, but a
  worker that is killed after a timeout pays it again (the sandbox allows 30 s
  to start). `tools/validate_submission.py` checks all of the above and reports
  the file's size and the worker's start-up time.

Embedding the parameters in the `.py` source (an array literal, or base64 read
through `io.BytesIO`) is still allowed and needs no weight file.

## Hidden information

- The **only** hidden information in the game is the **face-down plantation draw
  order**.
- This is **enforced by the forward model**, not left to the honor system. You
  legitimately know *which* tiles remain face-down (the multiset is deducible by
  counting the public tiles), but **never the order** they will be drawn in:
  - `forward_model.clone()` **determinizes** the hidden pile — every clone gets a
    fresh random order (multiset preserved), so search/simulation only ever plays
    out a random *consistent* world, never the true future draws. Cloning a clone
    keeps its order, so a search tree stays self-consistent within one move.
  - The raw `forward_model.game` / `forward_model.env` accessors on the live
    model also return a **determinized snapshot**, so the true order cannot be
    read directly without cloning.
  - The randomisation is independent of the real game (it never perturbs actual
    play) and reproducible under a fixed match seed.
  - In the official run the true order is not merely hidden behind an accessor:
    the copy shipped to your process has **already been reshuffled**, so the
    real order never leaves the organizer's process at all.

## How the official run is executed

The same rules apply in both settings below; they differ in how much is
**enforced** rather than trusted.

- **Your own testing** (`tools/play.py`, the web UI, the notebook) — agents run
  **in-process** (`tournament/match.py`). The time limit is measured after each
  move, and the forward model you are handed is a live object. The harness does
  verify after every move that you left the real game exactly as you found it,
  and reports any breach as `tampered` in the game record.
- **Official runs** — the public IEEE CoG run, and the CNU course's ladder rounds
  and final run (`tools/course_ladder.py`) — each entry runs in **its own
  process** (`tournament/sandbox.py`), as `tools/validate_submission.py` already
  does. The organizer's process owns the game; your agent never holds a
  reference to it. Consequences for you:
  - the per-move deadline is enforced by killing the worker (see above);
  - a planning agent still gets a forward model — a **determinized copy** of the
    game is handed to your process every move, and `clone()` / `step()` behave
    exactly as they do in-process (measured overhead ≈ 2 ms per move);
  - each worker gets a memory cap and single-threaded BLAS defaults, so budget
    your agent accordingly (say so in your submission if you need more);
  - a crash costs you one move, not the match.

Write and test your agent against the in-process harness; if it respects the
rules, isolation changes nothing you can observe.

## Fair play

- Submissions should be **your own work**.
- Be respectful of compute: an agent that reliably finishes its moves **well
  under the 0.1 s limit** is appreciated.

## See also

- [Submission guide](SUBMISSION_GUIDE.md)
- [Observation and action encoding](OBSERVATION_AND_ACTIONS.md)
- [Ranking](RANKING.md)
- [Game rules](GAME_RULES.md)
