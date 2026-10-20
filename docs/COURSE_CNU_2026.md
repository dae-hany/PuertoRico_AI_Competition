# Course edition — Chonnam National University, Reinforcement Learning (2026)

> **English** | [한국어](COURSE_CNU_2026.ko.md)

The 2p track is the **final project** of the Reinforcement Learning course in
the Department of Artificial Intelligence, Chonnam National University, in the
2026 autumn semester. It is an **individual** assignment. This page holds
**only what differs** from the general competition: everything else (the
engine, the observation and action encoding, the **0.1 s per-decision limit**,
legality, the sandbox, the hidden plantation order) is in
[COMPETITION_RULES.md](COMPETITION_RULES.md) and applies unchanged.

**Where the two disagree, this page wins** for the course. (Entrants in the
public IEEE CoG edition follow COMPETITION_RULES.md alone.)

| | Course (this page) | Public competition |
|---|---|---|
| Track | **2p only** | 2p and 3p |
| Ranking | **wins and rank** in games between students (a draw is ½ a win) | Elo (2p) / TrueSkill (3p) |
| Weight file | **one file allowed** (`weights.pt`) | no disk I/O |
| Deliverables | **two zips**: code (agent + README + optional weights), then write-up + presentation | agent file |
| Ladder | **4 rounds** before the deadline; wins and rank only | weekly; standings and replays |
| Channel | **Chonnam National University LMS** assignment submission page | to be announced |

All times are Korean Standard Time (KST).

## Grade

The course grade is **final exam 50% + final project 40% + attendance 10%**.
The final project's 40% splits in two:

| Part | Weight | What is graded |
|---|---:|---|
| Battle results | **20%** | games between student agents, seats swapped: **both the wins and the rank** count |
| Presentation and report | **20%** | `presentation.pdf` and `writeup.pdf`; final presentation on 3 or 5 December |

- Only **games between students** count. Each game is a win (1), a draw (½)
  or a loss (0); your **wins** are your wins plus half your draws.
- The bundled baselines are played too, and their results are published
  **separately, for reference only**.
- How the wins and the rank are turned into points will be **announced later**.
- There is **no bonus or penalty** for using, or not using, reinforcement
  learning.

## What you submit: two submissions

Both go to the **Chonnam National University LMS assignment submission page**.

### ① Code — Sunday 29 November 2026, 23:59

One zip named **`<student id>_<name>.zip`**, containing:

| File | |
|---|---|
| `my_agent.py` | exactly **one** `Agent` subclass, for the 2p track. It must be self-contained: do not import helper modules of your own, because the harness loads this single file directly. |
| `README.md` | how to run it, what it depends on, and whether it uses a weight file |
| `weights.pt` (optional) | one weight file, if your agent uses a learned model (see below) |

### ② Report and presentation — Monday 30 November 2026, 23:59

One zip named **`<student id>_<name>.zip`**, containing:

| File | |
|---|---|
| `writeup.pdf` | 1–2 pages: how your agent works, why you designed it that way, and what you measured yourself |
| `presentation.pdf` | the slides for your final presentation (3 or 5 December) |

### Your alias

Set a distinctive `name` on your class. That name, not your student id, is what
appears on the published board:

```python
class MyAgent(Agent):
    name = "Coffee Baron"       # your alias on the ladder
```

It may not be the same as another student's alias or a bundled baseline's
name; the ladder refuses to run until such a clash is fixed.

## Run rules

- **0.1 s per decision** (the same as the public competition). A decision that
  takes longer is replaced by a **random legal action**. Leave a margin in your slowest positions (p95, max),
  not just on average.
- Return an action the **mask allows**. An illegal action or an exception is
  also replaced by a random legal action.
- **Repeated violations can be grounds for disqualification.**
- Simulate on a **copy** of the forward model only:

  ```python
  sim = self.fm.clone()
  sim.step(a)
  ```

**Forbidden:** network access, running subprocesses, writing files, reading
other entrants' code, changing the real game state, and reading the hidden
plantation tile order.

**Allowed:** reading **one weight file** (the course exception below), and the
Python standard library, NumPy, PyTorch and this repository's modules.

The bundled `Search` (1500 nodes) and `MCTS` examples take more than 0.1 s per
decision; if you build on them, reduce the node or simulation count. (As bundled
baselines they keep their own 1 s budget; an entry built on their code does not.)

### The weight file (course exception)

COMPETITION_RULES.md forbids disk I/O. **The course allows exactly one weight
file** so that a learned agent can ship its parameters. Put it in the zip next
to your agent and load it **relative to your own file**, never relative to the
working directory:

```python
import os
HERE = os.path.dirname(os.path.abspath(__file__))
state = torch.load(os.path.join(HERE, "weights.pt"), map_location="cpu")
```

`tools/validate_submission.py` prints a **WARN** about file access for such an
agent. In the course that WARN is expected; a **FAIL** is not.

## Anything goes, but learning is recommended

Heuristics, adversarial search and reinforcement learning are all allowed, and
they can be combined. Since this is a reinforcement learning course, a
**learned component** (a value or policy network, self-play, a learned
evaluation inside a search) is recommended, but other methods are **not
penalized**, and no points are given for the name of an algorithm. The
bundled PPO checkpoint is for the 3p track; 2p training is supported
(`training/train_ppo.py --num_players 2`). See the "Recommended approach"
section of the [README](../README.md) and
[SEARCH_BASELINE_2P.md](SEARCH_BASELINE_2P.md).

## Schedule and ladder

| | Date |
|---|---|
| Ladder 1 (D-10) | Thursday 19 November 2026, entries due 17:00 |
| Ladder 2 (D-5) | Tuesday 24 November 2026, entries due 17:00 |
| Ladder 3 (D-3) | Thursday 26 November 2026, entries due 17:00 |
| Ladder 4 (D-1) | Saturday 28 November 2026, entries due 17:00 |
| ① Code deadline | Sunday 29 November 2026, 23:59 |
| ② Report and presentation deadline | Monday 30 November 2026, 23:59 |
| Final presentation | Thursday 3 December or Saturday 5 December 2026 |

### The ladder before the deadline

- Each round plays the entries submitted to the LMS **by 17:00 that day**
  against each other.
- Only the **wins and the rank** are published. Submitted code and game
  replays are **not** published.
- Every pair plays the public seeds **1–25**, each once in either seating:
  **25 seeds × 2 seatings = 50 games per opponent**. A seed fixes the game's
  random setup, so with the seats swapped each of you plays both sides of the
  same start.
- Ladder results do **not** count toward your grade. You may resubmit, and the
  **last code submitted before the deadline** is the one that is graded.

Submitting early tells you that your file runs and is in the right format, and
where you stand while there is still time to improve.

### The final grading run

Every pair plays **500 seeds × 2 seatings = 1000 games per opponent**. The 500
seeds are the 25 public ladder seeds plus **475 private seeds**, which are not
published. An improvement that only fits the public seeds may not carry over to
the final result.

## Before you submit

```bash
python tools/play.py --agent my_agent.py:MyAgent --games 20 --time-limit 0.1 --record
python tools/replay_game.py results/games/<game>.json --moves
python tools/validate_submission.py my_agent.py:MyAgent --track 2p --time-limit 0.1
```

`play.py` rotates the seats; `--vs all` plays every 2p baseline, and
`--games 20` is a quick check, not a statistically sufficient number of games.
The validator plays 2 games with your agent in its own process (the sandbox)
under the 0.1 s limit. Its last line must read `Result: READY`. Fix every FAIL,
and read every WARN.

## Contacts

| | |
|---|---|
| Teaching assistant | 최대한 |
| Questions | in class, or at the contact address on the course slides |

Announcements are made in class; this page is updated to match.
