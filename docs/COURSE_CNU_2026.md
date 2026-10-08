# Course edition — Chonnam National University, Reinforcement Learning (2026)

The 2p track is the **final project** of the Reinforcement Learning course in
the Department of Artificial Intelligence, Chonnam National University, in the
2026 autumn semester. This page holds **only what differs** from the general
competition: everything else — the engine, the observation and action encoding,
the 1 s move budget, legality, the sandbox, the hidden plantation order — is in
[COMPETITION_RULES.md](COMPETITION_RULES.md) and applies unchanged.

**Where the two disagree, this page wins** for the course. (Entrants in the
public IEEE CoG edition follow COMPETITION_RULES.md alone.)

| | Course (this page) | Public competition |
|---|---|---|
| Track | **2p only** | 2p and 3p |
| Ranking | **win count** (a draw is ½), students' games only | Elo (2p) / TrueSkill (3p) |
| Grade | from the **win count**, not from the rank | — |
| Weight file | **one file allowed** beside the agent | no disk I/O |
| Deliverables | agent + README + 1–2 page write-up | agent file |
| Ladder | 7 rounds before the deadline | weekly during the window |

## What you submit

One **zip** named `<student id>_<name>.zip`, containing:

1. **The agent** — one `.py` file with exactly **one** `Agent` subclass, for the
   **2p track**. It must be self-contained: helper modules in the zip are *not*
   importable, because the harness loads your single file directly.
2. **A README** — how to run it, what it depends on, and whether it ships a
   weight file.
3. **A write-up, 1–2 pages** — how the agent works, why you designed it that
   way, and what you measured. It carries no points of its own, but the
   write-up and the code are graded as one submission: **an entry without a
   write-up receives 0 points**.
4. **Optionally one weight file** (see below).

Set a distinctive `name` on your class. That name, not your student id, is what
appears on the published board:

```python
class MyAgent(Agent):
    name = "Coffee Baron"       # your alias on the ladder
```

Two students may not use the same alias and no alias may impersonate a bundled
baseline; the ladder refuses to run until such a clash is fixed.

This is an **individual** assignment.

## The weight file (course exception)

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

Everything else in the I/O rule still holds: no network, no subprocesses, no
reading other entrants' code, no writing files.

## Anything goes, but learning is the point

Heuristics, adversarial search and reinforcement learning are all allowed, and
they can be combined. Since this is the final project of a reinforcement
learning course, an entry that includes a **learned component** — a value or
policy network, self-play, a learned evaluation inside a search — is what the
assignment is aiming at. See the "Recommended approach" section of the
[README](../README.md) and [SEARCH_BASELINE_2P.md](SEARCH_BASELINE_2P.md).

## How you are ranked and graded

- Every entry plays a **seat-balanced round-robin** against every other entry,
  with enough games per pair that a single lucky game decides nothing. Every
  entry plays the **same number of games**.
- Each game is a **win (1)**, a **draw (½)** or a **loss (0)**. Your **win
  count** is your wins plus half your draws.
- **Only games between students count.** The bundled baselines are played too,
  as a fixed reference point, and shown in a separate column.
- The final standings list every entry from **1st to nth** by win count, with
  its wins, draws and losses. Because every entry plays the same number of
  games, this is the same order as the win rate.
- Timeouts and illegal moves are not scored separately — they cost you the
  moves they replace, and therefore games.

### Your grade follows your win count, not your rank

The grade is a continuous function of the win count, so one more win is worth
one game's share of the grade and nothing more. Two entries with 890 and 889
wins sit one place apart on the board but receive almost the same grade,
whereas a grade by rank could put a whole grade step between them. The formula
that turns a win count into points will be announced together with the
deadline. The write-up carries no points of its own, but it and the code form
one submission: **an entry without a write-up receives 0 points**.

The organizer runs this with `tools/course_ladder.py`, which also records Elo
for its own analysis; the published board shows wins, losses and win rate only.

## The ladder before the deadline

Counting back from the deadline, a ladder round runs at:

**D-14 · D-11 · D-8 · D-6 · D-4 · D-2 · D-1**

Each round plays every agent submitted **so far** and publishes the standings,
the win rates, **and every game's replay**. Replay a game move by move with:

```bash
python tools/replay_game.py <game>.json --moves
```

Two consequences worth thinking about:

- Submitting early tells you where you stand while there is still time to
  improve, and the replays show you exactly how you lost.
- Your play becomes **visible to everyone** once it is in a ladder round. If you
  would rather not reveal your strategy, submit at the deadline instead.

Ladder results do **not** count toward your grade. Resubmission is free and the
**last file received before the deadline** is the one that is graded.

## Before you submit

```bash
python tools/play.py --agent my_agent.py:MyAgent --vs Search --games 20
python tools/validate_submission.py my_agent.py:MyAgent --track 2p
```

The last line of the validator must read `Result: READY`. Fix every FAIL, and
read every WARN.

## Dates and channel

| | |
|---|---|
| Deadline | to be announced |
| Submission channel | to be announced |
| Teaching assistant | to be announced |

Announcements are made in class; this page is updated to match.
