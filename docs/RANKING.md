# How agents are ranked

Puerto Rico is a **general-sum** game, played here in two tracks — **2p**
(1‑vs‑1) and **3p** (3‑player). **Each track is ranked independently and has its
own leaderboard.** To rank a track, the organizer runs a **seat-balanced
round-robin** among that track's submitted agents (and a set of baselines): every
combination of agents is played across all seat assignments so that no agent is
advantaged by its position at the table. The 2p round-robin seats 2 agents per
game, the 3p round-robin seats 3 — the same runner with `n_seats=2` or `3`.

From the resulting games we compute, **per track**, an **official** skill rating
plus **win rate** as a companion view:

| Track | Official metric | Reported alongside |
|-------|-----------------|--------------------|
| **2p** (1‑vs‑1) | **Elo** | win rate (Wilson 95% CI) |
| **3p** | **TrueSkill** | win rate (Wilson 95% CI) |

**α-Rank** is an **optional analysis** metric (off by default), described at the
end. The combined table is produced by
[`tournament/leaderboard.py`](../tournament/leaderboard.py), ordered by the
track's official metric with win rate as an extra column. Output is written as
Markdown, CSV, and JSON.

---

## Elo — official for the 2p (1‑vs‑1) track

Elo is the classic **1-vs-1** rating: the gap between two ratings predicts the win
probability through the logistic curve

$$P(a \text{ beats } b) = \frac{1}{1 + 10^{-(R_a - R_b)/400}}.$$

A naive **sequential** Elo pass depends on the order games are processed and on
the K-factor — bad for a competition that must be reproducible. We instead report
the **converged** Elo: the **Bradley–Terry maximum-likelihood** rating the same
logistic model implies, computed **order-independently** from the round-robin win
counts (the MM algorithm, Hunter 2004). The field averages to **1500**; a light
prior keeps ratings finite when an agent wins or loses *every* game and shrinks
small-sample extremes toward the mean. A tie counts as **half a win** for each
side. Implemented in [`tournament/rankers/elo.py`](../tournament/rankers/elo.py).

Why Elo here: in a 1-vs-1 field it is the **standard, intuitive** rating, and it
weights wins by **opponent strength** (beating a strong agent moves you more than
beating a weak one). In a complete, balanced round-robin its ordering closely
tracks raw win rate — which is why win rate is shown alongside as a transparent
check.

## TrueSkill — official for the 3p track

**TrueSkill** (Herbrich et al., 2006) is a Bayesian skill-rating system built for
**free-for-all multiplayer** games — in effect the **multiplayer generalization of
Elo** — which fits 3-player Puerto Rico naturally. Each agent holds a rating
$N(\mu, \sigma^2)$; after each game the **finishing order** (1st / 2nd / 3rd) is
used to update every player's $\mu$ (estimated skill) and $\sigma$ (uncertainty).

Using the **full finishing order** is exactly why TrueSkill suits the 3p track:
coming **2nd is better than 3rd**, a distinction a pure "did you place 1st" win
rate throws away. The leaderboard value is the **conservative** score
$\mu - 3\sigma$ — a level we are ~99% confident the agent exceeds — so agents with
**few games** (high $\sigma$) are not over-rated.

TrueSkill **expresses uncertainty** and **updates incrementally** (useful when
submissions arrive over time). Its main **limitation** is that it assumes a
**single transitive skill scale**. Implemented in
[`tournament/rankers/trueskill_ranker.py`](../tournament/rankers/trueskill_ranker.py)
(uses the `trueskill` package).

## Win rate — reported in both tracks

An agent's win rate is the **fraction of games it wins** across the round-robin; a
tie **splits the win equally** among the tied winners. It is reported with a
**Wilson score 95% confidence interval** so close standings can be read with their
uncertainty. Win rate is **not** the official order (Elo / TrueSkill are), but it
is shown next to it as a **transparent, assumption-free** check — "who wins the
most games". Implemented in
[`tournament/rankers/win_rate.py`](../tournament/rankers/win_rate.py).

## α-Rank — optional analysis (opt-in, not official)

**α-Rank** (Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution",
*Scientific Reports*, 2019) ranks strategies by an **evolutionary process**
rather than a single scalar skill. Crucially, it does **not** assume a transitive
skill order, so it can reveal **non-transitive ("rock-paper-scissors") cycles**
among strategies — structure that scalar ratings hide.

How it is computed (**only when you pass `compute_alpha_rank=True`**):

1. Build a **monomorphic payoff matrix** $M$, where $M[s][r]$ is the win rate of
   a focal agent playing strategy $s$ when **every opponent** plays strategy $r$.
   Each entry is estimated by playing games of the form $[s, r, r]$ in the 3p
   track, or the head-to-head $[s, r]$ in the 2p track.
2. Run a **single-population evolutionary model** with finite population size $m$
   and ranking intensity $\alpha$. The population is **monomorphic**, and a lone
   mutant fixates with the **Moran fixation probability**. In words: a mutant
   that is fitter than the resident is more likely to take over, and the
   advantage is sharpened by $\alpha$ — the fixation chance grows with the
   payoff gap $M[s][r] - M[r][r]$, reducing to the neutral $1/m$ when they are
   equal.
3. The induced **Markov chain over strategies** has a **stationary
   distribution**; the **mass** on each strategy ranks the agents.

α-Rank is **kept out of the official standings on purpose**: it needs the **full
payoff matrix** (an extra round of games beyond the round-robin), it is
**sensitive to the intensity parameter** $\alpha$, and its stationary "mass" is
**unintuitive as a leaderboard number**. It is most useful in write-ups (e.g. the
IEEE CoG report) to **characterize the strategic structure** of the field —
whether the meta-game is transitive or cyclic. Implemented in
[`tournament/rankers/alpha_rank.py`](../tournament/rankers/alpha_rank.py)
(self-contained, NumPy only).

---

## Why these metrics

- The **official** metric is a **skill rating matched to the track**: **Elo** for
  1-vs-1, its multiplayer generalization **TrueSkill** for 3-player. Both weight
  results by **opponent strength**, and TrueSkill additionally uses the full
  **finishing order** (2nd vs 3rd), which a 1st-place-only win rate discards.
- **Win rate** is shown alongside as the **intuitive, assumption-free** check; in
  a balanced round-robin it usually agrees closely with the official order.
- **α-Rank** (opt-in) exposes **strategic structure** (non-transitivity) that
  scalar ratings cannot represent.

In practice, **the official rating and win rate usually agree closely**. Where
they diverge, it is typically on agents with **few or high-variance games**, for
which the rating's uncertainty handling and the Wilson interval on the win rate
react differently to limited data.

---

## Reference defaults

The bundled implementation ([`tournament/runner.py`](../tournament/runner.py)) uses
these defaults, all adjustable by the organizer:

- **Round-robin:** every group of `n_seats` agents (2 in the 2p track, 3 in the
  3p track) over all seatings, **1 game per seating** by default (raise it for
  tighter confidence intervals).
- **Elo (2p):** field mean **1500**, logistic scale **400**, computed as the
  converged Bradley–Terry MLE (order-independent).
- **α-Rank (opt-in):** ranking intensity **α = 1.0**, population size **m = 50**,
  payoff matrix from **4 games per ordered (s, r) pair**.

---

For full competition rules and submission requirements, see
[COMPETITION_RULES.md](./COMPETITION_RULES.md).
