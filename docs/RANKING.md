# How agents are ranked

Puerto Rico is a **3-player, general-sum** game. To rank submissions, the
organizer runs a **seat-balanced round-robin** among the submitted agents (and a
set of baselines): every combination of agents is played across all seat
assignments so that no agent is advantaged by its position at the table.

From the resulting games we compute **three** ranking metrics. **All three are
displayed**, but exactly **one is official**: the win rate. The other two are
secondary views that add information a single number cannot.

The combined table is produced by [`tournament/leaderboard.py`](../tournament/leaderboard.py),
ordered by the official metric, with TrueSkill and α-Rank shown as extra
columns. Output is written as Markdown, CSV, and JSON.

---

## Metric 1 — Win rate (official)

The **official** standings order. An agent's win rate is the **fraction of games
it wins** across the round-robin. A tie **splits the win equally** among the tied
winners (e.g. a 2-way tie gives each winner half a win).

Each agent's win rate is reported with a **Wilson score 95% confidence
interval**, so that close standings can be read together with their uncertainty
rather than as if the point estimates were exact.

Win rate is chosen as the official measure because it is **transparent and
intuitive**: it answers "who wins the most games" directly, with no model
assumptions. Implemented in
[`tournament/rankers/win_rate.py`](../tournament/rankers/win_rate.py).

## Metric 2 — TrueSkill (secondary)

**TrueSkill** (Herbrich et al., 2006) is a Bayesian skill-rating system built for
**free-for-all multiplayer** games, which fits 3-player Puerto Rico naturally.
Each agent holds a rating $N(\mu, \sigma^2)$; after each game, the **finishing
order** is used to update every player's $\mu$ (estimated skill) and $\sigma$
(uncertainty).

The leaderboard value is the **conservative** score $\mu - 3\sigma$ — a skill
level we are roughly 99% confident the agent exceeds. Using this lower bound
means agents with **few games** (and therefore high $\sigma$) are **not
over-rated**.

TrueSkill's strengths are that it handles **more than two players** natively,
**expresses uncertainty**, and **updates incrementally** — useful when
submissions arrive over time. Its main **limitation** is that it assumes a
**single transitive skill scale**: it presumes skill can be placed on one axis
where stronger always beats weaker. Implemented in
[`tournament/rankers/trueskill_ranker.py`](../tournament/rankers/trueskill_ranker.py)
(uses the `trueskill` package).

## Metric 3 — α-Rank (analysis only)

**α-Rank** (Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution",
*Scientific Reports*, 2019) ranks strategies by an **evolutionary process**
rather than a single scalar skill. Crucially, it does **not** assume a transitive
skill order, so it can reveal **non-transitive ("rock-paper-scissors") cycles**
among strategies — structure that scalar ratings hide.

How we compute it:

1. Build a **monomorphic payoff matrix** $M$, where $M[s][r]$ is the win rate of
   a focal agent playing strategy $s$ when **both opponents** play strategy $r$.
   Each entry is estimated by playing games of the form $[s, r, r]$.
2. Run a **single-population evolutionary model** with finite population size $m$
   and ranking intensity $\alpha$. The population is **monomorphic**, and a lone
   mutant fixates with the **Moran fixation probability**. In words: a mutant
   that is fitter than the resident is more likely to take over, and the
   advantage is sharpened by $\alpha$ — the fixation chance grows with the
   payoff gap $M[s][r] - M[r][r]$, reducing to the neutral $1/m$ when they are
   equal.
3. The induced **Markov chain over strategies** has a **stationary
   distribution**; the **mass** on each strategy ranks the agents.

α-Rank is **analysis-only**, not part of the live leaderboard, for two reasons:
it needs the **full payoff matrix** (extra games beyond the round-robin), and the
ranking can be **sensitive to the intensity parameter** $\alpha$. It is most
useful in write-ups (e.g. the IEEE CoG report) to **characterize the strategic
structure** of the field — whether the meta-game is transitive or cyclic.
Implemented in
[`tournament/rankers/alpha_rank.py`](../tournament/rankers/alpha_rank.py)
(self-contained, NumPy only).

---

## Why three metrics

Each metric answers a different question:

- **Win rate** is the intuitive, assumption-free **official** measure.
- **TrueSkill** adds **calibrated uncertainty** and scales gracefully to
  **rolling submissions**.
- **α-Rank** exposes **strategic structure** (non-transitivity) that scalar
  ratings cannot represent.

In practice, **win rate and TrueSkill usually agree closely**. Where they
diverge, it is typically on agents with **few or high-variance games**, for which
TrueSkill's conservative $\mu - 3\sigma$ and the Wilson interval on the win rate
react differently to limited data.

---

## Reference defaults

The bundled implementation ([`tournament/runner.py`](../tournament/runner.py)) uses
these defaults, all adjustable by the organizer:

- **Round-robin:** every group of 3 agents over all seatings, **1 game per
  seating** by default (raise it for tighter confidence intervals).
- **α-Rank:** ranking intensity **α = 1.0**, population size **m = 50**, with the
  payoff matrix estimated from **4 games per ordered (s, r) pair**.

---

For full competition rules and submission requirements, see
[COMPETITION_RULES.md](./COMPETITION_RULES.md).
