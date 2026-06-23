# The 2p search baseline (`SearchAgent`)

A reference agent for the **2p (1‑vs‑1) track**, meant to be **read, played
against, and improved**. It is deliberately a *clean* implementation of the
approach that works best here — adversarial search — rather than a black box.

- Code: [`agents/search_agent.py`](../agents/search_agent.py),
  [`agents/eval2p.py`](../agents/eval2p.py)
- Two tiers: **`SearchLiteAgent`** (a beginner target) and **`SearchAgent`**
  (the strong reference). `SearchAgent` beats every other bundled baseline; you
  should aim to beat `SearchLite` first, then `Search`.

## Why search (and not, say, plain RL) for 2p

1‑vs‑1 Puerto Rico is unusually friendly to tree search:

- **Two players, winner‑takes‑all → effectively zero‑sum.** One side's gain is
  the other's loss, so classical **minimax / alpha‑beta** applies directly.
- **Near‑perfect information.** The *only* hidden state is the order of the
  face‑down plantation stack (see `docs/COMPETITION_RULES.md`). Everything
  else — money, goods, buildings, ships — is fully observable.
- **A clone‑able forward model.** `ForwardModel.clone()` gives you an
  independent copy of the game you can simulate freely; `.step(a)` applies an
  action and auto‑resolves forced moves.
- **A 1 s/move budget** that comfortably fits a few thousand simulated nodes.

That combination is the classic home turf of *strong evaluation + adversarial
search*. (This is also why the README warns that plain RL underperforms here.)

## How `SearchAgent` works

1. **Alpha‑beta over the forward model.** At each decision node the agent looks
   at `model.current_player()`. If it is the agent, it **maximises** the root
   player's score margin; if it is the opponent, it **minimises** it. Note this
   is decided by *who is to move*, **not** by ply parity — in Puerto Rico a
   player often makes several decisions in a row within a phase (e.g. loading
   several ships in the Captain phase), so naive "alternate every ply" minimax
   would be wrong.
2. **Zero‑sum margin evaluation** (`agents/eval2p.py`). A leaf is scored as
   `h(me) − h(opp)`, where `h` is a per‑player "victory‑point potential"
   heuristic (realised VP + held goods/money + production capacity + building
   potential). A finished game returns the true win/loss (`±WIN_BASE + VP
   margin`) so a decided result always dominates any heuristic estimate. Using a
   *difference* makes the value genuinely zero‑sum and cancels much of the
   heuristic's absolute miscalibration.
3. **Iterative deepening.** Search depth 1, 2, 3, … keeping the best move from
   the deepest *completed* depth, and stop on the budget (below).
4. **Move ordering.** Children are tried best‑first using a cheap static action
   score, which makes alpha‑beta prune far more (more depth for the same budget).
5. **Determinization of the hidden draw.** Before searching, the agent reshuffles
   the face‑down plantation stack *in its own clone*, so it never reads the true
   draw order — that would be both against the rules and brittle.
6. **A fast clone.** `PuertoRicoEnv.__deepcopy__` shares the immutable
   observation/action spaces and copies only mutable game state, making
   `clone()` ~6–7× cheaper. Clone speed *is* your search depth, so this matters.

### Strength is set by a **node budget** (reproducible across hardware)

A search agent capped by **wall‑clock time** is stronger on a fast machine and
weaker on a slow one — so two entrants comparing against it would not be facing
the same opponent. To keep the baseline **fair and reproducible**, `SearchAgent`
is capped by a **node budget** (a fixed number of simulated nodes): it plays the
*same move on any machine*, only the wall time differs. A hard time cap remains
as a safety net so a move never exceeds the 1 s competition limit.

| Tier | `node_budget` | vs `TradeBuilding` (strongest heuristic) | typical move time |
|---|---:|---:|---:|
| `SearchLiteAgent` | 250 | ~80% | ~0.1 s |
| `SearchAgent` | 1500 | 100% | ~0.5 s |

For **maximum strength** (e.g. as a personal sparring partner) construct it with
`SearchAgent(node_budget=None)`, which uses the full time budget instead
(stronger, but hardware‑dependent).

## Using it

```python
from agents.search_agent import SearchAgent, SearchLiteAgent
agent = SearchAgent()          # strong, deterministic
weak  = SearchLiteAgent()      # beginner target
```

- **Play against it in the browser:** `python webui/server.py`, pick **2**
  players, set one seat to **Human** and the other to **`Search`** (or
  **`SearchLite`**).
- **Measure your agent against it:**
  `python tools/bench2p.py --agent MyAgent --opponents SearchLite,Search`
  (add your agent to the `REGISTRY` in `tools/bench2p.py`).
- **Full Elo board:** uncomment the `Search` / `SearchLite` lines in
  `examples/run_tournament.py` (`make_pool`) — note search agents make the
  round‑robin much slower.

## Make it stronger — your opportunity

`SearchAgent` is intentionally a *baseline*, not the ceiling. Ranked, concrete
improvements (verified to be real gaps in this implementation):

1. **Principal‑variation search (PVS / negascout).** Ordering is already strong;
   probing non‑first moves with a null window and re‑searching only on fail‑high
   typically buys +1–2 plies at the same node budget.
2. **Quiescence / phase‑boundary extension.** A leaf can land *mid‑phase* (e.g.
   after your Captain load but before the opponent's), which the static eval
   misjudges. Keep searching until the phase boundary before evaluating.
3. **Endgame‑aware evaluation.** Doubloons and unshipped goods help only via the
   tiebreak, not final VP; down‑weight them near the end so the search does not
   hoard liquid resources instead of scoring.
4. **2p denial terms.** In 2p the violet (purple) buildings come in *single*
   copies and quarries are scarce; explicitly valuing *denying* a contested
   building/quarry to the opponent is worth points the leaf eval ignores.
5. **A learned evaluation or policy** (e.g. AlphaZero‑style self‑play to train a
   value/policy network used inside the search) — the heavy but highest‑ceiling
   route.
6. **A faster clone / transposition table / multi‑determinization** — engineering
   that converts directly into depth or robustness.

Pick any of these, measure the change with `tools/bench2p.py` against
`SearchLite`/`Search`, and you have a competitive 2p entry.

## Rules reminder

The search must not read the hidden plantation draw order (it determinizes
instead), must return a legal action within 1 s, and must not mutate the live
forward model (always `clone()` first). See
[`docs/COMPETITION_RULES.md`](COMPETITION_RULES.md).
