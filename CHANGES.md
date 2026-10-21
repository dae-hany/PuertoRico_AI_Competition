# CHANGES

Changes that affect **results** (game rules, agent strength, tournament
guarantees). Cosmetic edits are not listed here.

House rule, inherited from the research engine this repo shares an ancestor
with: **no engine rule changes without a verbatim citation from the rulebook.**
An automated audit's opinion is not evidence. The costly precedent was three
"rule violations" that turned out to be correct engine behaviour, discarded a
suite of runs, and cost 8 GPU-hours.

---

## 2026-10-21 — the verified rules: accept/decline decisions, a 225/298 observation

The research engine this repo shares an ancestor with (`PuertoRico_ESWA`,
commit `11d6c11`) was checked rule by rule against the Puerto Rico Deluxe
rulebook and against an independent reference model, and this repo now plays
the same rules. The two engines had diverged, so nothing was copied wholesale:
each difference was found, judged against the rulebook, and ported into this
repo's own structure. Clause IDs below are those of
`verification/rule/RULES.md` (R…, O…) and `verification/rule/CONFORMANCE.md`
(E…).

**Results move.** The observation is now **225 (2p) / 298 (3p)** values and
there are two new actions, so an agent that reads the vector at hard-coded
offsets past 49, and any network trained before today, must be updated. Game
records from before today no longer replay (they are refused, with that
reason). docs/BASELINES.md was regenerated.

### How the differences were found

Before any change, the old engine and the reference model
(`verification/oracle.py`, written from the rule restatement without reference
to either engine) were played side by side on 4,000 seeded games, 1,000 per
player count. With five differences bridged, the legal actions and the full
state agreed before every decision of every game: the five below are all there
were.

### Rules

* **Optional benefits are decisions** (R4.5, R6.1.6; E9). The engine granted
  these without asking:
  * Mayor (R5.2.2): *"The player who chooses this role may first take one
    colonist from the colonist supply (not from the colonist ship!) as the
    privilege."*
  * Hospice (R6.3.5): *"they may take a colonist from the colonist supply and
    place it on the newly acquired plantation or quarry tile."*
  * University (R6.3.10): *"they may take a colonist from the colonist supply
    and place it on the newly acquired tile."*
  * Captain (R5.6.5): *"When loading the first kind of goods, the captain
    takes as the privilege 1 extra VP. The captain does not get this extra VP
    for the goods loaded in subsequent turns in this captain phase."* For every
    role, players *"may choose not to use the action or privilege of the
    placard chosen."* The rulebook's one exception, *"The action of a role
    placard is optional (exception: captain)"*, is the loading, which stays
    mandatory (R4.5). A declined privilege is gone.
  * Harbor (R6.3.11): *"Each time, during the captain phase, owners of an
    occupied harbor load goods on a cargo ship, they earn one extra VP"*, and
    *"Players are not required to use the special function of an occupied
    beige building if they do not want to."*

  Each is now offered, as it arises, as a decision with exactly two legal
  actions: **170 accept** and **171 decline**. Declining can matter, because
  these spend the colonists and VP chips whose exhaustion ends the game. The
  benefits that only pay doubloons (the builder's and trader's privileges, the
  markets, the factory, the prospector) stay automatic: refusing money never
  helps.
* **A wharf owner who passes may still use the wharf** (R5.6.1, R6.3.12; E10,
  O5). *"The wharf can only be used once per captain phase by its owner, but
  the player may choose when to use it, if at all"*, and *"each player may get
  several turns to load goods on the cargo ships."* A pass used to end a
  player's loading for the phase, so a wharf owner who could not load onto a
  cargo ship had to use the wharf at once or never. Now a pass with a usable
  wharf keeps the player's place, and any later load gives them another turn;
  loading ends once every player has passed or finished since the most recent
  load.
* **Goods that fit are kept without asking** (R5.6.7; E11, O8). *"the players
  must store the goods left on their windroses ... each player must find room
  in one of their warehouses"*; only *"When players cannot store all their
  extra goods, they may choose which goods to store and which to return to the
  goods supply."* The engine asked even when everything fitted, and accepted an
  assignment that threw goods away (1 corn and 3 sugar with a small warehouse:
  sugar on the windrose and corn in the warehouse loses 2 sugar).
* **126 VP chips with five players** (RULES.md §2, §3.1): *"with 5 players: 126
  total points"*, in chips of *"31 x 1 and 19 x 5"*. It was 122. Neither track
  is affected.
* **A decision with one legal action is taken automatically** in every phase
  (E2; no rulebook clause, since the same action is applied either way). The
  engine had still asked, for instance, for a Mayor placement whose only
  target was one kind of slot.

Stricter, where the code had accepted calls no rule allows: `env.step` raises
`ValueError` for an action outside the mask (an unmapped index used to be
ignored, and a role chosen in the middle of the Builder phase went through);
the engine refuses a role mid-phase, any other action while a benefit is
pending, and a cargo ship that does not exist; and it checks for a second
wharf use before anything changes.

### Interface

* Actions 170 / 171; while a benefit is pending they are the only legal
  actions. docs/OBSERVATION_AND_ACTIONS.md lists which decisions an agent is
  asked for.
* Observation: a global `pending_benefit` one-hot over the five benefits at
  offsets 50–54, right after `governor_idx`; every later global field moves by
  5, and the player blocks start at `79 + 73 i`. `current_player` (36) and
  `governor_idx` (49) do not move. `puerto_rico/observation.py` documents the
  contract, and `tests/test_game.py` now checks its offsets against the env.
* Engine: `Benefit`, `pending_benefits`, `resolve_benefit()`,
  `goods_all_fit()`; the captain's pass bookkeeping is `_captain_done` and
  `_captain_declined`. A hospice or university colonist's target is kept as a
  board index, because the hand-rolled `__deepcopy__` rebuilds the boards (an
  object reference would point outside a search clone).
* `ForwardModel.step` raises `ValueError` for an illegal action.
* Records carry `"rules": "2026-10-21"`, and older records are refused.
  `describe` names the benefit, the harness's tamper fingerprint covers the
  pending benefit, and the web UI asks a human player about each benefit.
* `PpoAgent` refuses a checkpoint of the old 220/293 layout, with a message to
  retrain it.
* The heuristic baselines accept every benefit (the old engine always granted
  them); MCTS and Search decide by searching, PPO by what it learned.

### Verification

* **Behaviour-preserving work first, on its own commit** (the observation
  built in one pass, the mask restructured, dead code removed): 2,800 seeded
  games, 916,122 recorded steps, identical observations, masks, rewards, infos
  and full engine state before and after. `tools/bench_engine.py`: observation
  0.060 → 0.043 ms, mask 0.0016 → 0.0008 ms, search proxy 1,965 → 2,471
  sims/s.
* **Reference model.** The model, its rule restatement and its specification
  are vendored unchanged in `verification/` (SHA-256 pinned by
  `tests/test_lockstep.py`); `verification/lockstep.py` is this repo's
  converter to the canonical state, and the driver. On the final engine:
  12,000 games (5,000 each with 2 and 3 players, 1,000 each with 4 and 5),
  4,678,135 decisions, benefits accepted 70% of the time: no difference.
  `tests/test_lockstep.py` replays 30 games per player count.
* Tests: 138 → 195.

### Decisions per game and the 0.1 s rule

The benefit decisions add decisions to a game; the automatic single actions
and the automatic storage take some away. Same pairings and seeds:

| Pairing | Decisions per game, before | after |
|---|---:|---:|
| 2p ActionValue vs TradeBuilding (40 games) | 322.8 | 346.4 |
| 2p ShippingRush vs Factory (40 games) | 318.6 | 321.6 |
| 3p ShippingRush, ActionValue, TradeBuilding (30 games) | 405.3 | 417.3 |
| longest of these games | 513 | 526 |

A benefit decision is one more decision with two legal actions, under the same
0.1 s limit as any other. The bundled baselines keep their 1 s budget and stay
far from it (mean / slowest move, before → after, on a shared AMD Threadripper
PRO 5995WX): Search 223 / 294 → 218 / 309 ms against TradeBuilding and
240 / 435 → 214 / 288 ms against PPO; SearchLite 39 / 84 → 38 / 69 ms; MCTS
120 / 175 → 95 / 138 ms in 2p and 130 / 183 → 104 / 175 ms in 3p; PPO and the
heuristics under 2 ms.

### Baselines

* **Both PPO checkpoints were retrained** with the command and seed that made
  them (5M steps of self-play, seed 42; 2.0 h each on 8 CPU threads, the two
  runs side by side), on an unchanged trainer. They are kept as trained, with
  no prior knowledge added: the baseline is what the bundled trainer reaches
  with its defaults under these rules. training/README.md has the command; the
  commit that installs them records how they compare with the old networks,
  and docs/BASELINES.md where they stand now.
* **docs/BASELINES.md was regenerated.** `tools/measure_baselines.py` now also
  plays the 2p PPO against each planner (Search1500 160 games, SearchLite250
  and MCTS60 40 each, with a Wilson interval), which training/README.md used to
  quote by hand. Strength numbers are kept there alone: the README, the course
  pages, training/README.md and the SearchLite docstring point to it instead of
  quoting it.

## 2026-10-20 — a 2p PPO baseline, and it is the strongest 2p agent

The 2p track had no RL baseline: the trainer supported `--num_players 2`, but
only the 3p checkpoint shipped. It now ships a 2p one too, and the claim that
`Search` tops the 2p track no longer holds.

* `training/checkpoints/ppo_baseline_2p.pt`: 5M steps of self-play with the
  trainer's default settings and seed 42 (`--num_players 2 --device cpu`),
  1.6 h on a CPU. Greedy, as an entrant under the 0.1 s rule: **67.5% vs
  `Search` over 160 games (95% CI [0.60, 0.74])**, 80% vs `SearchLite`, 89% vs
  `MCTS` (40 games each), and every game against every heuristic
  (docs/BASELINES.md). training/README.md has the learning curve.
* `PpoAgent` takes a list of checkpoints, one per track, and picks the network
  by the observation's length. The registry's `PPO` baseline loads both, so it
  now plays 2p as well: `baseline_names(2)` includes it, which adds it to
  `tools/run_ladder.py`'s default 2p field and to `tools/play.py --vs all`.
  `tools/course_ladder.py` keeps an explicit baseline list, and PPO is added
  to it as an anchor, so students see where they stand against the strongest
  2p baseline (a reference column only; the grade still counts student games
  alone).
* `examples/ppo_entry/my_agent.py`: an entry built on the 2p checkpoint, in the
  submission layout (copy the checkpoint next to it as `weights.pt`).
* `tools/measure_baselines.py` measures PPO in 2p too, and no longer crashes
  when it echoes its tables to a cp949 console (the file was written first).
  Regenerating docs/BASELINES.md reproduced every earlier cell; the `Search1500`
  row now has its full 8 games per cell (the earlier run's time cap cut some
  cells to 4–6).
* No engine, sandbox or ranking changes.

## 2026-10-20 — one weight file per entry, in both editions

The public competition forbade all disk I/O, so a learned agent could ship its
parameters only inside its `.py` source. It now follows the CNU course's rule,
with a size cap that applies to the course too (docs/COMPETITION_RULES.md,
*The weight file*):

* An entry is the agent `.py` plus **at most one read-only weight file**, next
  to it, **at most 50 MB**, located from `__file__` and read at import or in the
  constructor. Writing files and every other disk or network access stay
  forbidden. Since the harness builds a new agent every game, the docs
  recommend a module-level cache.
* `tools/validate_submission.py` no longer warns on every call named
  `open`/`load`/`save`. It watches the files the agent opens while it is
  imported and constructed (an audit hook) and **fails** an entry that reads a
  second file, a file outside its folder, a path relative to the working
  directory or a file over `WEIGHT_FILE_MAX_MB`, or that writes one; it
  reports the weight file's size as a PASS. A source scan still **warns** on
  writes anywhere, reads inside `act`, and reads whose path is not built from
  `__file__`, skipping in-memory buffers and code under
  `if __name__ == "__main__":`. Each track's summary also gives the worker's
  start-up time.
* The course's weight-file exception becomes the shared rule. **New for the
  course: the 50 MB cap**, and since `tools/course_ladder.py` leaves out an
  entry with a FAIL, a larger file now keeps an entry out of a ladder round.
* No engine, sandbox or ranking changes: the sandbox worker already reads its
  entry's folder (now covered by a test).

## 2026-10-19 — 0.1 s per decision for every entrant, both tracks

The public competition now uses the course's limit: **0.1 s per decision**
(was 1 s), in the 2p and the 3p track.

* `tournament.match.TIME_LIMIT_S = 0.1` is the default of `play_game`, the
  runner, the sandbox, `play.py`, `validate_submission.py`, `run_ladder.py`,
  `course_ladder.py`, `bench2p.py` and the web UI.
* **The bundled baselines keep 1 s** (`BASELINE_TIME_LIMIT_S`). Their strength
  is fixed by node and simulation budgets, and `Search` (1500 nodes) and `MCTS`
  need more than 0.1 s per move, so holding them to 0.1 s would replace many of
  their moves with random ones and change what they measure. The harness
  recognises a baseline by its exact class (`agents.registry.is_bundled_baseline`);
  a subclass, a copy, or a sandboxed entry gets 0.1 s. `play_game` also takes a
  per-seat list of limits, and `baseline_time_limit_s=None` holds everyone to
  the same limit.
* Baseline numbers in docs/BASELINES.md are unchanged (they were measured with a
  generous limit, and the baselines' budget did not change).

## 2026-10-19 — CNU course edition: 0.1 s, fixed seeds, wins and rank

No engine or rule changes. The course edition's run settings and boards
(docs/COURSE_CNU_2026.md):

* **0.1 s per decision** in the course; `tools/course_ladder.py` defaults to it.
  (The public competition adopted the same limit the same day, see above.)
* **Fixed seed list, both seatings.** `course_ladder.py` plays every seed of
  `--seeds` (default `1-25`) and `--seeds-file` once in each seating: a ladder
  round is 25 seeds × 2 = 50 games per pair; the final run adds 475 private
  seeds, drawn by the new `tools/make_private_seeds.py` and kept out of the
  repository: 500 seeds × 2 = 1000 games per pair.
* **Published board = rank and wins only** (a draw is half a win; equal wins
  share a rank). Results against the baselines go on a separate reference
  board; replays stay with the organizer.
* `validate_submission.py` now prints `Result: …` as its last line, and accepts
  an agent file given relative to any directory (the sandboxed games used to
  look for it in the repository root). `play.py` notes when an opponent ran out
  of time and played moves at random.

## 2026-08-20 — engine throughput pass (behavior-preserving)

No rule changes. Pure performance refactors, available to every entrant's
planning agent equally. Behavior is pinned by `tests/test_clone_equivalence.py`
(hand-rolled clones value-identical to generic `copy.deepcopy` with the hooks
disabled; fast mask identical to `observe()["action_mask"]` at every decision
of seeded games; clones share no mutable state with the live game; mid-game
envs still pickle for the sandbox). Tests: 76 → 88.

* **Hand-rolled `__deepcopy__` on `PuertoRicoGame`, `Player`, and the component
  dataclasses** (`slots=True` on the latter), replacing generic deepcopy's
  per-object dispatch; `PuertoRicoEnv.__deepcopy__` copies its bookkeeping
  dicts by kind and preserves the `infos` ↔ `episode_metrics` aliasing via the
  memo. `ForwardModel.clone()`: 0.30 ms → 0.049 ms (~6×).
* **`ForwardModel.action_mask()` / `legal_actions()` fast path** — computes
  just the mask via `valid_action_mask()` instead of building (and discarding)
  the full nested observation for every player: 0.042 ms → 0.002 ms (~20×).
* Net effect on a depth-10 clone+step+mask search workload: ~870 → ~2 900
  sims/s (measured by `tools/bench_engine.py`, new). `SearchAgent`
  (node_budget=1500): ~0.30 → ~0.18 s/move. Baseline strength is unchanged
  (node budgets, not time budgets, set baseline strength — by design);
  `tools/measure_baselines.py` numbers in docs/BASELINES.md remain valid.

## 2026-08-14 — correctness, reproducibility and isolation pass

Brought the competition repo in line with the corrections the research engine
received during its journal revision, and fixed what that comparison exposed.
Tests: 35 → 76.

### Rules

* **University now staffs the building it just built.** A large building
  occupies two city spaces, stored as `[building, OCCUPIED_SPACE]`, and the free
  colonist was placed on the *last* city slot — the dummy space. The large
  building stayed unstaffed and silently forfeited its end-game bonus, which is
  only scored "when they are occupied". Rulebook (Deluxe, University): *"they
  may take a colonist from the colonist supply and place it on the newly
  acquired tile."* `puerto_rico/engine.py`, test:
  `tests/test_engine_rules.py::test_university_staffs_the_building_it_just_built`

### Reproducibility and tournament integrity

* **Each game owns a private `random.Random`.** The engine drew from the global
  `random` module and `reset()` re-seeded the global `random` / `np.random`
  states. Consequences, all now gone: an entrant's agent could shift the deal it
  was about to be dealt (the 2p deck does exhaust and reshuffle mid-game); the
  environment silently reset the agent's own RNG stream at every game; and
  `make_env(seed=None)` was not random at all once any seeded env had been
  created. The competition promises seeded, reproducible games — it now keeps
  that promise across processes. `puerto_rico/{engine,env}.py`,
  `tests/test_reproducibility.py`
* **The bundled heuristics seed their own tie-breaking noise.** It had been
  reproducible only as a side effect of the environment clobbering the global
  numpy state.
* **A mask-legal action the engine rejects now raises.** It used to pay the
  actor −10 and terminate the game, turning a bug in this repo into a corrupted
  result. Probe: 0 occurrences in 800 random games (234k decisions).
  It paid for itself immediately — see the MCTS entry below. `puerto_rico/env.py`
* **Agents run in their own process for the official run** (`tournament/sandbox.py`).
  The per-move deadline is enforced by killing the worker rather than measured
  after the fact, and the organizer's process owns the game, so an isolated
  agent can neither corrupt it nor read the face-down draw order. Planning
  agents still get a forward model: a determinized snapshot is shipped each move
  (~2 ms against a 1 s budget). `tests/test_sandbox.py` tests these by breaking
  them — a hung agent, a crashing agent, a tampering agent, and one that reports
  the deck order it can see.
* **In-process games verify the same rule instead of trusting it.** Every agent
  now gets its own `ForwardModel` (they shared one), and the harness checks after
  each move that the real game — hidden draw order included — is untouched,
  reporting breaches as `tampered`. `tournament/match.py`

### A game the field would not end

`python examples/run_tournament.py` — the command the README tells entrants to
run — hung forever, and had done since before this pass.

The round-robin plays every group, self-matches included, and a table of three
`FactoryAgent`s never finishes. Puerto Rico ends when the VP chips, the colonist
supply, or a player's city runs out, and *which* of those drains depends on the
roles the agents pick. Factory wanted the Mayor only once it already held
colonists — circular, since the Mayor phase is the only source of them. Three
agents reasoning that way never pick it: measured 734 rounds, Mayor chosen 3
times, one player with an empty city. Fixed on both sides, because either alone
leaves the competition exposed:

* Factory values the Mayor when it has **empty colonist slots** (as ShippingRush
  already did). All-Factory games now end in 159–331 decisions in both tracks.
* `play_game` caps a game at `max_steps` (4000, ~8x the longest game seen between
  baselines), scores it where it stands and flags `truncated`. An organizer must
  not be able to lose a tournament to one submission's choice of roles, and no
  amount of fixing the bundled agents guarantees that for agents nobody has seen
  yet. `tournament/match.py`, test:
  `tests/test_tournament.py::test_every_baseline_self_match_finishes`

### Agents

* **MCTS determinized once per *simulation* instead of once per move.** Cloning
  the live model reshuffles the hidden pile, so every simulation ran in a
  different world while the tree still held actions chosen in an earlier one;
  the engine then refused a move the mask had allowed. Before the assertion
  above, that ended the simulated game with a large negative reward and quietly
  poisoned the value estimates the search was built on. Same defect and fix as
  the research engine's D5. `agents/mcts_agent.py`
* **MCTS default budget 200 → 60 simulations.** 200 measures ~1.1 s/move: the
  baseline was breaking the 1 s rule entrants are held to. 60 measures
  0.27–0.36 s/move with no move over the limit in either track. A search agent's
  strength is only defined at a stated budget, so the budget is now quoted with
  every number.
* **MCTS plays both tracks.** Its Max^N value vectors were hard-coded to three
  players, so in the 2p track every move raised and the harness silently
  substituted a random one — the "baseline" was a random agent.
* **ShippingRush scored Wharf loads at actions 74–78**, which the environment
  never emits, so the branch was dead — while the agent's building priority buys
  the Wharf *first*. Corrected to 59–63.
* **Factory read three stale action encodings**: the Quarry at 14 (it is 13),
  face-up plantations by slot index rather than by tile type (so its entire
  preference order applied to whichever crop happened to share an index), and
  hard-coded ship capacities `[4, 5, 6]`, which invents a third ship in the 2p
  track. Paired A/B on identical seeds, 60 games/cell: 3p vs Random 11.7 → 30.0%,
  3p vs TradeBuilding 3.3 → 17.5%, 2p vs Random 60.8 → 90.0%. One cell regressed
  (2p vs TradeBuilding 16.7 → 0.0%); a three-way variant test attributes that to
  the tile-mapping fix itself rather than to the Quarry rule.

### Training

* **`--env_mode aoe_ablation` removed.** It imported `env.aoe_ablation_env`,
  which this repo does not ship, so choosing it raised `ModuleNotFoundError` —
  a flag `--help` advertised and that could never work.
* **PBRS wrapper made correct** (unused by the bundled trainer, but kept honest):
  `gamma` 0.99 → 1.0, so it matches the discount the trainer actually uses, and
  Φ(terminal) is now 0. Potential-based shaping only leaves the optimal policy
  unchanged if both hold; with neither, the wrapper silently was not
  potential-based shaping. Same fix as the research engine's D10.
* **The trainer supports both tracks** (`--num_players {2,3}`); the observation
  width follows from the seat count. `PpoAgent` reads that width out of the
  checkpoint instead of assuming 3p, and says so clearly when handed the other
  track's observation. The 2p track still ships no RL baseline — an open target.

### Measurement

* `tools/measure_baselines.py` regenerates [`docs/BASELINES.md`](docs/BASELINES.md),
  which is what the README and the submission guide now quote. The strength
  claims in those documents had drifted apart from each other (the submission
  guide ranked PPO next to Random while the README called it the strongest 3p
  baseline) and could not be checked without re-running games by hand.
* What the first measured run (500 games in 2p, 1000 in 3p) showed, against
  what the docs had asserted:
  - The **heuristic ordering differs between the tracks**, which no document
    mentioned: `TradeBuilding` leads 1‑vs‑1 (Elo 1841) but is third in 3p, where
    `ShippingRush` leads. Tuning one agent for both tracks is a real trade-off.
  - `ShippingRush` was described as "strong"; it is **last but one** among the
    heuristics in 2p (Elo 1345, below `Factory`).
  - `Factory` was described as "weak"; it is **third** in 2p.
  - `SearchLite` was described as "clearly beatable"; it beats **every** bundled
    heuristic (88–100%). Relative to `SearchAgent`, not to the field.
  - `Search` tops 2p and `PPO` tops 3p, as claimed — those two held up.
