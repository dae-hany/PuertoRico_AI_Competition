# CHANGES

Changes that affect **results** (game rules, agent strength, tournament
guarantees). Cosmetic edits are not listed here.

House rule, inherited from the research engine this repo shares an ancestor
with: **no engine rule changes without a verbatim citation from the rulebook.**
An automated audit's opinion is not evidence. The costly precedent was three
"rule violations" that turned out to be correct engine behaviour, discarded a
suite of runs, and cost 8 GPU-hours.

---

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
