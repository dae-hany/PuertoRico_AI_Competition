# Engine Conformance

This document specifies how the environment turns the rules of [`RULES.md`](RULES.md) into a
sequence of single-player decisions, how it settles the questions that the rulebook leaves open,
and how the engine was checked against the rules.

## 1. Method

The engine is checked in two independent ways.

1. **Rule-level tests.** Each rule with a testable consequence is pinned by a scenario test that
   cites the rule identifier.
2. **Lockstep comparison with an independent reference model.** `verification/oracle.py`
   implements the rules from `RULES.md` and §2–§4 of this document without reference to the engine
   code. `verification/lockstep.py` plays seeded games in the engine and the reference model side by
   side. Before every decision it compares the set of legal actions and the full game state (§4.2),
   and at the end of the game the scores and winners.

## 2. Decision model

### 2.1 Turns and automatic moves

- **E1** Players are numbered 0 to n−1 in clockwise order. The environment always asks one player,
  the *current player*, for one action (§3).
- **E2** When exactly one action is legal, the environment takes it without asking. Role selection
  always offers at least two placards, and benefit decisions (E9) always offer two actions, so
  neither is ever automatic.
- **E3** The environment also takes, without asking, the steps that involve no choice: the
  prospector's privilege, production in the craftsman phase, colonist distribution, the end of each
  phase (R5.1.7, R5.2.7, R5.5.6, R5.6.8), and the end of each round (R4.6). The game ends after
  the round-end steps of R4.6 of the round in which a condition of R7.1 occurred, so the final
  state already shows the new governor and the doubloons placed on the unchosen placards.

### 2.2 Phases

- **E4 Role selection.** The current player chooses an available placard (R4.1–R4.3).
- **E5 Settler.** On a player's turn the hacienda draw (R6.3.2) is a separate decision, available
  until the player makes their main choice. The main choice is a face-up plantation by kind, a
  quarry, or pass. The turn ends with the main choice, after the hospice decision if one is
  offered (E9).
- **E6 Mayor.** If the colonist supply is not empty, the mayor first decides on the privilege
  colonist (E9). The colonists on the ship are then distributed (R5.2.3), and the players place
  colonists one at a time, the mayor first and then clockwise (O4). At the start of a player's
  placement turn all of that player's colonists return to their windrose, so the player places all
  of them again; this realises the free rearrangement of R5.2.4. Each action places one colonist on
  an unoccupied island tile of a given kind or on a building of a given kind with a free circle. A
  placement turn ends as soon as the player has no colonist left on the windrose or no empty
  circle. Whenever the player's colonists on the windrose are at least as many as their empty
  circles, every empty circle must be filled (R5.2.5), and the environment fills them without
  asking. The ship is refilled after the last player's turn (R5.2.7, O3).
- **E7 Builder, trader.** One action per player: a building or a good, or pass. The university
  decision (E9) follows a building action.
- **E8 Craftsman.** Production is automatic. The craftsman then chooses a privilege good among the
  kinds they produced in this phase that remain in the supply, or passes. If there is no such kind,
  the phase ends without a decision.
- **E9 Benefits.** The five optional benefits whose refusal can delay the end of the game are
  offered as a decision with two actions, *accept* and *decline*, immediately when they arise. The
  benefit's player is the current player, and no other action is legal.

  | Benefit | Rule | Offered to | Offered when | Effect of *accept* |
  |---|---|---|---|---|
  | Mayor's privilege | R5.2.2 | the mayor | right after the mayor is chosen, if the colonist supply is not empty | one colonist from the supply to the mayor's windrose |
  | Hospice | R6.3.5 | the player who placed the tile | right after a face-up plantation or a quarry is placed (not the hacienda tile), if the player's hospice is occupied and the supply or the ship holds a colonist | one colonist, from the supply if possible, otherwise from the ship, onto the new tile |
  | University | R6.3.10 | the player who built | right after a building is built, if the player's university was occupied before the build and the supply or the ship holds a colonist | one colonist, from the supply if possible, otherwise from the ship, onto the new building |
  | Captain's privilege | R5.6.5 | the captain | right after the captain's first load of the phase, by cargo ship or wharf (O6) | +1 VP from the VP supply |
  | Harbor | R6.3.11 | the loading player | right after every load, by cargo ship or wharf, if the player's harbor is occupied; after the captain's privilege when both arise | +1 VP from the VP supply |

  Declining has no effect; a declined captain's privilege is used up, as it belongs to the first
  load. The benefits that can only bring doubloons are applied automatically, because refusing
  doubloons never helps a player (no rule depends on the bank, O2): the builder's and trader's
  privileges, the markets, the factory, and the prospector.
- **E10 Captain.** On a loading turn the player must load onto a cargo ship if any load is possible;
  only ships that take the most goods of the chosen kind are offered (R5.6.4). An occupied, unused
  wharf adds one action per kind the player holds (R6.3.12). Pass is legal only when no cargo-ship
  load is possible. A player who passes while unable to use a wharf takes no further turn in the
  phase. A player who passes while still able to use their wharf keeps their place and gets another
  turn if any player loads after that pass. Loading ends as soon as every player has either taken
  their last turn or passed since the most recent load (O5); storage follows.
- **E11 Storage.** Starting with the captain and continuing clockwise, each player stores the goods
  left on their windrose (R5.6.7). Occupied warehouses provide warehouse slots, one for the small
  and two for the large warehouse; a warehouse slot keeps all goods of one kind, and the windrose
  keeps one good of one kind. The goods all fit when the player holds at most as many kinds as
  warehouse slots, or exactly one kind more and one of the kinds is a single good; this is decided
  at the start of the player's storage turn, and the goods do not change during it. Then the
  player's only legal action is pass, which keeps everything, so the environment stores it without
  asking. Otherwise the player chooses what to keep, one kind per action: a kind for a free
  warehouse slot, or a kind for the windrose while the windrose is free. Any choice is legal, even
  one that keeps fewer goods than another would (R5.6.7: the player "may choose which goods to
  store"). Pass is legal only once the windrose and every warehouse slot hold a kind; it ends the
  storage turn and returns the goods that are not kept to the supply. Goods leave the player only
  at this pass. Some of these decisions differ only in the order of the assignments, for example a
  kind for the warehouse slot and then another for the windrose, or the same two kinds the other
  way round; the environment still presents them, since each has at least two legal actions.

### 2.3 Settled open questions

| | Question (RULES.md §8) | Resolution |
|---|---|---|
| O1 | Tie after the tie-break | All tied players win. |
| O2 | Empty bank | The bank is unlimited. |
| O3 | Short colonist supply at the end of a mayor phase | The remaining colonists go onto the ship, and the game ends at the end of the round. |
| O4 | Simultaneous placement | Players place one at a time, the mayor first, then clockwise. |
| O5 | End of loading with a declining wharf owner | Loading ends when every player has taken their last turn or passed since the most recent load (E10). |
| O6 | Captain's privilege on a wharf load | It applies: the wharf load counts as a load. |
| O7 | Hidden VP chips | Every player's VP chips are part of the observation. |
| O8 | Optional beige functions against mandatory storage | Storage is mandatory as R5.6.7 states: a player keeps everything that fits, and otherwise fills the windrose and every warehouse slot (E11). |
| O9 | Optional actions against automatic production and distribution | Both happen automatically (E3), as the role texts describe them; the environment offers no way to decline them. |

## 3. Action encoding

The action space is `Discrete(200)`. Indices not listed are never legal.

| Indices | Action | Legal when |
|---|---|---|
| 0–7 | choose role r | role selection; placard r is available |
| 8–12 | settler: take a face-up plantation of kind k | main settler choice; a face-up tile of kind k; an empty island space |
| 13 | settler: take a quarry | main settler choice by the settler or an owner of an occupied construction hut; the quarry stack is not empty; an empty island space |
| 15 | pass | settler, builder, trader turns; the craftsman's privilege; captain turns without a cargo-ship load; storage turns under E11 (when the goods all fit, or once the windrose and every warehouse slot hold a kind) |
| 16–38 | build building b (16 + b) | builder turn; b in the supply; not owned; enough empty city spaces; affordable after reductions (R5.3.2) |
| 39–43 | sell good g (39 + g) | trader turn; the player holds g; fewer than four goods in the trading house; g not there, or the player's office is occupied |
| 44–58 | load good g on cargo ship s (44 + 5s + g) | captain turn; the player holds g; ship s is not full and carries g, or is empty while no other ship carries g; ship s takes the most goods of kind g among such ships |
| 59–63 | wharf: good g (59 + g) | captain turn; occupied wharf, not used in this phase; the player holds g |
| 64–68 | storage: windrose for good g (64 + g) | storage turn in which the goods do not all fit (E11); the player holds g; windrose unassigned; g not assigned to a warehouse |
| 93–97 | craftsman's privilege: good g (93 + g) | the craftsman produced g in this phase; g is in the supply |
| 105 | hacienda draw | settler turn before the main choice; occupied hacienda; not yet drawn in this turn; face-down supply not empty; an empty island space |
| 106–110 | storage: warehouse for good g (106 + g) | storage turn in which the goods do not all fit (E11); the player holds g; a free warehouse slot; g not assigned |
| 120–125 | mayor: colonist onto an unoccupied island tile of kind t (120 + t) | placement turn; a colonist on the windrose; an unoccupied tile of kind t |
| 140–162 | mayor: colonist onto building b (140 + b) | placement turn; a colonist on the windrose; the player owns b and b has a free circle |
| 170 | accept the offered benefit | benefit decision (E9) |
| 171 | decline the offered benefit | benefit decision (E9) |

Index orders:

- Roles r: 0 settler, 1 mayor, 2 builder, 3 craftsman, 4 trader, 5 captain, 6 first prospector,
  7 second prospector. With two or four players the prospector in play is placard 6.
- Goods g: 0 coffee, 1 tobacco, 2 corn, 3 sugar, 4 indigo.
- Island tile kinds t (and plantation kinds k): 0 coffee, 1 tobacco, 2 corn, 3 sugar, 4 indigo,
  5 quarry.
- Cargo ships s: in increasing order of capacity.
- Buildings b: 0 small indigo plant, 1 small sugar mill, 2 large indigo plant, 3 large sugar mill,
  4 tobacco storage, 5 coffee roaster, 6 small market, 7 hacienda, 8 construction hut, 9 small
  warehouse, 10 hospice, 11 office, 12 large market, 13 large warehouse, 14 factory,
  15 university, 16 harbor, 17 wharf, 18 guild hall, 19 residence, 20 fortress,
  21 customs house, 22 city hall.

## 4. Reference model interface

### 4.1 API

`verification/oracle.py` provides:

```python
class Oracle:
    @classmethod
    def new_game(cls, num_players, governor, plantation_stack, face_up): ...
    def legal_actions(self) -> list[int]: ...
    def step(self, action: int, shuffle=None) -> None: ...
    def canonical(self) -> dict: ...
```

- `new_game` builds the setup of RULES.md §3 itself. Only the random elements come from the caller:
  the governor, the face-down plantation supply after setup (`plantation_stack`, a list of kind
  names with the top tile last), and the face-up tiles. It raises `ValueError` if these tiles do not
  form exactly the plantations that the setup leaves.
- `legal_actions` returns the sorted legal action indices of the current decision, or `[]` when the
  game is over.
- `step` applies the current player's action and then every automatic step (E2, E3, E6) until the
  next decision or the end of the game. It raises `ValueError` for an illegal action. When the
  discard pile must be shuffled into a new face-down supply (R5.1.7), it calls
  `shuffle(tiles)` with the discard pile as a list of kind names and uses the returned list, top
  tile last, as the new supply; without `shuffle` it uses `random.shuffle` on a copy.

### 4.2 Canonical state

`canonical()` returns plain Python values, so two states compare with `==`. Names are the upper-case
English names used below; dictionaries list every key shown; "sorted" means Python's default
order of the name strings. The `shuffle` hook receives the discard pile sorted and must return a
permutation of it.

```python
{
  "num_players": int,
  "over": bool,                       # the game has ended
  "phase": str,                       # "ROLE", "SETTLER", "MAYOR", "BUILDER", "CRAFTSMAN",
                                      # "TRADER", "CAPTAIN", "STORE", or "OVER"
  "benefit": str | None,              # "MAYOR_PRIVILEGE", "HOSPICE", "UNIVERSITY",
                                      # "CAPTAIN_PRIVILEGE", "HARBOR", or None
  "current_player": int | None,       # None when over
  "governor": int,
  "rounds_completed": int,
  "active_role": str | None,          # role of the phase in progress; None in "ROLE" and "OVER"
  "role_player": int | None,          # who chose it; None in "ROLE" and "OVER"
  "roles_available": list[str],       # sorted
  "role_doubloons": dict[str, int],   # every placard in the game
  "vp_supply": int,                   # may become negative (R5.6.6)
  "colonist_supply": int,
  "colonist_ship": int,
  "quarry_supply": int,
  "goods_supply": dict[str, int],     # every good
  "building_supply": dict[str, int],  # every building
  "plantation_stack": list[str],      # face-down supply, top tile last
  "face_up": list[str],               # sorted
  "discard": list[str],               # sorted
  "trading_house": list[str],         # sorted
  "ships": list[list],                # [capacity, good or None, load], increasing capacity
  "players": [{
      "doubloons": int,
      "vp": int,                      # VP chips, including VP beyond the chip supply
      "goods": dict[str, int],        # every good
      "island": dict[str, list[int]], # tile kind -> [tiles, occupied tiles]; kinds with tiles only
      "city": dict[str, int],         # owned building -> colonists on it
      "unplaced": int,                # colonists on the windrose
  }, ...],
  "scores": list[list[int]] | None,   # when over: [VP total, doubloons + goods] per player
  "winners": list[int] | None,        # when over: sorted winners (R7.3, O1)
}
```

Names: roles `SETTLER, MAYOR, BUILDER, CRAFTSMAN, TRADER, CAPTAIN, PROSPECTOR_1, PROSPECTOR_2`;
goods `COFFEE, TOBACCO, CORN, SUGAR, INDIGO`; island kinds the goods plus `QUARRY`; buildings
`SMALL_INDIGO_PLANT, SMALL_SUGAR_MILL, INDIGO_PLANT, SUGAR_MILL, TOBACCO_STORAGE, COFFEE_ROASTER,
SMALL_MARKET, HACIENDA, CONSTRUCTION_HUT, SMALL_WAREHOUSE, HOSPICE, OFFICE, LARGE_MARKET,
LARGE_WAREHOUSE, FACTORY, UNIVERSITY, HARBOR, WHARF, GUILDHALL, RESIDENCE, FORTRESS, CUSTOMS_HOUSE,
CITY_HALL` (`INDIGO_PLANT` and `SUGAR_MILL` are the large indigo plant and the large sugar mill).

## 5. Rule-by-rule conformance

| Rules | Engine (`env/engine.py` unless stated) | Tests | Result |
|---|---|---|---|
| §3 setup (R3.1.1–R3.2.7) | `PuertoRicoGame.__init__`, `_setup_players`, `_init_roles`, `configs/constants.py` | `test_setup_matches_the_rulebook`; lockstep | Fixed: five players had 122 VP chips instead of 126. |
| §4 rounds (R4.1–R4.6) | `select_role`, `_end_phase`, `_end_round` | `test_round_end_places_doubloons_and_passes_the_governor`, `test_two_players_choose_three_roles_each`; lockstep | Match. |
| §5.1 settler (R5.1.1–R5.1.7) | `action_settler`, `action_hacienda_draw`, `_deal_face_up_plantations`; mask in `env/pr_env.py` | `test_plantation_supply_is_reshuffled_from_the_discard_pile`, hospice tests; lockstep | Fixed: the hospice colonist was given without a choice (E9). |
| §5.2 mayor (R5.2.1–R5.2.7) | `select_role`, `resolve_benefit`, `_distribute_colonists`, `action_mayor_place_colonist`, `_execute_phase_cleanup`; automatic placement in `env/pr_env.py` | `test_mayor_example_four_players`, `test_mayor_refills_the_ship_from_empty_building_circles`, `test_mayor_privilege_*`, `test_mayor_deadlock.py`; lockstep | Fixed: the privilege colonist was given without a choice (E9). |
| §5.3 builder (R5.3.1–R5.3.6) | `action_builder`; mask | `test_quarry_example`, `test_builder_privilege_and_quarries_add_up_but_not_below_zero`, university tests; lockstep | Fixed: the university colonist was given without a choice (E9). |
| §5.4 craftsman (R5.4.1–R5.4.4) | `_execute_craftsman_production`, `action_craftsman` | `test_craftsman_takes_goods_in_turn_order_from_a_short_supply`, `test_factory_example`; lockstep | Match. |
| §5.5 trader (R5.5.1–R5.5.7) | `action_trader`, `_execute_phase_cleanup`; mask | `test_trader_prices_privilege_markets_and_office`, `test_full_trading_house_is_emptied_at_the_end_of_the_phase`; lockstep | Match. |
| §5.6 captain (R5.6.1–R5.6.8) | `action_captain_load`, `action_captain_pass`, the storage actions, `_execute_phase_cleanup`; mask | `test_captain_example_four_players`, `test_harbor_and_wharf_example`, `test_engine_rules.py`, storage and wharf tests in `test_decisions.py`; lockstep | Fixed: the captain's privilege and the harbor VP were given without a choice (E9); a player who passed lost the later use of their wharf (E10); a player could discard goods that fit (E11). |
| §5.7 prospector (R5.7.1) | `select_role` | `test_prospector_gives_one_doubloon_and_no_action`; lockstep | Match. |
| §6 buildings (R6.1.1–R6.4.5) | the phase actions above, `get_scores` | the building examples in `test_rulebook_examples.py`; lockstep | Match, apart from the benefits listed above. |
| §7 game end and scoring (R7.1–R7.3) | `check_game_end`, `get_scores`; rewards in `env/pr_env.py` | `test_*_ends_the_game_after_the_round*`, `test_tie_break_counts_doubloons_and_goods`, `test_pr_env.py`; lockstep | Match. |

The environment also became stricter where the engine had accepted calls that no rule allows:
`PuertoRicoEnv.step` rejects actions outside the mask (it had ignored unmapped indices), the
engine rejects a role choice in the middle of a phase and any other action while a benefit
decision is pending, and a second use of the wharf is rejected before any state changes.

## 6. Verification results

Checked on 2026-10-21.

**Lockstep comparison.** 8,000 games, 2,000 for each player count from two to five, with
3,767,969 decisions. The legal actions and the canonical state agreed before every decision, and
the scores, winners, and rewards at the end of every game. The games used three random policies
(`verification/lockstep.py`: mostly non-pass actions, a builder-biased policy, and uniform play;
benefits accepted with probability 0.7). How often the rarer situations came up:

| Situation | Count |
|---|---|
| Benefit decisions: mayor's privilege, hospice, university, captain's privilege, harbor | 94,815; 47,946; 3,419; 51,311; 10,630 |
| Decisions with a usable wharf | 6,594 |
| Captain steps after a pass with a usable wharf | 5,804 |
| Office sale of a kind already in the trading house available | 13,154 |
| Hacienda draw available | 70,450 |
| Construction hut quarry available | 62,119 |
| Craftsman's privilege available | 65,366 |
| Large building affordable | 19,818 |
| Storage decisions with a free warehouse slot | 4,277 |
| Reshuffles of the plantation discard pile | 15,308 |
| Game ends by colonists, city, VP chips | 7,336; 410; 254 |

**Other checks.**
- `tests/` (92 tests) pins the rules and the decision model; `test_lockstep.py` replays 30 games
  per player count. `verification/test_oracle_self.py` (79 tests) checks the reference model
  against the rulebook examples and conservation laws; its author also confirmed that it detects
  77 deliberately planted rule errors.
- The optimizations of the observation, the action mask, and copying changed no behaviour: 1,800
  seeded games produced identical observations, masks, rewards, and states at every decision
  before and after them.

**Differences found and fixed.** The 5-player VP chips (R3.1.4); the five benefits of E9, which
the engine had given without a choice; the wharf after a pass (E10); and storage, where the
decision model itself let a player discard goods that fit (E11). The lockstep comparison cannot
detect an error shared by the specification and both implementations; the storage error was of
that kind and came to light when the reference model's author questioned the specification.

**Speed** (random play, decisions per second; `copy.deepcopy` of a mid-game environment):

| | 2 players | 3 players | 4 players | 5 players | deepcopy |
|---|---|---|---|---|---|
| Rule fixes only | 12,467 | 9,109 | 8,620 | 7,164 | 2,057 µs |
| Rule fixes and optimizations | 19,196 | 16,092 | 14,011 | 12,773 | 304 µs |

**Running the checks.**

```
python -m pytest tests/ verification/                         # about 2.5 minutes
LOCKSTEP_GAMES=500 python -m pytest tests/test_lockstep.py    # longer lockstep run
python -m verification.lockstep --games 2000 --players 2,3,4,5
```
