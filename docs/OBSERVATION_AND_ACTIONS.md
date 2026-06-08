# Observation and action encoding

Your agent's `act(observation, action_mask)` receives two NumPy arrays:

- `observation` — `float32`, shape **(293,)** — the full game state.
- `action_mask` — `int8`, shape **(200,)** — `action_mask[a] == 1` iff action `a`
  is currently legal. **You must return an `a` with `action_mask[a] == 1`.**

```python
import numpy as np
legal = np.where(action_mask > 0.5)[0]     # the actions you may choose from
```

You never *have* to decode the raw vector — the mask tells you what is legal, and
planning agents can read structured state from `forward_model.game`. But the
layout below lets you build features if you want to.

---

## Action space (`Discrete(200)`)

| Action index | Phase | Meaning |
|---|---|---|
| `0–7` | Role pick | choose a role: 0 Settler, 1 Mayor, 2 Builder, 3 Craftsman, 4 Trader, 5 Captain, 6 Prospector‑1, 7 Prospector‑2 |
| `8–12` | Settler | take a face‑up plantation: 8 Coffee, 9 Tobacco, 10 Corn, 11 Sugar, 12 Indigo |
| `13` | Settler | take a Quarry |
| `105` | Settler | Hacienda: draw one extra random plantation |
| `16–38` | Builder | build the building of type `a − 16` (BuildingType 0–22) |
| `39–43` | Trader | sell the good `a − 39` (Good 0–4) to the trading house |
| `44–58` | Captain | load a good on a ship: `ship = (a − 44) // 5`, `good = (a − 44) % 5` |
| `59–63` | Captain | load good `a − 59` using your **Wharf** |
| `64–68` | Captain (store) | keep good `a − 64` at the Windrose / Office |
| `106–110` | Captain (store) | keep good `a − 106` in your Warehouse |
| `93–97` | Craftsman | choose your privilege good `a − 93` |
| `120–125` | Mayor | place a colonist on an island tile of type `a − 120` (TileType 0–5) |
| `140–162` | Mayor | place a colonist on a building of type `a − 140` (BuildingType 0–22) |
| `15` | any | Pass / decline the current optional action |

Index ranges not listed are never legal; the `action_mask` is always the
authoritative source of legality.

**Good order:** `0 Coffee, 1 Tobacco, 2 Corn, 3 Sugar, 4 Indigo`.
**TileType order:** `0 Coffee, 1 Tobacco, 2 Corn, 3 Sugar, 4 Indigo plantation, 5 Quarry`.

---

## Observation vector (293 dims)

The vector is four concatenated blocks:

```
[ global (74) | player_0 (73) | player_1 (73) | player_2 (73) ]
   offset 0       offset 74       offset 147      offset 220
```

The per‑player blocks are **absolute** (player_0/1/2), not ego‑centric. On your
turn, the global `current_player` field (offset 36) tells you which block is you.

### Global block — offsets 0–73

| Offset | Field (size) | Meaning |
|---:|---|---|
| 0 | `cargo_ships_good_onehot` (18) | per ship (×3), one‑hot over {5 goods, empty} of what it carries |
| 18 | `cargo_ships_load` (3) | goods currently loaded on each ship |
| 21 | `cargo_ships_space` (3) | remaining capacity of each ship |
| 24 | `colonists_ship` (1) | colonists waiting on the colonist ship |
| 25 | `colonists_supply` (1) | colonists left in the general supply |
| 26 | `current_phase_onehot` (10) | one‑hot of the current phase |
| 36 | `current_player` (1) | index of the player to move (you, on your turn) |
| 37 | `face_up_plantation_counts` (6) | available face‑up plantations by type |
| 43 | `game_progress` (1) | normalised game progress |
| 44 | `goods_supply` (5) | remaining supply of each good |
| 49 | `governor_idx` (1) | current governor (first to pick a role this round) |
| 50 | `quarry_stack` (1) | quarries remaining |
| 51 | `role_doubloons` (8) | doubloons sitting on each unpicked role |
| 59 | `roles_available` (8) | 1 if that role can still be picked this round |
| 67 | `trading_house_count` (1) | number of goods in the trading house (0–4) |
| 68 | `trading_house_has_good` (5) | 1 per good type present in the trading house |
| 73 | `vp_chips` (1) | victory‑point chips left in the supply (a game‑end trigger) |

### Per‑player block — offsets 0–72 (within each player's 73‑dim block)

| Offset | Field (size) | Meaning |
|---:|---|---|
| 0 | `building_colonists` (23) | colonists placed in each building type |
| 23 | `doubloons` (1) | the player's money |
| 24 | `empty_city_spaces` (1) | free building slots in the city (of 12) |
| 25 | `goods` (5) | goods the player owns |
| 30 | `has_building` (23) | 1 if the player owns that building type |
| 53 | `island_empty_spaces` (1) | free plantation/quarry slots on the island |
| 54 | `island_tile_count` (6) | tiles owned, by type |
| 60 | `island_tile_occupied` (6) | staffed (colonist‑occupied) tiles, by type |
| 66 | `production_capacity` (5) | how many of each good the player can produce now |
| 71 | `unplaced_colonists` (1) | colonists held but not yet placed |
| 72 | `vp_chips` (1) | VP chips the player has earned (from shipping) |

So player *i*'s `doubloons` is at absolute index `74 + 73*i + 23`, and so on.

For the meaning of roles, goods, buildings, and scoring, see
[`GAME_RULES.md`](GAME_RULES.md).
