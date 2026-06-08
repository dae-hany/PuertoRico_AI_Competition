# Puerto Rico — rules for AI developers

Puerto Rico is an economic strategy board game (designer: Andreas Seyfarth). This
competition uses the **3‑player** variant; this repository is an independent
re‑implementation built for research and education. This page gives you enough of
the rules to reason about strategy — it is not a full rulebook. For the exact
encoding your agent sees, read [OBSERVATION_AND_ACTIONS.md](OBSERVATION_AND_ACTIONS.md).

## Goal

Score the most **Victory Points (VP)**. You develop a colony by repeatedly
choosing **roles**, then producing and shipping **goods** and constructing
**buildings**. VP comes from three sources:

1. **VP chips** earned by shipping goods (the Captain action).
2. The **printed VP value** of buildings you own.
3. The **end‑game bonuses** of the five large buildings (see below).

The highest total VP wins. Ties are broken by **leftover doubloons + goods**.

## Roles and the round structure

Each round, players take turns choosing a role that **nobody else has taken yet**
that round. When a role is chosen, **every** player performs that role's action in
turn order — but the player who *chose* it gets a small **privilege** (an extra
good, a discount, or shipping priority). Any role left unpicked at the end of a
round gains **one doubloon**, making it more attractive next round.

The engine indexes roles **0–7**:

| # | Role | Action (all players) | Chooser's privilege |
|---|------|----------------------|---------------------|
| 0 | Settler | Take one plantation tile (or a **Quarry**, which discounts buildings) onto your island. The **Hacienda** building lets you draw an extra random plantation. | Choice of tile / extra benefits via buildings |
| 1 | Mayor | Distribute new **colonists** (workers) from the colonist ship onto plantations and buildings. A tile or building only works if **staffed** by a colonist. | First / extra colonist |
| 2 | Builder | Buy **one building** with doubloons (quarries you own reduce the cost). | Build at a discount |
| 3 | Craftsman | Every player produces goods they have capacity for (staffed plantation **and** staffed production building). | Produce one extra **privilege good** of their choice |
| 4 | Trader | Sell **one** good to the trading house for doubloons. Prices rise with good value; the house holds up to **4** goods. | Trading priority |
| 5 | Captain | Ship goods onto the cargo ships for VP chips (**1 VP chip per good shipped**). | Shipping priority |
| 6–7 | Prospector‑1 / ‑2 | (No shared action.) | Chooser simply gains **one doubloon** |

## Goods

Five goods, in the engine's index order:

| # | Good | Notes |
|---|------|-------|
| 0 | Coffee | Higher value |
| 1 | Tobacco | Higher value |
| 2 | Corn | **Needs no production building** |
| 3 | Sugar | Lower value |
| 4 | Indigo | Lower value |

Except for **Corn**, each good requires a **matching production building plus a
staffed plantation** to produce. Roughly, Coffee/Tobacco are higher value and
Corn/Indigo lower.

## Buildings

There are **23 building types** (engine `BuildingType` 0–22). A player's city has
**12 building spaces**. Buildings include small/large production buildings (indigo,
sugar, tobacco, coffee) and "violet" buildings with special powers, for example:

- **Factory** — bonus doubloons for producing many *different* goods.
- **Harbor** — extra VP per shipment.
- **Wharf** — your own private one‑good ship.
- **Office** — sell duplicate goods.
- **Hospice** — free colonist when settling.
- **University** — free colonist when building.
- **Warehouse** — store goods past the Captain phase.

### Large buildings

Five **large** buildings each cost **10**, are worth **4 VP**, and grant an
end‑game bonus:

| Building | End‑game bonus |
|----------|----------------|
| Guildhall | VP per production building |
| Residence | VP for many island tiles |
| Fortress | VP per colonist |
| Customs House | VP per shipped VP chip |
| City Hall | VP per violet building |

## Shipping (Captain) details

There are **3 cargo ships** with fixed capacities, plus per‑player options
(Harbor, Wharf). Each ship can hold **only one good type at a time**. You earn
**1 VP chip per good shipped**.

## Game end

The current round finishes and the game ends when **any** of these occurs:

- the **VP‑chip supply** runs out,
- the **colonist supply** cannot refill the colonist ship, or
- any player fills **all 12** of their city building spaces.

Highest total VP wins; ties broken by leftover doubloons + goods.

## 3‑player setup constants

| Constant | Value |
|----------|-------|
| VP chips | 75 |
| Colonists in supply | 55 |
| Colonist ship capacity | 3 |
| Starting doubloons (each) | 2 |
| Cargo ship capacities | 4 / 5 / 6 |

## Further reading

- [OBSERVATION_AND_ACTIONS.md](OBSERVATION_AND_ACTIONS.md) — the observation vector and action encoding.
- [COMPETITION_RULES.md](COMPETITION_RULES.md) — competition format and submission rules.
