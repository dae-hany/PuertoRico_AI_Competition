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

| # | Good | Trading‑house price | How it is produced |
|---|------|--------------------:|--------------------|
| 0 | Coffee | **4** | Coffee Roaster + staffed Coffee plantation |
| 1 | Tobacco | **3** | Tobacco Storage + staffed Tobacco plantation |
| 2 | Corn | **0** | **No building needed** — a staffed Corn plantation produces Corn |
| 3 | Sugar | **2** | (Small) Sugar Mill + staffed Sugar plantation |
| 4 | Indigo | **1** | (Small) Indigo Plant + staffed Indigo plantation |

The price column is exactly what the **Trader** receives when selling that good
to the trading house (engine `GOOD_PRICES`). Except for **Corn**, a good is
produced only when you have **both** a matching production building **and** a
staffed plantation of that crop — both occupied by colonists.

## Buildings

There are **23 building types** (engine `BuildingType` 0–22). A player's city has
**12 building spaces**. A building only grants its ability/VP while it is
**staffed** (at least one colonist on it). Cost is in doubloons; *Cap* is how many
colonists it can hold. Values below are exact (engine `BUILDING_DATA`).

**Production buildings** (turn staffed plantations into goods during Craftsman):

| # | Building | Cost | VP | Cap | Produces |
|---|----------|-----:|---:|----:|----------|
| 0 | Small Indigo Plant | 1 | 1 | 1 | Indigo |
| 1 | Small Sugar Mill | 2 | 1 | 1 | Sugar |
| 2 | Indigo Plant | 3 | 2 | 3 | Indigo |
| 3 | Sugar Mill | 4 | 2 | 3 | Sugar |
| 4 | Tobacco Storage | 5 | 3 | 3 | Tobacco |
| 5 | Coffee Roaster | 6 | 3 | 2 | Coffee |

**Violet (special‑power) buildings:**

| # | Building | Cost | VP | Cap | Ability |
|---|----------|-----:|---:|----:|---------|
| 6 | Small Market | 1 | 1 | 1 | Trader: **+1** doubloon when you sell a good. |
| 7 | Hacienda | 2 | 1 | 1 | Settler: you may also draw one **extra random plantation**. |
| 8 | Construction Hut | 2 | 1 | 1 | Settler: you may take a **Quarry** instead of a plantation. |
| 9 | Small Warehouse | 3 | 1 | 1 | Captain: **keep 1 type** of good through the Captain phase. |
| 10 | Hospice | 4 | 2 | 1 | Settler: each new tile arrives with a **free colonist**. |
| 11 | Office | 5 | 2 | 1 | Trader: sell a good **even if its kind is already** in the house. |
| 12 | Large Market | 5 | 2 | 1 | Trader: **+2** doubloons when you sell a good. |
| 13 | Large Warehouse | 6 | 2 | 1 | Captain: **keep 2 types** of good through the Captain phase. |
| 14 | Factory | 7 | 3 | 1 | Craftsman: bonus doubloons for **different** goods produced — 2/3/4/5 kinds → **1/2/3/5**. |
| 15 | University | 8 | 3 | 1 | Builder: each building you construct arrives with a **free colonist**. |
| 16 | Harbor | 8 | 3 | 1 | Captain: **+1 VP** each time you ship goods. |
| 17 | Wharf | 9 | 3 | 1 | Captain: your **own private ship** (one good type, any amount), once per Captain phase. |

**Large buildings** — each costs **10**, is worth **4 VP**, capacity 1, and gives
an **end‑game bonus** (only if staffed):

| # | Building | End‑game bonus (exact) |
|---|----------|------------------------|
| 18 | Guildhall | **+1 VP** per small production building (Small Indigo/Sugar), **+2 VP** per large production building (Indigo Plant, Sugar Mill, Tobacco Storage, Coffee Roaster). |
| 19 | Residence | VP by filled island spaces: **≤9 → 4, 10 → 5, 11 → 6, 12 → 7**. |
| 20 | Fortress | **+1 VP** per **3** colonists you own (total, floored). |
| 21 | Customs House | **+1 VP** per **4** shipping VP chips you earned (floored). |
| 22 | City Hall | **+1 VP** per violet (non‑production) building you own. |

### Money, colonists, and quarries

- **Doubloons** are money (you start with 2). You spend them on buildings and gain
  them from the Trader, Prospector, unpicked‑role bonuses, and the Factory.
- **Colonists** are workers. The Mayor hands them out from the **colonist ship**,
  which is refilled from the supply each round. A plantation or building does
  nothing until a colonist is placed on it.
- **Quarries** discount building purchases: when you build, each **staffed**
  quarry lowers the price by **1 doubloon**, up to the building's printed VP value
  (so a 4‑VP building can be discounted by at most 4).

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
