# Puerto Rico: Base Game Rules

This document restates the rules of the *Puerto Rico* base game so that the engine can be checked
against them rule by rule. It follows the Rio Grande Games *Puerto Rico Deluxe* rulebook (English
edition, 2019). The plain-text extraction of that rulebook is kept in
[`puerto_rico_deluxe_rules_extracted.txt`](puerto_rico_deluxe_rules_extracted.txt); **[src a–b]**
cites its line numbers.

**Scope.** The base game for three to five players and the game for two players. The two
expansions, *The New Buildings* and *The Nobles* [src 1176–1530], are not covered; the engine does
not implement them.

**Conventions.**
- Rule identifiers such as **R5.6.4** are stable. Tests and [`CONFORMANCE.md`](CONFORMANCE.md)
  refer to them.
- *Must* and *may* keep the force they have in the rulebook. Quoted text is the rulebook's wording.
- *Beige buildings* is the Deluxe rulebook's name for the violet buildings of other editions. The
  *windrose* is the area of a player board that holds the player's doubloons, goods, unplaced
  colonists, and VP chips.
- Building data and goods prices are printed on the game board and tiles, not in the text
  extraction (§6, R5.5.2).
- Questions that the rulebook leaves open are listed in §8. They are not rules.

## 1. Overview [src 10–34, 160–182]

- **R1.1** The game is played over rounds. In each round every player chooses a role. The chooser
  receives the role's privilege, and every player, starting with the chooser and continuing
  clockwise, may take the role's action.
- **R1.2** Players earn victory points (VP) by shipping goods and from their buildings. The player
  with the most VP at the end of the game wins.

## 2. Components [src 36–73]

| Component | Supply in the box |
|---|---|
| Player boards | 5, each with 12 island spaces, 12 city spaces, and a windrose |
| Governor placard | 1 |
| Role placards | 8: settler, mayor, builder, craftsman, trader, captain, and 2 prospectors |
| Buildings | 49: 20 production buildings, 24 small beige buildings (2 of each of 12 kinds), 5 large beige buildings |
| Island tiles | 58: 8 quarries and 50 plantations (8 coffee, 9 tobacco, 10 corn, 11 sugar, 12 indigo) |
| Colonist ship | 1 |
| Cargo ships | 5, with 4, 5, 6, 7, and 8 cargo spaces |
| Trading house | 1, with 4 spaces |
| Colonists | 100 |
| Goods | 50: 9 coffee, 9 tobacco, 10 corn, 11 sugar, 11 indigo |
| VP chips | 50 chips worth 126 VP (31 of value 1, 19 of value 5) |
| Doubloons | 54 coins worth 86 doubloons (46 of value 1, 8 of value 5) |

## 3. Setup

### 3.1 Three to five players [src 98–153]

- **R3.1.1** The players choose the first governor by any method.
- **R3.1.2** Each player receives a player board, the starting doubloons, and one plantation tile,
  which they place on an island space. The starting plantations come from the 50 plantation tiles
  and are dealt in clockwise order beginning with the governor, as in the table.
- **R3.1.3** The remaining plantation tiles are shuffled face down to form the plantation supply.
  As many tiles as one more than the number of players are turned face up.
- **R3.1.4** The quarries form a face-up stack, and all goods form the goods supply. The trading
  house starts empty. The colonist ship starts with as many colonists as there are players.

| | 3 players | 4 players | 5 players |
|---|---|---|---|
| Starting doubloons per player | 2 | 3 | 4 |
| Starting plantations, governor first | indigo, indigo, corn | indigo, indigo, corn, corn | indigo, indigo, indigo, corn, corn |
| Face-up plantation tiles | 4 | 5 | 6 |
| Quarries | 8 | 8 | 8 |
| VP chips | 75 VP | 100 VP | 126 VP |
| Colonists on the colonist ship | 3 | 4 | 5 |
| Colonist supply | 55 | 75 | 95 |
| Cargo ships (cargo spaces) | 4, 5, 6 | 5, 6, 7 | 6, 7, 8 |
| Role placards | 6 (no prospector) | 7 (one prospector) | 8 (both prospectors) |
| Buildings, goods | all | all | all |

### 3.2 Two players [src 1103–1153]

All rules of the base game apply, with these changes:

- **R3.2.1** Each player receives 3 doubloons. The governor starts with an indigo plantation and the
  other player with a corn plantation.
- **R3.2.2** Three tiles of each plantation kind and three quarries are removed from the game. Three
  plantation tiles are face up (one more than the number of players).
- **R3.2.3** The building supply holds one of each beige building and two of each production
  building.
- **R3.2.4** The VP chips are worth 65 VP. The colonist supply holds 40 colonists and the colonist
  ship 2.
- **R3.2.5** The cargo ships have 4 and 6 cargo spaces.
- **R3.2.6** Two goods of each kind are removed, leaving 7 coffee, 7 tobacco, 8 corn, 9 sugar, and
  9 indigo.
- **R3.2.7** Seven role placards are used: all except one prospector.

## 4. Rounds and roles [src 160–198, 628–639, 1154–1165]

- **R4.1** The governor chooses the first role of the round. With three to five players, once the
  action of a role is finished, the next player clockwise chooses one of the remaining placards,
  until every player has chosen one. With two players, the players alternate, beginning with the
  governor, until each has chosen three placards, six of the seven.
- **R4.2** A player must choose a role on their turn [src 192–194]. A chosen placard stays with the
  chooser until the end of the round, and no other player may choose it in that round
  [src 197–198].
- **R4.3** The chooser takes all doubloons lying on the placard, in addition to the privilege
  [src 188–189].
- **R4.4** The chooser takes the role's action first; then each other player takes it once, in
  clockwise order (the captain's action can take several turns, R5.6.1). Only the chooser
  receives the privilege. The prospector has no action [src 184–191, 235–240].
- **R4.5** A player may decline the action and the privilege of the role they chose; the other
  players still take their turns at the action [src 192–194]. The action is optional for every
  player, except the captain's [src 195–196].
- **R4.6** The round ends when all roles of the round (six with two players) have been chosen and
  their actions completed. The governor places one doubloon on each placard that was not chosen in
  the round, regardless of how many doubloons already lie on it: three placards with three to five
  players, one with two players. The chosen placards are returned, the governor placard passes to
  the left, and the new governor begins the next round [src 177–182, 628–639, 1158–1160].

## 5. Roles

### 5.1 Settler [src 199–260]

- **R5.1.1** The settler may take either one face-up plantation tile or, as the privilege, one
  quarry, and place it on an empty island space.
- **R5.1.2** Then each other player, in clockwise order, may take one face-up plantation tile and
  place it on an empty island space. These players may not take a quarry (exception: construction
  hut, R6.3.3).
- **R5.1.3** Taking a tile is voluntary [src 258].
- **R5.1.4** A player whose 12 island spaces are filled may not take further tiles [src 217–218].
- **R5.1.5** When no quarries remain, neither the settler's privilege nor the construction hut can
  be used [src 259–260].
- **R5.1.6** The position of a tile on the island has no effect. Placed tiles are never removed
  [src 215–216].
- **R5.1.7** At the end of the phase the settler puts the untaken face-up tiles on the discard pile
  and turns up new tiles from the face-down supply, one more than the number of players. If the
  face-down supply runs short, the settler first turns up the remaining tiles, then shuffles the
  discard pile face down into a new supply and continues. If the row still cannot be filled, it
  stays incomplete [src 205–214].

### 5.2 Mayor [src 262–302, 329–352]

- **R5.2.1** Each plantation and quarry has one circle, and each building one to three (§6); each
  circle holds one colonist. A tile or building with at least one colonist is occupied. Only
  occupied tiles and buildings function [src 263–266].
- **R5.2.2** Privilege: the mayor may first take one colonist from the colonist supply, not from the
  colonist ship. When the colonist supply is empty, the privilege cannot be used
  [src 274–275, 298–299].
- **R5.2.3** Then the colonists on the colonist ship are taken one at a time, starting with the
  mayor and continuing clockwise, until the ship is empty [src 275–278]. Example: with four
  players and six colonists on the ship, the mayor takes three (two from the ship and the
  privilege colonist), the second player two, and the third and fourth players one each
  [src 329–346].
- **R5.2.4** Each player places the new colonists, together with all colonists acquired earlier, on
  empty circles of their tiles and buildings; colonists placed earlier may be moved. Colonists
  that cannot be placed stay on the windrose until a later mayor phase [src 279–284].
- **R5.2.5** "Players may not choose to place colonists on their windrose if they have empty
  circles available on their player board. All empty circles must be filled, if possible."
  Colonists are placed or moved only in the mayor phase [src 300–302].
- **R5.2.6** Players normally place at the same time. When their decisions depend on one another,
  the mayor places first and the others follow in clockwise order [src 292–295].
- **R5.2.7** As the last step, the mayor refills the colonist ship from the colonist supply: one
  colonist for each empty circle on the buildings of all players (circles on plantations and
  quarries do not count), but at least as many colonists as there are players [src 285–290].
  "When the colonist supply runs out, the mayor … does not refill the colonist ship"
  [src 298–299]. If the supply cannot provide the required number, the game ends at the end of the
  round (R7.1).

### 5.3 Builder [src 303–323, 354–372, 420–423]

- **R5.3.1** The builder may build one building and pays one doubloon less than its cost
  (privilege). Then each other player, in clockwise order, may build one building at its normal
  cost [src 304–311].
- **R5.3.2** Each occupied quarry of the building player reduces the cost by one doubloon, up to the
  limit of the building's column on the game board: one in the first column, two in the second,
  three in the third, and four in the fourth [src 316–322]. The builder's privilege applies in
  addition to the quarry reduction, and the cost cannot fall below zero [src 369–370]. Example: a
  player other than the builder with three occupied quarries pays 1 doubloon for a construction
  hut, 3 for an office, 5 for a harbor, and 7 for a city hall [src 370–372].
- **R5.3.3** The player pays the bank, takes the building from the supply, and places it on an
  empty city space. A large building needs two adjacent empty spaces [src 305–309], and buildings
  may be moved within the city to make room (R6.1.4).
- **R5.3.4** No player may build more than one building per round [src 314]. A player may own at
  most one building of each kind (R6.1.1), and only buildings left in the supply can be built.
- **R5.3.5** A city has 12 spaces. A player without the empty spaces a building needs cannot build
  it [src 422–423].
- **R5.3.6** If the builder does not build, they receive nothing for the privilege [src 421].

### 5.4 Craftsman [src 425–438, 727–750]

- **R5.4.1** Starting with the craftsman and continuing clockwise, each player takes from the goods
  supply the goods that they produce and places them on their windrose [src 426–430].
- **R5.4.2** An occupied corn plantation produces one corn. For indigo, sugar, tobacco, and coffee,
  a player produces as many goods of the kind as the smaller of two numbers: their occupied
  plantations of the kind and the colonists in their production buildings of the kind
  [src 737–750].
- **R5.4.3** If the supply of a kind runs short, the player takes what is left, possibly nothing,
  and may not take another kind instead [src 436–437, 965–969].
- **R5.4.4** Privilege: as the last step of the phase, the craftsman may take one more good of a
  kind they produced in this phase. A craftsman who produced nothing receives no extra good
  [src 432–433, 438].

### 5.5 Trader [src 440–503]

- **R5.5.1** Starting with the trader and continuing clockwise, each player may sell one good to the
  trading house and receives its price from the bank. The trader receives one extra doubloon with
  their sale (privilege) [src 441–447].
- **R5.5.2** The prices printed on the trading house are corn 0, indigo 1, sugar 2, tobacco 3, and
  coffee 4 doubloons (the text gives the range 0–4 [src 444]). Corn may be sold for nothing
  [src 502–503].
- **R5.5.3** The trading house holds four goods. When it is full, no more goods can be sold in the
  phase [src 451–452].
- **R5.5.4** The trading house buys only kinds that it does not already hold (exception: office,
  R6.3.6) [src 453–455].
- **R5.5.5** The phase ends when every player has had one turn or the trading house is full
  [src 447–449].
- **R5.5.6** As the last step, the trader empties the trading house if it holds four goods, and the
  goods return to the supply. Otherwise the goods stay for the next trader phase [src 456–460].
- **R5.5.7** If the trader does not sell, they receive no privilege doubloon [src 501].

### 5.6 Captain [src 510–618]

- **R5.6.1** The captain loads first, then the other players in clockwise order. The phase continues
  around the table as long as at least one player has goods they can load, so a player may have
  several loading turns [src 511–516].
- **R5.6.2** On their turn, a player must load if they can. They load goods of a single kind onto a
  single ship and choose freely among the kinds they can load; they need not choose the kind that
  loads the most goods [src 516–518, 525–526, 532–534].
- **R5.6.3** Each cargo ship carries goods of only one kind. A kind may not be loaded onto a ship
  while another cargo ship carries that kind, and nothing may be loaded onto a full ship
  [src 521–524].
- **R5.6.4** "When players load goods of a kind, they must load as many of that kind as they can.
  … If players have a kind of good that can be loaded on several empty ships, they must choose the
  ship where they can load the most goods, leaving none behind, if possible." [src 527–531]
- **R5.6.5** Each good loaded earns 1 VP, whatever its kind. With their first load in the phase,
  the captain takes 1 extra VP as the privilege. The captain gets no extra VP for later loads in
  the same phase, and none if they load nothing [src 536–541, 561–562].
- **R5.6.6** VP chips are kept face down [src 542–544]. "When the VP chips are exhausted, players
  earning VP thereafter should track them with pencil and paper." [src 664–665] Running out of
  chips ends the game at the end of the round (R7.1); it does not limit further VP.
- **R5.6.7** When no more goods can be loaded, each player must store their remaining goods: one
  good token on the windrose, plus all goods of the kinds that their occupied warehouses hold
  (R6.3.4, R6.3.8). A player whose goods all fit keeps them all. Otherwise the player fills the
  windrose and every warehouse with goods of their choice, and the rest return to the supply:
  "When players cannot store all their extra goods, they may choose which goods to store and which
  to return to the goods supply." [src 546–551, 559–560] Storage applies only at this
  point; goods acquired in other phases are kept without limit until the next captain phase
  [src 563–564].
- **R5.6.8** As the last step, the captain empties every full cargo ship, and the goods return to
  the supply. Partly loaded and empty ships keep their contents until the next captain phase
  [src 552–556].

Example with four players [src 583–618]. Anna, the captain, has 2 corn and 6 sugar; Bob has 2
sugar and 3 tobacco; Chris has 2 corn and 1 tobacco; David has 1 corn and 5 indigo. The 5- and
7-space ships are empty, and the 6-space ship carries 3 corn. Anna loads her 6 sugar onto the
7-space ship, because the 5-space ship would leave one behind, and scores 7 VP (6 + 1 privilege).
Bob loads 1 sugar, which fills the 7-space ship; he could have loaded his 3 tobacco onto the
5-space ship instead. Chris loads 1 tobacco onto the 5-space ship. David must load his 1 corn onto
the 6-space ship, since his indigo fits nowhere. Anna then must load her 2 corn onto the 6-space
ship, and Bob his 3 tobacco onto the 5-space ship. Nobody can load any more. The players store
their remaining goods, and Anna empties the full 6- and 7-space ships; the 5-space ship keeps its 4
tobacco. (The rulebook says that Anna and Bob have no goods left [src 611–612], but Bob still holds
the sugar that did not fit on the 7-space ship.)

### 5.7 Prospector [src 625–627]

- **R5.7.1** The prospector has no action. Its chooser receives 1 doubloon from the bank as the
  privilege. Prospectors are used only with two, four, and five players (§3.1 table, R3.2.7).

## 6. Buildings [src 701–1097]

### 6.1 General rules

- **R6.1.1** Each player may build each building only once [src 706].
- **R6.1.2** A building with at least one colonist is occupied. Only occupied buildings can be used
  or score bonuses. The VP printed on a building count at game end whether or not it is occupied
  [src 671, 707–708, 715–717].
- **R6.1.3** The cost of a building matters only when it is built [src 718–719].
- **R6.1.4** The position of a building in the city has no effect. A large building needs two
  adjacent empty city spaces, and buildings may be moved within the city to make room for it. No
  building may be removed [src 709–714].
- **R6.1.5** A large building occupies two city spaces but counts as one building [src 1038–1039].
- **R6.1.6** Players need not use the special function of an occupied beige building
  [src 873–874].

The cost, the VP (equal to the column number for every base-game building), the number of
circles, and the number of copies come from the game board and the building tiles.

### 6.2 Production buildings [src 727–750]

| Building | Cost | VP (column) | Circles | Copies, 3–5 players | Copies, 2 players | Good |
|---|---|---|---|---|---|---|
| Small indigo plant | 1 | 1 | 1 | 4 | 2 | indigo |
| Small sugar mill | 2 | 1 | 1 | 4 | 2 | sugar |
| Large indigo plant | 3 | 2 | 3 | 3 | 2 | indigo |
| Large sugar mill | 4 | 2 | 3 | 3 | 2 | sugar |
| Tobacco storage | 5 | 3 | 3 | 3 | 2 | tobacco |
| Coffee roaster | 6 | 3 | 2 | 3 | 2 | coffee |

- **R6.2.1** The circles of a production building are the most goods it can produce, and only
  circles holding colonists count. The player also needs an occupied plantation of the kind for
  each good (R5.4.2) [src 746–750].
- **R6.2.2** Corn needs no production building [src 737–739].

Example from the rulebook's illustration: three corn plantations of which two are occupied produce
2 corn; a tobacco storage with a colonist in only one of its circles limits tobacco to 1; four
sugar plantations of which three are occupied, with enough colonists in sugar mills, produce 3
sugar [src 759–762, 839–849].

### 6.3 Small beige buildings [src 866–1029]

Each small beige building has one circle. There are two copies of each, one in the two-player
game.

| Building | Cost | VP (column) | Function when occupied |
|---|---|---|---|
| Small market | 1 | 1 | one extra doubloon when selling |
| Hacienda | 2 | 1 | an extra face-down plantation in the settler phase |
| Construction hut | 2 | 1 | a quarry instead of a plantation in the settler phase |
| Small warehouse | 3 | 1 | store all goods of one kind |
| Hospice | 4 | 2 | a colonist on a newly placed island tile |
| Office | 5 | 2 | sell a kind already in the trading house |
| Large market | 5 | 2 | two extra doubloons when selling |
| Large warehouse | 6 | 2 | store all goods of two kinds |
| Factory | 7 | 3 | doubloons for producing several kinds |
| University | 8 | 3 | a colonist on a newly built building |
| Harbor | 8 | 3 | one extra VP per loading turn |
| Wharf | 9 | 3 | an unlimited private ship once per captain phase |

- **R6.3.1 Small market.** When the owner sells a good in the trader phase, they receive one extra
  doubloon [src 875–878].
- **R6.3.2 Hacienda.** On their settler turn, before taking a face-up tile, the owner may take the
  top tile of the face-down plantation supply. They must place it immediately on an empty island
  space and then take their normal turn. The extra tile is always a plantation: an owner of an
  occupied construction hut may not take a quarry instead of it, and a settler who owns a hacienda
  still takes at most one quarry. The hospice does not apply to the extra tile
  [src 879–887, 907–908].
- **R6.3.3 Construction hut.** In the settler phase the owner may take a quarry instead of a
  face-up plantation tile. A settler who owns one still takes only one quarry [src 888–891].
- **R6.3.4 Small warehouse.** At the end of the captain phase the owner may store, in addition to
  the one good on their windrose, all goods of one kind of their choice. The warehouse does not
  exempt goods from mandatory loading [src 892–902].
- **R6.3.5 Hospice.** In the settler phase, when the owner places a plantation or quarry tile, they
  may take a colonist from the colonist supply and place it on that tile. "If there are no more
  colonists in the colonist supply, the player may take one from the colonist ship. If there are
  also none there, the player takes none." The hospice does not apply to the hacienda's extra tile
  [src 903–910].
- **R6.3.6 Office.** When the owner sells, their good need not differ from the goods already in
  the trading house. A full trading house still accepts nothing [src 911–917].
- **R6.3.7 Large market.** When the owner sells a good, they receive two extra doubloons; an owner
  of both markets receives three [src 948–952].
- **R6.3.8 Large warehouse.** The owner may store all goods of two kinds of their choice in
  addition to the windrose good; an owner of both warehouses may store three kinds
  [src 953–958].
- **R6.3.9 Factory.** In the craftsman phase, an owner who produces goods of two, three, four, or
  five kinds receives 1, 2, 3, or 5 doubloons. Only the kinds actually taken from the supply
  count, not the number of goods. Example: an owner able to produce corn, sugar, and tobacco who
  finds no corn left in the supply produces two kinds and receives 1 doubloon [src 959–969].
- **R6.3.10 University.** In the builder phase, when the owner builds a building, they may take a
  colonist from the colonist supply and place it on the new building; they get only one colonist,
  even for a building with several circles. "If there are no more colonists in the colonist
  supply, they may take one from the colonist ship. If there are also none there, they take none."
  [src 970–976]
- **R6.3.11 Harbor.** Each time the owner loads goods in the captain phase, they earn one extra VP,
  including when they load with the wharf [src 977–985].
- **R6.3.12 Wharf.** Instead of loading onto a cargo ship, the owner may put all their goods of one
  kind into the goods supply and score VP as if they had loaded them onto a cargo ship. The
  imaginary ship takes any kind, including a kind that a cargo ship carries, and holds up to 11
  goods, which no player can exceed because the supply of each kind is at most 11. The owner may use the wharf once per captain phase and chooses when to use it, if at all;
  the rulebook's harbor example uses it on a turn when no cargo ship can take the owner's goods.
  "Players must load goods on a cargo ship whenever they can on their turn during the captain
  phase, except when they choose to use their wharf." [src 986–1001, 980–985]

### 6.4 Large buildings [src 1036–1097]

Each large building costs 10 doubloons, is worth 4 VP (fourth column), has one circle, occupies
two city spaces, and exists once, also in the two-player game. Its bonus counts only if it is
occupied at the end of the game; its 4 VP count either way [src 671, 686–688].

- **R6.4.1 Guild hall.** +1 VP for each small production building (small indigo plant, small sugar
  mill) and +2 VP for each large production building (large indigo plant, large sugar mill,
  tobacco storage, coffee roaster) in the owner's city, occupied or not. Example: a small and a
  large sugar mill, a small indigo plant, and a coffee roaster give +6 VP [src 1042–1050].
- **R6.4.2 Residence.** +4 VP for up to nine filled island spaces (plantations and quarries), +5 for
  ten, +6 for eleven, and +7 for twelve [src 1051–1057].
- **R6.4.3 Fortress.** +1 VP for every three colonists on the owner's player board, including those
  on the windrose. Example: 20 colonists give +6 VP [src 1058–1062].
- **R6.4.4 Customs house.** +1 VP for every four VP the owner acquired as VP chips during the game,
  including VP recorded on paper after the chips ran out; building VP do not count. Example: 23 VP
  in chips give +5 VP [src 1063–1070].
- **R6.4.5 City hall.** +1 VP for each beige building in the owner's city, occupied or not,
  including the city hall itself. Example: a hacienda, harbor, office, construction hut, large
  warehouse, and residence together with the city hall give +7 VP [src 1071–1075].

## 7. Game end and scoring [src 641–694, 1166–1169]

- **R7.1** The game ends at the end of the round in which at least one of the following happens:
  (a) at the end of a mayor phase, the colonist supply cannot fill the colonist ship as R5.2.7
  requires; (b) in a builder phase, a player builds on their twelfth city space; (c) in a captain
  phase, the last VP chip is taken.
- **R7.2** A player's score is the sum of their VP chips (including VP recorded on paper), the VP
  printed on their buildings, occupied or not, and the bonuses of their occupied large buildings
  [src 667–671].
- **R7.3** The player with the most VP wins. If several players tie for the most VP, the one among
  them with the most doubloons and goods together, one good counting as one doubloon, wins
  [src 689–691].

## 8. Questions the rulebook leaves open

These are not rules. [`CONFORMANCE.md`](CONFORMANCE.md) records how the environment settles each.

- **O1** A tie that remains after the tie-break of R7.3.
- **O2** Running out of doubloons. The bank holds 86 doubloons, and the rulebook does not say what
  happens when it is empty.
- **O3** A short colonist supply at the end of a mayor phase. R5.2.7 and R7.1(a) do not say whether
  the colonists that remain still go onto the ship.
- **O4** Placing colonists at the same time (R5.2.6) when the players act one at a time.
- **O5** The end of loading when no player can load onto a cargo ship but a wharf owner could still
  use the wharf and declines (R5.6.1, R6.3.12).
- **O6** Whether the captain's privilege applies when the captain's first load is made with the
  wharf. The wharf scores VP "as though they had loaded them on a cargo ship" (R6.3.12).
- **O7** Hidden VP chips (R5.6.6). Every VP-earning event happens in the open, so a player who
  remembers them knows every total.
- **O8** R6.1.6 lets a player leave the function of a beige building unused, while R5.6.7 requires
  players to store the goods that fit. The two clash for the warehouses.
- **O9** R4.5 makes the action of a role optional, while the craftsman's production (R5.4.1) and
  the distribution of colonists (R5.2.3) are described as steps that simply happen.
