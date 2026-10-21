"""Independent reference model ("oracle") of the Puerto Rico base game.

The model implements the rules of ``rule/RULES.md`` (rule identifiers such as ``R5.6.4``) as the
sequence of single-player decisions specified in ``rule/CONFORMANCE.md`` §2 (decision-model
identifiers such as ``E10``; settled open questions such as ``O3``), with the action encoding of
§3 and the interface and canonical state of §4.  It was written from those two documents (and,
for wording only, the rulebook extraction that they cite) without reference to the engine it is
meant to check.

How the model is organised
--------------------------
The game is a state machine that always rests at a decision point.  Between calls, the state
names the phase (``phase``), the player who is asked (``current_player``), an optional benefit
decision (``benefit``, E9) and the bookkeeping of the phase in progress.

* ``legal_actions`` derives the legal action indices from that state (CONFORMANCE §3).
* ``step`` hands the action to the handler of the current decision.  Each handler applies the
  rule and moves the state on to the next decision.  On the way it performs every step that
  involves no choice (E3: the prospector's privilege, production, colonist distribution, the
  end of each phase, the end of the round and of the game; E6: the mayor's automatic filling).
* ``_advance`` then takes every forced action, i.e. every decision with exactly one legal action
  that is neither a role selection nor a benefit decision (E2).

All state is plain Python data, so ``copy.deepcopy`` clones a game.  Only the standard library
is used.
"""

from __future__ import annotations

import operator
import random
from collections import Counter
from typing import NamedTuple, Optional

# =================================================================================================
# Names and index orders (CONFORMANCE §3 "Index orders", §4.2 "Names")
# =================================================================================================

#: Role placards in the order of the role index r (actions 0-7).
ROLES = ("SETTLER", "MAYOR", "BUILDER", "CRAFTSMAN", "TRADER", "CAPTAIN",
         "PROSPECTOR_1", "PROSPECTOR_2")

#: Goods in the order of the good index g; also the plantation kinds k.
GOODS = ("COFFEE", "TOBACCO", "CORN", "SUGAR", "INDIGO")

#: Island tile kinds in the order of the tile index t.
ISLAND_KINDS = GOODS + ("QUARRY",)

#: Buildings in the order of the building index b.  INDIGO_PLANT and SUGAR_MILL are the large
#: indigo plant and the large sugar mill.
BUILDINGS = (
    "SMALL_INDIGO_PLANT", "SMALL_SUGAR_MILL", "INDIGO_PLANT", "SUGAR_MILL", "TOBACCO_STORAGE",
    "COFFEE_ROASTER", "SMALL_MARKET", "HACIENDA", "CONSTRUCTION_HUT", "SMALL_WAREHOUSE", "HOSPICE",
    "OFFICE", "LARGE_MARKET", "LARGE_WAREHOUSE", "FACTORY", "UNIVERSITY", "HARBOR", "WHARF",
    "GUILDHALL", "RESIDENCE", "FORTRESS", "CUSTOMS_HOUSE", "CITY_HALL",
)

#: The phase name (canonical "phase") of each role that has an action.
ROLE_PHASE = {"SETTLER": "SETTLER", "MAYOR": "MAYOR", "BUILDER": "BUILDER",
              "CRAFTSMAN": "CRAFTSMAN", "TRADER": "TRADER", "CAPTAIN": "CAPTAIN"}

# =================================================================================================
# Action indices (CONFORMANCE §3).  Indices not listed are never legal.
# =================================================================================================

NUM_ACTIONS = 200
ROLE_ACTION = 0              # 0-7      choose role r
SETTLE_ACTION = 8            # 8-12     settler: take a face-up plantation of kind k
QUARRY_ACTION = 13           # 13       settler: take a quarry
PASS = 15                    # 15       pass
BUILD_ACTION = 16            # 16-38    build building b
SELL_ACTION = 39             # 39-43    sell good g
LOAD_ACTION = 44             # 44-58    load good g on cargo ship s: 44 + 5 s + g
WHARF_ACTION = 59            # 59-63    wharf: good g
WINDROSE_ACTION = 64         # 64-68    storage: windrose for good g
CRAFT_ACTION = 93            # 93-97    craftsman's privilege: good g
HACIENDA_ACTION = 105        # 105      hacienda draw
WAREHOUSE_ACTION = 106       # 106-110  storage: warehouse for good g
PLACE_TILE_ACTION = 120      # 120-125  mayor: colonist onto an unoccupied island tile of kind t
PLACE_BUILDING_ACTION = 140  # 140-162  mayor: colonist onto building b
ACCEPT = 170                 # 170      accept the offered benefit (E9)
DECLINE = 171                # 171      decline the offered benefit (E9)

# =================================================================================================
# Building data (RULES.md §6).  The VP of every base-game building equals its column on the game
# board (§6.1); the column is also the limit of the quarry reduction (R5.3.2).
# =================================================================================================


class BuildingData(NamedTuple):
    cost: int                 # doubloons (R6.1.3)
    vp: int                   # printed VP = column (§6.1)
    circles: int              # colonist circles (R5.2.1)
    copies: int               # copies with three to five players
    copies_2p: int            # copies with two players (R3.2.3)
    size: int                 # city spaces occupied (R6.1.5)
    category: str             # "production", "small" (small beige) or "large" (large beige)
    good: Optional[str]       # good produced, for production buildings


BUILDING_DATA = {
    # Production buildings (§6.2 table)
    "SMALL_INDIGO_PLANT": BuildingData(1, 1, 1, 4, 2, 1, "production", "INDIGO"),
    "SMALL_SUGAR_MILL":   BuildingData(2, 1, 1, 4, 2, 1, "production", "SUGAR"),
    "INDIGO_PLANT":       BuildingData(3, 2, 3, 3, 2, 1, "production", "INDIGO"),
    "SUGAR_MILL":         BuildingData(4, 2, 3, 3, 2, 1, "production", "SUGAR"),
    "TOBACCO_STORAGE":    BuildingData(5, 3, 3, 3, 2, 1, "production", "TOBACCO"),
    "COFFEE_ROASTER":     BuildingData(6, 3, 2, 3, 2, 1, "production", "COFFEE"),
    # Small beige buildings (§6.3 table): one circle; two copies, one with two players
    "SMALL_MARKET":       BuildingData(1, 1, 1, 2, 1, 1, "small", None),
    "HACIENDA":           BuildingData(2, 1, 1, 2, 1, 1, "small", None),
    "CONSTRUCTION_HUT":   BuildingData(2, 1, 1, 2, 1, 1, "small", None),
    "SMALL_WAREHOUSE":    BuildingData(3, 1, 1, 2, 1, 1, "small", None),
    "HOSPICE":            BuildingData(4, 2, 1, 2, 1, 1, "small", None),
    "OFFICE":             BuildingData(5, 2, 1, 2, 1, 1, "small", None),
    "LARGE_MARKET":       BuildingData(5, 2, 1, 2, 1, 1, "small", None),
    "LARGE_WAREHOUSE":    BuildingData(6, 2, 1, 2, 1, 1, "small", None),
    "FACTORY":            BuildingData(7, 3, 1, 2, 1, 1, "small", None),
    "UNIVERSITY":         BuildingData(8, 3, 1, 2, 1, 1, "small", None),
    "HARBOR":             BuildingData(8, 3, 1, 2, 1, 1, "small", None),
    "WHARF":              BuildingData(9, 3, 1, 2, 1, 1, "small", None),
    # Large buildings (§6.4): cost 10, 4 VP, one circle, two city spaces, one copy
    "GUILDHALL":          BuildingData(10, 4, 1, 1, 1, 2, "large", None),
    "RESIDENCE":          BuildingData(10, 4, 1, 1, 1, 2, "large", None),
    "FORTRESS":           BuildingData(10, 4, 1, 1, 1, 2, "large", None),
    "CUSTOMS_HOUSE":      BuildingData(10, 4, 1, 1, 1, 2, "large", None),
    "CITY_HALL":          BuildingData(10, 4, 1, 1, 1, 2, "large", None),
}

#: Production buildings of each good (R5.4.2, R6.2.1).  Corn needs none (R6.2.2).
PRODUCTION_BUILDINGS = {
    g: tuple(b for b in BUILDINGS if BUILDING_DATA[b].good == g) for g in GOODS
}

#: Prices printed on the trading house (R5.5.2).
TRADE_PRICE = {"CORN": 0, "INDIGO": 1, "SUGAR": 2, "TOBACCO": 3, "COFFEE": 4}

#: Factory doubloons by the number of kinds produced (R6.3.9).
FACTORY_BONUS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5}

# =================================================================================================
# Setup (RULES.md §2-§3)
# =================================================================================================

#: Island tiles and goods in the box (§2).
BOX_PLANTATIONS = {"COFFEE": 8, "TOBACCO": 9, "CORN": 10, "SUGAR": 11, "INDIGO": 12}
BOX_QUARRIES = 8
BOX_GOODS = {"COFFEE": 9, "TOBACCO": 9, "CORN": 10, "SUGAR": 11, "INDIGO": 11}

ISLAND_SPACES = 12           # R5.1.4
CITY_SPACES = 12             # R5.3.5
TRADING_HOUSE_SPACES = 4     # R5.5.3


class Setup(NamedTuple):
    doubloons: int                     # starting doubloons per player
    starting_plantations: tuple        # dealt clockwise beginning with the governor
    vp_chips: int                      # VP in chips
    colonist_ship: int
    colonist_supply: int
    ship_capacities: tuple             # cargo ships in increasing order of capacity
    roles: tuple                       # role placards in the game
    tiles_removed: int                 # removed of each plantation kind and of the quarries
    goods_removed: int                 # removed of each good


SETUP = {
    # §3.1 table.  When one prospector is used it is taken to be PROSPECTOR_1.
    3: Setup(2, ("INDIGO", "INDIGO", "CORN"), 75, 3, 55, (4, 5, 6), ROLES[:6], 0, 0),
    4: Setup(3, ("INDIGO", "INDIGO", "CORN", "CORN"), 100, 4, 75, (5, 6, 7), ROLES[:7], 0, 0),
    5: Setup(4, ("INDIGO", "INDIGO", "INDIGO", "CORN", "CORN"), 126, 5, 95, (6, 7, 8), ROLES,
             0, 0),
    # §3.2: R3.2.1 doubloons and plantations, R3.2.2 three tiles of each kind and three quarries
    # removed, R3.2.4 VP and colonists, R3.2.5 ships, R3.2.6 two goods of each kind removed,
    # R3.2.7 seven placards.
    2: Setup(3, ("INDIGO", "CORN"), 65, 2, 40, (4, 6), ROLES[:7], 3, 2),
}


def plantations_in_game(num_players):
    """Plantation tiles of each kind in the game (§2, R3.2.2)."""
    removed = SETUP[num_players].tiles_removed
    return {k: c - removed for k, c in BOX_PLANTATIONS.items()}


def quarries_in_game(num_players):
    """Quarry tiles in the game (§3.1 table, R3.2.2)."""
    return BOX_QUARRIES - SETUP[num_players].tiles_removed


def goods_in_game(num_players):
    """Goods of each kind in the game (§2, R3.2.6)."""
    removed = SETUP[num_players].goods_removed
    return {g: c - removed for g, c in BOX_GOODS.items()}


def building_copies(num_players, building):
    """Copies of a building in the supply at setup (§6.2-§6.4, R3.2.3)."""
    data = BUILDING_DATA[building]
    return data.copies_2p if num_players == 2 else data.copies


def picks_per_round(num_players):
    """Placards chosen in a round: one per player, six with two players (R4.1)."""
    return 6 if num_players == 2 else num_players


# =================================================================================================
# Player board
# =================================================================================================


class Player:
    """One player board: island, city, and windrose (doubloons, goods, colonists, VP chips).

    Every island tile has one circle (R5.2.1 with CONFORMANCE §3 and §4.2, which treat a tile as
    either occupied or unoccupied), so ``island`` stores, per tile kind, [tiles, occupied tiles].
    The position of a tile or building has no effect (R5.1.6, R6.1.4), so counts suffice.
    """

    def __init__(self):
        self.doubloons = 0
        self.vp = 0                                  # VP chips, including VP on paper (R5.6.6)
        self.goods = {g: 0 for g in GOODS}
        self.island = {}                             # tile kind -> [tiles, occupied tiles]
        self.city = {}                               # owned building -> colonists on it
        self.unplaced = 0                            # colonists on the windrose

    # Island ---------------------------------------------------------------------------------

    def island_tiles(self):
        return sum(tiles for tiles, _ in self.island.values())

    def has_island_space(self):
        return self.island_tiles() < ISLAND_SPACES                       # R5.1.4

    def add_tile(self, kind):
        """Place a tile, unoccupied, on an empty island space (R5.1.1)."""
        self.island.setdefault(kind, [0, 0])[0] += 1

    def occupied_tiles(self, kind):
        return self.island.get(kind, [0, 0])[1]

    # City -----------------------------------------------------------------------------------

    def city_spaces(self):
        """City spaces in use; a large building takes two (R6.1.5)."""
        return sum(BUILDING_DATA[b].size for b in self.city)

    def occupied(self, building):
        """Only an owned building with at least one colonist functions (R5.2.1, R6.1.2)."""
        return self.city.get(building, 0) > 0

    def colonists_in(self, buildings):
        return sum(self.city.get(b, 0) for b in buildings)

    # Colonists ------------------------------------------------------------------------------

    def colonists_on_board(self):
        """Colonists on island tiles and buildings (not on the windrose)."""
        return sum(occ for _, occ in self.island.values()) + sum(self.city.values())

    def empty_island_circles(self):
        return sum(tiles - occ for tiles, occ in self.island.values())

    def empty_building_circles(self):
        return sum(BUILDING_DATA[b].circles - c for b, c in self.city.items())

    def empty_circles(self):
        return self.empty_island_circles() + self.empty_building_circles()

    def return_colonists_to_windrose(self):
        """E6: at the start of a placement turn all of the player's colonists go back to the
        windrose, which realises the free rearrangement of R5.2.4."""
        self.unplaced += self.colonists_on_board()
        for counts in self.island.values():
            counts[1] = 0
        for b in self.city:
            self.city[b] = 0

    def fill_all_circles(self):
        """R5.2.5: every empty circle is filled (the caller checks there are enough colonists)."""
        self.unplaced -= self.empty_circles()
        for counts in self.island.values():
            counts[1] = counts[0]
        for b in self.city:
            self.city[b] = BUILDING_DATA[b].circles

    # Goods ----------------------------------------------------------------------------------

    def production_capacity(self):
        """Goods of each kind the player can produce (R5.4.2, R6.2.1, R6.2.2).

        Corn: one per occupied corn plantation.  Other kinds: the smaller of the occupied
        plantations of the kind and the colonists in the production buildings of the kind.
        """
        capacity = {}
        for g in GOODS:
            if g == "CORN":
                capacity[g] = self.occupied_tiles("CORN")                # R6.2.2
            else:
                capacity[g] = min(self.occupied_tiles(g),
                                  self.colonists_in(PRODUCTION_BUILDINGS[g]))
        return capacity

    def goods_count(self):
        return sum(self.goods.values())

    def warehouse_slots(self):
        """E11: occupied warehouses provide one slot (small) and two slots (large)
        (R6.3.4, R6.3.8)."""
        return ((1 if self.occupied("SMALL_WAREHOUSE") else 0)
                + (2 if self.occupied("LARGE_WAREHOUSE") else 0))

    def canonical(self):
        return {
            "doubloons": self.doubloons,
            "vp": self.vp,
            "goods": dict(self.goods),
            "island": {k: [t, o] for k, (t, o) in self.island.items() if t > 0},
            "city": dict(self.city),
            "unplaced": self.unplaced,
        }


#: R6.4.1: the small and the large production buildings.
SMALL_PRODUCTION = ("SMALL_INDIGO_PLANT", "SMALL_SUGAR_MILL")
LARGE_PRODUCTION = ("INDIGO_PLANT", "SUGAR_MILL", "TOBACCO_STORAGE", "COFFEE_ROASTER")

#: R6.4.2: residence bonus by filled island spaces (up to nine: 4).
RESIDENCE_BONUS = {10: 5, 11: 6, 12: 7}


def victory_points(player):
    """A player's final score (R7.2): VP chips (including VP on paper), the printed VP of every
    building, occupied or not (R6.1.2), and the bonuses of the occupied large buildings (§6.4)."""
    total = player.vp + sum(BUILDING_DATA[b].vp for b in player.city)
    if player.occupied("GUILDHALL"):                                           # R6.4.1
        total += sum(1 for b in player.city if b in SMALL_PRODUCTION)
        total += sum(2 for b in player.city if b in LARGE_PRODUCTION)
    if player.occupied("RESIDENCE"):                                           # R6.4.2
        total += RESIDENCE_BONUS.get(player.island_tiles(), 4)
    if player.occupied("FORTRESS"):                                            # R6.4.3
        total += (player.colonists_on_board() + player.unplaced) // 3
    if player.occupied("CUSTOMS_HOUSE"):                                       # R6.4.4
        total += player.vp // 4
    if player.occupied("CITY_HALL"):                                           # R6.4.5: beige,
        total += sum(1 for b in player.city                                    # itself included
                     if BUILDING_DATA[b].category in ("small", "large"))
    return total


class Ship:
    """A cargo ship: capacity, the kind it carries (None when empty) and its load."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.good = None
        self.load = 0

    def is_full(self):
        return self.load == self.capacity


# =================================================================================================
# The game
# =================================================================================================


class Oracle:
    """Reference model of one game, with the interface of CONFORMANCE §4.1."""

    # ---------------------------------------------------------------------------------------------
    # Interface (CONFORMANCE §4.1)
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def new_game(cls, num_players, governor, plantation_stack, face_up):
        """Build the setup of RULES.md §3.  Only the random elements come from the caller: the
        governor, the face-down plantation supply (kind names, top tile last) and the face-up
        tiles.  Raises ValueError if the tiles are not exactly the plantations that the setup
        leaves."""
        return cls(num_players, governor, plantation_stack, face_up)

    def legal_actions(self):
        """The sorted legal action indices of the current decision; [] when the game is over."""
        if self.over:
            return []
        if self.benefit is not None:
            return [ACCEPT, DECLINE]                                           # E9
        legal = {
            "ROLE": self._role_legal,
            "SETTLER": self._settler_legal,
            "MAYOR": self._mayor_legal,
            "BUILDER": self._builder_legal,
            "CRAFTSMAN": self._craftsman_legal,
            "TRADER": self._trader_legal,
            "CAPTAIN": self._captain_legal,
            "STORE": self._store_legal,
        }[self.phase]()
        return sorted(legal)

    def step(self, action, shuffle=None):
        """Apply the current player's action, then every automatic step (E2, E3, E6) until the
        next decision or the end of the game.  Raises ValueError for an illegal action.

        ``shuffle(tiles)`` is called when the plantation discard pile must become a new
        face-down supply (R5.1.7); it receives the discard pile as a list of kind names (in
        sorted order) and must return the new supply, top tile last.  Without it,
        ``random.shuffle`` is applied to a copy.
        """
        try:
            action = operator.index(action)
        except TypeError:
            raise ValueError(f"not an action index: {action!r}") from None
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action} (legal: {self.legal_actions()})")
        self._shuffle_hook = shuffle
        try:
            self._apply(action)
            self._advance()
        finally:
            self._shuffle_hook = None

    def canonical(self):
        """The canonical state of CONFORMANCE §4.2, as plain Python values."""
        return {
            "num_players": self.n,
            "over": self.over,
            "phase": self.phase,
            "benefit": self.benefit,
            "current_player": self.current_player,
            "governor": self.governor,
            "rounds_completed": self.rounds_completed,
            "active_role": self.active_role,
            "role_player": self.role_player,
            "roles_available": sorted(self.roles_available),
            "role_doubloons": dict(self.role_doubloons),
            "vp_supply": self.vp_supply,
            "colonist_supply": self.colonist_supply,
            "colonist_ship": self.colonist_ship,
            "quarry_supply": self.quarry_supply,
            "goods_supply": dict(self.goods_supply),
            "building_supply": dict(self.building_supply),
            "plantation_stack": list(self.plantation_stack),
            "face_up": sorted(self.face_up),
            "discard": sorted(self.discard),
            "trading_house": sorted(self.trading_house),
            "ships": [[s.capacity, s.good, s.load] for s in self.ships],
            "players": [p.canonical() for p in self.players],
            "scores": None if self.scores is None else [list(s) for s in self.scores],
            "winners": None if self.winners is None else list(self.winners),
        }

    # ---------------------------------------------------------------------------------------------
    # Setup (RULES.md §3)
    # ---------------------------------------------------------------------------------------------

    def __init__(self, num_players, governor, plantation_stack, face_up):
        if num_players not in SETUP:
            raise ValueError(f"the game is for 2 to 5 players, not {num_players!r}")
        try:
            governor = operator.index(governor)
        except TypeError:
            raise ValueError(f"governor must be a player index, not {governor!r}") from None
        if not 0 <= governor < num_players:
            raise ValueError(f"governor {governor} is not a player of a {num_players}-player game")
        setup = SETUP[num_players]
        plantation_stack = list(plantation_stack)
        face_up = list(face_up)

        # R3.1.2 / R3.2.1: the starting plantations come out of the plantation tiles; R3.1.3 /
        # R3.2.2: the rest form the face-down supply and the face-up row of n + 1 tiles.
        leftover = Counter(plantations_in_game(num_players))
        leftover.subtract(setup.starting_plantations)
        for kind in plantation_stack + face_up:
            if kind not in GOODS:
                raise ValueError(f"{kind!r} is not a plantation kind")
        if Counter(plantation_stack) + Counter(face_up) != +leftover:
            raise ValueError("the plantation tiles are not the plantations that the setup leaves")
        if len(face_up) != num_players + 1:
            raise ValueError(f"{num_players + 1} plantation tiles must be face up, not "
                             f"{len(face_up)}")

        self.n = num_players
        self.governor = governor                        # R3.1.1
        self.rounds_completed = 0
        self.roles_in_game = setup.roles
        self.roles_available = list(setup.roles)        # in role-index order
        self.role_doubloons = {r: 0 for r in setup.roles}
        self.vp_supply = setup.vp_chips
        self.colonist_supply = setup.colonist_supply
        self.colonist_ship = setup.colonist_ship        # R3.1.4
        self.quarry_supply = quarries_in_game(num_players)
        self.goods_supply = goods_in_game(num_players)
        self.building_supply = {b: building_copies(num_players, b) for b in BUILDINGS}
        self.plantation_stack = plantation_stack        # face-down supply, top tile last
        self.face_up = face_up
        self.discard = []
        self.trading_house = []                         # R3.1.4: starts empty
        self.ships = [Ship(c) for c in setup.ship_capacities]
        self.players = [Player() for _ in range(num_players)]
        for player in self.players:
            player.doubloons = setup.doubloons
        for i, kind in enumerate(setup.starting_plantations):
            self.players[(governor + i) % num_players].add_tile(kind)

        # Decision state.  The game starts with the governor choosing a role (R4.1).
        self.phase = "ROLE"
        self.benefit = None
        self.current_player = governor
        self.active_role = None
        self.role_player = None
        self.picks = 0                     # placards chosen so far in this round
        self.turn = 0                      # position in the phase's turn order (0 = chooser)
        self.end_triggered = False         # a condition of R7.1 has occurred in this round
        self.over = False
        self.scores = None
        self.winners = None
        # Bookkeeping of particular phases
        self.hacienda_drawn = False        # settler: the hacienda draw was made in this turn
        self.new_tile = None               # settler: kind of the tile the hospice applies to
        self.new_building = None           # builder: the building the university applies to
        self.craftsman_kinds = ()          # craftsman: kinds the craftsman produced
        self.done = [False] * num_players         # captain: took their last turn (E10)
        self.passed = [False] * num_players       # captain: passed since the most recent load
        self.wharf_used = [False] * num_players   # captain: wharf used in this phase (R6.3.12)
        self.captain_loaded = False        # captain: the captain has made their first load
        self.pending_benefits = []         # captain: benefits still to offer after a load
        self.windrose_kind = None          # storage: kind assigned to the windrose (E11)
        self.warehouse_kinds = []          # storage: kinds assigned to warehouse slots (E11)
        self._shuffle_hook = None

    # ---------------------------------------------------------------------------------------------
    # Decision dispatch and automatic moves
    # ---------------------------------------------------------------------------------------------

    @property
    def _player(self):
        """The current player's board."""
        return self.players[self.current_player]

    def _turn_player(self):
        """The player whose turn it is in the phase: the chooser first, then clockwise (R4.4)."""
        return (self.role_player + self.turn) % self.n

    def _apply(self, action):
        if self.benefit is not None:
            self._resolve_benefit(action == ACCEPT)
            return
        {
            "ROLE": self._role_act,
            "SETTLER": self._settler_act,
            "MAYOR": self._mayor_act,
            "BUILDER": self._builder_act,
            "CRAFTSMAN": self._craftsman_act,
            "TRADER": self._trader_act,
            "CAPTAIN": self._captain_act,
            "STORE": self._store_act,
        }[self.phase](action)

    def _advance(self):
        """E2: when exactly one action is legal, take it without asking.  Role selection and
        benefit decisions are never taken automatically."""
        while not self.over and self.phase != "ROLE" and self.benefit is None:
            legal = self.legal_actions()
            if not legal:
                raise AssertionError(f"decision without legal actions in phase {self.phase}")
            if len(legal) > 1:
                return
            self._apply(legal[0])

    def _resolve_benefit(self, accept):
        """E9: accept or decline the offered benefit; declining has no effect."""
        benefit = self.benefit
        self.benefit = None
        if benefit == "MAYOR_PRIVILEGE":
            self._mayor_privilege(accept)
        elif benefit == "HOSPICE":
            self._hospice(accept)
        elif benefit == "UNIVERSITY":
            self._university(accept)
        else:                                          # CAPTAIN_PRIVILEGE or HARBOR
            self._captain_benefit(accept)

    def _colonist_available(self):
        """Hospice and university: a colonist in the supply or, failing that, on the ship."""
        return self.colonist_supply > 0 or self.colonist_ship > 0

    def _take_colonist(self):
        """R6.3.5, R6.3.10: one colonist from the supply if possible, otherwise from the ship."""
        if self.colonist_supply > 0:
            self.colonist_supply -= 1
        else:
            self.colonist_ship -= 1

    # ---------------------------------------------------------------------------------------------
    # Rounds and roles (RULES.md §4; E4)
    # ---------------------------------------------------------------------------------------------

    def _role_legal(self):
        return [ROLE_ACTION + ROLES.index(r) for r in self.roles_available]     # E4

    def _role_act(self, action):
        role = ROLES[action - ROLE_ACTION]
        chooser = self.current_player
        self.roles_available.remove(role)            # R4.2: the placard stays with the chooser
        self.players[chooser].doubloons += self.role_doubloons[role]           # R4.3
        self.role_doubloons[role] = 0
        self.picks += 1
        self.active_role = role
        self.role_player = chooser
        self.turn = 0
        if role == "SETTLER":
            self._settler_begin()
        elif role == "MAYOR":
            self._mayor_begin()
        elif role == "BUILDER":
            self._builder_begin()
        elif role == "CRAFTSMAN":
            self._craftsman_begin()
        elif role == "TRADER":
            self._trader_begin()
        elif role == "CAPTAIN":
            self._captain_begin()
        else:
            self._prospector()

    def _role_finished(self):
        """R4.1: once a role's action is finished, the next chooser chooses; after the last
        placard of the round, the round ends."""
        self.phase = "ROLE"
        self.benefit = None
        self.active_role = None
        self.role_player = None
        if self.picks < picks_per_round(self.n):
            # R4.1: clockwise from the governor; with two players the two alternate.
            self.current_player = (self.governor + self.picks) % self.n
        else:
            self._end_round()

    def _end_round(self):
        """R4.6 (automatic, E3)."""
        for role in self.roles_available:            # the placards not chosen in the round
            self.role_doubloons[role] += 1
        self.roles_available = list(self.roles_in_game)        # chosen placards are returned
        self.picks = 0
        self.governor = (self.governor + 1) % self.n            # passes to the left
        self.rounds_completed += 1
        if self.end_triggered:
            self._end_game()                                    # R7.1, E3
        else:
            self.phase = "ROLE"
            self.current_player = self.governor                 # the new governor begins

    def _end_game(self):
        """R7.1-R7.3, O1: the final state after the round-end steps of the last round."""
        self.over = True
        self.phase = "OVER"
        self.benefit = None
        self.current_player = None
        self.active_role = None
        self.role_player = None
        # R7.3: most VP; tie-break on doubloons + goods (one good = one doubloon).
        self.scores = [[victory_points(p), p.doubloons + p.goods_count()] for p in self.players]
        best = max(self.scores)
        self.winners = [i for i, s in enumerate(self.scores) if s == best]     # O1: all win

    # ---------------------------------------------------------------------------------------------
    # Prospector (R5.7.1)
    # ---------------------------------------------------------------------------------------------

    def _prospector(self):
        """No action; the chooser's privilege of 1 doubloon is applied automatically (E3, E9)."""
        self.players[self.role_player].doubloons += 1
        self._role_finished()

    # ---------------------------------------------------------------------------------------------
    # Settler (RULES.md §5.1; E5)
    # ---------------------------------------------------------------------------------------------

    def _settler_begin(self):
        self.phase = "SETTLER"
        self._settler_turn_begin()

    def _settler_turn_begin(self):
        self.current_player = self._turn_player()        # R5.1.1-R5.1.2: settler first
        self.hacienda_drawn = False
        self.new_tile = None

    def _settler_legal(self):
        player = self._player
        legal = [PASS]                                    # R5.1.3: taking a tile is voluntary
        if not player.has_island_space():                 # R5.1.4
            return legal
        for k, kind in enumerate(GOODS):                  # a face-up plantation, by kind
            if kind in self.face_up:
                legal.append(SETTLE_ACTION + k)
        # R5.1.1 privilege, R5.1.2 / R6.3.3 construction hut, R5.1.5 no quarry left
        if self.quarry_supply > 0 and (self.current_player == self.role_player
                                       or player.occupied("CONSTRUCTION_HUT")):
            legal.append(QUARRY_ACTION)
        # R6.3.2, E5: the hacienda draw comes before the main choice, once per turn
        if (not self.hacienda_drawn and player.occupied("HACIENDA")
                and self.plantation_stack):
            legal.append(HACIENDA_ACTION)
        return legal

    def _settler_act(self, action):
        player = self._player
        if action == HACIENDA_ACTION:
            # R6.3.2: the top face-down tile is placed at once; the normal turn follows.  The
            # hospice does not apply to it (R6.3.5).
            player.add_tile(self.plantation_stack.pop())
            self.hacienda_drawn = True
            return
        if action == PASS:
            self._settler_turn_end()
            return
        if action == QUARRY_ACTION:
            kind = "QUARRY"
            self.quarry_supply -= 1
        else:
            kind = GOODS[action - SETTLE_ACTION]
            self.face_up.remove(kind)
        player.add_tile(kind)
        # E9 / R6.3.5: the hospice decision follows a face-up plantation or a quarry.
        if player.occupied("HOSPICE") and self._colonist_available():
            self.new_tile = kind
            self.benefit = "HOSPICE"
        else:
            self._settler_turn_end()

    def _hospice(self, accept):
        if accept:
            self._take_colonist()
            self._player.island[self.new_tile][1] += 1   # the colonist goes onto the new tile
        self._settler_turn_end()

    def _settler_turn_end(self):
        self.new_tile = None
        self.turn += 1
        if self.turn < self.n:
            self._settler_turn_begin()
        else:
            self._settler_refill()
            self._role_finished()

    def _settler_refill(self):
        """R5.1.7 (automatic, E3): discard the untaken face-up tiles, then turn up n + 1 tiles;
        when the face-down supply runs short, turn up what is left, shuffle the discard pile
        into a new supply and continue; if the row still cannot be filled it stays incomplete."""
        self.discard.extend(self.face_up)
        self.face_up = []
        wanted = self.n + 1
        self._turn_up(wanted)
        if len(self.face_up) < wanted and self.discard:
            self.plantation_stack = self._shuffle_discard()
            self.discard = []
            self._turn_up(wanted)

    def _turn_up(self, wanted):
        while len(self.face_up) < wanted and self.plantation_stack:
            self.face_up.append(self.plantation_stack.pop())    # the top tile is last

    def _shuffle_discard(self):
        """CONFORMANCE §4.1: the shuffle hook gets the discard pile and returns the new supply."""
        tiles = sorted(self.discard)
        if self._shuffle_hook is None:
            new_stack = list(tiles)
            random.shuffle(new_stack)
            return new_stack
        new_stack = self._shuffle_hook(list(tiles))
        if new_stack is None or Counter(new_stack) != Counter(tiles):
            raise ValueError("shuffle must return a permutation of the discard pile")
        return list(new_stack)

    # ---------------------------------------------------------------------------------------------
    # Mayor (RULES.md §5.2; E6)
    # ---------------------------------------------------------------------------------------------

    def _mayor_begin(self):
        self.phase = "MAYOR"
        self.current_player = self.role_player
        if self.colonist_supply > 0:
            self.benefit = "MAYOR_PRIVILEGE"         # R5.2.2, E6, E9: decided first
        else:
            self._mayor_distribute()                 # R5.2.2: the privilege cannot be used

    def _mayor_privilege(self, accept):
        if accept:
            self.colonist_supply -= 1                # from the supply, not from the ship
            self.players[self.role_player].unplaced += 1
        self._mayor_distribute()

    def _mayor_distribute(self):
        """R5.2.3 (automatic, E3): the ship's colonists are taken one at a time, starting with
        the mayor and continuing clockwise, until the ship is empty.  They go onto the windrose;
        the boards of players whose placement turn has not started are not touched."""
        i = 0
        while self.colonist_ship > 0:
            self.players[(self.role_player + i) % self.n].unplaced += 1
            self.colonist_ship -= 1
            i += 1
        self.turn = 0
        self._mayor_next_turn()

    def _mayor_next_turn(self):
        """E6, O4: placement turns, the mayor first and then clockwise.  A turn that needs no
        decision is completed at once; after the last player's turn the ship is refilled."""
        while self.turn < self.n:
            self.current_player = self._turn_player()
            self._player.return_colonists_to_windrose()          # E6, R5.2.4
            if not self._mayor_turn_over():
                return                                           # the player must decide
            self.turn += 1
        self._mayor_refill()
        self._role_finished()

    def _mayor_turn_over(self):
        """E6 / R5.2.5: with at least as many colonists on the windrose as empty circles, every
        circle is filled without asking.  The turn ends when the player has no colonist on the
        windrose or no empty circle."""
        player = self._player
        if player.unplaced >= player.empty_circles():
            player.fill_all_circles()
        return player.unplaced == 0 or player.empty_circles() == 0

    def _mayor_legal(self):
        player = self._player
        legal = []
        if player.unplaced > 0:
            for t, kind in enumerate(ISLAND_KINDS):
                tiles, occupied = player.island.get(kind, [0, 0])
                if occupied < tiles:
                    legal.append(PLACE_TILE_ACTION + t)
            for b, building in enumerate(BUILDINGS):
                if building in player.city and \
                        player.city[building] < BUILDING_DATA[building].circles:
                    legal.append(PLACE_BUILDING_ACTION + b)
        return legal

    def _mayor_act(self, action):
        player = self._player
        if action >= PLACE_BUILDING_ACTION:
            player.city[BUILDINGS[action - PLACE_BUILDING_ACTION]] += 1
        else:
            player.island[ISLAND_KINDS[action - PLACE_TILE_ACTION]][1] += 1
        player.unplaced -= 1
        if self._mayor_turn_over():
            self.turn += 1
            self._mayor_next_turn()

    def _mayor_refill(self):
        """R5.2.7 (automatic, E3): one colonist per empty building circle of all players, at
        least one per player.  O3: a short supply puts what remains on the ship, and the game
        ends at the end of the round (R7.1(a))."""
        required = max(self.n, sum(p.empty_building_circles() for p in self.players))
        if self.colonist_supply >= required:
            self.colonist_supply -= required
            self.colonist_ship += required
        else:
            self.colonist_ship += self.colonist_supply
            self.colonist_supply = 0
            self.end_triggered = True

    # ---------------------------------------------------------------------------------------------
    # Builder (RULES.md §5.3; E7)
    # ---------------------------------------------------------------------------------------------

    def _builder_begin(self):
        self.phase = "BUILDER"
        self.current_player = self.role_player        # R5.3.1: the builder first

    def _build_cost(self, player_index, building):
        """R5.3.1-R5.3.2: the cost, less one for the builder (privilege), less one per occupied
        quarry up to the building's column; never below zero."""
        data = BUILDING_DATA[building]
        reduction = min(self.players[player_index].occupied_tiles("QUARRY"), data.vp)
        if player_index == self.role_player:
            reduction += 1
        return max(0, data.cost - reduction)

    def _can_build(self, player_index, building):
        player = self.players[player_index]
        return (self.building_supply[building] > 0                       # R5.3.4: in the supply
                and building not in player.city                          # R6.1.1: once each
                and CITY_SPACES - player.city_spaces() >= BUILDING_DATA[building].size  # R5.3.5
                and player.doubloons >= self._build_cost(player_index, building))

    def _builder_legal(self):
        legal = [PASS]                                    # R4.5: the action is optional
        for b, building in enumerate(BUILDINGS):
            if self._can_build(self.current_player, building):
                legal.append(BUILD_ACTION + b)
        return legal

    def _builder_act(self, action):
        if action == PASS:                     # R5.3.6: no building, nothing for the privilege
            self._builder_turn_end()
            return
        player = self._player
        building = BUILDINGS[action - BUILD_ACTION]
        university_was_occupied = player.occupied("UNIVERSITY")
        player.doubloons -= self._build_cost(self.current_player, building)   # R5.3.3
        self.building_supply[building] -= 1
        player.city[building] = 0
        if player.city_spaces() == CITY_SPACES:
            self.end_triggered = True                                        # R7.1(b)
        # E9 / R6.3.10: the university decision follows the build.
        if university_was_occupied and self._colonist_available():
            self.new_building = building
            self.benefit = "UNIVERSITY"
        else:
            self._builder_turn_end()

    def _university(self, accept):
        if accept:
            self._take_colonist()
            self._player.city[self.new_building] += 1    # one colonist onto the new building
        self._builder_turn_end()

    def _builder_turn_end(self):
        self.new_building = None
        self.turn += 1                       # R5.3.1, R5.3.4: one building per player
        if self.turn < self.n:
            self.current_player = self._turn_player()
        else:
            self._role_finished()

    # ---------------------------------------------------------------------------------------------
    # Craftsman (RULES.md §5.4; E8)
    # ---------------------------------------------------------------------------------------------

    def _craftsman_begin(self):
        """R5.4.1 (automatic, E3, E8): production, the craftsman first and then clockwise."""
        self.phase = "CRAFTSMAN"
        self.current_player = self.role_player
        for i in range(self.n):
            kinds = self._produce((self.role_player + i) % self.n)
            if i == 0:
                self.craftsman_kinds = kinds
        if not self._craftsman_privilege_kinds():
            self._role_finished()           # E8: no such kind, the phase ends without decision

    def _produce(self, player_index):
        """R5.4.2-R5.4.3: take the goods produced, as many as the supply holds; R6.3.9: the
        factory pays for the kinds actually taken.  Returns the kinds taken."""
        player = self.players[player_index]
        kinds = []
        for g, capacity in player.production_capacity().items():
            amount = min(capacity, self.goods_supply[g])      # R5.4.3: what is left
            if amount > 0:
                self.goods_supply[g] -= amount
                player.goods[g] += amount
                kinds.append(g)
        if player.occupied("FACTORY"):
            player.doubloons += FACTORY_BONUS[len(kinds)]    # automatic (E9)
        return tuple(kinds)

    def _craftsman_privilege_kinds(self):
        """E8 / R5.4.4: kinds the craftsman produced in this phase that remain in the supply."""
        return [g for g in self.craftsman_kinds if self.goods_supply[g] > 0]

    def _craftsman_legal(self):
        return [PASS] + [CRAFT_ACTION + GOODS.index(g) for g in self._craftsman_privilege_kinds()]

    def _craftsman_act(self, action):
        if action != PASS:
            g = GOODS[action - CRAFT_ACTION]
            self.goods_supply[g] -= 1
            self._player.goods[g] += 1
        self._role_finished()

    # ---------------------------------------------------------------------------------------------
    # Trader (RULES.md §5.5; E7)
    # ---------------------------------------------------------------------------------------------

    def _trader_begin(self):
        self.phase = "TRADER"
        self.current_player = self.role_player          # R5.5.1: the trader first

    def _trader_legal(self):
        player = self._player
        legal = [PASS]
        if len(self.trading_house) < TRADING_HOUSE_SPACES:                      # R5.5.3
            for g, good in enumerate(GOODS):
                if player.goods[good] > 0 and (good not in self.trading_house      # R5.5.4
                                               or player.occupied("OFFICE")):     # R6.3.6
                    legal.append(SELL_ACTION + g)
        return legal

    def _trader_act(self, action):
        if action != PASS:
            player = self._player
            good = GOODS[action - SELL_ACTION]
            player.goods[good] -= 1
            self.trading_house.append(good)
            price = TRADE_PRICE[good]                                 # R5.5.2
            if self.current_player == self.role_player:
                price += 1                                            # R5.5.1 privilege, R5.5.7
            if player.occupied("SMALL_MARKET"):
                price += 1                                            # R6.3.1
            if player.occupied("LARGE_MARKET"):
                price += 2                                            # R6.3.7
            player.doubloons += price
        self.turn += 1
        # R5.5.5: the phase ends when every player has had a turn or the house is full.
        if self.turn < self.n and len(self.trading_house) < TRADING_HOUSE_SPACES:
            self.current_player = self._turn_player()
            return
        if len(self.trading_house) == TRADING_HOUSE_SPACES:          # R5.5.6 (E3)
            for good in self.trading_house:
                self.goods_supply[good] += 1
            self.trading_house = []
        self._role_finished()

    # ---------------------------------------------------------------------------------------------
    # Captain: loading (RULES.md §5.6; E10)
    # ---------------------------------------------------------------------------------------------

    def _captain_begin(self):
        self.phase = "CAPTAIN"
        self.current_player = self.role_player          # R5.6.1: the captain loads first
        self.done = [False] * self.n
        self.passed = [False] * self.n
        self.wharf_used = [False] * self.n
        self.captain_loaded = False
        self.pending_benefits = []

    def _cargo_loads(self, player_index):
        """The legal cargo-ship loads (R5.6.2-R5.6.4, E10): for each good the player holds, the
        ships that are not full and carry that good, or are empty while no other ship carries
        it; among them, only the ships that take the most goods of that kind."""
        player = self.players[player_index]
        loads = []
        for g, good in enumerate(GOODS):
            held = player.goods[good]
            if held == 0:
                continue
            carried_elsewhere = any(ship.good == good for ship in self.ships)
            candidates = [s for s, ship in enumerate(self.ships)
                          if not ship.is_full()
                          and (ship.good == good or (ship.good is None and not carried_elsewhere))]
            if not candidates:
                continue
            amounts = {s: min(held, self.ships[s].capacity - self.ships[s].load)
                       for s in candidates}
            most = max(amounts.values())
            loads.extend(LOAD_ACTION + 5 * s + g for s in candidates if amounts[s] == most)
        return loads

    def _wharf_actions(self, player_index):
        """R6.3.12, E10: an occupied, unused wharf adds one action per kind the player holds."""
        player = self.players[player_index]
        if not player.occupied("WHARF") or self.wharf_used[player_index]:
            return []
        return [WHARF_ACTION + g for g, good in enumerate(GOODS) if player.goods[good] > 0]

    def _captain_legal(self):
        loads = self._cargo_loads(self.current_player)
        wharf = self._wharf_actions(self.current_player)
        if loads:
            return loads + wharf               # R5.6.2: must load if they can (or use the wharf)
        return wharf + [PASS]                  # E10: pass only when no cargo-ship load is possible

    def _captain_act(self, action):
        player_index = self.current_player
        player = self._player
        if action == PASS:
            self._captain_pass()
        elif WHARF_ACTION <= action < WHARF_ACTION + len(GOODS):
            # R6.3.12: all goods of the kind go to the supply and score as if shipped.
            good = GOODS[action - WHARF_ACTION]
            amount = player.goods[good]
            player.goods[good] = 0
            self.goods_supply[good] += amount
            self.wharf_used[player_index] = True
            self._loaded(amount)
        else:
            ship_index, g = divmod(action - LOAD_ACTION, 5)
            ship, good = self.ships[ship_index], GOODS[g]
            amount = min(player.goods[good], ship.capacity - ship.load)     # R5.6.4: all it can
            ship.good = good
            ship.load += amount
            player.goods[good] -= amount
            self._loaded(amount)

    def _award_vp(self, player_index, amount):
        """R5.6.5-R5.6.6: VP chips from the supply; past its end they are recorded on paper and
        the supply goes negative.  R7.1(c): taking the last chip ends the game at round end."""
        self.players[player_index].vp += amount
        self.vp_supply -= amount
        if self.vp_supply <= 0:
            self.end_triggered = True

    def _loaded(self, amount):
        """After a load by cargo ship or wharf: 1 VP per good (R5.6.5), then the benefits of
        E9 in order: the captain's privilege (the captain's first load only, O6), then the
        harbor (R6.3.11)."""
        player_index = self.current_player
        self._award_vp(player_index, amount)
        self.passed = [False] * self.n      # E10: those who passed with a wharf get another turn
        if player_index == self.role_player and not self.captain_loaded:
            self.captain_loaded = True      # E9: the privilege belongs to the first load
            self.pending_benefits.append("CAPTAIN_PRIVILEGE")
        if self._player.occupied("HARBOR"):
            self.pending_benefits.append("HARBOR")
        self._captain_continue()

    def _captain_benefit(self, accept):
        if accept:
            self._award_vp(self.current_player, 1)
        self._captain_continue()

    def _captain_continue(self):
        if self.pending_benefits:
            self.benefit = self.pending_benefits.pop(0)
        else:
            self._captain_next_turn()

    def _captain_pass(self):
        """E10, O5: a pass while unable to use a wharf is the player's last turn; a pass while
        still able to use the wharf keeps the player's place.  Loading ends as soon as every
        player has taken their last turn or passed since the most recent load."""
        player_index = self.current_player
        if self._wharf_actions(player_index):
            self.passed[player_index] = True
        else:
            self.done[player_index] = True
        if all(self.done[i] or self.passed[i] for i in range(self.n)):
            self._store_begin()
        else:
            self._captain_next_turn()

    def _captain_next_turn(self):
        """R5.6.1: the turn goes clockwise to the next player who has not taken their last turn.
        (A player who passed since the most recent load is never reached: before the turn
        could come back to them every other player would have passed, ending the loading.)"""
        nxt = self.current_player
        for _ in range(self.n):
            nxt = (nxt + 1) % self.n
            if not self.done[nxt]:
                break
        assert not self.done[nxt] and not self.passed[nxt]
        self.current_player = nxt

    # ---------------------------------------------------------------------------------------------
    # Captain: storage and the end of the phase (R5.6.7-R5.6.8; E11)
    # ---------------------------------------------------------------------------------------------

    def _store_begin(self):
        self.phase = "STORE"
        self.turn = 0
        self._store_turn_begin()

    def _store_turn_begin(self):
        self.current_player = self._turn_player()          # E11: the captain first
        self.windrose_kind = None
        self.warehouse_kinds = []

    @staticmethod
    def _goods_all_fit(player):
        """E11 / R5.6.7: the goods all fit when the player holds at most as many kinds as
        warehouse slots, or exactly one kind more and one of the kinds is a single good (which
        the windrose keeps).  Goods and slots do not change during a storage turn, so neither
        does the answer."""
        kinds = [g for g in GOODS if player.goods[g] > 0]
        slots = player.warehouse_slots()
        return (len(kinds) <= slots
                or (len(kinds) == slots + 1 and any(player.goods[g] == 1 for g in kinds)))

    def _store_legal(self):
        player = self._player
        if self._goods_all_fit(player):
            return [PASS]                 # E11: pass keeps everything; taken without asking (E2)
        # E11: the goods do not all fit, so the player chooses what to keep, one kind per action.
        unassigned = [g for g in GOODS
                      if player.goods[g] > 0 and g != self.windrose_kind
                      and g not in self.warehouse_kinds]
        free_slots = player.warehouse_slots() - len(self.warehouse_kinds)
        legal = []
        if free_slots > 0:
            legal += [WAREHOUSE_ACTION + GOODS.index(g) for g in unassigned]
        if self.windrose_kind is None:
            legal += [WINDROSE_ACTION + GOODS.index(g) for g in unassigned]
        elif free_slots == 0:
            legal.append(PASS)            # E11: every warehouse slot and the windrose hold a kind
        return legal

    def _store_act(self, action):
        if WINDROSE_ACTION <= action < WINDROSE_ACTION + len(GOODS):
            self.windrose_kind = GOODS[action - WINDROSE_ACTION]
            return
        if WAREHOUSE_ACTION <= action < WAREHOUSE_ACTION + len(GOODS):
            self.warehouse_kinds.append(GOODS[action - WAREHOUSE_ACTION])
            return
        # PASS ends the storage turn; goods leave the player only now (E11).  When the goods all
        # fit, everything is kept.  Otherwise the warehouse kinds are kept in full and one good
        # of the windrose kind; every other good returns to the supply (R5.6.7, R6.3.4, R6.3.8).
        player = self._player
        if self._goods_all_fit(player):
            kept = dict(player.goods)
        else:
            kept = {good: (player.goods[good] if good in self.warehouse_kinds
                           else 1 if good == self.windrose_kind else 0) for good in GOODS}
        for good in GOODS:
            self.goods_supply[good] += player.goods[good] - kept[good]
            player.goods[good] = kept[good]
        self.windrose_kind = None
        self.warehouse_kinds = []
        self.turn += 1
        if self.turn < self.n:
            self._store_turn_begin()
        else:
            self._unload_full_ships()
            self._role_finished()

    def _unload_full_ships(self):
        """R5.6.8 (automatic, E3): full ships are emptied; the others keep their cargo."""
        for ship in self.ships:
            if ship.is_full():
                self.goods_supply[ship.good] += ship.load
                ship.good = None
                ship.load = 0
