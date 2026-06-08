"""
FactoryAgent — Factory Diversification Strategy (v2)

Core loop:
  1. Early game: Settler to build diverse plantations + Builder for production buildings
  2. Mid game: Trade high-value goods → accumulate doubloons → buy Factory
  3. Post-Factory: Craftsman to trigger Factory bonus (multi-good production)
  4. Post-Harbor: Captain for VP + continue Factory income

Key insight: You MUST have plantations to produce goods, and you MUST produce goods
to trade and earn doubloons. Without Settler, the entire Factory economy collapses.
"""
import numpy as np

from agents.base import Agent
from puerto_rico.constants import BUILDING_DATA, BuildingType, TileType, Good

# ── Pre-computed look-ups ─────────────────────────────────────────────────────
_COST: dict[BuildingType, int] = {
    b: BUILDING_DATA[b][0] for b in BuildingType
    if b not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)
}

# Plantation tile → primary production building (None = no building needed)
_TILE_TO_PROD: dict[int, BuildingType | None] = {
    int(TileType.INDIGO_PLANTATION):  BuildingType.SMALL_INDIGO_PLANT,
    int(TileType.TOBACCO_PLANTATION): BuildingType.TOBACCO_STORAGE,
    int(TileType.SUGAR_PLANTATION):   BuildingType.SMALL_SUGAR_MILL,
    int(TileType.COFFEE_PLANTATION):  BuildingType.COFFEE_ROASTER,
    int(TileType.CORN_PLANTATION):    None,
    int(TileType.QUARRY):             None,
}

# Settler plantation priority: Diverse high-value goods for Factory bonus
# Coffee(4) > Tobacco(3) > Sugar(2) > Corn(0, but free production) > Indigo(1)
_SETTLER_PREF = [
    int(TileType.COFFEE_PLANTATION),   # Highest value: 4 doubloons
    int(TileType.TOBACCO_PLANTATION),  # High value: 3 doubloons  
    int(TileType.SUGAR_PLANTATION),    # Medium value: 2 doubloons
    int(TileType.CORN_PLANTATION),     # Free production (no building needed)
    int(TileType.INDIGO_PLANTATION),   # Low value: 1 doubloon
]

# Target plantation composition for Factory strategy
# Goal: 4+ different goods to maximize Factory bonus (+3 doubloons for 4 types)
_TARGET_COUNTS = {
    int(TileType.COFFEE_PLANTATION):  2,   # High priority
    int(TileType.TOBACCO_PLANTATION): 2,   # High priority
    int(TileType.SUGAR_PLANTATION):   2,   # Medium priority
    int(TileType.CORN_PLANTATION):    2,   # Free production
    int(TileType.INDIGO_PLANTATION):  2,   # Fill remaining
}

# Goods array index → Good enum value (obs order: Coffee=0,Tobacco=1,Corn=2,Sugar=3,Indigo=4)
# Trade actions: 39=Coffee, 40=Tobacco, 41=Corn, 42=Sugar, 43=Indigo
# Craftsman privilege: 93=Coffee, 94=Tobacco, 95=Corn, 96=Sugar, 97=Indigo

# Large building purchase order
_LARGE_BLDG_ORDER = [
    BuildingType.GUILDHALL,
    BuildingType.CUSTOMS_HOUSE,
    BuildingType.CITY_HALL,
]


# ── Helper ────────────────────────────────────────────────────────────────────
def _iv(x) -> int:
    return int(x.item()) if hasattr(x, "item") else int(x)


class FactoryAgent(Agent):
    """
    Deterministic Factory diversification strategy.
    reset_strategy() is a no-op (single strategy, no randomisation).
    """

    name = "Factory"

    def __init__(self, action_dim: int = 200):
        super().__init__()
        self.action_dim = action_dim
        self._env = None

    def on_game_start(self, forward_model=None):
        self._env = forward_model

    def set_env(self, env):   # backward-compatible alias
        self._env = env

    def reset_strategy(self):
        pass  # Deterministic

    # ── Static board queries ──────────────────────────────────────────────────

    @staticmethod
    def _has(city_b: np.ndarray, b: BuildingType) -> bool:
        return int(b) in city_b

    @staticmethod
    def _active(city_b: np.ndarray, city_c: np.ndarray, b: BuildingType) -> bool:
        for slot, bv in enumerate(city_b):
            if bv == int(b) and city_c[slot] > 0:
                return True
        return False

    @staticmethod
    def _tile_counts(island_tiles: np.ndarray) -> dict[int, int]:
        counts: dict[int, int] = {}
        for t in island_tiles:
            v = _iv(t)
            if v <= 5:
                counts[v] = counts.get(v, 0) + 1
        return counts

    @staticmethod
    def _tradeable_actions(goods: np.ndarray, trading_house) -> list[int]:
        """Return Trader action ids (39–43) for goods not already in the house."""
        in_house = {_iv(g) for g in trading_house if 0 <= _iv(g) <= 4}
        return [
            39 + i
            for i in range(5)
            if _iv(goods[i]) > 0 and i not in in_house
        ]

    def _unmatched_tiles(self, city_b: np.ndarray, island_tiles: np.ndarray) -> list[int]:
        """Plantation types present but missing their production building."""
        counts = self._tile_counts(island_tiles)
        return [
            t for t, bldg in _TILE_TO_PROD.items()
            if bldg is not None
            and counts.get(t, 0) > 0
            and not self._has(city_b, bldg)
        ]

    @staticmethod
    def _starting_with_corn(island_tiles: np.ndarray) -> bool:
        """True if only plantation is corn (corn-start scenario)."""
        tile_vals = [_iv(t) for t in island_tiles if _iv(t) <= 5]
        return (
            tile_vals.count(int(TileType.CORN_PLANTATION)) > 0
            and tile_vals.count(int(TileType.INDIGO_PLANTATION)) == 0
            and len(tile_vals) <= 2
        )

    # ── Building selection ────────────────────────────────────────────────────

    def _choose_building(self, doubloons: int, city_b: np.ndarray,
                          has_factory: bool, has_harbor: bool,
                          factory_active: bool,
                          unmatched: list[int],
                          corn_start_no_small_market: bool,
                          has_sm_mkt: bool = False,
                          num_prod_buildings: int = 0) -> BuildingType | None:
        """
        Decide which building to buy (None = nothing affordable/needed).
        
        Strategy: Build 1-2 production buildings early, then rush Factory.
        After Factory, build Harbor for VP, then large buildings.
        """
        # 1. Small Market early (huge ROI for trade-focused strategy)
        if not has_sm_mkt and doubloons >= _COST[BuildingType.SMALL_MARKET]:
            return BuildingType.SMALL_MARKET

        # 2. First 1-2 production buildings (need some production to trade)
        if num_prod_buildings < 2:
            # Priority: cheap buildings first to get economy running
            prod_priority = [
                int(TileType.INDIGO_PLANTATION),   # cost 1 - quick start
                int(TileType.SUGAR_PLANTATION),    # cost 2
                int(TileType.TOBACCO_PLANTATION),  # cost 5
                int(TileType.COFFEE_PLANTATION),   # cost 6
            ]
            for tile_t in prod_priority:
                if tile_t not in unmatched:
                    continue
                bldg = _TILE_TO_PROD[tile_t]
                if bldg is not None and doubloons >= _COST[bldg]:
                    return bldg

        # 3. FACTORY — top priority once we have basic production
        if not has_factory and doubloons >= _COST[BuildingType.FACTORY]:
            return BuildingType.FACTORY

        # 4. More production buildings AFTER Factory (for Factory bonus)
        if has_factory and not has_harbor:
            for tile_t in [int(TileType.TOBACCO_PLANTATION), int(TileType.COFFEE_PLANTATION),
                           int(TileType.SUGAR_PLANTATION), int(TileType.INDIGO_PLANTATION)]:
                if tile_t not in unmatched:
                    continue
                bldg = _TILE_TO_PROD[tile_t]
                if bldg is not None and doubloons >= _COST[bldg]:
                    return bldg

        # 5. Harbor (after Factory built)
        if has_factory and not has_harbor:
            if doubloons >= _COST[BuildingType.HARBOR]:
                return BuildingType.HARBOR

        # 6. Large buildings
        if has_harbor or has_factory:
            for lb in _LARGE_BLDG_ORDER:
                if not self._has_cached(city_b, lb) and doubloons >= _COST[lb]:
                    return lb

        return None

    def _has_cached(self, city_b: np.ndarray, b: BuildingType) -> bool:
        return int(b) in city_b

    # ── Settler preference ────────────────────────────────────────────────────

    def _settler_action(self, mask: np.ndarray, face_up: np.ndarray,
                        tile_counts: dict[int, int]) -> int | None:
        """
        If have Settler privilege (action 14 available): always take Quarry.
        Otherwise: pick from face-up tiles according to priority,
                   skipping types we already hold in target quantity.
        """
        # Settler privilege → Quarry
        if mask[14]:
            return 14

        # Build priority list: tile types we still need more of
        wanted = []
        for tile_t in _SETTLER_PREF:
            have = tile_counts.get(tile_t, 0)
            need = _TARGET_COUNTS.get(tile_t, 1)
            if have < need:
                wanted.append(tile_t)

        for tile_t in wanted:
            for slot_idx, ft in enumerate(face_up):
                if _iv(ft) == tile_t and mask[8 + slot_idx]:
                    return 8 + slot_idx

        # Fallback: any available face-up tile
        for slot_idx in range(len(face_up)):
            if mask[8 + slot_idx]:
                return 8 + slot_idx

        return None

    # ── Mayor strategy ────────────────────────────────────────────────────────

    def _mayor_action(self, mask: np.ndarray, harbor_active: bool) -> int | None:
        """
        Pre-Harbor:  BUILDING_FOCUS (71) → fill production buildings + Factory
        Post-Harbor: TRADE_FACTORY_FOCUS (70) → then CAPTAIN_FOCUS (69) for Harbor VP
        """
        if harbor_active:
            for a in (69, 70, 71):
                if mask[a]:
                    return a
        else:
            for a in (71, 70, 69):
                if mask[a]:
                    return a
        return None

    # ── Active production check ───────────────────────────────────────────────

    def _has_active_production(self, island_tiles: np.ndarray,
                                island_occ: np.ndarray,
                                city_b: np.ndarray,
                                city_c: np.ndarray) -> bool:
        """True if at least one commodity can currently be produced."""
        for tile, occ in zip(island_tiles, island_occ):
            t = _iv(tile)
            is_occ = bool(_iv(occ))
            if not is_occ:
                continue
            if t == int(TileType.CORN_PLANTATION):
                return True  # Corn needs no building
            prod_bldg = _TILE_TO_PROD.get(t)
            if prod_bldg is not None and self._active(city_b, city_c, prod_bldg):
                return True
        return False

    # ── Opponent analysis helpers ────────────────────────────────────────────

    def _get_opponent_goods(self, obs_dict: dict, player_idx: int) -> dict:
        """Get total goods for each opponent."""
        if not hasattr(self, '_env') or not self._env: return {}
        game = self._env.game
        opp_goods = {}
        for i, p in enumerate(game.players):
            if i != player_idx:
                opp_goods[i] = sum(p.goods.values())
        return opp_goods

    def _get_game_progress(self, obs_dict: dict) -> dict:
        """Analyze game state to predict end timing."""
        if not hasattr(self, '_env') or not self._env:
            return {"vp_chips": 50, "vp_critical": False, "city_critical": False, "endgame": False, "max_city_fill": 0}
        game = self._env.game
        vp_chips = game.vp_chips
        
        max_city_fill = max(sum(1 for b in p.city_board if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)) for p in game.players)
        
        vp_critical = vp_chips <= 15
        city_critical = max_city_fill >= 10
        
        return {
            "vp_chips": vp_chips,
            "vp_critical": vp_critical,
            "city_critical": city_critical,
            "endgame": vp_critical or city_critical,
            "max_city_fill": max_city_fill
        }

    def _get_best_shipping_action(self, mask: np.ndarray, goods: np.ndarray,
                                   cargo_ships_good: np.ndarray,
                                   cargo_ships_load: np.ndarray,
                                   has_harbor: bool) -> tuple[int, float]:
        """Find the best shipping action that maximizes VP."""
        ship_capacities = [4, 5, 6]
        best_action = -1
        best_score = 0.0
        
        for ship_idx in range(3):
            ship_good = _iv(cargo_ships_good[ship_idx])
            ship_load = _iv(cargo_ships_load[ship_idx])
            ship_cap = ship_capacities[ship_idx] if ship_idx < len(ship_capacities) else 6
            ship_space = ship_cap - ship_load
            
            if ship_space <= 0:
                continue
            
            for good_idx in range(5):
                action = 44 + ship_idx * 5 + good_idx
                if not mask[action]:
                    continue
                
                my_amount = _iv(goods[good_idx])
                if my_amount <= 0:
                    continue
                
                can_load = (ship_good == 5 or ship_good == good_idx)
                if not can_load:
                    continue
                
                load_amount = min(my_amount, ship_space)
                vp = load_amount
                if has_harbor:
                    vp += 1
                
                score = vp * 10 + load_amount
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        return best_action, best_score

    # ── Main inference ────────────────────────────────────────────────────────

    def act(self, observation, action_mask):
        # This strategy reads structured fields, so use the nested observation
        # (the flat `observation` arg is not enough here).
        obs_dict = self._env.observation_dict() if getattr(self, "_env", None) is not None else None
        priority = np.full(self.action_dim, 3.0, dtype=np.float32)
        mask_np = np.asarray(action_mask).flatten()
        valid_actions = np.where(mask_np > 0.5)[0]
        mask = mask_np
        priority[15] = 0.5

        for a in range(16, 39):
            priority[a] = 0.08

        game = self._env.game if hasattr(self, '_env') and self._env else None

        # Resolve the acting player from the engine's current-player pointer.
        player_idx = game.current_player_idx if game is not None else None

        if game is None or player_idx is None:
            priority += np.random.uniform(0, 0.05, self.action_dim)
            priority[mask == 0] = -1e9
            return int(np.argmax(priority))

        p = game.players[player_idx]

        doubloons     = p.doubloons
        goods         = np.array([p.goods[Good.COFFEE], p.goods[Good.TOBACCO], p.goods[Good.CORN], p.goods[Good.SUGAR], p.goods[Good.INDIGO]])
        total_goods   = int(np.sum(goods))
        
        city_b        = np.array([b.building_type.value for b in p.city_board])
        city_c        = np.array([b.colonists for b in p.city_board])
        island_tiles  = np.array([t.tile_type.value if t.tile_type != TileType.EMPTY else 255 for t in p.island_board])
        island_occ    = np.array([1 if t.is_occupied else 0 for t in p.island_board])
        
        face_up       = np.array([t.value for t in game.face_up_plantations])
        trading_house = np.array([g.value for g in game.trading_house])
        unplaced_col  = p.unplaced_colonists
        
        cargo_ships_good = np.array([s.good_type.value if s.good_type is not None else 5 for s in game.cargo_ships])
        cargo_ships_load = np.array([s.current_load for s in game.cargo_ships])

        tile_counts = self._tile_counts(island_tiles)
        num_plantations = sum(tile_counts.values())
        has_factory = self._has(city_b, BuildingType.FACTORY)
        has_harbor  = self._has(city_b, BuildingType.HARBOR)
        has_sm_mkt  = self._has(city_b, BuildingType.SMALL_MARKET)
        factory_act = self._active(city_b, city_c, BuildingType.FACTORY)
        harbor_act  = self._active(city_b, city_c, BuildingType.HARBOR)
        unmatched   = self._unmatched_tiles(city_b, island_tiles)
        
        prod_bldg_types = [BuildingType.SMALL_INDIGO_PLANT, BuildingType.INDIGO_PLANT,
                          BuildingType.SMALL_SUGAR_MILL, BuildingType.SUGAR_MILL,
                          BuildingType.TOBACCO_STORAGE, BuildingType.COFFEE_ROASTER]
        num_prod_buildings = sum(1 for b in prod_bldg_types if self._has(city_b, b))

        target_now = self._choose_building(
            doubloons, city_b, has_factory, has_harbor,
            factory_act, unmatched, False, has_sm_mkt, num_prod_buildings
        )
        target_with_discount = self._choose_building(
            doubloons + 1, city_b, has_factory, has_harbor,
            factory_act, unmatched, False, has_sm_mkt, num_prod_buildings
        )

        trade_actions = self._tradeable_actions(goods, trading_house)
        can_trade     = len(trade_actions) > 0
        has_prod      = self._has_active_production(island_tiles, island_occ, city_b, city_c)
        
        # Game state analysis
        game_progress = self._get_game_progress(obs_dict)
        endgame = game_progress["endgame"]
        vp_critical = game_progress["vp_critical"]
        
        # Opponent analysis
        opp_goods = self._get_opponent_goods(obs_dict, player_idx)
        max_opp_goods = max(opp_goods.values()) if opp_goods else 0
        
        early_game = num_plantations < 4
        mid_game = 4 <= num_plantations < 7
        late_game = num_plantations >= 7 or has_harbor or endgame

        # ── Role priorities ───────────────────────────────────────────────────

        if harbor_act or (has_harbor and endgame):
            # ── ENDGAME: Captain for VP, Builder for large buildings ──────
            if target_with_discount is not None and mask[2]:
                priority[2] = 210.0
            if mask[5]:
                base_captain = 190.0
                if total_goods >= 2:
                    base_captain += 20.0
                priority[5] = base_captain
            if mask[3]:
                priority[3] = 160.0
            if mask[1]:
                priority[1] = 80.0
            if mask[0]:
                priority[0] = 60.0    # Settler (still useful in endgame)
        else:
            # ── GROWTH PHASE ──────────────────────────────────────────────
            
            # SETTLER: CRITICAL for Factory strategy - need diverse plantations!
            # Without plantations, no production → no trade → no money → no Factory
            if mask[0]:
                if early_game:
                    priority[0] = 180.0  # HIGH priority early
                elif mid_game:
                    priority[0] = 120.0  # Still important
                else:
                    priority[0] = 60.0   # Less critical late
            
            # Builder: when there is a STRATEGIC building to buy
            if target_with_discount is not None:
                bp = (
                    220.0 if target_with_discount == BuildingType.FACTORY else
                    200.0 if target_with_discount == BuildingType.HARBOR  else
                    185.0 if target_with_discount == BuildingType.SMALL_MARKET else
                    170.0  # production building
                )
                if mask[2]:
                    priority[2] = bp
            else:
                priority[2] = 1.0  # Avoid if nothing to build

            # Trader: high priority when tradeable good in hand
            if can_trade and mask[4]:
                # Higher value goods = higher priority
                high_value = any(_iv(goods[i]) > 0 for i in [0, 1])  # Coffee or Tobacco
                priority[4] = 210.0 if high_value else 160.0

            # Craftsman: produce goods when we can
            if mask[3]:
                if has_prod and not can_trade:
                    priority[3] = 175.0  # High when we need goods
                elif has_prod:
                    priority[3] = 100.0  # Medium when we already have goods
                else:
                    priority[3] = 40.0   # Low if no production

            # Mayor: place colonists when needed
            if unplaced_col > 0 and mask[1]:
                priority[1] = 130.0
            elif mask[1]:
                priority[1] = 50.0
            
            # Captain: Use opponent awareness - pre-empt if opponent has more goods
            if mask[5]:
                if has_harbor and total_goods >= 1:
                    priority[5] = 175.0  # High priority with Harbor
                elif total_goods >= 2 and max_opp_goods >= 3:
                    # Opponent has lots of goods - force them to ship
                    priority[5] = 155.0
                elif total_goods >= 2 and not can_trade:
                    priority[5] = 110.0  # Ship if can't trade
                else:
                    priority[5] = 30.0   # Avoid otherwise

        priority[6] = 8.0   # Prospectors: fallback
        priority[7] = 8.0

        # ── Specific action priorities ────────────────────────────────────────

        # Builder phase: ONLY the target building gets elevated.
        # All other building purchase actions stay at 0.08 (below pass).
        # target_now = what we can afford RIGHT NOW (in the Builder phase,
        # we may or may not have the role discount depending on who selected it).
        if target_now is not None:
            act_bldg = 16 + int(target_now)
            if mask[act_bldg]:
                priority[act_bldg] = max(priority[2], 165.0) + 10.0

        # Trader: rank sellable goods by value
        good_values = [4, 3, 0, 2, 1]  # Coffee, Tobacco, Corn, Sugar, Indigo
        for act in trade_actions:
            gi = act - 39
            if mask[act]:
                priority[act] = 215.0 + good_values[gi]

        # Settler: specific tile selection
        settler_act = self._settler_action(mask, face_up, tile_counts)
        if settler_act is not None and mask[settler_act]:
            priority[settler_act] = priority[0] + 5.0

        # Captain shipping: Use optimized shipping action selection
        best_ship_action, ship_score = self._get_best_shipping_action(
            mask, goods, cargo_ships_good, cargo_ships_load, has_harbor or harbor_act
        )
        if best_ship_action >= 0:
            priority[best_ship_action] = 300.0 + ship_score
        
        # Other captain actions get lower priority
        for i in range(44, 64):
            if mask[i] and priority[i] < 280.0:
                if has_harbor or harbor_act:
                    priority[i] = 270.0
                else:
                    priority[i] = 55.0
        
        # Wharf: use when have lots of goods and have harbor
        has_wharf = self._active(city_b, city_c, BuildingType.WHARF)
        for i in range(59, 64):
            if mask[i]:
                good_idx = i - 59
                amount = _iv(goods[good_idx])
                if (harbor_act or has_harbor) and amount >= 2:
                    priority[i] = 295.0 + amount * 5
                else:
                    priority[i] = 50.0
        
        # Captain store: keep goods if possible
        for i in range(64, 69):
            if mask[i]:
                priority[i] = 45.0

        # Craftsman privilege: prefer highest-value good for Factory bonus
        for i, gv in enumerate(good_values):
            if mask[93 + i]:
                priority[93 + i] = 60.0 + gv

        # ── Mayor strategy (120-125 Island by Type, 140-162 City by Type) ──────────
        for t_val in range(6):
            act_isl = 120 + t_val
            if mask[act_isl]:
                t_type = TileType(t_val)
                base = 220.0
                if t_type == TileType.QUARRY:
                    base = 225.0
                elif t_type in (TileType.COFFEE_PLANTATION, TileType.TOBACCO_PLANTATION):
                    base = 235.0
                else:
                    base = 230.0
                priority[act_isl] = base + np.random.uniform(0, 5.0)

        for b_val in range(23):
            act_city = 140 + b_val
            if mask[act_city]:
                b_type = BuildingType(b_val)
                base = 230.0
                if b_type == BuildingType.FACTORY:
                    base = 260.0
                elif b_type in (BuildingType.HARBOR, BuildingType.WHARF):
                    base = 250.0 if harbor_act else 240.0
                elif b_type in [BuildingType.SMALL_INDIGO_PLANT, BuildingType.INDIGO_PLANT, BuildingType.SMALL_SUGAR_MILL, BuildingType.SUGAR_MILL, BuildingType.TOBACCO_STORAGE, BuildingType.COFFEE_ROASTER]:
                    base = 245.0
                priority[act_city] = base + np.random.uniform(0, 5.0)

        # ── Finalize ─────────────────────────────────────────────────────────
        priority += np.random.uniform(0, 0.05, self.action_dim)
        priority[mask == 0] = -1e9

        chosen = int(np.argmax(priority))
        return chosen
