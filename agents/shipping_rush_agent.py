import numpy as np
import random

from agents.base import Agent
from puerto_rico.constants import BUILDING_DATA, BuildingType, Good, TileType


# Pre-compute building costs for threshold checks
_COST = {b: BUILDING_DATA[b][0] for b in BuildingType if b not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)}

# Helper to extract int from numpy/tensor
def _iv(x) -> int:
    return int(x.item()) if hasattr(x, "item") else int(x)


class ShippingRushAgent(Agent):
    """
    A heuristic agent with Shipping Rush strategy.
    
    Prioritizes shipping goods for VP over building or trading.
    Designed as a baseline opponent for PPO agent evaluation.
    
    v2 Improvements:
    - Opponent state awareness (goods, buildings)
    - Game end prediction (VP chips, city slots)
    - Optimized Captain shipping (maximize VP per shipment)
    """

    name = "ShippingRush"

    def __init__(self, action_dim: int = 200, fixed_strategy: int | None = None):
        super().__init__()
        self.action_dim = action_dim
        self.fixed_strategy = fixed_strategy
        self.strategy = 0
        self._env = None
        self.reset_strategy()

    def on_game_start(self, forward_model=None):
        self._env = forward_model

    def set_env(self, env):   # backward-compatible alias
        self._env = env

    def reset_strategy(self):
        """Only Shipping Rush strategy (strategy 0)."""
        if self.fixed_strategy is not None:
            self.strategy = self.fixed_strategy
        else:
            self.strategy = 0  # Only Shipping Rush

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_building(has_building_arr: np.ndarray, b_type: BuildingType) -> bool:
        """Check if player owns a building using binary has_building vector."""
        return bool(has_building_arr[b_type.value] > 0)

    @staticmethod
    def _building_colonists_count(building_colonists_arr: np.ndarray,
                                  b_type: BuildingType) -> int:
        """Get colonist count for a building using per-type colonist vector."""
        return int(building_colonists_arr[b_type.value])

    @staticmethod
    def _is_active(has_building_arr: np.ndarray, building_colonists_arr: np.ndarray,
                   b_type: BuildingType) -> bool:
        """Building is owned AND has at least one colonist."""
        return bool(has_building_arr[b_type.value] > 0 and building_colonists_arr[b_type.value] > 0)

    def _set_role(self, priority: np.ndarray, mask: np.ndarray, role_id: int, p: float):
        if mask[role_id]:
            priority[role_id] = p

    def _set_bldg(self, priority: np.ndarray, mask: np.ndarray,
                  b_list: list[int], base: float = 200.0):
        for rank, b in enumerate(b_list):
            action = 16 + b
            if mask[action]:
                priority[action] = base - rank * 5.0

    def _set_settler(self, priority: np.ndarray, mask: np.ndarray,
                     face_up_counts: np.ndarray, wanted_tiles: list[int], base: float = 150.0):
        """Assign settler priorities for desired plantation types.
        face_up_counts: shape (6,), count per TileType (0-5).
        wanted_tiles: list of TileType int values in priority order.
        Action mapping: 8=Coffee(0), 9=Tobacco(1), 10=Corn(2), 11=Sugar(3), 12=Indigo(4), 13=Quarry(5).
        """
        for rank, tile_type in enumerate(wanted_tiles):
            action = 8 + tile_type  # Direct tile_type to action mapping
            if tile_type < 6 and face_up_counts[tile_type] > 0 and mask[action]:
                priority[action] = base - rank * 5.0
        # Quarry (tile_type 5)
        if 5 in wanted_tiles:
            if mask[13]:
                priority[13] = base + 10.0
            elif mask[14]:
                priority[14] = base + 10.0

    # ------------------------------------------------------------------
    # Opponent analysis helpers
    # ------------------------------------------------------------------

    def _get_opponent_goods(self, obs_dict: dict, player_idx: int) -> dict:
        """Get total goods for each opponent."""
        opp_goods = {}
        for i in range(len(obs_dict["players"])):
            if i != player_idx:
                opp_state = obs_dict["players"][f"player_{i}"]
                opp_goods[i] = int(np.sum(opp_state["goods"]))
        return opp_goods

    def _opponent_wants_captain(self, obs_dict: dict, player_idx: int) -> bool:
        """Check if any opponent has more goods than us."""
        my_goods = int(np.sum(obs_dict["players"][f"player_{player_idx}"]["goods"]))
        for i, goods in self._get_opponent_goods(obs_dict, player_idx).items():
            if goods > my_goods:
                return True
        return False

    def _get_game_progress(self, obs_dict: dict, player_idx: int) -> dict:
        """Analyze game state to predict end timing using new obs encoding."""
        global_s = obs_dict["global_state"]
        vp_chips = _iv(global_s["vp_chips"][0])

        # Check city fill status using empty_city_spaces
        min_empty_city = 12
        for i in range(len(obs_dict["players"])):
            p_state = obs_dict["players"][f"player_{i}"]
            empty = _iv(p_state["empty_city_spaces"][0])
            min_empty_city = min(min_empty_city, empty)
        max_city_fill = 12 - min_empty_city

        vp_critical = vp_chips <= 15
        city_critical = max_city_fill >= 10

        return {
            "vp_chips": vp_chips,
            "vp_critical": vp_critical,
            "city_critical": city_critical,
            "endgame": vp_critical or city_critical,
            "max_city_fill": max_city_fill
        }

    # ------------------------------------------------------------------
    # Captain shipping optimization
    # ------------------------------------------------------------------

    def _get_best_shipping_action(self, mask: np.ndarray, goods: np.ndarray,
                                   cargo_ships_good_onehot: np.ndarray,
                                   cargo_ships_load: np.ndarray,
                                   cargo_ships_space: np.ndarray,
                                   has_harbor: bool, has_wharf: bool,
                                   wharf_used: bool) -> tuple[int, float]:
        """
        Find the best shipping action that maximizes VP.
        cargo_ships_good_onehot: shape (18,) — 3 ships × 6 classes (5 goods + none).
        cargo_ships_space: shape (3,) — remaining capacity per ship.
        Returns (action_id, priority_score).
        """
        best_action = -1
        best_score = 0.0

        for ship_idx in range(3):
            ship_space = int(cargo_ships_space[ship_idx])
            if ship_space <= 0:
                continue

            # Decode ship's good type from one-hot
            ship_onehot = cargo_ships_good_onehot[ship_idx * 6:(ship_idx + 1) * 6]
            ship_good = int(np.argmax(ship_onehot))  # 0-4 = good, 5 = none/empty

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

        # Check Wharf option (74-78 for wharf shipping)
        if has_wharf and not wharf_used:
            for good_idx in range(5):
                action = 74 + good_idx
                if not mask[action]:
                    continue

                my_amount = _iv(goods[good_idx])
                if my_amount <= 0:
                    continue

                vp = my_amount
                if has_harbor:
                    vp += 1

                score = vp * 10 + my_amount + 5
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action, best_score

    # ------------------------------------------------------------------
    # Mayor strategy selection
    # ------------------------------------------------------------------

    def _choose_mayor_strategy(self, my_state: dict, mask: np.ndarray,
                                priority: np.ndarray, player_idx: int):
        """Mayor actions: 120-131 (Island), 140-151 (City).
        Uses has_building (binary) and building_colonists (per-type) from new obs.
        Since Mayor phase recalls colonists, we use the mask to determine valid slots.
        Priority is assigned by building category importance mapping through env.game.
        """
        has_bldg = my_state["has_building"]
        bldg_col = my_state["building_colonists"]

        shipping_bldgs = {BuildingType.HARBOR, BuildingType.WHARF,
                          BuildingType.SMALL_WAREHOUSE, BuildingType.LARGE_WAREHOUSE}
        trade_bldgs    = {BuildingType.SMALL_MARKET, BuildingType.LARGE_MARKET,
                          BuildingType.OFFICE, BuildingType.FACTORY}
        production_bldgs = {BuildingType.SMALL_INDIGO_PLANT, BuildingType.INDIGO_PLANT,
                             BuildingType.SMALL_SUGAR_MILL, BuildingType.SUGAR_MILL,
                             BuildingType.TOBACCO_STORAGE, BuildingType.COFFEE_ROASTER}

        game = self._env.game if hasattr(self, '_env') and self._env else None

        # City building colonist placement (140-162): use mask to find valid targets by BuildingType
        for b_val in range(23):
            action_id = 140 + b_val
            if mask[action_id]:
                base = 230.0
                b_type = BuildingType(b_val)
                if b_type in shipping_bldgs:
                    base = 260.0
                elif b_type in production_bldgs:
                    base = 240.0
                elif b_type in trade_bldgs:
                    base = 235.0
                priority[action_id] = base + np.random.uniform(0, 5.0)

        # Island tile colonist placement (120-125): use mask to find valid targets by TileType
        for t_val in range(6):
            action_id = 120 + t_val
            if mask[action_id]:
                base = 220.0
                t_type = TileType(t_val)
                if t_type == TileType.QUARRY:
                    base = 225.0
                elif t_type in (TileType.COFFEE_PLANTATION, TileType.TOBACCO_PLANTATION, TileType.SUGAR_PLANTATION):
                    base = 230.0
                priority[action_id] = base + np.random.uniform(0, 5.0)

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def act(self, observation, action_mask):
        priority = np.full(self.action_dim, 10.0, dtype=np.float32)
        mask_np = np.asarray(action_mask).flatten()
        mask = mask_np

        priority[15] = 1.0  # Pass is weakest
        priority[16:39] = 0.5  # Suppress random building

        # Rebuild the structured observation that this strategy reads from.
        fm = self._env
        obs_dict = None
        player_idx = None
        if fm is not None:
            player_idx = fm.current_player()
            obs_dict = fm.observation_dict()

        if obs_dict is not None and player_idx is not None:
            my_state = obs_dict["players"][f"player_{player_idx}"]
            global_s = obs_dict["global_state"]

            # ── Scalar extraction ──────────────────────────────────────
            doubloons = _iv(my_state["doubloons"][0])
            goods: np.ndarray = my_state["goods"]
            total_goods = int(goods.sum())
            unplaced_col = _iv(my_state["unplaced_colonists"][0])

            has_bldg = my_state["has_building"]
            bldg_col = my_state["building_colonists"]

            empty_city_slots = int(my_state["empty_city_spaces"][0])
            empty_island_slots = int(my_state["island_empty_spaces"][0])
            occupied_city = 12 - empty_city_slots
            occupied_island = 12 - empty_island_slots

            # ── Active building checks (new binary encoding) ──────────
            has_harbor = self._is_active(has_bldg, bldg_col, BuildingType.HARBOR)
            has_wharf = self._is_active(has_bldg, bldg_col, BuildingType.WHARF)
            has_sm_mkt = self._is_active(has_bldg, bldg_col, BuildingType.SMALL_MARKET)
            has_harbor_built = self._has_building(has_bldg, BuildingType.HARBOR)
            has_wharf_built = self._has_building(has_bldg, BuildingType.WHARF)

            # ── Game state analysis ────────────────────────────────────
            trading_house_count = int(global_s["trading_house_count"][0])
            is_trader_open = trading_house_count < 4
            
            game_progress = self._get_game_progress(obs_dict, player_idx)
            endgame = game_progress["endgame"]
            vp_critical = game_progress["vp_critical"]
            
            # Opponent analysis
            opp_wants_captain = self._opponent_wants_captain(obs_dict, player_idx)
            opp_goods = self._get_opponent_goods(obs_dict, player_idx)
            max_opp_goods = max(opp_goods.values()) if opp_goods else 0

            face_up_counts = global_s["face_up_plantation_counts"]
            cargo_ships_good_onehot = global_s["cargo_ships_good_onehot"]
            cargo_ships_load = global_s["cargo_ships_load"]
            cargo_ships_space = global_s["cargo_ships_space"]

            # ══════════════════════════════════════════════════════════
            # SHIPPING RUSH STRATEGY
            # ══════════════════════════════════════════════════════════

            # Role priorities with opponent awareness
            self._set_role(priority, mask, 6, 20.0)  # Prospectors
            self._set_role(priority, mask, 7, 20.0)

            # 1. MAYOR: FIXED (Pick if vacant slots exist, even if unplaced is 0)
            game_obj = self._env.game if (hasattr(self, '_env') and self._env) else None
            has_vacant_slots = False
            if game_obj:
                p_obj = game_obj.players[player_idx]
                vacant_b = sum(BUILDING_DATA[b.building_type][2] - b.colonists for b in p_obj.city_board if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE))
                vacant_i = sum(1 for t in p_obj.island_board if t.tile_type != TileType.EMPTY and not t.is_occupied)
                has_vacant_slots = (vacant_b + vacant_i) > 0

            if unplaced_col > 0 or has_vacant_slots:
                priority[1] = 110.0 if (has_vacant_slots and unplaced_col == 0) else 90.0

            # 2. CAPTAIN
            if total_goods >= 2:
                base_captain = 140.0
                if has_harbor: base_captain += 40.0
                if endgame: base_captain += 30.0
                if opp_wants_captain: base_captain += 20.0
                self._set_role(priority, mask, 5, base_captain)
            elif total_goods == 1:
                self._set_role(priority, mask, 5, 100.0 if has_harbor else 70.0)

            # 3. SETTLER
            if empty_island_slots > 0:
                priority[0] = 115.0 if occupied_island < 4 else 80.0

            # 4. CRAFTSMAN: FIXED (Only produce if I have capacity)
            prod_cap = my_state["production_capacity"].sum() if "production_capacity" in my_state else 0
            if prod_cap > 0:
                if total_goods == 0:
                    priority[3] = 125.0
                elif total_goods <= 2:
                    priority[3] = 100.0
            else:
                self._set_role(priority, mask, 3, 10.0) # Avoid suicide Craftsman

            # 5. TRADER
            if total_goods > 0 and is_trader_open:
                priority[4] = 105.0 if (doubloons < 5 and not has_harbor_built) else 75.0

            # 6. BUILDER
            if empty_city_slots > 0:
                if not has_wharf_built and doubloons >= _COST[BuildingType.WHARF]:
                    self._set_role(priority, mask, 2, 145.0)
                elif not has_harbor_built and doubloons >= _COST[BuildingType.HARBOR]:
                    self._set_role(priority, mask, 2, 135.0)
                elif not has_sm_mkt and doubloons >= _COST[BuildingType.SMALL_MARKET]:
                    self._set_role(priority, mask, 2, 80.0)

            # 7. TILE SELECTION: Semantic usage
            self._set_settler(priority, mask, face_up_counts, [2, 4, 3, 5])

            # Building priorities
            bldg_priority = [
                BuildingType.WHARF,
                BuildingType.HARBOR,
                BuildingType.LARGE_WAREHOUSE,
                BuildingType.SMALL_MARKET,
                BuildingType.SMALL_SUGAR_MILL,
                BuildingType.SMALL_INDIGO_PLANT,
                BuildingType.SMALL_WAREHOUSE,
            ]
            self._set_bldg(priority, mask, bldg_priority, base=230.0)

            # Large buildings in endgame
            if endgame or occupied_city >= 8:
                self._set_bldg(priority, mask, [
                    BuildingType.CUSTOMS_HOUSE,
                    BuildingType.FORTRESS,
                    BuildingType.GUILDHALL,
                ], base=200.0)

            # ── Phase-specific actions ──────────────────────────────────

            # Optimized Captain shipping
            best_ship_action, ship_score = self._get_best_shipping_action(
                mask, goods, cargo_ships_good_onehot, cargo_ships_load,
                cargo_ships_space, has_harbor, has_wharf, False
            )
            if best_ship_action >= 0:
                priority[best_ship_action] = 320.0 + ship_score
            
            # Other captain actions get lower priority
            for i in range(44, 64):
                if mask[i] and priority[i] < 300.0:
                    priority[i] = 280.0

            # Mayor strategy
            self._choose_mayor_strategy(my_state, mask, priority, player_idx)

            # Trader: prefer high-value goods
            for i, good_val in enumerate([4, 3, 0, 2, 1]):
                action = 39 + i
                if mask[action]:
                    priority[action] = 100.0 + good_val * 8

            # Craftsman privilege: prefer high-value goods
            for i, good_val in enumerate([4, 3, 0, 2, 1]):
                action = 93 + i
                if mask[action]:
                    priority[action] = 80.0 + good_val * 5

            # Captain store: keep goods
            for i in range(64, 69):
                if mask[i]:
                    priority[i] = 55.0 + i
            for i in range(106, 111):
                if mask[i]:
                    priority[i] = 55.0 + i

        priority += np.random.uniform(0, 0.1, size=self.action_dim)
        priority[mask_np <= 0.5] = -1e9

        valid_actions = np.where(mask_np > 0.5)[0]

        chosen_act = int(np.argmax(priority))
        return int(chosen_act)


# Backward compatibility alias (deprecated - will be removed in future versions)
AdvancedRuleBasedAgent = ShippingRushAgent
