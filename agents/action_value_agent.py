import numpy as np
import copy
from typing import Optional

from agents.base import Agent
from puerto_rico.constants import (
    Phase, Role, Good, TileType, BuildingType,
    BUILDING_DATA, VP_CHIPS_SETUP, COLONIST_SUPPLY_SETUP
)


class ActionValueAgent(Agent):
    name = "ActionValue"
    """
    Action-Value Heuristic Agent for Puerto Rico.
    
    For each legal action, computes base state heuristic plus action-specific
    bonus using one-step lookahead evaluation. Designed as a strong baseline 
    for PPO agent evaluation.
    
    Strategy: Evaluates all legal actions by simulating them and computing
    the resulting state value using a comprehensive heuristic function.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Core conversion constants
    # ═══════════════════════════════════════════════════════════════════════════

    # Doubloon -> VP conversion ratio (4 doubloons = 1 VP, Puerto Rico statistics)
    _DOUBLOON_TO_VP = 0.25

    # Good trade prices (doubloons)
    _GOOD_TRADE_PRICES = {
        Good.COFFEE: 4,
        Good.TOBACCO: 3,
        Good.CORN: 0,
        Good.SUGAR: 2,
        Good.INDIGO: 1
    }
    
    # Good unit value (VP equivalent)
    # 1 VP when shipped, price x _DOUBLOON_TO_VP when sold
    # ** Revised: reflect trade value so expensive goods are worth more early **
    # V_goods = Σ qty(g) x unit_value(g) x P_ship x weak_decay
    # Here unit_value = max(1.0, trade_value) also reflects the trade option value
    _GOOD_UNIT_VALUES = {
        Good.COFFEE: 1.0,   # max(1.0, 4*0.25) = 1.0 (ship == sell)
        Good.TOBACCO: 1.0,  # max(1.0, 3*0.25) = 1.0
        Good.SUGAR: 1.0,    # max(1.0, 2*0.25) = 1.0
        Good.INDIGO: 1.0,   # max(1.0, 1*0.25) = 1.0
        Good.CORN: 1.0,     # max(1.0, 0*0.25) = 1.0
    }

    # Good extra value (flexibility of the trade option)
    # Expensive goods yield doubloons when sold -> can buy buildings -> extra value
    # Meaningful in the early game; shipping VP matters more late game
    _GOOD_TRADE_BONUS = {
        Good.COFFEE: 0.5,   # sellable for 4 doubloons -> high flexibility
        Good.TOBACCO: 0.4,  # 3 doubloons
        Good.SUGAR: 0.2,    # 2 doubloons
        Good.INDIGO: 0.1,   # 1 doubloon
        Good.CORN: 0.0,     # cannot be sold
    }

    # Plantation type -> good type mapping
    _PLANTATION_TO_GOOD = {
        TileType.COFFEE_PLANTATION: Good.COFFEE,
        TileType.TOBACCO_PLANTATION: Good.TOBACCO,
        TileType.CORN_PLANTATION: Good.CORN,
        TileType.SUGAR_PLANTATION: Good.SUGAR,
        TileType.INDIGO_PLANTATION: Good.INDIGO,
    }
    
    # Production building -> good type mapping
    _PRODUCTION_BUILDING_TO_GOOD = {
        BuildingType.SMALL_INDIGO_PLANT: Good.INDIGO,
        BuildingType.INDIGO_PLANT: Good.INDIGO,
        BuildingType.SMALL_SUGAR_MILL: Good.SUGAR,
        BuildingType.SUGAR_MILL: Good.SUGAR,
        BuildingType.TOBACCO_STORAGE: Good.TOBACCO,
        BuildingType.COFFEE_ROASTER: Good.COFFEE,
    }
    
    # Expected per-use VP value of commercial buildings (revised)
    _COMMERCIAL_ABILITY_VALUES = {
        BuildingType.SMALL_MARKET: 0.25,   # 1 doubloon = 0.25 VP
        BuildingType.LARGE_MARKET: 0.50,   # 2 doubloons = 0.5 VP
        BuildingType.OFFICE: 0.20,         # duplicate-sale value
        BuildingType.HARBOR: 1.0,          # extra VP per average shipment
        BuildingType.WHARF: 1.5,           # free-shipping flexibility
        BuildingType.SMALL_WAREHOUSE: 0.3, # good preservation value
        BuildingType.LARGE_WAREHOUSE: 0.5, # good preservation value
        BuildingType.FACTORY: 0.5,         # dynamic calculation (default)
        BuildingType.HACIENDA: 0.15,       # free plantation
        BuildingType.CONSTRUCTION_HUT: 0.15, # quarry access
        BuildingType.HOSPICE: 0.20,        # free colonist
        BuildingType.UNIVERSITY: 0.20,     # colonist on building
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # Game parameters (3-player basis)
    # ═══════════════════════════════════════════════════════════════════════════

    # Total role selections: 17 rounds x 3 players = 51
    # Uniform distribution assumption: each role 51/6 ≈ 8.5 times
    _TOTAL_ROLE_SELECTIONS = 51.0
    _NUM_ROLES = 6.0
    _EXPECTED_ROLE_USES_BASE = _TOTAL_ROLE_SELECTIONS / _NUM_ROLES  # ≈ 8.5

    # Good shipping success probability (conservative estimate)
    # Risks: insufficient ship space, competition, forced discard
    _SHIPPING_SUCCESS_PROB = 0.7

    # Building -> related role mapping
    _BUILDING_TO_ROLE = {
        BuildingType.HARBOR: 'captain',
        BuildingType.WHARF: 'captain',
        BuildingType.SMALL_WAREHOUSE: 'captain',
        BuildingType.LARGE_WAREHOUSE: 'captain',
        BuildingType.SMALL_MARKET: 'trader',
        BuildingType.LARGE_MARKET: 'trader',
        BuildingType.OFFICE: 'trader',
        BuildingType.FACTORY: 'craftsman',
        BuildingType.HACIENDA: 'settler',
        BuildingType.CONSTRUCTION_HUT: 'settler',
        BuildingType.HOSPICE: 'settler',
        BuildingType.UNIVERSITY: 'builder',
    }
    
    def __init__(self, action_dim: int = 200, use_role_doubloon_value: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self._env = None  # Will be set externally
        self._use_role_doubloon_value = use_role_doubloon_value
        
    def on_game_start(self, forward_model=None):
        """Set the ForwardModel reference for state access."""
        self._env = forward_model

    def set_env(self, env):
        """Backward-compatible alias for on_game_start."""
        self._env = env

    def act(self, observation, action_mask):
        """
        Select action using one-step lookahead heuristic evaluation.

        Args:
            observation: Observation (not used - we access env directly)
            action_mask: Action mask (1 = valid, 0 = invalid)

        Returns:
            int: a legal action index
        """
        if self._env is None:
            raise RuntimeError("Environment not set. Call set_env() first.")

        # Get valid actions from mask
        mask_np = np.asarray(action_mask).flatten()
        valid_actions = np.where(mask_np > 0.5)[0]

        if len(valid_actions) == 0:
            # No valid actions - return pass (action 15)
            return 15

        if len(valid_actions) == 1:
            # Only one valid action - no need to evaluate
            return int(valid_actions[0])

        # Evaluate each valid action using heuristic
        game = self._env.game
        current_player_idx = game.current_player_idx

        best_action = valid_actions[0]
        best_value = float('-inf')

        # Current state heuristic (for comparison)
        current_heuristic = self._compute_heuristic(game, current_player_idx)

        action_values = []
        for action_idx in valid_actions:
            # Estimate the heuristic value after taking this action
            # We use action-specific heuristic bonuses instead of full simulation
            action_value = self._estimate_action_value(
                game, current_player_idx, action_idx, current_heuristic
            )
            action_values.append(action_value)

        action_values = np.array(action_values)

        best_action = valid_actions[np.argmax(action_values)]

        return int(best_action)
    
    def _estimate_action_value(self, game, player_idx: int, action_idx: int, 
                                base_heuristic: float) -> float:
        """
        Estimate the heuristic value after taking an action.
        
        Instead of full simulation, we compute incremental changes based on
        action semantics. This is faster and avoids state modification issues.
        """
        p = game.players[player_idx]
        progress = self._game_progress(game)
        decay = max(0.0, 1.0 - progress)
        
        bonus = 0.0
        
        # ═══ Role Selection (0-7) ═══
        if action_idx < 8:
            role = Role(action_idx)
            bonus = self._role_selection_bonus(game, player_idx, role, decay)
        
        # ═══ Settler Phase: Plantation Selection (8-14) ═══
        elif 8 <= action_idx < 15:
            tile_idx = action_idx - 8
            tile_types = [
                TileType.COFFEE_PLANTATION,
                TileType.TOBACCO_PLANTATION,
                TileType.CORN_PLANTATION,
                TileType.SUGAR_PLANTATION,
                TileType.INDIGO_PLANTATION,
                TileType.QUARRY,
            ]
            if tile_idx < len(tile_types):
                bonus = self._plantation_bonus(game, player_idx, tile_types[tile_idx], decay)
        
        # ═══ Pass Action (15) ═══
        elif action_idx == 15:
            bonus = -0.1  # Small penalty for passing (when other options exist)
        
        # ═══ Builder Phase: Building Selection (16-38) ═══
        elif 16 <= action_idx < 39:
            building_idx = action_idx - 16
            building_types = list(BuildingType)[:23]  # Exclude EMPTY and OCCUPIED_SPACE
            if building_idx < len(building_types):
                bonus = self._building_bonus(game, player_idx, building_types[building_idx], decay)
        
        # ═══ Trader Phase: Good Selection (39-43) ═══
        elif 39 <= action_idx < 44:
            good_idx = action_idx - 39
            good = Good(good_idx)
            bonus = self._trade_bonus(game, player_idx, good, decay)
        
        # ═══ Captain Phase: Ship Selection (44-63) ═══
        elif 44 <= action_idx < 64:
            ship_good_idx = action_idx - 44
            ship_idx = ship_good_idx // 5
            good_idx = ship_good_idx % 5
            good = Good(good_idx)
            bonus = self._shipping_bonus(game, player_idx, ship_idx, good, decay)
        
        # ═══ Captain Store Phase: Good Selection (64-68) ═══
        elif 64 <= action_idx < 69:
            good_idx = action_idx - 64
            good = Good(good_idx)
            bonus = self._store_bonus(game, player_idx, good, decay)
            
        # ═══ Mayor Phase: Colonist Placement ═══
        elif 120 <= action_idx < 126:
            t_type = TileType(action_idx - 120)
            bonus = 0.5
            if t_type == TileType.QUARRY:
                bonus = 1.0
            else:
                good = self._PLANTATION_TO_GOOD.get(t_type)
                if good == Good.CORN:
                    bonus = 0.8
                elif good is not None and self._has_production_building(game, player_idx, good):
                    bonus = 0.9
            bonus *= decay
            
        elif 140 <= action_idx < 163:
            b_type = BuildingType(action_idx - 140)
            bonus = 1.0
            if b_type in self._COMMERCIAL_ABILITY_VALUES:
                bonus = 1.5
            elif b_type in self._PRODUCTION_BUILDING_TO_GOOD:
                bonus = 1.2
            if BUILDING_DATA.get(b_type, [0,0,0,0,False])[4]: # is_large 
                bonus = 2.0
            bonus *= decay
        
        # ═══ Wharf Phase (74-78) ═══
        elif 74 <= action_idx < 79:
            good_idx = action_idx - 74
            good = Good(good_idx)
            qty = p.goods[good]
            bonus = qty * 1.0  # Direct VP from wharf shipping
        
        return base_heuristic + bonus
    
    def _role_selection_bonus(self, game, player_idx: int, role: Role, decay: float) -> float:
        """Estimate value of selecting a role."""
        p = game.players[player_idx]
        bonus = 0.0
        
        # Role-specific bonuses based on current state
        if role == Role.SETTLER:
            # Value depends on available plantations and island space
            if p.empty_island_spaces > 0:
                bonus = 0.3 * decay
        
        elif role == Role.MAYOR:
            # Value depends on unplaced colonists and empty slots
            empty_slots = self._count_empty_slots(game, player_idx)
            bonus = min(empty_slots, game.colonists_supply) * 0.15 * decay
        
        elif role == Role.BUILDER:
            # Value depends on doubloons and available buildings
            if p.doubloons >= 1:
                bonus = 0.5 * decay
        
        elif role == Role.CRAFTSMAN:
            # Value depends on production capacity
            total_capacity = sum(
                self._production_capacity(game, player_idx, g) for g in Good
            )
            bonus = total_capacity * 0.3 * decay
        
        elif role == Role.TRADER:
            # Value depends on goods and trade probability
            for good in Good:
                if p.goods[good] > 0:
                    trade_prob = self._trade_probability(game, player_idx, good)
                    price = self._GOOD_TRADE_PRICES[good]
                    bonus += trade_prob * price * self._DOUBLOON_TO_VP
        
        elif role == Role.CAPTAIN:
            # Value depends on shippable goods
            total_goods = sum(p.goods.values())
            bonus = total_goods * 0.4 * decay
        
        elif role in (Role.PROSPECTOR_1, Role.PROSPECTOR_2):
            # Prospector gives 1 doubloon
            bonus = self._DOUBLOON_TO_VP * decay
        
        # Add role money bonus (if any on the role card)
        # Disabled when use_role_doubloon_value=False (for ablation)
        if self._use_role_doubloon_value:
            role_doubloons = game.role_doubloons.get(role, 0)
            bonus += role_doubloons * self._DOUBLOON_TO_VP * decay
        
        return bonus
    
    def _plantation_bonus(self, game, player_idx: int, tile_type: TileType, decay: float) -> float:
        """Estimate value of taking a plantation."""
        if tile_type == TileType.QUARRY:
            # Quarry is valuable for building discounts
            return 0.8 * decay
        
        good = self._PLANTATION_TO_GOOD.get(tile_type)
        if good is None:
            return 0.0
        
        # Value based on good price and whether we have matching production building
        price = self._GOOD_TRADE_PRICES[good]
        has_building = self._has_production_building(game, player_idx, good)
        
        # Higher value if we can produce this good
        if good == Good.CORN:
            # Corn doesn't need building
            return (0.3 + price * 0.1) * decay
        elif has_building:
            return (0.4 + price * 0.15) * decay
        else:
            return (0.2 + price * 0.05) * decay
    
    def _building_bonus(self, game, player_idx: int, building_type: BuildingType, decay: float) -> float:
        """Estimate value of building construction."""
        if building_type in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
            return 0.0
        
        data = BUILDING_DATA.get(building_type)
        if data is None:
            return 0.0
        
        cost, vp, capacity, max_count, is_large, good_produced = data
        
        # Base value from VP
        bonus = vp
        
        # Large building bonus
        if is_large:
            bonus += 2.0  # Large buildings have high end-game potential
        
        # Production building bonus
        if good_produced is not None:
            price = self._GOOD_TRADE_PRICES[good_produced]
            bonus += price * self._DOUBLOON_TO_VP * decay
        
        # Commercial building bonus
        if building_type in self._COMMERCIAL_ABILITY_VALUES:
            # ** Revised: Factory uses dynamic value calculation **
            if building_type == BuildingType.FACTORY:
                ability_value = self._factory_bonus_value(game, player_idx)
            else:
                ability_value = self._COMMERCIAL_ABILITY_VALUES[building_type]
            
            expected_uses = self._expected_role_uses(decay)
            bonus += ability_value * expected_uses
        
        return bonus
    
    def _shipping_bonus(self, game, player_idx: int, ship_idx: int, good: Good, decay: float) -> float:
        """Estimate value of shipping goods."""
        p = game.players[player_idx]
        
        if ship_idx >= len(game.cargo_ships):
            return 0.0
        
        ship = game.cargo_ships[ship_idx]
        qty = min(p.goods[good], ship.capacity - ship.current_load)
        
        # Base VP from shipping
        bonus = qty * 1.0
        
        # Harbor bonus
        has_harbor = any(
            cb.building_type == BuildingType.HARBOR and cb.colonists > 0
            for cb in p.city_board
        )
        if has_harbor:
            bonus += qty * 1.0  # Additional VP per shipment
        
        return bonus
    
    def _store_bonus(self, game, player_idx: int, good: Good, decay: float) -> float:
        """Estimate value of storing a good (Captain Store phase)."""
        p = game.players[player_idx]
        qty = p.goods[good]
        price = self._GOOD_TRADE_PRICES[good]
        
        # Value of keeping the good for future trade/ship
        return qty * max(0.3, price * self._DOUBLOON_TO_VP * 0.6) * decay
    
    def _trade_bonus(self, game, player_idx: int, good: Good, decay: float) -> float:
        """Estimate value of trading a good."""
        p = game.players[player_idx]
        
        if p.goods[good] <= 0:
            return 0.0
        
        price = self._GOOD_TRADE_PRICES[good]
        
        # Check for market bonuses
        small_market_bonus = any(
            cb.building_type == BuildingType.SMALL_MARKET and cb.colonists > 0
            for cb in p.city_board
        )
        large_market_bonus = any(
            cb.building_type == BuildingType.LARGE_MARKET and cb.colonists > 0
            for cb in p.city_board
        )
        
        total_price = price
        if small_market_bonus:
            total_price += 1
        if large_market_bonus:
            total_price += 2
        
        # Convert doubloons to VP equivalent
        return total_price * self._DOUBLOON_TO_VP * decay
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Heuristic Computation (Main Function) - Revised v2.0
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _compute_heuristic(self, game, player_idx: int) -> float:
        """
        Compute the full heuristic value for a player's state.
        
        H(s) = V_realized + V_potential
        
        V_realized: confirmed VP (no decay applied)
        V_potential: potential VP (decay applied)
        """
        p = game.players[player_idx]
        progress = self._game_progress(game)
        decay = max(0.0, 1.0 - progress)

        # ═══════════════════════════════════════════════════════════════════════
        # 1. V_realized (confirmed value) - VP certainly gained at game end
        # ═══════════════════════════════════════════════════════════════════════

        # 1.1 VP chips
        chip_vp = p.vp_chips

        # 1.2 Building base VP and classification
        building_vp = 0.0
        num_violet = 0
        num_large_prod = 0
        num_small_prod = 0

        for b in p.city_board:
            if b.building_type in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                continue
            b_type = b.building_type
            building_vp += BUILDING_DATA[b_type][1]

            # Building classification (for Guildhall, City Hall bonuses)
            if b_type.value in (0, 1):  # SMALL_INDIGO_PLANT, SMALL_SUGAR_MILL
                num_small_prod += 1
            elif b_type.value in (2, 3, 4, 5):  # INDIGO_PLANT ~ COFFEE_ROASTER
                num_large_prod += 1
            elif b_type.value >= 6:  # Violet buildings
                num_violet += 1
        
        # 1.3 Dynamic bonus from activated large buildings
        dynamic_large_vp = 0.0
        for b in p.city_board:
            b_type = b.building_type
            if b_type in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                continue

            # Only when the large building is activated (colonists > 0)
            if b.colonists > 0 and BUILDING_DATA[b_type][4]:  # is_large
                if b_type == BuildingType.CITY_HALL:
                    dynamic_large_vp += num_violet
                elif b_type == BuildingType.CUSTOMS_HOUSE:
                    dynamic_large_vp += chip_vp // 4
                elif b_type == BuildingType.FORTRESS:
                    total_colonists = self._count_total_colonists(p)
                    dynamic_large_vp += total_colonists // 3
                elif b_type == BuildingType.RESIDENCE:
                    island_tiles = sum(1 for tb in p.island_board if tb.tile_type != TileType.EMPTY)
                    dynamic_large_vp += self._residence_bonus(island_tiles)
                elif b_type == BuildingType.GUILDHALL:
                    dynamic_large_vp += (num_large_prod * 2) + (num_small_prod * 1)
        
        v_realized = chip_vp + building_vp + dynamic_large_vp
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. V_potential (potential value) - resources convertible to VP by future actions
        # ═══════════════════════════════════════════════════════════════════════

        # 2.1 V_goods: expected VP conversion of held goods
        # *** Revised: reflect shipping success probability ***
        # Goods become VP only when shipped. Risks: ship space, competition, forced discard
        v_goods = 0.0
        for good in Good:
            qty = p.goods[good]
            if qty > 0:
                unit_value = self._GOOD_UNIT_VALUES[good]
                # Apply shipping success probability
                ship_value = qty * unit_value * self._SHIPPING_SUCCESS_PROB
                # ** Revised: add trade option value of expensive goods (matters early) **
                trade_bonus = qty * self._GOOD_TRADE_BONUS[good] * decay
                v_goods += ship_value + trade_bonus
        # Goods are used in the near future, so apply only weak decay
        v_goods *= (0.5 + 0.5 * decay)

        # 2.2 V_doubloons: expected VP conversion of doubloons
        v_doubloons = p.doubloons * self._DOUBLOON_TO_VP * decay

        # 2.3 V_production: expected future VP of currently active production capacity
        # ** Avoid overlap with V_infrastructure: count only current capacity **
        v_production = 0.0
        expected_craftsman = self._expected_role_uses(decay)

        for good in Good:
            capacity = self._production_capacity(game, player_idx, good)
            if capacity > 0:
                unit_value = self._GOOD_UNIT_VALUES[good]
                # Reflect success probability of the produce -> good -> ship chain
                v_production += capacity * unit_value * expected_craftsman * self._SHIPPING_SUCCESS_PROB

        # 2.4 V_commercial: expected future VP of activated commercial buildings
        v_commercial = 0.0
        expected_uses = self._expected_role_uses(decay)  # uniform distribution assumption

        for cb in p.city_board:
            b_type = cb.building_type
            if b_type in self._COMMERCIAL_ABILITY_VALUES and cb.colonists > 0:
                # Factory uses dynamic calculation
                if b_type == BuildingType.FACTORY:
                    ability_vp = self._factory_bonus_value(game, player_idx)
                else:
                    ability_vp = self._COMMERCIAL_ABILITY_VALUES[b_type]
                
                v_commercial += ability_vp * expected_uses
        
        # 2.5 V_infrastructure: infrastructure potential value (deduplicated)
        v_infrastructure = self._compute_infrastructure_value(game, player_idx, decay)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Total Heuristic
        # ═══════════════════════════════════════════════════════════════════════
        v_potential = v_goods + v_doubloons + v_production + v_commercial + v_infrastructure
        total = v_realized + v_potential
        
        return total
    
    def _compute_infrastructure_value(self, game, player_idx: int, decay: float) -> float:
        """
        Expected VP from activating empty slots / inactive buildings.
        ** Count only "additional production" to avoid overlap with V_production **
        """
        p = game.players[player_idx]
        v_infra = 0.0

        # Reflect colonist placement uncertainty
        COLONIST_DISCOUNT = 0.5

        expected_craftsman = self._expected_role_uses(decay)

        # ─────────────────────────────────────────────────────────────────────
        # (a) Potential production value of empty plantations
        # Needs a matching building slot to increase production when activated
        # ─────────────────────────────────────────────────────────────────────

        # Compute current state per good
        for good in Good:
            # Current occupied plantation count
            occupied_plantations = self._count_occupied_plantations(p, good)
            # Current building slot count (Corn needs no building)
            if good == Good.CORN:
                building_slots = float('inf')  # Corn has no limit
            else:
                building_slots = self._count_building_slots(p, good)

            # Current capacity (already reflected in V_production)
            current_capacity = min(occupied_plantations, building_slots)

            # Empty plantation count
            unoccupied_plantations = self._count_unoccupied_plantations(p, good)

            if unoccupied_plantations > 0:
                # Production that activating empty plantations could add
                # = min(unoccupied, building_slots - current_capacity)
                if good == Good.CORN:
                    additional_capacity = unoccupied_plantations
                else:
                    available_building_headroom = max(0, building_slots - current_capacity)
                    additional_capacity = min(unoccupied_plantations, available_building_headroom)

                if additional_capacity > 0:
                    unit_value = self._GOOD_UNIT_VALUES[good]
                    # Reflect produce -> ship chain success probability
                    v_infra += additional_capacity * unit_value * expected_craftsman * self._SHIPPING_SUCCESS_PROB * COLONIST_DISCOUNT

        # ─────────────────────────────────────────────────────────────────────
        # (b) Potential value of empty building slots
        # ─────────────────────────────────────────────────────────────────────

        for cb in p.city_board:
            b_type = cb.building_type
            if b_type in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                continue

            capacity = BUILDING_DATA[b_type][2]
            empty_slots = capacity - cb.colonists

            if empty_slots <= 0:
                continue

            # (b-1) Production building: valuable only with enough matching plantations
            if b_type in self._PRODUCTION_BUILDING_TO_GOOD:
                good = self._PRODUCTION_BUILDING_TO_GOOD[b_type]

                # Current occupied plantation count
                occupied_plantations = self._count_occupied_plantations(p, good)
                # Current building colonist count (this + other buildings of same good)
                current_building_colonists = self._count_building_slots(p, good)

                # Filling empty slots gains no production if plantations are insufficient
                # effective empty = min(empty_slots, occupied_plantations - current_building_colonists)
                unmatched_plantation_headroom = max(0, occupied_plantations - current_building_colonists)
                effective_empty = min(empty_slots, unmatched_plantation_headroom)

                if effective_empty > 0:
                    unit_value = self._GOOD_UNIT_VALUES[good]
                    v_infra += effective_empty * unit_value * expected_craftsman * self._SHIPPING_SUCCESS_PROB * COLONIST_DISCOUNT

            # (b-2) Commercial building: only the first colonist activates it
            elif b_type in self._COMMERCIAL_ABILITY_VALUES:
                if cb.colonists == 0:  # not yet activated
                    ability_vp = self._COMMERCIAL_ABILITY_VALUES[b_type]
                    expected_uses = self._expected_role_uses(decay)
                    v_infra += ability_vp * expected_uses * COLONIST_DISCOUNT

            # (b-3) Large building: expected bonus VP when activated
            elif BUILDING_DATA[b_type][4]:  # is_large
                if cb.colonists == 0:  # not yet activated
                    estimated_bonus = self._estimate_large_building_bonus(game, player_idx, b_type)
                    v_infra += estimated_bonus * COLONIST_DISCOUNT
        
        return v_infra
    
    def _expected_role_uses(self, decay: float) -> float:
        """
        Expected number of selections of a given role over the remaining game.
        Uniform distribution assumption: every role chosen with equal probability.

        Total role selections: 51 (17 rounds x 3 players)
        Number of roles: 6
        Expected value: 51/6 ≈ 8.5 times x decay
        """
        return self._EXPECTED_ROLE_USES_BASE * decay
    
    def _count_total_colonists(self, p) -> int:
        """Total number of colonists for the player."""
        return (
            p.unplaced_colonists +
            sum(1 for tb in p.island_board if tb.is_occupied) +
            sum(cb.colonists for cb in p.city_board 
                if cb.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE))
        )
    
    def _residence_bonus(self, island_tiles: int) -> int:
        """Compute bonus VP of the Residence building."""
        if island_tiles <= 9:
            return 4
        elif island_tiles == 10:
            return 5
        elif island_tiles == 11:
            return 6
        else:
            return 7
    
    def _count_occupied_plantations(self, p, good: Good) -> int:
        """Occupied plantation count for a given good."""
        for tile_type, g in self._PLANTATION_TO_GOOD.items():
            if g == good:
                return sum(1 for tb in p.island_board 
                          if tb.tile_type == tile_type and tb.is_occupied)
        return 0
    
    def _count_unoccupied_plantations(self, p, good: Good) -> int:
        """Unoccupied plantation count for a given good."""
        for tile_type, g in self._PLANTATION_TO_GOOD.items():
            if g == good:
                return sum(1 for tb in p.island_board 
                          if tb.tile_type == tile_type and not tb.is_occupied)
        return 0
    
    def _count_building_slots(self, p, good: Good) -> int:
        """Total occupied slot count of production buildings for a given good."""
        slots = 0
        for cb in p.city_board:
            if cb.building_type in self._PRODUCTION_BUILDING_TO_GOOD:
                if self._PRODUCTION_BUILDING_TO_GOOD[cb.building_type] == good:
                    slots += cb.colonists
        return slots
    
    def _estimate_large_building_bonus(self, game, player_idx: int, b_type: BuildingType) -> float:
        """Expected bonus VP of an inactive large building."""
        p = game.players[player_idx]

        if b_type == BuildingType.CITY_HALL:
            # Estimate based on current violet building count
            num_violet = sum(1 for b in p.city_board
                           if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)
                           and b.building_type.value >= 6)
            return float(num_violet) + 2.0  # small growth expectation

        elif b_type == BuildingType.CUSTOMS_HOUSE:
            # Estimate based on current VP chips
            return (p.vp_chips // 4) + 2.0

        elif b_type == BuildingType.FORTRESS:
            # Estimate based on current colonist count
            total = self._count_total_colonists(p)
            return (total // 3) + 1.0

        elif b_type == BuildingType.RESIDENCE:
            # Estimate based on current island tile count
            island_tiles = sum(1 for tb in p.island_board if tb.tile_type != TileType.EMPTY)
            return float(self._residence_bonus(island_tiles))

        elif b_type == BuildingType.GUILDHALL:
            # Estimate based on current production building count
            num_small = sum(1 for b in p.city_board if b.building_type.value in (0, 1))
            num_large = sum(1 for b in p.city_board if b.building_type.value in (2, 3, 4, 5))
            return float(num_large * 2 + num_small * 1) + 2.0

        return 5.0  # default estimate
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _game_progress(self, game) -> float:
        """Calculate game progress (0.0 = start, 1.0 = near end)."""
        num_players = len(game.players)
        
        # VP chips depletion
        initial_vp = VP_CHIPS_SETUP.get(num_players, 75)
        vp_progress = 1.0 - (game.vp_chips / max(initial_vp, 1))
        
        # City fill progress
        max_city_fill = 0
        for p in game.players:
            filled = sum(1 for b in p.city_board 
                        if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE))
            max_city_fill = max(max_city_fill, filled)
        city_progress = max_city_fill / 12.0
        
        # Colonist depletion
        initial_colonists = COLONIST_SUPPLY_SETUP.get(num_players, 55)
        colonist_progress = 1.0 - (game.colonists_supply / max(initial_colonists, 1))
        
        return max(vp_progress, city_progress, colonist_progress)
    
    def _production_capacity(self, game, player_idx: int, good: Good) -> int:
        """Calculate production capacity for a specific good."""
        p = game.players[player_idx]
        
        # Find plantation type for this good
        plantation_type = None
        for tile_type, g in self._PLANTATION_TO_GOOD.items():
            if g == good:
                plantation_type = tile_type
                break
        
        if plantation_type is None:
            return 0
        
        # Count occupied plantations
        occupied_plantations = sum(
            1 for tb in p.island_board 
            if tb.tile_type == plantation_type and tb.is_occupied
        )
        
        # Corn doesn't need building
        if good == Good.CORN:
            return occupied_plantations
        
        # Count occupied building slots
        building_slots = 0
        for cb in p.city_board:
            if cb.building_type in self._PRODUCTION_BUILDING_TO_GOOD:
                if self._PRODUCTION_BUILDING_TO_GOOD[cb.building_type] == good:
                    building_slots += cb.colonists
        
        return min(occupied_plantations, building_slots)
    
    def _trade_probability(self, game, player_idx: int, good: Good) -> float:
        """Estimate probability of successfully trading a good."""
        p = game.players[player_idx]
        
        # Check for Office
        has_office = any(
            cb.building_type == BuildingType.OFFICE and cb.colonists > 0
            for cb in p.city_board
        )
        
        # Check if good is already in trading house
        if hasattr(game, 'trading_house') and good in game.trading_house:
            if not has_office:
                return 0.0
        
        # Count competitors ahead in turn order
        # ** Revised: check actual held goods, not production_capacity **
        ahead_competitors = 0
        governor_idx = game.governor_idx
        num_players = len(game.players)
        
        for i in range(num_players):
            if i == player_idx:
                continue
            
            my_order = (player_idx - governor_idx) % num_players
            other_order = (i - governor_idx) % num_players
            
            if other_order < my_order:
                # Check whether the opponent actually holds this good
                other_player = game.players[i]
                if other_player.goods[good] > 0:
                    ahead_competitors += 1
        
        return 1.0 / (1.0 + ahead_competitors * 0.3)
    
    def _has_production_building(self, game, player_idx: int, good: Good) -> bool:
        """Check if player has a production building for the given good."""
        p = game.players[player_idx]
        
        for cb in p.city_board:
            if cb.building_type in self._PRODUCTION_BUILDING_TO_GOOD:
                if self._PRODUCTION_BUILDING_TO_GOOD[cb.building_type] == good:
                    return True
        return False
    
    def _count_empty_slots(self, game, player_idx: int) -> int:
        """Count empty colonist slots (plantations + buildings)."""
        p = game.players[player_idx]
        
        # Empty plantation slots
        empty_plantation_slots = sum(
            1 for tb in p.island_board if not tb.is_occupied
        )
        
        # Empty building slots
        empty_building_slots = 0
        for cb in p.city_board:
            if cb.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                capacity = BUILDING_DATA[cb.building_type][2]
                empty_building_slots += capacity - cb.colonists
        
        return empty_plantation_slots + empty_building_slots
    
    def _factory_bonus_value(self, game, player_idx: int) -> float:
        """
        Calculate Factory building dynamic value.
        Factory gives doubloons based on number of different goods produced.
        """
        goods_types = set()
        for good in Good:
            if self._production_capacity(game, player_idx, good) > 0:
                goods_types.add(good)
        
        num_types = len(goods_types)
        # Factory bonus: 0/1/2/3/5 doubloons for 1/2/3/4/5 good types
        if num_types <= 1:
            doubloon_bonus = 0.0
        elif num_types == 2:
            doubloon_bonus = 1.0
        elif num_types == 3:
            doubloon_bonus = 2.0
        elif num_types == 4:
            doubloon_bonus = 3.0
        else:
            doubloon_bonus = 5.0
        
        return doubloon_bonus * self._DOUBLOON_TO_VP


# ═══════════════════════════════════════════════════════════════════════════════
# Simplified Lookahead Agent (Alternative: Pure State Evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

class ActionValueAgentSimple(ActionValueAgent):
    """
    Simplified version that uses only the base heuristic without action-specific
    bonuses. Useful for comparison or when action semantics are unclear.
    """
    
    def _estimate_action_value(self, game, player_idx: int, action_idx: int,
                                base_heuristic: float) -> float:
        """Simply return base heuristic with small random tie-breaking."""
        return base_heuristic + np.random.uniform(0, 0.001)


# Backward compatibility aliases (deprecated - will be removed in future versions)
DaehanHeuristicAgent = ActionValueAgent
DaehanHeuristicAgentSimple = ActionValueAgentSimple
