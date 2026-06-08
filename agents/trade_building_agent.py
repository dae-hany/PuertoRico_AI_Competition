"""
ActionValueAgent_TradeBuild — Trade→Building VP Maximization Variant

═══════════════════════════════════════════════════════════════════════════════
STRATEGIC IDENTITY
═══════════════════════════════════════════════════════════════════════════════

This agent inherits the proven ActionValueAgent engine and modifies its
heuristic weights to favor the Trade→Building VP conversion path over
the default balanced approach.

  Path A (ActionValueAgent default):  Goods → Captain → VP chips  (balanced)
  Path B (THIS AGENT):               Goods → Trader → Doubloons → Buildings → Building VP

Key weight modifications vs ActionValueAgent:

  1. DOUBLOON VALUE AMPLIFIED
     - _DOUBLOON_TO_VP: 0.25 → 0.45  (doubloons are worth more because
       they convert into buildings which give permanent VP)

  2. TRADE INFRASTRUCTURE BOOSTED
     - Small Market ability: 0.25 → 0.60  (+1 doubloon per trade = huge ROI)
     - Large Market ability: 0.50 → 1.00  (+2 doubloons per trade)
     - Office ability:       0.20 → 0.50  (duplicate good trading flexibility)
     - Good trade bonuses:   ×2.5         (Coffee 0.5→1.2, Tobacco 0.4→1.0, etc.)

  3. SHIPPING VALUE REDUCED
     - _SHIPPING_SUCCESS_PROB: 0.70 → 0.50  (shipping is less attractive)
     - Harbor ability:  1.0 → 0.4  (less reason to pursue Harbor)
     - Wharf ability:   1.5 → 0.6  (less reason to pursue Wharf)

  4. BUILDING VALUE BOOSTED
     - Large buildings: 2.0 → 3.5 bonus  (endgame building VP amplified)
     - Quarry plantation: 0.8 → 1.3      (quarries = building discounts)
     - Builder role: effective bonus ×1.8

  5. CAPTAIN ROLE DEVALUED
     - Captain role selection bonus: goods × 0.4 → goods × 0.25
     - Captain still participates when forced (phase actions unchanged)

═══════════════════════════════════════════════════════════════════════════════
DESIGN RATIONALE
═══════════════════════════════════════════════════════════════════════════════

  vs ActionValueAgent (default):
    - Default agent balances all VP axes equally
    - This variant biases toward Trade→Build path, creating a distinct
      strategic profile that the PPO agent has never trained against

  vs ShippingRushAgent:
    - ShippingRush maximizes Captain/shipping VP
    - This agent is orthogonal: maximizes Trader→Builder VP

  Why it should pass benchmark:
    - Inherits ALL phase handling from ActionValueAgent (proven to work)
    - Only heuristic weights change, not the decision architecture
    - Still ships goods when Captain phase occurs (action bonuses unchanged)
    - The bias is moderate enough to not cripple play

═══════════════════════════════════════════════════════════════════════════════
INTERFACE
═══════════════════════════════════════════════════════════════════════════════

Identical to ActionValueAgent — requires on_game_start() before use:
  - act(observation, action_mask) -> int
  - on_game_start(forward_model=None)  (set_env alias)
  - Compatible with GameEvaluator (same dispatch as ActionValueAgent)

═══════════════════════════════════════════════════════════════════════════════
"""
from puerto_rico.constants import BuildingType, Good
from agents.action_value_agent import ActionValueAgent


class TradeBuildingAgent(ActionValueAgent):
    """
    ActionValueAgent variant biased toward Trade→Building VP path.
    Inherits proven one-step lookahead engine, only modifies heuristic weights.
    """

    name = "TradeBuilding"

    # ═══════════════════════════════════════════════════════════════════════════
    # Override: Doubloon value amplified (buildings are the goal)
    # ═══════════════════════════════════════════════════════════════════════════
    _DOUBLOON_TO_VP = 0.45  # Default: 0.25 — doubloons convert to permanent VP

    # ═══════════════════════════════════════════════════════════════════════════
    # Override: Shipping probability reduced (we prefer trading over shipping)
    # ═══════════════════════════════════════════════════════════════════════════
    _SHIPPING_SUCCESS_PROB = 0.50  # Default: 0.70

    # ═══════════════════════════════════════════════════════════════════════════
    # Override: Trade bonus amplified (expensive goods → high doubloon trades)
    # ═══════════════════════════════════════════════════════════════════════════
    _GOOD_TRADE_BONUS = {
        Good.COFFEE:  1.2,   # Default: 0.5 — Coffee trade = 4+ doubloons
        Good.TOBACCO: 1.0,   # Default: 0.4 — Tobacco trade = 3+ doubloons
        Good.SUGAR:   0.5,   # Default: 0.2
        Good.INDIGO:  0.25,  # Default: 0.1
        Good.CORN:    0.0,   # Default: 0.0 — Corn can't be traded
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # Override: Commercial building ability values
    # Trade infrastructure boosted, shipping infrastructure reduced
    # ═══════════════════════════════════════════════════════════════════════════
    _COMMERCIAL_ABILITY_VALUES = {
        # Trade infrastructure — BOOSTED
        BuildingType.SMALL_MARKET:     0.60,   # Default: 0.25 — +1 doubloon/trade
        BuildingType.LARGE_MARKET:     1.00,   # Default: 0.50 — +2 doubloons/trade
        BuildingType.OFFICE:           0.50,   # Default: 0.20 — duplicate good trading
        BuildingType.FACTORY:          0.80,   # Default: 0.50 — doubloon machine

        # Shipping infrastructure — REDUCED
        BuildingType.HARBOR:           0.40,   # Default: 1.0 — less shipping focus
        BuildingType.WHARF:            0.60,   # Default: 1.5 — still useful but deprioritized
        BuildingType.SMALL_WAREHOUSE:  0.20,   # Default: 0.3
        BuildingType.LARGE_WAREHOUSE:  0.30,   # Default: 0.5

        # Settler infrastructure — slightly boosted (quarries help building)
        BuildingType.HACIENDA:         0.20,   # Default: 0.15
        BuildingType.CONSTRUCTION_HUT: 0.30,   # Default: 0.15 — quarry access = discounts
        BuildingType.HOSPICE:          0.25,   # Default: 0.20
        BuildingType.UNIVERSITY:       0.30,   # Default: 0.20 — free colonist on build
    }

    def _role_selection_bonus(self, game, player_idx, role, decay):
        """
        Override role selection to bias toward Trade→Build cycle.
        
        Key changes vs parent:
        - Builder bonus: ×1.8 (when affordable)
        - Trader bonus: trade income valued higher
        - Captain bonus: reduced (goods × 0.25 instead of 0.40)
        - Quarry on Settler: boosted
        """
        from puerto_rico.constants import Role

        p = game.players[player_idx]
        bonus = 0.0

        if role == Role.SETTLER:
            if p.empty_island_spaces > 0:
                bonus = 0.4 * decay  # Default: 0.3

        elif role == Role.MAYOR:
            empty_slots = self._count_empty_slots(game, player_idx)
            bonus = min(empty_slots, game.colonists_supply) * 0.15 * decay

        elif role == Role.BUILDER:
            # BOOSTED: building is our primary VP engine
            if p.doubloons >= 1:
                bonus = 0.9 * decay  # Default: 0.5

        elif role == Role.CRAFTSMAN:
            total_capacity = sum(
                self._production_capacity(game, player_idx, g) for g in Good
            )
            bonus = total_capacity * 0.3 * decay  # Same as default

        elif role == Role.TRADER:
            # BOOSTED: trade income valued higher via _DOUBLOON_TO_VP
            for good in Good:
                if p.goods[good] > 0:
                    trade_prob = self._trade_probability(game, player_idx, good)
                    price = self._GOOD_TRADE_PRICES[good]
                    bonus += trade_prob * price * self._DOUBLOON_TO_VP  # 0.45 vs 0.25

        elif role == Role.CAPTAIN:
            # REDUCED: shipping is secondary VP path for this variant
            total_goods = sum(p.goods.values())
            bonus = total_goods * 0.25 * decay  # Default: 0.40

        elif role in (Role.PROSPECTOR_1, Role.PROSPECTOR_2):
            bonus = self._DOUBLOON_TO_VP * decay  # Higher with our _DOUBLOON_TO_VP

        # Role money bonus (accumulated doubloons on role card)
        role_doubloons = game.role_doubloons.get(role, 0)
        bonus += role_doubloons * self._DOUBLOON_TO_VP * decay

        return bonus

    def _plantation_bonus(self, game, player_idx, tile_type, decay):
        """Override: Quarry value boosted (building discounts are key)."""
        from puerto_rico.constants import TileType as TT

        if tile_type == TT.QUARRY:
            return 1.3 * decay  # Default: 0.8 — quarries = cheaper buildings

        good = self._PLANTATION_TO_GOOD.get(tile_type)
        if good is None:
            return 0.0

        price = self._GOOD_TRADE_PRICES[good]
        has_building = self._has_production_building(game, player_idx, good)

        if good == Good.CORN:
            return (0.3 + price * 0.1) * decay
        elif has_building:
            # Boost plantation value when matching building exists
            return (0.5 + price * 0.20) * decay  # Default: 0.4 + price*0.15
        else:
            return (0.2 + price * 0.05) * decay

    def _building_bonus(self, game, player_idx, building_type, decay):
        """Override: Large building bonus increased."""
        from puerto_rico.constants import BUILDING_DATA, BuildingType as BT

        if building_type in (BT.EMPTY, BT.OCCUPIED_SPACE):
            return 0.0

        data = BUILDING_DATA.get(building_type)
        if data is None:
            return 0.0

        cost, vp, capacity, max_count, is_large, good_produced = data

        bonus = vp

        # BOOSTED large building bonus
        if is_large:
            bonus += 3.5  # Default: 2.0

        if good_produced is not None:
            price = self._GOOD_TRADE_PRICES[good_produced]
            bonus += price * self._DOUBLOON_TO_VP * decay

        if building_type in self._COMMERCIAL_ABILITY_VALUES:
            if building_type == BT.FACTORY:
                ability_value = self._factory_bonus_value(game, player_idx)
            else:
                ability_value = self._COMMERCIAL_ABILITY_VALUES[building_type]

            expected_uses = self._expected_role_uses(decay)
            bonus += ability_value * expected_uses

        return bonus
