"""Tests pinning the decision model: which decisions a player is asked to make,
and which the environment settles on its own.

Each rule names the rulebook passage it follows (Puerto Rico Deluxe rulebook).
The model and these tests follow the research engine this repo shares an
ancestor with (PuertoRico_ESWA, rule/CONFORMANCE.md §2, where the rules are
numbered E1-E11).
"""
import random

import numpy as np
import pytest

from puerto_rico.constants import BuildingType, Good, Phase
from puerto_rico.engine import PuertoRicoGame
from puerto_rico.env import PuertoRicoEnv


def _game(n=3, seed=0):
    g = PuertoRicoGame(n, rng=random.Random(seed))
    g.start_game()
    return g


def _occupy(p, b_type, colonists=1):
    p.build_building(b_type)
    next(b for b in p.city_board if b.building_type == b_type).colonists = colonists


# ── storage keeps whatever fits ──────────────────────────────────────────────

def _storage_turn(goods, warehouses=()):
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    g = env.game
    p = g.players[0]
    for good in Good:
        p.goods[good] = goods.get(good, 0)
    for b in warehouses:
        _occupy(p, b)
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.CAPTAIN_STORE, 0, 0
    g._storage_assignments = {i: {'windrose': None, 'warehouses': []} for i in range(3)}
    env.agent_selection = "player_0"
    return env, g, p


SMALL, LARGE = BuildingType.SMALL_WAREHOUSE, BuildingType.LARGE_WAREHOUSE
C, T, K, S, I = Good.COFFEE, Good.TOBACCO, Good.CORN, Good.SUGAR, Good.INDIGO


@pytest.mark.parametrize("goods, warehouses", [
    ({K: 5}, (SMALL,)),                     # one kind, one slot
    ({K: 2, C: 1}, (SMALL,)),               # corn in the warehouse, the coffee on the windrose
    ({S: 1}, ()),                           # a single good on the windrose
    ({C: 2, T: 2, K: 2}, (SMALL, LARGE)),   # three kinds, three slots
    ({K: 1, S: 3}, (SMALL,)),               # sugar in the warehouse, the corn on the windrose
    ({}, ()),
])
def test_goods_that_fit_are_all_kept_without_a_decision(goods, warehouses):
    """Captain, goods storage: players "must store the goods left on their
    windroses", and only "when players cannot store all their extra goods, they
    may choose which goods to store and which to return to the goods supply".

    The engine used to ask even when everything fitted, and then accepted an
    assignment that threw goods away (1 corn and 3 sugar with a small warehouse:
    sugar on the windrose, corn in the warehouse -> 2 sugar lost).
    """
    env, g, p = _storage_turn(goods, warehouses)
    assert np.flatnonzero(env.valid_action_mask()).tolist() == [15]
    with pytest.raises(ValueError):
        g.action_captain_store_windrose(0, next(iter(goods), Good.CORN))
    assert env._execute_auto_actions() is True
    assert {good: p.goods[good] for good in Good if p.goods[good]} == goods
    assert g.current_player_idx == 1


@pytest.mark.parametrize("goods, warehouses, legal", [
    ({K: 5, S: 2}, (SMALL,), {64 + K, 64 + S, 106 + K, 106 + S}),
    ({K: 5}, (), {64 + K}),
    ({C: 2, T: 2, K: 2, S: 2}, (SMALL, LARGE), {64 + C, 64 + T, 64 + K, 64 + S,
                                                106 + C, 106 + T, 106 + K, 106 + S}),
])
def test_goods_that_do_not_fit_leave_a_choice(goods, warehouses, legal):
    """"When players cannot store all their extra goods, they may choose"."""
    env, g, p = _storage_turn(goods, warehouses)
    assert set(np.flatnonzero(env.valid_action_mask()).tolist()) == legal


def test_both_outcomes_of_a_real_storage_choice():
    for windrose, warehouse, kept in ((S, K, {K: 5, S: 1}), (K, S, {K: 1, S: 2})):
        env, g, p = _storage_turn({K: 5, S: 2}, (SMALL,))
        g.action_captain_store_windrose(0, windrose)
        with pytest.raises(ValueError):
            g.action_captain_store_pass(0)          # the warehouse slot is still free
        g.action_captain_store_warehouse(0, warehouse)
        g.action_captain_store_pass(0)
        assert {good: p.goods[good] for good in Good if p.goods[good]} == kept
