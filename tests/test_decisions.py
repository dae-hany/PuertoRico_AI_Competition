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


# ── captain turns with a wharf ───────────────────────────────────────────────

def _captain_phase(g, captain):
    g.current_phase = Phase.CAPTAIN
    g.active_role_player = captain
    g.current_player_idx = captain
    g._captain_privilege_used = False
    g._captain_done = set()
    g._captain_declined = set()
    g._wharf_used = {i: False for i in range(g.num_players)}
    for p in g.players:
        for good in Good:
            p.goods[good] = 0
    for ship in g.cargo_ships:
        ship.current_load = 0
        ship.good_type = None


def _blocked_coffee_setup():
    """Player 1 owns an occupied wharf and coffee that no cargo ship can take."""
    g = _game()
    _captain_phase(g, 0)
    _occupy(g.players[1], BuildingType.WHARF)
    g.players[1].goods[Good.COFFEE] = 3
    s0, s1, s2 = g.cargo_ships
    s0.good_type, s0.current_load = Good.CORN, s0.capacity       # full
    s1.good_type, s1.current_load = Good.SUGAR, 1
    s2.good_type, s2.current_load = Good.INDIGO, 1
    return g


def test_wharf_owner_who_passed_gets_another_turn_after_a_load():
    """Wharf: "the player may choose when to use it, if at all". A pass used to
    end the owner's loading for the whole phase, so "later" never came."""
    g = _blocked_coffee_setup()
    g.players[2].goods[Good.SUGAR] = 2

    g.action_captain_pass(0)                  # no goods: no further turn
    assert g.current_player_idx == 1
    g.action_captain_pass(1)                  # declines the wharf for now
    g.action_captain_load(2, 1, Good.SUGAR)   # a load after player 1's pass
    assert g.current_player_idx == 1, "player 1 may still choose to use the wharf"
    coffee_supply = g.goods_supply[Good.COFFEE]
    g.action_captain_load(1, -1, Good.COFFEE)
    assert g.players[1].vp_chips == 3
    assert g.goods_supply[Good.COFFEE] == coffee_supply + 3, "wharf goods return to the supply"


def test_loading_ends_after_a_full_cycle_without_a_load():
    g = _blocked_coffee_setup()
    g.action_captain_pass(0)
    g.action_captain_pass(1)
    g.action_captain_pass(2)
    assert g.current_phase == Phase.CAPTAIN_STORE
    assert g.players[1].goods[Good.COFFEE] == 3


# ── a single legal action is taken without asking ────────────────────────────

def test_single_mayor_placement_is_automatic():
    """One colonist and two empty corn plantations: one legal action. The env
    used to ask the agent for it; every other single-action decision was
    already automatic."""
    from puerto_rico.components import IslandTile
    from puerto_rico.constants import TileType

    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    g = env.game
    p = g.players[g.current_player_idx]
    p.island_board = [IslandTile(TileType.CORN_PLANTATION), IslandTile(TileType.CORN_PLANTATION)]
    p.unplaced_colonists = 1
    g.current_phase = Phase.MAYOR
    assert np.flatnonzero(env.valid_action_mask()).tolist() == [120 + TileType.CORN_PLANTATION]
    assert env._execute_auto_actions() is True
    assert sum(t.is_occupied for t in p.island_board) == 1


@pytest.mark.parametrize("num_players", [2, 3])
def test_no_agent_is_asked_a_decision_with_one_legal_action(num_players):
    """Role selection always offers at least two placards, so it is never
    automatic; every other single-action decision is."""
    for seed in range(6):
        env = PuertoRicoEnv(num_players=num_players)
        env.reset(seed=seed)
        rng = random.Random(seed)
        while env.agents:
            name = env.agent_selection
            if env.terminations.get(name, False) or env.truncations.get(name, False):
                env.step(None)
                continue
            legal = np.flatnonzero(env.observe(name)["action_mask"]).tolist()
            assert len(legal) >= 2, (env.game.current_phase, legal)
            env.step(rng.choice(legal))


# ── illegal actions are refused ──────────────────────────────────────────────

def test_step_rejects_actions_outside_the_mask():
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    for action in (14, 15, 199, 170, 16):
        with pytest.raises(ValueError):
            env.step(action)
    env.step(0)                               # Settler
    with pytest.raises(ValueError):
        env.step(1)                           # a role in the middle of a phase


def test_engine_refuses_a_role_in_the_middle_of_a_phase():
    from puerto_rico.constants import Role

    g = _game()
    g.select_role(g.current_player_idx, Role.SETTLER)
    with pytest.raises(ValueError, match="between phases"):
        g.select_role(g.current_player_idx, Role.MAYOR)


def test_second_wharf_use_is_refused_before_anything_changes():
    g = _game()
    _captain_phase(g, 0)
    cap = g.players[0]
    _occupy(cap, BuildingType.WHARF)
    cap.goods[Good.COFFEE] = 2
    g._wharf_used[0] = True
    with pytest.raises(ValueError, match="once per Captain phase"):
        g.action_captain_load(0, -1, Good.COFFEE)
    assert g._captain_privilege_used is False, "the privilege must not be spent"
    assert cap.goods[Good.COFFEE] == 2 and cap.vp_chips == 0


def test_engine_refuses_a_cargo_ship_that_does_not_exist():
    g = _game(n=2)
    _captain_phase(g, 0)
    g.players[0].goods[Good.CORN] = 1
    with pytest.raises(ValueError, match="cargo ship index"):
        g.action_captain_load(0, 2, Good.CORN)    # the 2p game has two ships
