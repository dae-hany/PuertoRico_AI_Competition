"""Tests pinning the decision model: which decisions a player is asked to make,
and which the environment settles on its own.

Each rule names the rulebook passage it follows (Puerto Rico Deluxe rulebook).
The model and these tests follow the research engine this repo shares an
ancestor with (PuertoRico_ESWA, rule/CONFORMANCE.md §2, where the rules are
numbered E1-E11).
"""
import copy
import pickle
import random

import numpy as np
import pytest

from puerto_rico import flatten_observation
from puerto_rico.constants import Benefit, BuildingType, Good, Phase, Role, TileType
from puerto_rico.engine import PuertoRicoGame
from puerto_rico.env import ACCEPT_BENEFIT, DECLINE_BENEFIT, PuertoRicoEnv
from puerto_rico.observation import PENDING_BENEFIT_OFFSET


def _game(n=3, seed=0):
    g = PuertoRicoGame(n, rng=random.Random(seed))
    g.start_game()
    return g


def _occupy(p, b_type, colonists=1):
    p.build_building(b_type)
    next(b for b in p.city_board if b.building_type == b_type).colonists = colonists


# ── optional benefits are accept/decline decisions ──────────────────────────

@pytest.mark.parametrize("accept", [True, False])
def test_mayor_privilege_is_offered_before_distribution(accept):
    """Mayor: "The player who chooses this role may first take one colonist from
    the colonist supply (not from the colonist ship!) as the privilege."
    """
    g = _game()
    mayor = g.current_player_idx
    supply, ship = g.colonists_supply, g.colonists_ship
    g.select_role(mayor, Role.MAYOR)
    assert g.pending_benefits == [Benefit.MAYOR_PRIVILEGE]
    assert all(p.unplaced_colonists == 0 for p in g.players), "distribution waits for the decision"

    g.resolve_benefit(mayor, accept=accept)
    assert g.colonists_supply == supply - (1 if accept else 0)
    assert g.colonists_ship == 0
    shares = [g.players[(mayor + k) % 3].unplaced_colonists for k in range(3)]
    assert shares == [ship // 3 + (1 if accept else 0), ship // 3, ship // 3]


def test_mayor_privilege_is_not_offered_from_an_empty_supply():
    """Mayor: "When the colonist supply runs out, the mayor may not use this
    privilege"."""
    g = _game()
    g.colonists_supply = 0
    mayor = g.current_player_idx
    g.select_role(mayor, Role.MAYOR)
    assert not g.pending_benefits
    assert g.players[mayor].unplaced_colonists == 1


@pytest.mark.parametrize("accept", [True, False])
def test_hospice_colonist_is_optional(accept):
    """Hospice: "they may take a colonist from the colonist supply and place it on
    the newly acquired plantation or quarry tile."
    """
    g = _game()
    p = g.players[0]
    _occupy(p, BuildingType.HOSPICE)
    g.face_up_plantations = [TileType.SUGAR_PLANTATION]
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 1
    supply = g.colonists_supply

    g.action_settler(0, tile_choice=0)
    assert g.pending_benefits == [Benefit.HOSPICE]
    assert g.current_player_idx == 0, "the turn waits for the decision"
    g.resolve_benefit(0, accept=accept)
    assert p.island_board[-1].is_occupied is accept
    assert g.colonists_supply == supply - (1 if accept else 0)
    assert g.current_player_idx == 1


def test_hospice_does_not_apply_to_the_hacienda_tile():
    """Hospice: "players who also own an occupied hacienda and choose to take the
    additional facedown or quarry tile do not get a colonist for the extra tile."
    """
    g = _game()
    p = g.players[0]
    _occupy(p, BuildingType.HOSPICE)
    _occupy(p, BuildingType.HACIENDA)
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 1
    g.action_hacienda_draw(0)
    assert not g.pending_benefits
    assert not p.island_board[-1].is_occupied


@pytest.mark.parametrize("choice, placed", [
    (0, TileType.SUGAR_PLANTATION),           # a face-up plantation
    (-1, TileType.QUARRY),                    # a quarry (the settler's privilege)
    (-2, None),                               # pass: only the hacienda tile is placed
])
def test_hospice_colonist_goes_on_the_regular_tile_after_a_hacienda_draw(choice, placed):
    """Hospice: the colonist goes on "the newly acquired plantation or quarry
    tile", never on the hacienda's extra tile, which is placed first.
    """
    g = _game()
    p = g.players[0]
    _occupy(p, BuildingType.HOSPICE)
    _occupy(p, BuildingType.HACIENDA)
    g.face_up_plantations = [TileType.SUGAR_PLANTATION]
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 0
    start, supply = len(p.island_board), g.colonists_supply

    g.action_hacienda_draw(0)
    hacienda_tile = p.island_board[start]
    g.action_settler(0, tile_choice=choice)
    if placed is None:
        assert not g.pending_benefits
        assert len(p.island_board) == start + 1
    else:
        assert g.pending_benefits == [Benefit.HOSPICE]
        g.resolve_benefit(0, accept=True)
        assert p.island_board[-1].tile_type == placed
        assert p.island_board[-1].is_occupied
        assert g.colonists_supply == supply - 1
    assert not hacienda_tile.is_occupied


def test_hospice_and_university_fall_back_to_the_colonist_ship():
    """Hospice, University: "If there are no more colonists in the colonist
    supply, the player may take one from the colonist ship. If there are also
    none there, the player takes none."
    """
    g = _game()
    p = g.players[0]
    _occupy(p, BuildingType.HOSPICE)
    g.face_up_plantations = [TileType.CORN_PLANTATION, TileType.CORN_PLANTATION]
    g.colonists_supply, g.colonists_ship = 0, 3
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 1
    g.action_settler(0, tile_choice=0)
    g.resolve_benefit(0, accept=True)
    assert p.island_board[-1].is_occupied and g.colonists_ship == 2

    g.colonists_ship = 0                      # nothing anywhere: nothing offered
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 1
    g.action_settler(0, tile_choice=0)
    assert not g.pending_benefits and not p.island_board[-1].is_occupied

    g = _game()
    p = g.players[0]
    p.doubloons = 10
    _occupy(p, BuildingType.UNIVERSITY)
    g.colonists_supply, g.colonists_ship = 0, 3
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.BUILDER, 0, 1
    g.action_builder(0, BuildingType.SMALL_MARKET)
    g.resolve_benefit(0, accept=True)
    market = next(b for b in p.city_board if b.building_type == BuildingType.SMALL_MARKET)
    assert market.colonists == 1 and g.colonists_ship == 2


@pytest.mark.parametrize("accept", [True, False])
def test_university_colonist_is_optional(accept):
    """University: "they may take a colonist from the colonist supply and place it
    on the newly acquired tile."
    """
    g = _game()
    p = g.players[0]
    p.doubloons = 10
    _occupy(p, BuildingType.UNIVERSITY)
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.BUILDER, 0, 1
    supply = g.colonists_supply

    g.action_builder(0, BuildingType.SMALL_MARKET)
    assert g.pending_benefits == [Benefit.UNIVERSITY]
    g.resolve_benefit(0, accept=accept)
    market = next(b for b in p.city_board if b.building_type == BuildingType.SMALL_MARKET)
    assert market.colonists == (1 if accept else 0)
    assert g.colonists_supply == supply - (1 if accept else 0)
    assert g.current_player_idx == 1


def test_a_university_built_now_offers_nothing():
    g = _game()
    p = g.players[0]
    p.doubloons = 10
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.BUILDER, 0, 1
    g.action_builder(0, BuildingType.UNIVERSITY)
    assert not g.pending_benefits


@pytest.mark.parametrize("accept", [True, False])
def test_captain_privilege_belongs_to_the_first_load(accept):
    """Captain: "When loading the first kind of goods, the captain takes as the
    privilege 1 extra VP. The captain does not get this extra VP for the goods
    loaded in subsequent turns". Optional, as every privilege is ("they may
    choose not to use the action or privilege of the placard chosen"); a
    declined privilege is used up."""
    g = _game()
    _captain_phase(g, 0)
    cap = g.players[0]
    cap.goods[Good.CORN] = 2
    cap.goods[Good.SUGAR] = 1
    vp_supply = g.vp_chips

    g.action_captain_load(0, 0, Good.CORN)
    assert g.pending_benefits == [Benefit.CAPTAIN_PRIVILEGE]
    g.resolve_benefit(0, accept=accept)
    assert cap.vp_chips == 2 + (1 if accept else 0)

    # Others cannot load; the captain loads again without a second privilege.
    for _ in range(2):
        g.action_captain_pass(g.current_player_idx)
    g.action_captain_load(0, 1, Good.SUGAR)
    assert not g.pending_benefits
    assert cap.vp_chips == 3 + (1 if accept else 0)
    assert g.vp_chips == vp_supply - cap.vp_chips


def test_captain_privilege_comes_before_the_harbor():
    """Harbor: "Each time ... owners of an occupied harbor load goods on a cargo
    ship, they earn one extra VP"; beige functions are optional ("Players are not
    required to use the special function of an occupied beige building")."""
    g = _game()
    _captain_phase(g, 0)
    cap = g.players[0]
    _occupy(cap, BuildingType.HARBOR)
    cap.goods[Good.CORN] = 1
    g.action_captain_load(0, 0, Good.CORN)
    assert g.pending_benefits == [Benefit.CAPTAIN_PRIVILEGE, Benefit.HARBOR]
    g.resolve_benefit(0, accept=False)
    assert g.pending_benefits == [Benefit.HARBOR]
    g.resolve_benefit(0, accept=True)
    assert cap.vp_chips == 2
    assert g.current_player_idx == 1


def test_benefit_decision_admits_only_accept_and_decline():
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    mayor = env.game.current_player_idx
    env.step(Role.MAYOR)
    obs = env.observe(env.agent_selection)
    assert env.agent_selection == f"player_{mayor}"
    assert np.flatnonzero(obs["action_mask"]).tolist() == [ACCEPT_BENEFIT, DECLINE_BENEFIT]
    pending = obs["observation"]["global_state"]["pending_benefit"]
    assert pending.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    flat = flatten_observation(obs["observation"])
    assert flat[PENDING_BENEFIT_OFFSET:PENDING_BENEFIT_OFFSET + 5].tolist() == pending.tolist()
    with pytest.raises(ValueError):
        env.step(120 + TileType.INDIGO_PLANTATION)
    with pytest.raises(ValueError):
        env.game.action_mayor_place_colonist(mayor, is_city=False, slot_idx=0)
    env.step(DECLINE_BENEFIT)
    assert not env.game.pending_benefits
    assert not env.observe(env.agent_selection)["observation"]["global_state"]["pending_benefit"].any()


def _pending_hospice_game():
    g = _game()
    p = g.players[0]
    _occupy(p, BuildingType.HOSPICE)
    g.face_up_plantations = [TileType.CORN_PLANTATION]
    g.current_phase, g.current_player_idx, g.active_role_player = Phase.SETTLER, 0, 1
    g.action_settler(0, tile_choice=0)
    assert g.pending_benefits == [Benefit.HOSPICE]
    return g


@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
def test_a_copy_resolves_its_pending_benefit_on_its_own_board(how):
    """The colonist's target is kept as a board position, not an object: the
    hand-rolled __deepcopy__ rebuilds the boards, so an object reference would
    have pointed outside the copy and the colonist would have gone nowhere."""
    g = _pending_hospice_game()
    clone = copy.deepcopy(g) if how == "deepcopy" else pickle.loads(pickle.dumps(g))
    clone.resolve_benefit(0, accept=True)
    assert clone.players[0].island_board[-1].is_occupied
    assert not g.players[0].island_board[-1].is_occupied
    assert g.pending_benefits == [Benefit.HOSPICE]


def test_forward_model_clones_resolve_benefits_like_the_real_game():
    """Search clones the env at benefit decisions (hospice and university
    included): the clone must play on exactly as the real game would."""
    from puerto_rico import ForwardModel, make_env

    seen = set()
    for seed in range(40):
        env = make_env(seed=seed, num_players=3)
        fm = ForwardModel(env)
        rng = random.Random(seed)
        while not fm.is_terminal():
            game = env.unwrapped.game
            if game.pending_benefits and game.pending_benefits[0] in (Benefit.HOSPICE,
                                                                      Benefit.UNIVERSITY):
                seen.add(game.pending_benefits[0])
                clone = copy.deepcopy(env)
                for e in (env, clone):
                    e.step(ACCEPT_BENEFIT)
                assert flatten_observation(clone.observe(clone.agent_selection)["observation"]
                                           ).tobytes() == flatten_observation(
                    env.observe(env.agent_selection)["observation"]).tobytes()
                continue
            fm.step(rng.choice(fm.legal_actions()))
        if len(seen) == 2:
            break
    assert seen == {Benefit.HOSPICE, Benefit.UNIVERSITY}


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
    """Captain: "When players cannot store all their extra goods, they may
    choose"."""
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
