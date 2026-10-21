"""Rule-fidelity tests pinned to the official rulebook, plus the mask/engine invariant.

Each rule test names the rulebook passage it encodes. The engine is not changed
on a hunch: if a test here and the rulebook disagree, the rulebook wins.
"""
import numpy as np
import pytest

from puerto_rico import make_env
from puerto_rico.constants import BuildingType, Phase


def _builder_turn(env, player_idx=0):
    """Put ``player_idx`` on turn in the Builder phase with money to spend."""
    game = env.unwrapped.game
    game.current_phase = Phase.BUILDER
    game.active_role_player = player_idx
    game.current_player_idx = player_idx
    game.players_taken_action = 0
    game.players[player_idx].doubloons = 50
    return game


def test_university_staffs_the_building_it_just_built():
    """University: "...place it on the newly acquired tile."

    (Deluxe rulebook, University.) A large building occupies two city spaces,
    which the engine stores as ``[building, OCCUPIED_SPACE]``. Placing the free
    colonist on the *last* city slot therefore dropped it on the dummy space and
    left the large building unstaffed — silently forfeiting its end-game bonus,
    which is only scored "when they are occupied".
    """
    env = make_env(seed=0, num_players=3)
    game = _builder_turn(env)
    p = game.players[0]

    p.build_building(BuildingType.UNIVERSITY)
    next(b for b in p.city_board
         if b.building_type == BuildingType.UNIVERSITY).colonists = 1

    supply_before = game.colonists_supply
    game.action_builder(0, BuildingType.GUILDHALL)

    guildhall = next(b for b in p.city_board
                     if b.building_type == BuildingType.GUILDHALL)
    dummy = next(b for b in p.city_board
                 if b.building_type == BuildingType.OCCUPIED_SPACE)
    assert guildhall.colonists == 1, "the free colonist must staff the new building"
    assert dummy.colonists == 0, "not its second, dummy city space"
    assert game.colonists_supply == supply_before - 1


def test_university_free_colonist_also_works_for_small_buildings():
    env = make_env(seed=0, num_players=3)
    game = _builder_turn(env)
    p = game.players[0]

    p.build_building(BuildingType.UNIVERSITY)
    next(b for b in p.city_board
         if b.building_type == BuildingType.UNIVERSITY).colonists = 1

    game.action_builder(0, BuildingType.SMALL_MARKET)
    market = next(b for b in p.city_board
                  if b.building_type == BuildingType.SMALL_MARKET)
    assert market.colonists == 1


def test_staffed_large_building_scores_its_end_game_bonus():
    """The consequence the University bug used to hide: an unstaffed large
    building scores its printed 4 VP but none of its bonus."""
    env = make_env(seed=0, num_players=3)
    game = env.unwrapped.game
    p = game.players[0]
    p.build_building(BuildingType.CITY_HALL)
    p.build_building(BuildingType.SMALL_MARKET)      # one violet building

    unstaffed = game.get_scores()[0]
    next(b for b in p.city_board
         if b.building_type == BuildingType.CITY_HALL).colonists = 1
    staffed = game.get_scores()[0]

    assert staffed[4] > unstaffed[4], "City Hall bonus only counts when staffed"


def test_engine_rejecting_a_move_is_a_hard_failure_not_a_silent_loss():
    """A move the engine refuses used to be swallowed as "-10 and end the game".

    That turned a bug in this repo into a corrupted result. It must raise.
    Agents cannot reach this path: ``tournament.match`` replaces an illegal
    agent move with a random legal one before the env ever sees it, and the
    env itself refuses a move outside the mask (next test). So the mask is
    made to lie here, to reach the case it guards: a mask/engine mismatch.
    """
    env = make_env(seed=0, num_players=3)
    base = env.unwrapped
    assert base.game.current_phase == Phase.END_ROUND   # role selection

    lying = np.zeros(200, dtype=bool)
    lying[39] = True              # "sell a good" while nobody is in the Trader phase
    base.valid_action_mask = lambda: lying
    with pytest.raises(AssertionError, match="engine rejected it"):
        env.step(39)


def test_an_action_outside_the_mask_is_refused_and_changes_nothing():
    env = make_env(seed=0, num_players=3)
    base = env.unwrapped
    before = (base.game.current_phase, base.game.current_player_idx,
              base.game.round_number, dict(base._cumulative_rewards))
    for action in (39, 14, 15, 199, 200, -1):
        with pytest.raises(ValueError, match="Illegal action"):
            env.step(action)
    assert (base.game.current_phase, base.game.current_player_idx,
            base.game.round_number, dict(base._cumulative_rewards)) == before


@pytest.mark.parametrize("num_players", [2, 3])
def test_mask_and_engine_agree_over_random_play(num_players):
    """The invariant the assertion above protects: every mask-legal action is
    one the engine accepts."""
    for g in range(12):
        env = make_env(seed=900 + g, num_players=num_players)
        rng = np.random.default_rng(g)
        while env.agents:
            name = env.agent_selection
            if env.terminations.get(name, False) or env.truncations.get(name, False):
                env.step(None)
                continue
            mask = np.asarray(env.observe(name)["action_mask"])
            env.step(int(rng.choice(np.where(mask > 0.5)[0])))


# ── setup ────────────────────────────────────────────────────────────────────

# doubloons, starting plantations from the governor, face-up tiles, quarries,
# VP chips, colonist ship, colonist supply, cargo ships, role placards
_SETUP = {
    2: (3, ["INDIGO_PLANTATION", "CORN_PLANTATION"], 3, 5, 65, 2, 40, [4, 6], 7),
    3: (2, ["INDIGO_PLANTATION"] * 2 + ["CORN_PLANTATION"], 4, 8, 75, 3, 55, [4, 5, 6], 6),
    4: (3, ["INDIGO_PLANTATION"] * 2 + ["CORN_PLANTATION"] * 2, 5, 8, 100, 4, 75,
        [5, 6, 7], 7),
    5: (4, ["INDIGO_PLANTATION"] * 3 + ["CORN_PLANTATION"] * 2, 6, 8, 126, 5, 95,
        [6, 7, 8], 8),
}


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_setup_matches_the_rulebook(n):
    """Deluxe rulebook, Preparation: "the VP chips ... with 3 players: 75 total
    points - with 4 players: 100 total points - with 5 players: 126 total points"
    (the box holds "31 x 1 and 19 x 5" chips, 126 VP), and the 2-player
    variant's "65 total VP". The engine had 122 for five players.
    """
    import random
    from collections import Counter

    from puerto_rico.constants import BUILDING_DATA, Good, Role, TileType
    from puerto_rico.engine import PuertoRicoGame

    doubloons, starts, face_up, quarries, vp, ship, supply, ships, roles = _SETUP[n]
    g = PuertoRicoGame(n, rng=random.Random(7))
    g.start_game()

    assert all(p.doubloons == doubloons for p in g.players)
    for k, kind in enumerate(starts):
        p = g.players[(g.governor_idx + k) % n]
        assert [(t.tile_type, t.is_occupied) for t in p.island_board] == [(TileType[kind], False)]
    assert len(g.face_up_plantations) == face_up
    assert g.quarry_stack == quarries
    assert g.vp_chips == vp
    assert g.colonists_ship == ship
    assert g.colonists_supply == supply
    assert [s.capacity for s in g.cargo_ships] == ships
    assert len(g.available_roles) == roles
    prospectors = {2: 1, 3: 0, 4: 1, 5: 2}[n]
    assert sum(r in (Role.PROSPECTOR_1, Role.PROSPECTOR_2) for r in g.available_roles) == prospectors

    removed = 2 if n == 2 else 0
    assert g.goods_supply == {Good.COFFEE: 9 - removed, Good.TOBACCO: 9 - removed,
                              Good.CORN: 10 - removed, Good.SUGAR: 11 - removed,
                              Good.INDIGO: 11 - removed}
    plantations = {TileType.COFFEE_PLANTATION: 8, TileType.TOBACCO_PLANTATION: 9,
                   TileType.CORN_PLANTATION: 10, TileType.SUGAR_PLANTATION: 11,
                   TileType.INDIGO_PLANTATION: 12}
    expected = Counter({k: c - (3 if n == 2 else 0) for k, c in plantations.items()})
    expected.subtract(Counter(TileType[k] for k in starts))
    assert Counter(g.plantation_stack) + Counter(g.face_up_plantations) == +expected
    assert not g.plantation_discard

    for b in BuildingType:
        if b in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
            continue
        production = BUILDING_DATA[b][5] is not None
        if n == 2:
            copies = 2 if production else 1
        elif production:
            copies = 4 if b in (BuildingType.SMALL_INDIGO_PLANT, BuildingType.SMALL_SUGAR_MILL) else 3
        else:
            copies = 1 if BUILDING_DATA[b][4] else 2
        assert g.building_supply[b] == copies, b.name
