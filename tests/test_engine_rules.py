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
    agent move with a random legal one before the env ever sees it.
    """
    env = make_env(seed=0, num_players=3)
    assert env.unwrapped.game.current_phase == Phase.END_ROUND   # role selection

    with pytest.raises(AssertionError, match="engine rejected it"):
        env.step(39)              # "sell a good" while nobody is in the Trader phase


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
