"""Play seeded games in this repo's engine and the reference model side by side.

``verification/oracle.py`` is an independent model of the rules, written from
``verification/rule/RULES.md`` and ``verification/rule/CONFORMANCE.md`` without
reference to any engine code (see ``verification/README.md``). Before every
decision the two must agree on the set of legal actions and on the canonical
state of CONFORMANCE.md §4.2; at the end of the game also on the scores, the
winners and the rewards. Usage:

    python -m verification.lockstep --games 500 --players 2,3 --workers 8
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from puerto_rico.constants import BuildingType, Good, Phase, TileType
from puerto_rico.env import ACCEPT_BENEFIT, DECLINE_BENEFIT, PuertoRicoEnv
from verification.oracle import Oracle

TILE_NAME = {
    TileType.COFFEE_PLANTATION: "COFFEE", TileType.TOBACCO_PLANTATION: "TOBACCO",
    TileType.CORN_PLANTATION: "CORN", TileType.SUGAR_PLANTATION: "SUGAR",
    TileType.INDIGO_PLANTATION: "INDIGO", TileType.QUARRY: "QUARRY",
}
BUILDINGS = [b for b in BuildingType if b not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)]
PASS = 15
STYLES = ("default", "builder", "uniform")


def engine_canonical(env: PuertoRicoEnv) -> dict:
    """This engine's state in the canonical form of CONFORMANCE.md §4.2.

    The engine waits for a role choice in ``Phase.END_ROUND`` (``None`` before
    the first one) and stores in ``Phase.CAPTAIN_STORE``; the canonical phases
    are ``ROLE`` and ``STORE``. An island tile is one circle, so the island is
    counted per kind as [tiles, occupied tiles]. A large building's second
    space (``OCCUPIED_SPACE``) is not a building.
    """
    g = env.game
    over = bool(env.terminations) and all(env.terminations.values())
    if over:
        phase = "OVER"
    elif g.current_phase in (None, Phase.END_ROUND):
        phase = "ROLE"
    elif g.current_phase == Phase.CAPTAIN_STORE:
        phase = "STORE"
    else:
        phase = g.current_phase.name
    in_phase = phase not in ("ROLE", "OVER")
    players = []
    for p in g.players:
        island = {}
        for t in p.island_board:
            entry = island.setdefault(TILE_NAME[t.tile_type], [0, 0])
            entry[0] += 1
            entry[1] += int(t.is_occupied)
        players.append({
            "doubloons": p.doubloons,
            "vp": p.vp_chips,
            "goods": {good.name: p.goods[good] for good in Good},
            "island": island,
            "city": {b.building_type.name: b.colonists for b in p.city_board
                     if b.building_type != BuildingType.OCCUPIED_SPACE},
            "unplaced": p.unplaced_colonists,
        })
    scores = winners = None
    if over:
        scores = [[s[0], s[1]] for s in g.get_scores()]
        best = max(tuple(s) for s in scores)
        winners = [i for i, s in enumerate(scores) if tuple(s) == best]
    return {
        "num_players": g.num_players,
        "over": over,
        "phase": phase,
        "benefit": g.pending_benefits[0].name if g.pending_benefits and not over else None,
        "current_player": None if over else g.current_player_idx,
        "governor": g.governor_idx,
        "rounds_completed": g.round_number,
        "active_role": g.active_role.name if in_phase else None,
        "role_player": g.active_role_player if in_phase else None,
        "roles_available": sorted(r.name for r in g.available_roles),
        "role_doubloons": {r.name: v for r, v in g.role_doubloons.items()},
        "vp_supply": g.vp_chips,
        "colonist_supply": g.colonists_supply,
        "colonist_ship": g.colonists_ship,
        "quarry_supply": g.quarry_stack,
        "goods_supply": {good.name: g.goods_supply[good] for good in Good},
        "building_supply": {b.name: g.building_supply[b] for b in BUILDINGS},
        "plantation_stack": [TILE_NAME[t] for t in g.plantation_stack],
        "face_up": sorted(TILE_NAME[t] for t in g.face_up_plantations),
        "discard": sorted(TILE_NAME[t] for t in g.plantation_discard),
        "trading_house": sorted(good.name for good in g.trading_house),
        "ships": [[s.capacity, s.good_type.name if s.good_type is not None else None,
                   s.current_load] for s in g.cargo_ships],
        "players": players,
        "scores": scores,
        "winners": winners,
    }


def first_difference(a, b, path="state"):
    """Path and values of the first difference between two canonical values, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                return f"{path}.{key}", a.get(key, "<missing>"), b.get(key, "<missing>")
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else (path, a, b)


class Mismatch(AssertionError):
    pass


def choose_action(rng: random.Random, legal: list[int], style: str) -> int:
    """Random policies that keep reaching the less common rules. Benefits are
    accepted 70% of the time, so declining is exercised too."""
    if legal == [ACCEPT_BENEFIT, DECLINE_BENEFIT]:
        return ACCEPT_BENEFIT if rng.random() < 0.7 else DECLINE_BENEFIT
    if style == "uniform":
        return rng.choice(legal)
    others = [a for a in legal if a != PASS]
    if PASS in legal and (not others or rng.random() < 0.15):
        return PASS
    builds = [a for a in others if 16 <= a <= 38]
    if style == "builder" and builds:
        return max(builds) if rng.random() < 0.5 else rng.choice(builds)
    return rng.choice(others)


def play_game(num_players: int, seed: int, style: str, stats: Counter) -> int:
    """Play one game in lockstep; returns the number of decisions. Raises Mismatch."""
    env = PuertoRicoEnv(num_players=num_players)
    env.reset(seed=seed)
    g = env.game
    # The reference model takes the engine's shuffles of the plantation discard
    # pile as given: it gets the same random tiles, and checks they are the right ones.
    shuffled: list[list[str]] = []
    engine_shuffle = g.rng.shuffle

    def record_shuffle(tiles, *args, **kwargs):
        engine_shuffle(tiles, *args, **kwargs)
        shuffled.append([TILE_NAME[t] for t in tiles])

    g.rng.shuffle = record_shuffle

    def replay_shuffle(tiles):
        if not shuffled:
            raise Mismatch("the reference model shuffled the discard pile but the engine did not")
        order = shuffled.pop(0)
        if sorted(order) != sorted(tiles):
            raise Mismatch(f"shuffled tiles differ: engine {sorted(order)}, oracle {sorted(tiles)}")
        stats["reshuffles"] += 1
        return list(order)

    oracle = Oracle.new_game(num_players, g.governor_idx,
                             [TILE_NAME[t] for t in g.plantation_stack],
                             [TILE_NAME[t] for t in g.face_up_plantations])
    rng = random.Random(seed * 7919 + num_players)
    history: list[tuple] = []
    decisions = 0
    while True:
        mine, theirs = engine_canonical(env), oracle.canonical()
        diff = first_difference(mine, theirs)
        if diff:
            raise Mismatch(f"state differs at {diff[0]}: engine {diff[1]!r}, oracle {diff[2]!r}; "
                           f"last actions {history[-6:]}")
        if mine["over"]:
            rewards = [env.rewards[f"player_{i}"] for i in range(num_players)]
            if [i for i, r in enumerate(rewards) if r > 0] != mine["winners"]:
                raise Mismatch(f"rewards {rewards} disagree with winners {mine['winners']}")
            stats["games"] += 1
            stats["end: colonists" if g._colonists_ship_underfilled else
                  "end: VP chips" if g.vp_chips <= 0 else "end: city"] += 1
            return decisions
        legal = sorted(np.flatnonzero(env.observe(env.agent_selection)["action_mask"]).tolist())
        if legal != oracle.legal_actions():
            raise Mismatch(f"legal actions differ in phase {mine['phase']} "
                           f"(benefit {mine['benefit']}): engine {legal}, "
                           f"oracle {oracle.legal_actions()}; last actions {history[-6:]}")
        _count(stats, mine, legal)
        action = choose_action(rng, legal, style)
        history.append((mine["phase"], mine["current_player"], action))
        decisions += 1
        env.step(action)
        oracle.step(action, shuffle=replay_shuffle)
        if g._captain_declined:
            stats["captain: pass with a usable wharf"] += 1


def _count(stats: Counter, state: dict, legal: list[int]) -> None:
    """Tally decision situations, to show which rules the games reached."""
    stats["decisions"] += 1
    if state["benefit"]:
        stats[f"benefit offered: {state['benefit']}"] += 1
    if 105 in legal:
        stats["hacienda draw available"] += 1
    if 13 in legal and state["current_player"] != state["role_player"]:
        stats["construction hut quarry available"] += 1
    if any(59 <= a <= 63 for a in legal):
        stats["wharf available"] += 1
    if any(106 <= a <= 110 for a in legal):
        stats["warehouse slot available"] += 1
    if any(93 <= a <= 97 for a in legal):
        stats["craftsman privilege available"] += 1
    if any(34 <= a <= 38 for a in legal):
        stats["large building affordable"] += 1
    if state["phase"] == "TRADER" and any(
            39 + i in legal and name in state["trading_house"] for i, name in enumerate(
                ["COFFEE", "TOBACCO", "CORN", "SUGAR", "INDIGO"])):
        stats["office duplicate sale available"] += 1


def _run_chunk(args) -> Counter:
    n, seeds, styles = args
    stats: Counter = Counter()
    for k, seed in seeds:
        style = styles[k % len(styles)]
        try:
            stats["decisions in games"] += play_game(n, seed, style, stats)
        except Mismatch as e:
            raise Mismatch(f"{n} players, seed {seed}, policy {style}: {e}") from None
    return stats


def run(players: list[int], games: int, seed: int = 0, styles=STYLES,
        workers: int = 1) -> dict:
    """Lockstep ``games`` games per player count; returns {num_players: Counter}."""
    results = {}
    for n in players:
        seeds = [(k, seed + k) for k in range(games)]
        chunks = [(n, seeds[i::workers], tuple(styles)) for i in range(max(1, workers))]
        stats: Counter = Counter()
        if workers <= 1:
            stats.update(_run_chunk(chunks[0]))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for part in pool.map(_run_chunk, chunks):
                    stats.update(part)
        results[n] = stats
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=200, help="games per player count")
    ap.add_argument("--players", default="2,3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        results = run([int(x) for x in args.players.split(",")], args.games, args.seed,
                      args.styles.split(","), args.workers)
    except Mismatch as e:
        print(f"MISMATCH: {e}")
        return 1
    for n, stats in results.items():
        print(f"== {n} players")
        for key in sorted(stats):
            print(f"{stats[key]:>11,}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
