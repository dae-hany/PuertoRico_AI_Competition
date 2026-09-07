"""Replay a recorded game (see puerto_rico/records.py) and inspect it.

    python tools/replay_game.py results/webui_games/game_20260907_101500_12345.json
        verify the record replays exactly and print the final scores

    python tools/replay_game.py <record.json> --moves
        also narrate every decision in English

    python tools/replay_game.py <record.json> --stop 40
        replay the first 40 decisions, then show the position the next player
        sees: who is to move, the phase, the legal actions, and the flat
        observation vector your agent would receive

Every game finished in the web UI is saved automatically under
results/webui_games/. To record your own games elsewhere, use
puerto_rico.records.build_record / save_record.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from puerto_rico import flatten_observation
from puerto_rico.describe import describe_action
from puerto_rico.records import ReplayError, game_over, load_record, replay_record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="path to a game record (.json)")
    ap.add_argument("--moves", action="store_true", help="narrate every decision")
    ap.add_argument("--stop", type=int, default=None, metavar="N",
                    help="replay only the first N decisions and show the position")
    args = ap.parse_args()

    rec = load_record(args.record)
    labels = rec.get("player_labels") or [f"seat {i}" for i in range(rec["num_players"])]
    print(f"record: {args.record}")
    print(f"seed {rec['seed']}, {rec['num_players']} players, "
          f"{len(rec['actions'])} decisions, played {rec.get('time', '?')}")
    for i, lab in enumerate(labels):
        print(f"  seat {i}: {lab}")

    def narrate(i, seat, action, env):
        if args.moves:
            print(f"{i:4d}  {describe_action(seat, action, env)}")

    try:
        env = replay_record(rec, on_step=narrate, stop_after=args.stop)
    except ReplayError as e:
        sys.exit(f"REPLAY FAILED: {e}")

    if args.stop is not None and not game_over(env):
        name = env.agent_selection
        seat = env.agent_name_mapping[name]
        obs = env.observe(name)
        mask = np.asarray(obs["action_mask"])
        flat = flatten_observation(obs["observation"])
        phase = env.game.current_phase
        print(f"\nafter {args.stop} decisions: seat {seat} ({labels[seat]}) to move, "
              f"phase {phase.name if phase else 'ROLE_SELECT'}, round {env.game.round_number}")
        print("legal actions:")
        for a in np.where(mask > 0.5)[0]:
            print(f"  {int(a):3d}  {describe_action(seat, int(a), env)}")
        np.set_printoptions(linewidth=100, precision=3, suppress=True)
        print(f"flat observation (len {len(flat)}):\n{flat}")
        return

    scores = env.game.get_scores()
    print("\nfinal scores (vp, tiebreak, ...):")
    for i, s in enumerate(scores):
        mark = "  <- winner" if i in rec.get("winners", []) else ""
        print(f"  seat {i} {labels[i]:<24} {tuple(int(x) for x in s)}{mark}")
    recorded = rec.get("scores")
    if recorded is not None:
        same = [list(map(int, s)) for s in scores] == recorded
        print("replays exactly: scores match the record" if same
              else f"WARNING: replayed scores differ from the record {recorded}")
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
