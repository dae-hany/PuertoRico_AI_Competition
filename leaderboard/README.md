# Ladder standings

During the competition window the organizer runs every entry received so far
against every other entry and the bundled baselines, and publishes the result
here. Nothing has been run yet: the first round appears once the window opens.

```
leaderboard/
  2p/latest.md                 the newest 2p standings
  2p/<date>/leaderboard.md     that round's standings (also .csv / .json)
  2p/<date>/games/*.json       every game of the round, replayable
  3p/...                       the same for the 3p track
```

Replay any game, move by move, with

```bash
python tools/replay_game.py leaderboard/2p/<date>/games/game_00012_12.json --moves
```

The rounds are produced by `python tools/run_ladder.py` (see its docstring);
the official metric is Elo in the 2p track and TrueSkill in the 3p track, as
described in [`../docs/RANKING.md`](../docs/RANKING.md).
