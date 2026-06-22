"""Play one game in each track between baseline agents and print the result.

Run from the repo root:  python examples/play_one_game.py

The seat count is just the number of agents you pass to ``play_game``:
2 agents -> 1-vs-1 (2p track), 3 agents -> 3p track.
"""
import os
import sys

# Make the repo root importable when run as a script from the examples/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ActionValueAgent, RandomAgent, ShippingRushAgent
from tournament.match import play_game


def show(label, agents, seed):
    names = [a.name for a in agents]
    result = play_game(agents, seed=seed)

    print(f"\n{label}")
    print("Players     :", names)
    print("Final VP    :", result["scores"])
    winners = result["winners"]
    if len(winners) == 1:
        print("Winner      :", names[winners[0]])
    else:
        print("Winners (tie):", [names[w] for w in winners])
    print("Game length :", result["steps"], "decisions")


def main():
    # 2p track — 1 vs 1
    show("[2p track] 1-vs-1",
         [ActionValueAgent(), ShippingRushAgent()], seed=42)

    # 3p track — 3 players
    show("[3p track] 3-player",
         [ActionValueAgent(), ShippingRushAgent(), RandomAgent(seed=0)], seed=42)


if __name__ == "__main__":
    main()
