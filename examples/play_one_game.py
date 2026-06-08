"""Play a single 3-player game between baseline agents and print the result.

Run from the repo root:  python examples/play_one_game.py
"""
import os
import sys

# Make the repo root importable when run as a script from the examples/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ActionValueAgent, RandomAgent, ShippingRushAgent
from tournament.match import play_game


def main():
    agents = [ActionValueAgent(), ShippingRushAgent(), RandomAgent(seed=0)]
    names = [a.name for a in agents]

    result = play_game(agents, seed=42)

    print("Players     :", names)
    print("Final VP    :", result["scores"])
    winners = result["winners"]
    if len(winners) == 1:
        print("Winner      :", names[winners[0]])
    else:
        print("Winners (tie):", [names[w] for w in winners])
    print("Game length :", result["steps"], "decisions")


if __name__ == "__main__":
    main()
