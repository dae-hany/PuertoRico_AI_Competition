"""
A 2p entry built on a trained PPO network: the layout of a learned submission.

The folder holds what a learned entry ships: this file and ONE weight file next
to it. Put the bundled 2p checkpoint there first:

    cp training/checkpoints/ppo_baseline_2p.pt examples/ppo_entry/weights.pt
    python tools/validate_submission.py examples/ppo_entry/my_agent.py --track 2p

or point it at a checkpoint of your own (``training/train_ppo.py --num_players
2`` writes ``<out_dir>/checkpoints/ckpt_final.pt``). The validator should report
``[PASS] weight file``.

As shipped it plays exactly like the PPO baseline: the most probable legal
action. ``act`` is where to improve on it, e.g. by using the network's policy
and value inside a search over ``forward_model`` (see
docs/SEARCH_BASELINE_2P.md), or by fine-tuning the weights against the
baselines that beat it.
"""
import os

import numpy as np
import torch

from agents.base import Agent
from agents.ppo_agent import PpoNetwork
from puerto_rico.observation import egocentric_view

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.pt")    # from __file__, never from the cwd

# The harness builds a new agent every game, so the file is read once per
# process and the network kept here.
_NET = None


def _net():
    global _NET
    if _NET is None:
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        net = PpoNetwork(obs_dim=int(state["embed.0.weight"].shape[1]))
        net.load_state_dict(state)
        net.eval()
        _NET = net
    return _NET


class MyPpoAgent(Agent):
    name = "MyPpoAgent"     # shown on the board; it may not be a baseline's name

    def __init__(self):
        super().__init__()
        self.net = _net()     # at construction, never inside act()

    def act(self, observation, action_mask):
        # train_ppo.py trains on the egocentric view (block 0 = the acting
        # player), so the same rotation must be applied here.
        obs = torch.as_tensor(egocentric_view(observation), dtype=torch.float32)
        with torch.no_grad():
            logits = self.net.actor_logits(obs.unsqueeze(0))[0].numpy()
        logits = np.where(action_mask > 0.5, logits, -np.inf)
        return int(np.argmax(logits))
