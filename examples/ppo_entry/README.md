# A learned 2p entry

`my_agent.py` plays the bundled 2p PPO network greedily (the most probable
legal move), laid out the way a learned submission is: the agent file plus
**one weight file next to it**, read from `__file__` once per process
([The weight file](../../docs/COMPETITION_RULES.md#the-weight-file)).

The weight file is not in git; copy the bundled checkpoint in first:

```bash
cp training/checkpoints/ppo_baseline_2p.pt examples/ppo_entry/weights.pt
python tools/validate_submission.py examples/ppo_entry/my_agent.py --track 2p
python tools/play.py --agent examples/ppo_entry/my_agent.py:MyPpoAgent --vs PPO,Search
```

As shipped it plays exactly like the `PPO` baseline, which beats `Search` in
about two games of three ([training/README.md](../../training/README.md)), so
against `PPO` it scores about 50%. That is the point to improve from: use the
network's policy and value inside a search over `forward_model`, fine-tune it
against the opponents it still loses to, or train your own and drop your
`ckpt_final.pt` in as `weights.pt`.

It depends on PyTorch (`pip install -e ".[rl]"`) and imports `PpoNetwork` from
this repository, which the rules allow. If you change the network's shape,
copy the class into your file instead.
