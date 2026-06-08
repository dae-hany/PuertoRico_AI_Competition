# Training an RL agent (optional)

This folder holds the PPO self-play trainer used to produce a learned agent. It
is **optional** — the competition only needs the core package, the baselines, and
the tournament. Reinforcement learning is provided as a *starting point* because
Puerto Rico is hard for plain RL (the benchmark's central finding: a trained PPO
agent plays at roughly random strength). Beating the heuristic and MCTS baselines
with RL is an open challenge.

## Train

```bash
# A quick (weak) run, just to see it working:
python training/train_ppo.py --total_timesteps 50000 --num_envs 4 \
    --ckpt_interval 25000 --out_dir results/ppo_run

# Longer runs produce stronger (but still modest) agents — see --help for flags.
```

Checkpoints are written to `<out_dir>/checkpoints/ckpt_*.pt` as
`{"model_state": <state_dict>, ...}`.

## Play your trained agent

```python
from agents.ppo_agent import PpoAgent
agent = PpoAgent("results/ppo_run/checkpoints/ckpt_final.pt")
```

`PpoAgent` loads the network and plays greedily (highest-probability legal move).

## Files

- `train_ppo.py` — PPO self-play training loop (parameter-sharing, 3 agents).
- `wrapper.py` — flattens the env observation for the network (training only).
- `random_bot.py` — a random opponent used by the `fixed_random` training mode.

> Note: the `--env_mode aoe_ablation` option from the research code is not
> shipped here; use the default `standard` mode.
