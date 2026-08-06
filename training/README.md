# Training an RL agent (optional)

This folder holds the PPO self-play trainer used to produce the learned baseline.
It is **optional** — the competition only needs the core package, the baselines,
and the tournament.

`checkpoints/ppo_baseline.pt` is 5M steps of parameter-sharing self-play
(~2.6 h on 8 CPU threads). Measured over 60 seat-rotated 3p games per matchup,
greedy play, no illegal moves:

| PPO (player) vs 2× | win rate | PPO mean VP | opponent mean VP |
|---|---:|---:|---:|
| `RandomAgent` | 100.0% | 69.5 | 25.2 |
| `FactoryAgent` | 100.0% | 79.8 | 22.7 |
| `TradeBuildingAgent` | 93.3% | 44.4 | 23.4 |
| `ShippingRushAgent` | 88.3% | 53.3 | 28.0 |
| `ActionValueAgent` | 90.0% | 51.2 | 32.2 |
| `ActionValue` + `ShippingRush` | 98.3% | 54.1 | — |

An agent of average strength wins ≈ 33.3% of 3-player games.

Against `MctsAgent` at the web UI's practical budget (30 simulations, rollout
depth 40) it wins **10/12** with 58.9 VP to 30.4. That is a small sample, and
`MctsAgent`'s default 200-simulation budget was not measured — a single game at
that setting takes tens of minutes, which is why the web UI does not use it
either.

## Train

```bash
# A quick run, just to see it working (a few minutes on CPU):
python training/train_ppo.py --total_timesteps 200000 --num_envs 4 \
    --eval_interval 50000 --eval_episodes 100 --out_dir results/ppo_run
```

```bash
# The full run that produced the bundled baseline:
python training/train_ppo.py --total_timesteps 5000000 --num_envs 8 \
    --eval_interval 250000 --ckpt_interval 250000 --eval_episodes 100 \
    --out_dir results/ppo_v3
```

Checkpoints are written to `<out_dir>/checkpoints/ckpt_*.pt` as
`{"model_state": <state_dict>, "args": {...}, ...}`. Two CSVs are written
alongside them: `training_log.csv` (one row per PPO update, including
`approx_kl`, `clipfrac` and `illegal_ends`) and `eval_log.csv` (win rate vs two
`RandomBot`s, reported both greedily and with sampling).

## Play your trained agent

```python
from agents.ppo_agent import PpoAgent
agent = PpoAgent("results/ppo_v3/checkpoints/ckpt_final.pt")
```

`PpoAgent` loads the network and plays greedily (highest-probability legal move).
It reads the `egocentric` flag out of the checkpoint and applies the matching
observation rotation, so training and play always agree. A checkpoint saved
before the rotation existed has no such flag and is read back as absolute-seat,
so old and new weights both keep playing correctly — but they are *different*
encodings, and weights cannot be mixed between them.

## How the trainer handles self-play

A 3-player AEC game differs from a single-agent CleanRL loop in ways that are
easy to get subtly wrong. The four that matter here:

**The rollout buffer is not one trajectory.** Steps arrive interleaved
(p0 → p1 → p2 → p0 → …), so GAE is computed over *each seat's own
sub-sequence*. From player 0's point of view, "the next state" is player 0's next
decision point — the opponents' moves in between are environment dynamics.

**The terminal reward arrives for every seat at once.** The env writes ±1 for all
three players on the single step that ends the game, and PettingZoo then deletes
those entries during the dead-step drain. The trainer snapshots them at that
moment and attributes each seat's reward back to *that seat's* last transition.

**The observation is ego-centric.** `flatten_observation` lays the player blocks
out in absolute seat order, so one parameter-sharing network would otherwise have
to learn a seat-selection rule ("if `current_player == 1`, read block 1") from a
single raw scalar. `puerto_rico.observation.egocentric_view` rotates the view so
block 0 is always the acting player; `--no_egocentric` restores the old encoding
for comparison.

**`gae_lambda` stays at 0.95.** With a terminal-only reward and `gamma=1.0` it is
tempting to argue for λ = 1 (the return becomes the plain Monte-Carlo game
outcome, and at λ = 0.95 the terminal reward reaches a seat's opening moves with
weight ≈ `0.95^100`). Measured over two 200k-step runs at the same seed, that
argument does not hold up — the bootstrapped critic carries the signal past the
λ horizon, and the variance reduction wins:

| step | λ = 0.95 | λ = 1.0 |
|---:|---|---|
| 102k | 0.98 win / 42.6 VP | 0.84 win / 44.2 VP |
| 200k | **0.99 win / 51.6 VP** | 0.98 win / 47.9 VP |

(win rate = greedy play as player 0 against two `RandomBot`s, 100 games; a random
policy scores ≈ 0.33.)

## Files

- `train_ppo.py` — PPO self-play training loop (parameter-sharing, 3 agents).
- `wrapper.py` — flattens (and rotates) the env observation for the network.
- `random_bot.py` — a random opponent used by evaluation and the
  `fixed_random` training mode.
- `checkpoints/ppo_baseline.pt` — the bundled baseline that `PpoAgent` and the
  web UI load by default.

> Note: the `--env_mode aoe_ablation` option from the research code is not
> shipped here; use the default `standard` mode.
