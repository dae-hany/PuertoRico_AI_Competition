# Training an RL agent (optional)

This folder holds the PPO self-play trainer used to produce the learned baseline.
It is **optional** — the competition only needs the core package, the baselines,
and the tournament. It needs PyTorch: `pip install -e ".[rl]"`.

The trainer works for **either track**: `--num_players 3` (default) or
`--num_players 2`. The observation width follows from the seat count
(`74 + 73 × num_players`, so 293 or 220), and `PpoAgent` reads that width back
out of the checkpoint — but **a checkpoint only plays the track it was trained
on**. One checkpoint per track is bundled, both from the full-run command below
with default settings (5M steps of parameter-sharing self-play, seed 42):

| Checkpoint | Track | Training time |
|---|---|---|
| `checkpoints/ppo_baseline_2p.pt` | 2p | 1.6 h on CPU (`--device cpu`, ~860 steps/s) |
| `checkpoints/ppo_baseline.pt` | 3p | ~2.6 h on 8 CPU threads |

The registry's `PPO` baseline loads both and plays whichever track it is seated
in.

### 2p: the strongest bundled agent

Greedy play, seat-balanced, under the entrant's 0.1 s rule (through
[`../examples/ppo_entry/`](../examples/ppo_entry/); about 0.5 ms a move):

| 2p PPO vs | win rate | games |
|---|---:|---:|
| `SearchAgent` (1500 nodes) | **67.5%** (95% CI [0.60, 0.74]) | 160 |
| `SearchLiteAgent` (250 nodes) | 80% | 40 |
| `MctsAgent` (60 simulations) | 89% | 40 |
| every heuristic | 100% | 24 each |

The heuristic row comes from [`../docs/BASELINES.md`](../docs/BASELINES.md). The
planner rows come from `tools/play.py --agent examples/ppo_entry/my_agent.py
--vs SearchLite,Search,MCTS --games 40`, plus four 30-game runs against
`Search` at `--seed` 10000, 20000, 30000 and 40000.

It gets there fast. The same run's intermediate checkpoints
(`--ckpt_interval 250000`), 40 games per cell, same setup:

| steps | Factory | TradeBuilding | ActionValue | ShippingRush | Search |
|---:|---:|---:|---:|---:|---:|
| 262k | 20% | 78% | 28% | 75% | 0% |
| 508k | 70% | 88% | 50% | 100% | 25% |
| 754k | 92% | 98% | 70% | 100% | 39% |
| 1.02M | 80% | 86% | 75% | 100% | 38% |
| 1.26M | 92% | 82% | 82% | 92% | 40% |
| 1.51M | 96% | 98% | 68% | 100% | 50% |
| 5.01M (bundled) | 100% | 98% | 98% | 100% | 65% |

So plain self-play PPO, with no search and no reward shaping, outplays
alpha-beta search in this game. Treat it as a starting point rather than a
ceiling: the network's policy and value can also guide a search, and fine-tuning
against the opponents it still loses to is another way up.

### 3p

`checkpoints/ppo_baseline.pt` against each heuristic, greedy play,
seat-balanced, 24 games per matchup:

| PPO vs 2× | win rate |
|---|---:|
| `RandomAgent` | 100% |
| `FactoryAgent` | 100% |
| `TradeBuildingAgent` | 100% |
| `ShippingRushAgent` | 83% |
| `ActionValueAgent` | 83% |

An agent of average strength wins ≈ 33.3% of 3-player games, so PPO is the agent
to beat in the 3p track. These numbers come from
[`../docs/BASELINES.md`](../docs/BASELINES.md); regenerate them with
`python tools/measure_baselines.py` after changing anything that could move
them. (The table published before that tool existed was measured on an engine
and a set of baselines that have since been corrected — see
[`../CHANGES.md`](../CHANGES.md) — so it is not comparable.)

For head-to-head numbers against `MctsAgent` — and for every other baseline
pairing — see [`../docs/BASELINES.md`](../docs/BASELINES.md), which
`tools/measure_baselines.py` regenerates. MCTS is quoted there **with its search
budget**, because that is what defines its strength.

## Train

```bash
# A quick run, just to see it working (a few minutes on CPU):
python training/train_ppo.py --total_timesteps 200000 --num_envs 4 \
    --eval_interval 50000 --eval_episodes 100 --out_dir results/ppo_run
```

```bash
# The full run that produced the bundled baselines (the 2p one added
# --num_players 2 --device cpu --out_dir results/ppo_2p):
python training/train_ppo.py --total_timesteps 5000000 --num_envs 8 \
    --eval_interval 250000 --ckpt_interval 250000 --eval_episodes 100 \
    --out_dir results/ppo_v3
```

Checkpoints are written to `<out_dir>/checkpoints/ckpt_*.pt` as
`{"model_state": <state_dict>, "args": {...}, ...}`. Two CSVs are written
alongside them: `training_log.csv` (one row per PPO update, including
`approx_kl`, `clipfrac` and `illegal_ends`) and `eval_log.csv` (win rate vs
`RandomBot`s in the other seats, reported both greedily and with sampling). In
2p that number reaches 100% within ~0.5M steps, long before the policy stops
improving, so measure a checkpoint against the baselines with `tools/play.py`.

## Play your trained agent

```python
from agents.ppo_agent import PpoAgent
agent = PpoAgent("results/ppo_v3/checkpoints/ckpt_final.pt")

# one checkpoint per track: the agent picks by observation length
agent = PpoAgent(["training/checkpoints/ppo_baseline_2p.pt",
                  "training/checkpoints/ppo_baseline.pt"])
```

To submit a trained network as an entry, copy
[`../examples/ppo_entry/`](../examples/ppo_entry/): it loads one `weights.pt`
from its own folder, as the weight-file rule requires.

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

- `train_ppo.py` — PPO self-play training loop (parameter-sharing; `--num_players`
  picks the track).
- `wrapper.py` — flattens (and rotates) the env observation for the network.
- `random_bot.py` — a random opponent used by evaluation and the
  `fixed_random` training mode.
- `checkpoints/ppo_baseline_2p.pt`, `checkpoints/ppo_baseline.pt` — the bundled
  2p and 3p baselines, which the registry's `PPO` and the web UI load.

> Note: the research code's AOE-ablation environment is not part of this repo.
> The `--env_mode` flag that used to offer it has been removed — it imported a
> module that does not ship, so selecting it only ever raised `ModuleNotFoundError`.
