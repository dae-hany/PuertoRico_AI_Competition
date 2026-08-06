"""
train_ppo.py — PPO Self-Play Training for Puerto Rico (paper hyperparameters)

All 3 agents share a single PPOAgent network (parameter-sharing self-play).
Experiences from num_envs independent environments are pooled into one rollout
buffer (n_steps * num_envs total steps) before each PPO update.

Multi-agent credit assignment
-----------------------------
An AEC rollout interleaves the seats (p0 → p1 → p2 → p0 → …), so the raw buffer
is *not* a single trajectory. Two things follow, and both are handled here:

  * Every transition is tagged with its owning seat, and GAE runs over each
    seat's own sub-sequence. "The next state" for one of p0's decisions is p0's
    *next* decision point — the opponents' moves in between are environment
    dynamics from p0's point of view.
  * The terminal reward is written by the env for all seats at once, on the step
    that ends the game. It is snapshotted there (PettingZoo deletes it during the
    dead-step drain) and attributed back to each seat's own last transition.

The observation is egocentric (see ``puerto_rico.observation.egocentric_view``):
block 0 is always the acting player. A parameter-sharing policy cannot work well
without this.

Key hyperparameter defaults match the paper specification:
  gamma=1.0  (undiscounted — sparse terminal reward)
  n_steps=2048, n_epochs=10, lr=3e-4, num_envs=8

Usage:
  python training/train_ppo.py                     # full run
  python training/train_ppo.py \\
    --total_timesteps 10000 --num_envs 2 --n_steps 256 \\
    --eval_interval 5000 --eval_episodes 10 \\
    --out_dir results/smoke_test

Outputs (in --out_dir):
  training_log.csv   — one row per PPO update
  eval_log.csv       — one row per evaluation checkpoint
  checkpoints/       — ckpt_XXXXXXXX.pt every --ckpt_interval steps
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Repo root (parent of training/) — needed on sys.path for `puerto_rico.*` and
# `training.*` imports, and used as the cwd so relative --out_dir paths resolve
# against the project root rather than this file's directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from puerto_rico.env import PuertoRicoEnv
from training.wrapper import CleanRLAECWrapper
from agents.ppo_agent import PpoNetwork as PPOAgent
from training.random_bot import RandomBot
from puerto_rico.constants import Role

OBS_DIM   = 293
ACT_DIM   = 200
N_PLAYERS = 3


# ── environment factory ────────────────────────────────────────────────────────

def make_env(obs_mode: str = 'full', env_mode: str = 'standard',
             egocentric: bool = True):
    if env_mode == 'aoe_ablation':
        from env.aoe_ablation_env import AOEAblationEnv
        env = AOEAblationEnv(num_players=N_PLAYERS, random_seed_mode=True)
    else:
        env = PuertoRicoEnv(num_players=N_PLAYERS, random_seed_mode=True)
    return CleanRLAECWrapper(env, obs_mode=obs_mode, egocentric=egocentric)


class EnvRunner:
    """One environment plus the bookkeeping that must survive across updates.

    Environments are *not* reset between rollouts: an episode that is still in
    progress when the buffer fills simply continues into the next one. Resetting
    every update would throw the late game away and skew the state distribution
    toward opening positions.
    """

    def __init__(self, env, n_players: int = N_PLAYERS):
        self.env = env
        self.n_players = n_players
        self.ep_steps = 0
        self._started = False

    def ensure_ready(self):
        if not self._started or not self.env.agents:
            self.env.reset()
            self._started = True
            self.ep_steps = 0


# ── rollout collection ────────────────────────────────────────────────────────

def collect_rollout(runner: EnvRunner, agent, num_steps: int, device: str,
                    training_mode: str = 'self_play'):
    """
    Collect num_steps stored transitions from one environment.

    training_mode='self_play'   — all seats use the shared policy (standard)
    training_mode='fixed_random'— only player_0 is trained; opponents are
                                  RandomBots and their steps are not stored
                                  (so exactly num_steps player_0 transitions)
    """
    env = runner.env
    n   = runner.n_players

    obs_buf   = np.zeros((num_steps, OBS_DIM), dtype=np.float32)
    act_buf   = np.zeros(num_steps,            dtype=np.int64)
    logp_buf  = np.zeros(num_steps,            dtype=np.float32)
    val_buf   = np.zeros(num_steps,            dtype=np.float32)
    rew_buf   = np.zeros(num_steps,            dtype=np.float32)
    done_buf  = np.zeros(num_steps,            dtype=np.float32)
    mask_buf  = np.zeros((num_steps, ACT_DIM), dtype=np.float32)
    owner_buf = np.full(num_steps, -1,         dtype=np.int64)

    ep_wins, ep_vps, ep_lens = [], [], []
    illegal_ends = 0

    random_bots = [RandomBot() for _ in range(n - 1)] if training_mode == 'fixed_random' else None

    runner.ensure_ready()

    # Buffer index of each seat's most recent transition in the *current*
    # episode; -1 means "nothing stored yet" (seat has not moved since the
    # episode started, or its moves predate this rollout).
    last_idx = [-1] * n
    step = 0

    def finish_episode_if_over():
        """Attribute terminal rewards, record stats, drain dead steps, reset."""
        nonlocal last_idx, illegal_ends
        base = env.unwrapped
        # The env flags every seat at once, so "over" means every live agent is
        # flagged; testing all() (not any()) also keeps the drain below safe.
        if not env.agents or not all(base.terminations.get(a, False) or
                                     base.truncations.get(a, False)
                                     for a in env.agents):
            return

        # Snapshot before draining: PettingZoo's _was_dead_step deletes each
        # agent's entry from `rewards`, so this is the only chance to read them.
        final_rewards = {k: float(v) for k, v in base.rewards.items()}
        if any("error" in base.infos.get(f"player_{p}", {}) for p in range(n)):
            illegal_ends += 1

        for p in range(n):
            i = last_idx[p]
            if i >= 0:
                rew_buf[i] += final_rewards.get(f"player_{p}", 0.0)
                done_buf[i] = 1.0

        scores = base.game.get_scores()
        best = max((s[0], s[1]) for s in scores)
        ep_wins.append(1.0 if (scores[0][0], scores[0][1]) == best else 0.0)
        ep_vps.append(scores[0][0])
        ep_lens.append(runner.ep_steps)

        while env.agents:               # drain the dead-step queue
            env.step(None)
        env.reset()
        runner.ep_steps = 0
        last_idx = [-1] * n

    while step < num_steps:
        agent_name = env.agent_selection
        p_idx      = env.unwrapped.agent_name_mapping[agent_name]

        obs  = env.observe(agent_name)
        flat = obs["observation"].astype(np.float32)
        mask = obs["action_mask"].astype(np.float32)

        obs_t  = torch.FloatTensor(flat).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        if training_mode == 'fixed_random' and p_idx != 0:
            with torch.no_grad():
                action_t, _, _, _ = random_bots[p_idx - 1].get_action_and_value(obs_t, mask_t)
            env.step(int(action_t.item()))
            runner.ep_steps += 1
            finish_episode_if_over()
            continue

        with torch.no_grad():
            action, logp, _, value = agent.get_action_and_value(obs_t, mask_t)

        a = int(action.item())
        env.step(a)
        runner.ep_steps += 1

        obs_buf[step]   = flat
        act_buf[step]   = a
        logp_buf[step]  = float(logp.item())
        val_buf[step]   = float(value.item())
        rew_buf[step]   = 0.0           # non-terminal steps score nothing;
        done_buf[step]  = 0.0           # both are back-filled on game over
        mask_buf[step]  = mask
        owner_buf[step] = p_idx
        last_idx[p_idx] = step
        step += 1

        finish_episode_if_over()

    # Per-seat bootstrap for a rollout that ended mid-episode. Seats whose last
    # stored transition is terminal ignore this (GAE zeroes it out).
    boot = np.zeros(n, dtype=np.float32)
    base = env.unwrapped
    for p in range(n):
        name = f"player_{p}"
        if name in env.agents and not (base.terminations.get(name, False) or
                                       base.truncations.get(name, False)):
            o = env.observe(name)
            o_t = torch.FloatTensor(o["observation"].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                boot[p] = float(agent.get_value(o_t).item())

    return (obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf,
            owner_buf, boot, ep_wins, ep_vps, ep_lens, illegal_ends)


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rew_buf, val_buf, done_buf, bootstrap_val, gamma, gae_lambda):
    """GAE over a single agent's trajectory.

    ``done_buf[t]`` means "the episode ended as a result of the action at t", so
    ``1 - done_buf[t]`` — not ``done_buf[t+1]`` — is the mask that both drops the
    bootstrap and cuts the advantage chain at t.
    """
    n        = len(rew_buf)
    adv_buf  = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        nonterminal = 1.0 - done_buf[t]
        next_val    = bootstrap_val if t == n - 1 else val_buf[t + 1]
        delta       = rew_buf[t] + gamma * next_val * nonterminal - val_buf[t]
        last_gae    = delta + gamma * gae_lambda * nonterminal * last_gae
        adv_buf[t]  = last_gae

    ret_buf = adv_buf + val_buf
    return adv_buf, ret_buf


def compute_gae_per_seat(rew_buf, val_buf, done_buf, owner_buf, boot,
                         n_players, gamma, gae_lambda):
    """Split the interleaved buffer by owning seat and run GAE on each part."""
    adv_buf = np.zeros_like(rew_buf)
    ret_buf = np.zeros_like(rew_buf)

    for p in range(n_players):
        idx = np.nonzero(owner_buf == p)[0]
        if idx.size == 0:
            continue
        adv, ret = compute_gae(rew_buf[idx], val_buf[idx], done_buf[idx],
                               float(boot[p]), gamma, gae_lambda)
        adv_buf[idx] = adv
        ret_buf[idx] = ret

    return adv_buf, ret_buf


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(agent, optimizer, obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
               mask_buf, clip_coef, ent_coef, vf_coef, n_epochs, batch_size,
               device, max_grad_norm):
    n = len(obs_buf)
    obs_t    = torch.FloatTensor(obs_buf).to(device)
    act_t    = torch.LongTensor(act_buf).to(device)
    logp_old = torch.FloatTensor(logp_buf).to(device)
    adv_t    = torch.FloatTensor(adv_buf).to(device)
    ret_t    = torch.FloatTensor(ret_buf).to(device)
    mask_t   = torch.FloatTensor(mask_buf).to(device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    pg_losses, v_losses, ent_losses, kls, clipfracs = [], [], [], [], []

    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            _, new_logp, entropy, new_val = agent.get_action_and_value(
                obs_t[b], mask_t[b], action=act_t[b]
            )
            new_val = new_val.squeeze(-1)

            logratio = new_logp - logp_old[b]
            ratio    = logratio.exp()
            adv_b    = adv_t[b]

            pg1 = -adv_b * ratio
            pg2 = -adv_b * ratio.clamp(1 - clip_coef, 1 + clip_coef)
            pg_loss  = torch.max(pg1, pg2).mean()
            v_loss   = 0.5 * (new_val - ret_t[b]).pow(2).mean()
            ent_loss = entropy.mean()
            loss     = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # Schulman's low-variance approximate KL
                kls.append(float(((ratio - 1) - logratio).mean().item()))
                clipfracs.append(float(((ratio - 1).abs() > clip_coef).float().mean().item()))

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(ent_loss.item())

    return (float(np.mean(pg_losses)), float(np.mean(v_losses)),
            float(np.mean(ent_losses)), float(np.mean(kls)), float(np.mean(clipfracs)))


# ── in-training evaluation (PPO vs Random × 2) ────────────────────────────────

def run_eval(agent, n_episodes: int, device: str, obs_mode: str = 'full',
             env_mode: str = 'standard', egocentric: bool = True,
             greedy: bool = True) -> dict:
    """Run n_episodes of PPO (player_0) vs 2 RandomBots. Returns stats dict.

    ``greedy=True`` matches how ``agents.ppo_agent.PpoAgent`` actually plays
    (argmax over legal actions), so eval numbers describe the deployed policy
    rather than the exploration policy.
    """
    random_agents = [RandomBot(), RandomBot()]
    wins, vps, ep_lens = [], [], []
    env = make_env(obs_mode=obs_mode, env_mode=env_mode, egocentric=egocentric)

    for _ in range(n_episodes):
        env.reset()
        ep_step = 0

        while env.agents:
            agent_name = env.agent_selection
            term  = env.terminations.get(agent_name, False)
            trunc = env.truncations.get(agent_name, False)
            if term or trunc:
                env.step(None)
                continue

            obs  = env.observe(agent_name)
            flat = obs["observation"].astype(np.float32)
            mask = obs["action_mask"].astype(np.float32)
            obs_t  = torch.FloatTensor(flat).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

            p_idx = env.unwrapped.agent_name_mapping[agent_name]
            if p_idx == 0:
                with torch.no_grad():
                    if greedy:
                        logits = agent.actor_logits(obs_t)
                        neg    = torch.tensor(-1e8, dtype=logits.dtype, device=device)
                        action = int(torch.argmax(
                            torch.where(mask_t > 0.5, logits, neg), dim=1).item())
                    else:
                        action_t, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                        action = int(action_t.item())
            else:
                action_t, _, _, _ = random_agents[p_idx - 1].get_action_and_value(obs_t, mask_t)
                action = int(action_t.item())

            env.step(action)
            ep_step += 1

        scores = env.unwrapped.game.get_scores()
        best = max((s[0], s[1]) for s in scores)
        wins.append(1.0 if (scores[0][0], scores[0][1]) == best else 0.0)
        vps.append(scores[0][0])
        ep_lens.append(ep_step)

    return {
        "win_rate":    float(np.mean(wins)),
        "mean_vp":     float(np.mean(vps)),
        "std_vp":      float(np.std(vps)),
        "mean_ep_len": float(np.mean(ep_lens)),
    }


# ── main training loop ────────────────────────────────────────────────────────

def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_csv = os.path.join(args.out_dir, "training_log.csv")
    eval_csv  = os.path.join(args.out_dir, "eval_log.csv")

    with open(train_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "pg_loss", "v_loss", "ent_loss", "approx_kl", "clipfrac", "lr",
            "episodes", "win_rate_rolling", "mean_vp_rolling", "mean_ep_len",
            "illegal_ends",
        ])
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "win_rate", "mean_vp", "std_vp", "mean_ep_len",
            "win_rate_sampled", "mean_vp_sampled",
        ])

    # Optional WandB
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project="puerto_rico_ppo", config=vars(args))

    # Agent + optimizer
    agent     = PPOAgent(obs_dim=OBS_DIM, action_dim=ACT_DIM).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # LR: linear decay to 0 over the run. The loop runs ceil() updates, so
    # total_updates must round up too — flooring it drives lr to 0 early (and,
    # for short runs, on the very first update).
    steps_per_update = args.n_steps * args.num_envs
    total_updates    = max(1, math.ceil(args.total_timesteps / steps_per_update))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: max(0.0, 1.0 - u / total_updates)
    )

    # Build envs once; they persist across rollouts (episodes are not restarted).
    runners = [EnvRunner(make_env(obs_mode=args.obs_mode, env_mode=args.env_mode,
                                  egocentric=args.egocentric))
               for _ in range(args.num_envs)]

    global_step      = 0
    update_count     = 0
    next_ckpt_step   = args.ckpt_interval
    next_eval_step   = args.eval_interval

    recent_wins  = deque(maxlen=500)
    recent_vps   = deque(maxlen=500)
    recent_lens  = deque(maxlen=500)

    t0 = time.time()
    print(f"Training PPO | total_timesteps={args.total_timesteps:,} | "
          f"num_envs={args.num_envs} | n_steps={args.n_steps} | "
          f"updates={total_updates} | gamma={args.gamma} | lr={args.lr} | "
          f"egocentric={args.egocentric}")

    while global_step < args.total_timesteps:
        agent.eval()

        # Collect n_steps from each env, pool all experiences
        all_obs, all_act, all_logp, all_val = [], [], [], []
        all_rew, all_done, all_mask         = [], [], []
        all_owner, all_boot                 = [], []
        illegal_ends                        = 0

        for runner in runners:
            (o, a, lp, v, r, d, m, own, bv,
             ew, evp, el, ill) = collect_rollout(runner, agent, args.n_steps, device,
                                                 training_mode=args.training_mode)
            all_obs.append(o);    all_act.append(a);     all_logp.append(lp)
            all_val.append(v);    all_rew.append(r);     all_done.append(d)
            all_mask.append(m);   all_owner.append(own); all_boot.append(bv)
            recent_wins.extend(ew)
            recent_vps.extend(evp)
            recent_lens.extend(el)
            illegal_ends += ill

        steps_this_update = args.n_steps * args.num_envs
        global_step      += steps_this_update
        update_count     += 1

        # GAE per env, per seat, then concatenate
        adv_parts, ret_parts = [], []
        for i in range(args.num_envs):
            adv, ret = compute_gae_per_seat(
                all_rew[i], all_val[i], all_done[i], all_owner[i], all_boot[i],
                N_PLAYERS, args.gamma, args.gae_lambda,
            )
            adv_parts.append(adv)
            ret_parts.append(ret)

        obs_buf  = np.concatenate(all_obs)
        act_buf  = np.concatenate(all_act)
        logp_buf = np.concatenate(all_logp)
        mask_buf = np.concatenate(all_mask)
        adv_buf  = np.concatenate(adv_parts)
        ret_buf  = np.concatenate(ret_parts)

        agent.train()
        pg_l, v_l, ent_l, kl, clipfrac = ppo_update(
            agent, optimizer, obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
            mask_buf, args.clip_coef, args.ent_coef, args.vf_coef,
            args.n_epochs, args.batch_size, device, args.max_grad_norm,
        )
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        wr     = float(np.mean(recent_wins))  if recent_wins  else 0.0
        mvp    = float(np.mean(recent_vps))   if recent_vps   else 0.0
        mlen   = float(np.mean(recent_lens))  if recent_lens  else 0.0
        sps    = global_step / (time.time() - t0)

        with open(train_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                global_step, round(pg_l, 5), round(v_l, 5), round(ent_l, 5),
                round(kl, 6), round(clipfrac, 4), round(cur_lr, 8),
                len(recent_wins), round(wr, 4), round(mvp, 2), round(mlen, 1),
                illegal_ends,
            ])

        if use_wandb:
            import wandb
            wandb.log({"step": global_step, "pg_loss": pg_l, "v_loss": v_l,
                       "ent_loss": ent_l, "approx_kl": kl, "clipfrac": clipfrac,
                       "lr": cur_lr, "win_rate_rolling": wr,
                       "mean_vp_rolling": mvp}, step=global_step)

        eta = (args.total_timesteps - global_step) / max(sps, 1e-9) / 60.0
        print(f"step={global_step:>9,} | wr={wr:.3f} | vp={mvp:.1f} | "
              f"pg={pg_l:.4f} v={v_l:.4f} kl={kl:.4f} | "
              f"sps={sps:.0f} | eta={eta:.0f}m", flush=True)

        # Evaluation checkpoint
        if global_step >= next_eval_step:
            agent.eval()
            print(f"  [eval] running {args.eval_episodes} games…", flush=True)
            t_eval = time.time()
            # Greedy is how PpoAgent actually plays; the sampled number is logged
            # alongside it so a degenerate argmax is distinguishable from a
            # genuinely weak policy.
            stats = run_eval(agent, args.eval_episodes, device, obs_mode=args.obs_mode,
                             env_mode=args.env_mode, egocentric=args.egocentric,
                             greedy=True)
            samp  = run_eval(agent, args.eval_episodes, device, obs_mode=args.obs_mode,
                             env_mode=args.env_mode, egocentric=args.egocentric,
                             greedy=False)
            elapsed_eval = time.time() - t_eval
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    global_step,
                    round(stats["win_rate"],    4),
                    round(stats["mean_vp"],     2),
                    round(stats["std_vp"],      2),
                    round(stats["mean_ep_len"], 1),
                    round(samp["win_rate"],     4),
                    round(samp["mean_vp"],      2),
                ])
            print(f"  [eval] step={global_step:,} | "
                  f"win_rate={stats['win_rate']:.3f} (greedy) / "
                  f"{samp['win_rate']:.3f} (sampled) | "
                  f"mean_vp={stats['mean_vp']:.1f} | "
                  f"ep_len={stats['mean_ep_len']:.0f} | "
                  f"elapsed={elapsed_eval:.0f}s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"eval/win_rate": stats["win_rate"],
                           "eval/mean_vp": stats["mean_vp"],
                           "eval/mean_ep_len": stats["mean_ep_len"]}, step=global_step)
            next_eval_step += args.eval_interval

        # Training checkpoint
        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step:08d}.pt")
            torch.save({
                "step":        global_step,
                "model_state": agent.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "args":        vars(args),
            }, ckpt_path)
            print(f"  [ckpt] saved → {ckpt_path}")
            next_ckpt_step += args.ckpt_interval

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    torch.save({
        "step":        global_step,
        "model_state": agent.state_dict(),
        "args":        vars(args),
    }, final_path)
    print(f"\nTraining done. Final checkpoint → {final_path}")
    print(f"Total time: {(time.time()-t0)/3600:.2f} h")

    if use_wandb:
        import wandb
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPO Self-Play Training — Puerto Rico (paper defaults)")
    p.add_argument("--total_timesteps", type=int,   default=5_000_000)
    p.add_argument("--n_steps",         type=int,   default=2048,
                   help="Rollout steps per env per update")
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--n_epochs",        type=int,   default=10)
    p.add_argument("--gamma",           type=float, default=1.0,
                   help="Discount factor (1.0 = undiscounted, as in paper)")
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_coef",       type=float, default=0.2)
    p.add_argument("--ent_coef",        type=float, default=0.01)
    p.add_argument("--vf_coef",         type=float, default=0.5)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--max_grad_norm",   type=float, default=0.5)
    p.add_argument("--num_envs",        type=int,   default=8,
                   help="Number of independent environments for rollout pooling")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--eval_interval",   type=int,   default=50_000,
                   help="Run evaluation every N timesteps")
    p.add_argument("--eval_episodes",   type=int,   default=200,
                   help="Number of evaluation games (PPO vs 2 Random)")
    p.add_argument("--ckpt_interval",   type=int,   default=500_000)
    p.add_argument("--out_dir",         type=str,   default="results/ppo_v3")
    p.add_argument("--device",          type=str,   default="auto")
    p.add_argument("--use_wandb",       action="store_true", default=False)
    p.add_argument("--obs_mode",        type=str, default="full",
                   choices=["full", "self_only"],
                   help="Observation mode: full (all players visible) or "
                        "self_only (opponent dims zeroed, for RQ3-C2 ablation)")
    p.add_argument("--no_egocentric",   dest="egocentric", action="store_false",
                   default=True,
                   help="Disable the egocentric observation rotation (absolute "
                        "seat order). Only for reproducing the old encoding — "
                        "parameter-sharing self-play needs the rotation.")
    p.add_argument("--training_mode",   type=str, default="self_play",
                   choices=["self_play", "fixed_random"],
                   help="self_play: shared policy for all agents (standard self-play); "
                        "fixed_random: only player_0 trained, opponents are fixed RandomBot")
    p.add_argument("--env_mode",        type=str, default="standard",
                   choices=["standard", "aoe_ablation"],
                   help="standard: original Puerto Rico; "
                        "aoe_ablation: selector-only phases (AOE ablated)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT)
    train(args)
