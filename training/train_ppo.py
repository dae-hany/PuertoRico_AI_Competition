"""
train_ppo.py — PPO Self-Play Training for Puerto Rico (paper hyperparameters)

All 3 agents share a single PPOAgent network (parameter-sharing self-play).
Experiences from num_envs independent environments are pooled into one rollout
buffer (n_steps * num_envs total steps) before each PPO update.

Key hyperparameter defaults match the paper specification:
  gamma=1.0  (undiscounted — sparse terminal reward)
  n_steps=2048, n_epochs=10, lr=3e-4, num_envs=8

Usage:
  python train_ppo.py                              # full run
  python train_ppo.py --total_timesteps 10000 \\   # smoke test
    --num_envs 2 --n_steps 256 \\
    --eval_interval 5000 --eval_episodes 10 \\
    --out_dir results/smoke_test

Outputs (in --out_dir):
  training_log.csv   — one row per PPO update
  eval_log.csv       — one row per evaluation checkpoint
  checkpoints/       — ckpt_XXXXXXXX.pt every --ckpt_interval steps
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.dirname(os.path.abspath(__file__))
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

def make_env(obs_mode: str = 'full', env_mode: str = 'standard'):
    if env_mode == 'aoe_ablation':
        from env.aoe_ablation_env import AOEAblationEnv
        env = AOEAblationEnv(num_players=N_PLAYERS, random_seed_mode=True)
    else:
        env = PuertoRicoEnv(num_players=N_PLAYERS, random_seed_mode=True)
    return CleanRLAECWrapper(env, obs_mode=obs_mode)


# ── single-env rollout collection ─────────────────────────────────────────────

def collect_rollout_one_env(env, agent, num_steps: int, device: str,
                             training_mode: str = 'self_play'):
    """
    Collect num_steps AEC steps from one environment using the shared policy.

    training_mode='self_play'   — all agents use shared policy (standard)
    training_mode='fixed_random'— only player_0 trained; opponents use RandomBot
                                  (exactly num_steps player_0 transitions stored)
    """
    obs_buf  = np.zeros((num_steps, OBS_DIM), dtype=np.float32)
    act_buf  = np.zeros(num_steps,            dtype=np.int64)
    logp_buf = np.zeros(num_steps,            dtype=np.float32)
    val_buf  = np.zeros(num_steps,            dtype=np.float32)
    rew_buf  = np.zeros(num_steps,            dtype=np.float32)
    done_buf = np.zeros(num_steps,            dtype=np.float32)
    mask_buf = np.zeros((num_steps, ACT_DIM), dtype=np.float32)

    ep_wins   = []
    ep_vps    = []
    ep_lens   = []
    ep_step_count = 0

    random_bots = [RandomBot(), RandomBot()] if training_mode == 'fixed_random' else None

    env.reset()
    step = 0

    while step < num_steps:
        agent_name = env.agent_selection
        term  = env.terminations.get(agent_name, False)
        trunc = env.truncations.get(agent_name, False)

        if term or trunc:
            if agent_name == "player_0":
                base   = env.unwrapped
                scores = base.game.get_scores()
                best_vp = max(s[0] for s in scores)
                best_tb = max(s[1] for s in scores if s[0] == best_vp)
                winners = [i for i, s in enumerate(scores)
                           if s[0] == best_vp and s[1] == best_tb]
                ep_wins.append(1.0 if 0 in winners else 0.0)
                ep_vps.append(scores[0][0])
                ep_lens.append(ep_step_count)
                ep_step_count = 0
            env.step(None)
            if not env.agents:
                env.reset()
            continue

        obs  = env.observe(agent_name)
        flat = obs["observation"].astype(np.float32)
        mask = obs["action_mask"].astype(np.float32)

        obs_t  = torch.FloatTensor(flat).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        if training_mode == 'fixed_random':
            p_idx = int(agent_name.split('_')[1])
            if p_idx != 0:
                with torch.no_grad():
                    action_t, _, _, _ = random_bots[p_idx - 1].get_action_and_value(obs_t, mask_t)
                env.step(int(action_t.item()))
                ep_step_count += 1
                if not env.agents:
                    env.reset()
                    ep_step_count = 0
                continue

        with torch.no_grad():
            action, logp, _, value = agent.get_action_and_value(obs_t, mask_t)

        a = int(action.item())
        env.step(a)
        ep_step_count += 1

        rew  = float(env.unwrapped.rewards.get(agent_name, 0.0))
        done = float(env.terminations.get(agent_name, False) or
                     env.truncations.get(agent_name, False))

        obs_buf[step]  = flat
        act_buf[step]  = a
        logp_buf[step] = float(logp.item())
        val_buf[step]  = float(value.item())
        rew_buf[step]  = rew
        done_buf[step] = done
        mask_buf[step] = mask
        step += 1

        if done and not env.agents:
            env.reset()
            ep_step_count = 0

    # Bootstrap value at end of rollout
    bootstrap_val = 0.0
    if env.agents:
        if training_mode == 'fixed_random':
            # Always bootstrap from player_0's perspective (rollout may end on an opponent's turn)
            if ("player_0" in env.agents and
                    not env.terminations.get("player_0", False) and
                    not env.truncations.get("player_0", False)):
                obs = env.observe("player_0")
                obs_t  = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(obs["action_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    bootstrap_val = float(agent.get_value(obs_t).item())
        else:
            cur = env.agent_selection
            if not env.terminations.get(cur, False) and not env.truncations.get(cur, False):
                obs  = env.observe(cur)
                obs_t  = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(obs["action_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    bootstrap_val = float(agent.get_value(obs_t).item())

    return (obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf,
            bootstrap_val, ep_wins, ep_vps, ep_lens)


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rew_buf, val_buf, done_buf, bootstrap_val, gamma, gae_lambda):
    n        = len(rew_buf)
    adv_buf  = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        next_val  = bootstrap_val if t == n - 1 else val_buf[t + 1]
        next_done = 0.0          if t == n - 1 else done_buf[t + 1]
        delta      = rew_buf[t] + gamma * next_val * (1 - next_done) - val_buf[t]
        last_gae   = delta + gamma * gae_lambda * (1 - next_done) * last_gae
        adv_buf[t] = last_gae

    ret_buf = adv_buf + val_buf
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

    pg_losses, v_losses, ent_losses = [], [], []

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

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(ent_loss.item())

    return float(np.mean(pg_losses)), float(np.mean(v_losses)), float(np.mean(ent_losses))


# ── in-training evaluation (PPO vs Random × 2) ────────────────────────────────

def run_eval(agent, n_episodes: int, device: str, obs_mode: str = 'full',
             env_mode: str = 'standard') -> dict:
    """Run n_episodes of PPO (player_0) vs 2 RandomBots. Returns stats dict."""
    random_agents = [RandomBot(), RandomBot()]
    wins, vps, ep_lens = [], [], []

    for _ in range(n_episodes):
        env = make_env(obs_mode=obs_mode, env_mode=env_mode)
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
                    action_t, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                action = int(action_t.item())
            else:
                action_t, _, _, _ = random_agents[p_idx - 1].get_action_and_value(obs_t, mask_t)
                action = int(action_t.item())

            env.step(action)
            ep_step += 1

        base   = env.unwrapped
        scores = base.game.get_scores()
        best_vp = max(s[0] for s in scores)
        best_tb = max(s[1] for s in scores if s[0] == best_vp)
        winners = [i for i, s in enumerate(scores)
                   if s[0] == best_vp and s[1] == best_tb]
        wins.append(1.0 if 0 in winners else 0.0)
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
            "step", "pg_loss", "v_loss", "ent_loss", "lr",
            "episodes", "win_rate_rolling", "mean_vp_rolling", "mean_ep_len",
        ])
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "win_rate", "mean_vp", "std_vp", "mean_ep_len",
        ])

    # Optional WandB
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project="puerto_rico_ppo", config=vars(args))

    # Agent + optimizer
    agent     = PPOAgent(obs_dim=OBS_DIM, action_dim=ACT_DIM).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # LR: linear decay to 0 over total_timesteps
    total_updates = args.total_timesteps // (args.n_steps * args.num_envs)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: max(0.0, 1.0 - u / max(1, total_updates))
    )

    # Build envs once; reuse across rollouts
    envs = [make_env(obs_mode=args.obs_mode, env_mode=args.env_mode) for _ in range(args.num_envs)]

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
          f"gamma={args.gamma} | lr={args.lr}")

    while global_step < args.total_timesteps:
        agent.eval()

        # Collect n_steps from each env, pool all experiences
        all_obs, all_act, all_logp, all_val = [], [], [], []
        all_rew, all_done, all_mask         = [], [], []
        all_boot                             = []

        for env in envs:
            (o, a, lp, v, r, d, m, bv,
             ew, evp, el) = collect_rollout_one_env(env, agent, args.n_steps, device,
                                                     training_mode=args.training_mode)
            all_obs.append(o);  all_act.append(a);   all_logp.append(lp)
            all_val.append(v);  all_rew.append(r);   all_done.append(d)
            all_mask.append(m); all_boot.append(bv)
            recent_wins.extend(ew)
            recent_vps.extend(evp)
            recent_lens.extend(el)

        steps_this_update = args.n_steps * args.num_envs
        global_step      += steps_this_update
        update_count     += 1

        # Compute GAE per env, then concatenate
        adv_parts, ret_parts = [], []
        for i in range(args.num_envs):
            adv, ret = compute_gae(
                all_rew[i], all_val[i], all_done[i], all_boot[i],
                args.gamma, args.gae_lambda,
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
        pg_l, v_l, ent_l = ppo_update(
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
                round(cur_lr, 7), len(recent_wins),
                round(wr, 4), round(mvp, 2), round(mlen, 1),
            ])

        if use_wandb:
            import wandb
            wandb.log({"step": global_step, "pg_loss": pg_l, "v_loss": v_l,
                       "ent_loss": ent_l, "lr": cur_lr,
                       "win_rate_rolling": wr, "mean_vp_rolling": mvp}, step=global_step)

        print(f"step={global_step:>9,} | wr={wr:.3f} | vp={mvp:.1f} | "
              f"pg={pg_l:.4f} v={v_l:.4f} | sps={sps:.0f}")

        # Evaluation checkpoint
        if global_step >= next_eval_step:
            agent.eval()
            print(f"  [eval] running {args.eval_episodes} games…", flush=True)
            t_eval = time.time()
            stats = run_eval(agent, args.eval_episodes, device, obs_mode=args.obs_mode,
                             env_mode=args.env_mode)
            elapsed_eval = time.time() - t_eval
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    global_step,
                    round(stats["win_rate"],    4),
                    round(stats["mean_vp"],     2),
                    round(stats["std_vp"],      2),
                    round(stats["mean_ep_len"], 1),
                ])
            print(f"  [eval] step={global_step:,} | "
                  f"win_rate={stats['win_rate']:.3f} | "
                  f"mean_vp={stats['mean_vp']:.1f} | "
                  f"ep_len={stats['mean_ep_len']:.0f} | "
                  f"elapsed={elapsed_eval:.0f}s")
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
    p.add_argument("--out_dir",         type=str,   default="results/ppo_v2")
    p.add_argument("--device",          type=str,   default="auto")
    p.add_argument("--use_wandb",       action="store_true", default=False)
    p.add_argument("--obs_mode",        type=str, default="full",
                   choices=["full", "self_only"],
                   help="Observation mode: full (all players visible) or "
                        "self_only (opponent dims zeroed, for RQ3-C2 ablation)")
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
