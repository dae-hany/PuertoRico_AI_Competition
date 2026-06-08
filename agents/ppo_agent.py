"""
agents/ppo_agent.py — PPO self-play network + inference agent.

``PpoNetwork`` is the actor-critic trained by ``training/train_ppo.py``.
``PpoAgent`` loads a trained checkpoint and plays greedily: at each move it takes
the highest-probability *legal* action. This is the only baseline that needs
PyTorch.

Reality check: the bundled checkpoint plays at roughly *random* strength. Puerto
Rico is hard for plain RL — that is the central finding of the benchmark this
competition is built on. The PPO agent is therefore a **starting point to
improve on**, not a strong opponent. The strong baselines are the heuristics and
MCTS.
"""
import numpy as np
import torch
import torch.nn as nn

from agents.base import Agent

OBS_DIM = 293
ACTION_DIM = 200


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight init — standard for PPO networks."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """Pre-norm residual block: ``x + MLP(LayerNorm(x))``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PpoNetwork(nn.Module):
    """PPO actor-critic with a shared residual-MLP trunk.

    ``obs(293) -> Embed -> N x ResidualBlock -> separate actor / critic heads``.
    Attribute names are kept stable so saved checkpoints load cleanly.
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 512, num_res_blocks: int = 3):
        super().__init__()
        self.embed = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.shared_trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _shared_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_trunk(self.embed(x))

    def actor_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self._shared_features(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._shared_features(x))

    def get_action_and_value(self, x, action_mask, action=None):
        """Training-time API: sample a legal action and return value/log-prob."""
        from torch.distributions.categorical import Categorical
        features = self._shared_features(x)
        logits = self.actor_head(features)
        neg = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask > 0.5, logits, neg)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(features)


class PpoAgent(Agent):
    """Plays a trained :class:`PpoNetwork` greedily (highest-probability legal move)."""

    name = "PPO"

    def __init__(self, checkpoint_path: str = None, obs_dim: int = OBS_DIM,
                 action_dim: int = ACTION_DIM, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.net = PpoNetwork(obs_dim, action_dim).to(device)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.net.load_state_dict(state)
        self.net.eval()

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.net.actor_logits(obs_t)
            neg = torch.tensor(-1e8, dtype=logits.dtype, device=self.device)
            masked = torch.where(mask_t > 0.5, logits, neg)
            return int(torch.argmax(masked, dim=1).item())
