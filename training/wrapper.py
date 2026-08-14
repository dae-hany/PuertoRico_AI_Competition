import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

from puerto_rico.constants import BUILDING_DATA, BuildingType, TileType
from puerto_rico.observation import (GLOBAL_DIM, PER_PLAYER_DIM,
                                     egocentric_view, flatten_observation)

class CleanRLAECWrapper(BaseWrapper):
    """
    Wrapper to make PuertoRicoEnv compatible with typical CleanRL PPO implementations.
    Specifically, this flattens the nested dictionary observation into a 1D array
    while keeping the action_mask available in the info dict or separate,
    depending on how CleanRL handles invalid action masking.

    Standard CleanRL MARL for PettingZoo expects the observation to be flattened.

    egocentric:
      True (default) — rotate the player blocks so block 0 is always the observing
      player (see ``puerto_rico.observation.egocentric_view``). Parameter-sharing
      self-play needs this: without it one network has to serve three different
      seat-to-offset mappings selected by a single raw scalar. Agents that consume
      a checkpoint must apply the same rotation — ``agents.ppo_agent.PpoAgent``
      does. Set False only to reproduce the old absolute-seat encoding.

    obs_mode:
      'full'      — all player states visible (default)
      'self_only' — opponent dims zeroed out (RQ3-C2 ablation)
    """
    # Obs layout constants (kept as attributes for backwards compatibility)
    _GLOBAL_DIM     = GLOBAL_DIM
    _PER_PLAYER_DIM = PER_PLAYER_DIM

    def __init__(self, env: AECEnv, obs_mode: str = 'full', egocentric: bool = True):
        super().__init__(env)
        assert obs_mode in ('full', 'self_only'), f"Unknown obs_mode: {obs_mode}"
        self.obs_mode = obs_mode
        self.egocentric = egocentric

        # Flatten the observation space, except the action_mask
        self._orig_obs_space = env.observation_space(env.possible_agents[0])

        # Calculate flattened size
        self._flat_size = self._get_flat_size(self._orig_obs_space["observation"])

        # Redefine observation spaces
        self.observation_spaces = {}
        for agent in env.possible_agents:
            self.observation_spaces[agent] = spaces.Dict({
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self._flat_size,), dtype=np.float32),
                "action_mask": self._orig_obs_space["action_mask"]
            })
            
    def _get_flat_size(self, space):
        if isinstance(space, spaces.Box):
            return np.prod(space.shape)
        elif isinstance(space, spaces.MultiBinary):
            return np.prod(space.shape) if isinstance(space.shape, tuple) else space.n
        elif isinstance(space, spaces.Dict):
            return sum(self._get_flat_size(s) for s in space.spaces.values())
        return 0
        
    def _flatten_obs(self, obs_dict):
        # Delegate to the canonical encoder so training and the competition
        # harness can never drift apart.
        return flatten_observation(obs_dict)

    def observe(self, agent: str):
        obs = self.env.observe(agent)
        flat_obs = self._flatten_obs(obs["observation"])
        n_players = self.env.unwrapped.num_players
        p_idx = self.env.unwrapped.agent_name_mapping[agent]

        if self.egocentric:
            # Centre on the *requested* agent, not on whoever is to move — this
            # method is also called off-turn (e.g. to bootstrap a value estimate).
            flat_obs = egocentric_view(flat_obs, self_idx=p_idx,
                                       num_players=n_players)
            self_block = 0
        else:
            flat_obs = flat_obs.copy()
            self_block = p_idx

        if self.obs_mode == 'self_only':
            # Zero out every opponent block; keep global + own player dims.
            for j in range(n_players):
                if j != self_block:
                    lo = GLOBAL_DIM + j * PER_PLAYER_DIM
                    flat_obs[lo:lo + PER_PLAYER_DIM] = 0.0

        return {
            "observation": flat_obs,
            "action_mask": obs["action_mask"]
        }

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

class PBRSWrapper(BaseWrapper):
    """
    Potential-Based Reward Shaping Wrapper.
    Adds dense rewards to the base environment's sparse rewards based on
    the change in game potential Φ(s') - Φ(s).

    Not used by the bundled trainer, but kept correct so that anyone who reaches
    for it gets the guarantee it is named after. Ng et al.'s result — that
    potential-based shaping leaves the optimal policy unchanged — holds only if

      * ``gamma`` is the **same discount the learner uses** (this repo trains at
        ``gamma=1.0``, so that is the default here; 0.99 quietly broke it), and
      * ``Φ(terminal) = 0``, otherwise the final transition pays out a shaped
        reward that no future state can cancel, which shifts the optimum.
    """
    def __init__(self, env: AECEnv, w_ship: float = 1.0, w_bldg: float = 1.0, w_doub: float = 1.0, gamma: float = 1.0):
        super().__init__(env)
        self.w_ship = w_ship
        self.w_bldg = w_bldg
        self.w_doub = w_doub
        self.gamma = gamma
        self._prev_potentials = {}
        
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self._prev_potentials = {
            f"player_{i}": self._compute_potential(i) for i in range(self.env.unwrapped.num_players)
        }
        
    def step(self, action):
        agent = self.env.agent_selection
        
        # Dead step check
        if self.env.terminations[agent] or self.env.truncations[agent]:
            self.env.step(action)
            return
            
        player_idx = self.env.unwrapped.agent_name_mapping[agent]
        
        self.env.step(action)
        
        # Add PBRS to the acting player. Φ(terminal) is 0 by definition, so a
        # move that ends the game is shaped by -Φ(s) alone.
        terminal = not self.env.agents or all(
            self.env.terminations.get(a, False) or self.env.truncations.get(a, False)
            for a in self.env.agents)
        new_potential = 0.0 if terminal else self._compute_potential(player_idx)
        old_potential = self._prev_potentials[f"player_{player_idx}"]

        shaping_reward = (self.gamma * new_potential) - old_potential
        
        self.env.rewards[agent] += shaping_reward
        
        # Update potentials for all players
        for i in range(self.env.unwrapped.num_players):
            self._prev_potentials[f"player_{i}"] = self._compute_potential(i)
            
    def _compute_potential(self, player_idx: int) -> float:
        game = self.env.unwrapped.game
        if game is None:
            return 0.0
            
        p = game.players[player_idx]
        
        # 1. Base components
        shipping_vp = p.vp_chips * self.w_ship
        building_vp = sum(BUILDING_DATA[b.building_type][1] for b in p.city_board) * self.w_bldg
        doubloons = p.doubloons * self.w_doub
        
        # 2. Large building anticipated bonus
        large_bonus_vp = 0.0
        if p.is_building_occupied(BuildingType.GUILDHALL):
            prod_count = sum(1 for b in p.city_board if 0 <= b.building_type.value <= 5)
            large_bonus_vp = (prod_count * 1.5) * self.w_bldg
        elif p.is_building_occupied(BuildingType.CITY_HALL):
            violet_count = sum(1 for b in p.city_board if 6 <= b.building_type.value <= 22)
            large_bonus_vp = (violet_count * 1.0) * self.w_bldg
        elif p.is_building_occupied(BuildingType.CUSTOMS_HOUSE):
            large_bonus_vp = (p.vp_chips // 4) * self.w_bldg
        elif p.is_building_occupied(BuildingType.RESIDENCE):
            island_count = sum(1 for t in p.island_board if t.tile_type != TileType.EMPTY)
            large_bonus_vp = (max(0, island_count - 9) * 1.5) * self.w_bldg
        elif p.is_building_occupied(BuildingType.FORTRESS):
            large_bonus_vp = (p.total_colonists_owned // 3) * self.w_bldg
            
        return shipping_vp + building_vp + doubloons + large_bonus_vp
