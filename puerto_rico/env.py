import copy
import random

from pettingzoo import AECEnv
try:
    from pettingzoo.utils.agent_selector import AgentSelector  # pettingzoo >= 1.24.4
except ImportError:  # older pettingzoo only exposes the lowercase name
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
from gymnasium import spaces
import numpy as np

from puerto_rico.engine import PuertoRicoGame
from puerto_rico.constants import Phase, Role, Good, TileType, BuildingType, BUILDING_DATA, VP_CHIPS_SETUP, COLONIST_SUPPLY_SETUP

_GOODS = tuple(Good)
_ROLES = tuple(Role)
_NOT_BUILDINGS = (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE)
_PRODUCES = {b: data[5] for b, data in BUILDING_DATA.items() if data[5] is not None}

# Flat layout of the observation: (key, size) in key order; roles_available is int8
# and kept apart from the float32 buffer (size None).
_GLOBAL_LAYOUT = (
    ("cargo_ships_good_onehot", 18), ("cargo_ships_load", 3), ("cargo_ships_space", 3),
    ("colonists_ship", 1), ("colonists_supply", 1), ("current_phase_onehot", 10),
    ("current_player", 1), ("face_up_plantation_counts", 6), ("game_progress", 1),
    ("goods_supply", 5), ("governor_idx", 1),
    ("quarry_stack", 1), ("role_doubloons", 8), ("roles_available", None),
    ("trading_house_count", 1), ("trading_house_has_good", 5), ("vp_chips", 1),
)
_PLAYER_LAYOUT = (
    ("building_colonists", 23), ("doubloons", 1), ("empty_city_spaces", 1), ("goods", 5),
    ("has_building", 23), ("island_empty_spaces", 1), ("island_tile_count", 6),
    ("island_tile_occupied", 6), ("production_capacity", 5), ("unplaced_colonists", 1),
    ("vp_chips", 1),
)


def _copy_nested(v):
    """Fast deep copy of JSON-like bookkeeping (dicts/lists of scalars).

    Used for ``episode_metrics``, which generic ``copy.deepcopy`` walks one
    dispatch at a time — measurable when planning agents clone the env
    thousands of times per move. Tuples of scalars are shared (immutable).
    """
    if isinstance(v, dict):
        return {k: _copy_nested(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy_nested(x) for x in v]
    if isinstance(v, tuple):
        if all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
            return v
        return tuple(_copy_nested(x) for x in v)
    return v


# Sparse rewards for benchmark
class PuertoRicoEnv(AECEnv):
    metadata = {'render.modes': ['human'], 'name': 'puerto_rico_v0'}

    def __init__(self, num_players: int = 3, random_seed_mode: bool = True, fixed_seed: int = 42):
        super(PuertoRicoEnv, self).__init__()
        self.num_players = num_players
        self.random_seed_mode = random_seed_mode
        self.fixed_seed = fixed_seed
        self.game = None

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.num_players))))

        self._action_spaces = {agent: self._define_action_space() for agent in self.possible_agents}
        self._observation_spaces = {agent: self._define_observation_space() for agent in self.possible_agents}

        # Slices of the flat observation buffer that _get_obs builds in one pass.
        offset = 0
        self._obs_global_slices = []
        for key, size in _GLOBAL_LAYOUT:
            if size is None:
                self._obs_global_slices.append((key, None, None))
            else:
                self._obs_global_slices.append((key, offset, offset + size))
                offset += size
        self._obs_player_slices = []
        for agent in self.possible_agents:
            slices = []
            for key, size in _PLAYER_LAYOUT:
                slices.append((key, offset, offset + size))
                offset += size
            self._obs_player_slices.append((agent, slices))
        space = self._observation_spaces[self.possible_agents[0]]["observation"]
        assert [k for k, _ in _GLOBAL_LAYOUT] == list(space["global_state"].spaces)
        assert all(size is None or space["global_state"][k].shape == (size,)
                   for k, size in _GLOBAL_LAYOUT)
        assert [k for k, _ in _PLAYER_LAYOUT] == list(space["players"]["player_0"].spaces)
        assert all(space["players"]["player_0"][k].shape == (size,) for k, size in _PLAYER_LAYOUT)

    # Static config built once in __init__ and never mutated during a game.
    # Sharing (not deep-copying) these makes ForwardModel.clone() ~7x faster,
    # since the gymnasium observation Dict-space is by far the costliest object
    # to deep-copy (it dominates a full env deepcopy). Planning agents clone the
    # env thousands of times per move, so this directly sets the search budget.
    _CLONE_SHARED_ATTRS = frozenset({
        "_action_spaces", "_observation_spaces",
        "possible_agents", "agent_name_mapping",
        "_obs_global_slices", "_obs_player_slices",
    })

    # Flat {agent_name: scalar} bookkeeping — a shallow dict copy is a deep copy.
    _CLONE_FLAT_DICT_ATTRS = frozenset({
        "rewards", "_cumulative_rewards", "terminations", "truncations",
    })

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        d = self.__dict__
        # episode_metrics is seeded into the memo *before* infos is copied:
        # _finalize_episode aliases the same metrics object into every agent's
        # info dict, and that aliasing must survive the clone exactly as the
        # generic deepcopy would preserve it.
        em = d.get("episode_metrics")
        if em is not None and id(em) not in memo:
            memo[id(em)] = _copy_nested(em)
        for k, v in d.items():
            if k in self._CLONE_SHARED_ATTRS:
                new.__dict__[k] = v                       # share immutable config
            elif k in self._CLONE_FLAT_DICT_ATTRS:
                new.__dict__[k] = dict(v)
            elif k == "agents":
                new.__dict__[k] = list(v)
            elif k == "episode_metrics":
                new.__dict__[k] = memo[id(v)]
            elif v is None or isinstance(v, (int, float, str, bool)):
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)  # copy mutable game state
        return new

    def _define_action_space(self) -> spaces.Discrete:
        # === Action Mapping ===
        # 0-7:     Pick Role (Role.SETTLER=0 ~ Role.PROSPECTOR_2=7)
        # 8-12:    Settler - Face up plantation by TileType 0~4 (8=Coffee, 9=Tobacco, 10=Corn, 11=Sugar, 12=Indigo)
        # 13:      Settler - Take Quarry from quarry stack (role holder / Construction Hut)
        # 15:      Pass (phase-dependent: Settler/Builder/Trader/Captain/Mayor/Store/Craftsman)
        # 16-38:   Builder - Build building (BuildingType 0~22)
        # 39-43:   Trader - Sell good (Good 0~4)
        # 44-58:   Captain - Load (ship_idx * 5 + good_type)
        # 59-63:   Captain - Load via Wharf (Good 0~4)
        # 64-68:   Captain Store Windrose - Keep Good (Good 0~4)
        # 93-97:   Craftsman - Privilege good selection (Good 0~4)
        # 105:     Settler - Hacienda (draw extra plantation from stack)
        # 106-110: Captain Store Warehouse (Good 0~4)
        # 120-125: Mayor - Place colonist on Island by TileType (120+TileType.value)
        # 140-162: Mayor - Place colonist on City by BuildingType (140+BuildingType.value)
        return spaces.Discrete(200)

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def _define_observation_space(self) -> spaces.Dict:
        """
        Semantic observation space with proper encoding for categorical variables.

        Key changes from raw-integer scalar encoding:
        - island_tiles: per-type count (6) + per-type occupied count (6)  [was: 12 ordinals]
        - city_buildings: binary has_building (23) + colonists (23)       [was: 12 ordinals + 12 ints]
        - cargo_ships_good: one-hot per ship (3x6=18)                    [was: 3 ordinals]
        - trading_house: binary per good type (5) + fill count (1)       [was: 4 ordinals]
        - face_up_plantations: per-type count (6)                        [was: 4 ordinals]
        - current_phase: one-hot (10)                                    [was: 1 ordinal]
        - NEW derived: production_capacity(5), game_progress(1),
                       cargo_ship_space(3), island/city empty spaces

        Total obs_dim: 74 (global) + 73 x num_players (per-player) = 293 for 3P
        """
        obs_space = {
            "observation": spaces.Dict({
                "global_state": spaces.Dict({
                    # Cargo ships: one-hot good type (6 classes: 5 goods + none) per ship
                    "cargo_ships_good_onehot": spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32),
                    "cargo_ships_load": spaces.Box(low=0, high=15, shape=(3,), dtype=np.float32),
                    "cargo_ships_space": spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32),
                    "colonists_ship": spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
                    "colonists_supply": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                    # Phase: one-hot (9 phases + 1 for None/INIT)
                    "current_phase_onehot": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                    "current_player": spaces.Box(low=0, high=self.num_players - 1, shape=(1,), dtype=np.float32),
                    # Face-up plantations: count per tile type (0-5, no per-slot ordinals)
                    "face_up_plantation_counts": spaces.Box(low=0, high=self.num_players + 1, shape=(6,), dtype=np.float32),
                    "game_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    "goods_supply": spaces.Box(low=0, high=15, shape=(5,), dtype=np.float32),
                    "governor_idx": spaces.Box(low=0, high=self.num_players - 1, shape=(1,), dtype=np.float32),
                    "quarry_stack": spaces.Box(low=0, high=9, shape=(1,), dtype=np.float32),
                    "role_doubloons": spaces.Box(low=0, high=20, shape=(8,), dtype=np.float32),
                    "roles_available": spaces.MultiBinary(8),
                    # Trading house: binary flags per good type (not per-slot ordinals)
                    "trading_house_count": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
                    "trading_house_has_good": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
                    "vp_chips": spaces.Box(low=-50, high=200, shape=(1,), dtype=np.float32),
                }),
                "players": spaces.Dict({
                    f"player_{i}": spaces.Dict({
                        # Per building type: colonist count (0 if not owned)
                        "building_colonists": spaces.Box(low=0, high=3, shape=(23,), dtype=np.float32),
                        "doubloons": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                        "empty_city_spaces": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),
                        "goods": spaces.Box(low=0, high=15, shape=(5,), dtype=np.float32),
                        # Binary: does player own BuildingType(j)? (j=0..22)
                        "has_building": spaces.Box(low=0, high=1, shape=(23,), dtype=np.float32),
                        "island_empty_spaces": spaces.Box(low=0, high=12, shape=(1,), dtype=np.float32),
                        # Count of each tile type (Coffee/Tobacco/Corn/Sugar/Indigo/Quarry)
                        "island_tile_count": spaces.Box(low=0, high=12, shape=(6,), dtype=np.float32),
                        # Occupied count per tile type
                        "island_tile_occupied": spaces.Box(low=0, high=12, shape=(6,), dtype=np.float32),
                        # Derived: min(occupied_plantations, building_capacity) per good
                        "production_capacity": spaces.Box(low=0, high=12, shape=(5,), dtype=np.float32),
                        "unplaced_colonists": spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
                        "vp_chips": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
                    }) for i in range(self.num_players)
                })
            }),
            "action_mask": spaces.Box(low=0, high=1, shape=(200,), dtype=np.int8)
        }
        return spaces.Dict(obs_space)

    def _episode_rng(self, seed) -> random.Random:
        """Build the RNG this episode's engine will own.

        The process-global ``random`` / ``np.random`` states are never touched,
        so env seeding cannot couple with an agent's own RNG use (MCTS
        determinization, exploration noise, a heuristic's tie-breaks, ...) in
        either direction.

            reset(seed=S)           -> Random(S)           reproducible game
            random_seed_mode=True   -> Random()            fresh deal each reset
            random_seed_mode=False  -> Random(fixed_seed)  identical episodes
        """
        if seed is not None:
            return random.Random(seed)
        if self.random_seed_mode:
            return random.Random()
        return random.Random(self.fixed_seed)

    def _make_game(self, rng: random.Random) -> PuertoRicoGame:
        return PuertoRicoGame(self.num_players, rng=rng)

    def reset(self, seed=None, options=None):
        rng = self._episode_rng(seed)
        # Remember the seed so ForwardModel can derive a reproducible — but
        # engine-independent — RNG for determinizing the hidden plantation order.
        self._seed = seed if self.random_seed_mode else self.fixed_seed

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.episode_metrics = {
            "role_selections": {r.name: 0 for r in Role},
            "role_selections_by_round": {},
            "player_stats": {}
        }

        self.game = self._make_game(rng)
        self.game.start_game()
        self._game_step_count = 0

        # Determine starting player based on engine
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = f"player_{self.game.current_player_idx}"

        # Populate initial info
        for agent in self.agents:
            self.infos[agent] = self._get_info()

    def observe(self, agent: str):
        obs = self._get_obs()

        # Action mask should be generated for the requested agent
        # If it's not their turn, their mask is all 0.
        agent_idx = self.agent_name_mapping[agent]
        if agent_idx == self.game.current_player_idx and not (self.terminations.get(agent, True) or self.truncations.get(agent, True)):
            mask = self.valid_action_mask().astype(np.int8)
        else:
            mask = np.zeros(200, dtype=np.int8)

        # PettingZoo standard convention requires action_mask inside the top-level Dict
        return {
            "observation": obs,
            "action_mask": mask
        }

    def _mayor_empty_slots(self, p):
        """Colonist-placeable slots the player has right now, as (is_city, idx)."""
        slots = []
        for i, t in enumerate(p.island_board):
            if t.tile_type != TileType.EMPTY and not t.is_occupied:
                slots.append((False, i))
        for i, b in enumerate(p.city_board):
            if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                cap = BUILDING_DATA[b.building_type][2]
                for _ in range(cap - b.colonists):
                    slots.append((True, i))
        return slots

    def _execute_auto_actions(self):
        """
        Automatically execute actions when there's no meaningful choice.
        Returns True if an auto-action was executed, False otherwise.
        """
        if self.game.current_phase is None:
            return False

        player_idx = self.game.current_player_idx
        p = self.game.players[player_idx]
        phase = self.game.current_phase

        # 1. Mayor is the only phase with no Pass action, so a player with
        #    nothing to place has a legitimately empty action mask — and would
        #    stall the phase rather than be auto-advanced. This check therefore
        #    has to run *before* the empty-mask early return below, otherwise it
        #    is unreachable. (Engine invariants make the state unreachable today:
        #    every seat is recalled at the start of its Mayor turn, which frees
        #    all of its slots, and no seat can own zero colonists or zero tiles.
        #    This keeps the env safe if those invariants ever change.)
        empty_slots = self._mayor_empty_slots(p) if phase == Phase.MAYOR else []
        if phase == Phase.MAYOR and (p.unplaced_colonists == 0 or len(empty_slots) == 0):
            self.game._advance_phase_turn()
            return True

        # 2. Check for pass-only or single-choice situations
        mask = self.valid_action_mask()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            return False

        if len(valid_actions) == 1:
            action = valid_actions[0]

            # Auto-execute the only valid action
            if action == 15:  # Pass
                self._handle_pass(player_idx)
                return True

            # Captain forced shipping (only one ship/good combo)
            if phase == Phase.CAPTAIN and 44 <= action <= 63:
                if 44 <= action <= 58:
                    idx = action - 44
                    ship_idx = idx // 5
                    g_type = Good(idx % 5)
                    self.game.action_captain_load(player_idx, ship_idx, g_type)
                else:  # Wharf
                    g_type = Good(action - 59)
                    self.game.action_captain_load(player_idx, -1, g_type)
                return True

            # Captain Store forced choice
            if phase == Phase.CAPTAIN_STORE and action != 15:
                if 64 <= action <= 68:
                    g_type = Good(action - 64)
                    self.game.action_captain_store_windrose(player_idx, g_type)
                    return True
                elif 106 <= action <= 110:
                    g_type = Good(action - 106)
                    self.game.action_captain_store_warehouse(player_idx, g_type)
                    return True

            # Craftsman auto-pass for non-privilege holders
            if phase == Phase.CRAFTSMAN and action == 15:
                self.game.action_craftsman(player_idx, privilege_good=None)
                return True

        # 3. Mayor phase auto-fill: more colonists than slots, so where each one
        #    goes carries no decision — place them without asking.
        if phase == Phase.MAYOR and p.unplaced_colonists >= len(empty_slots):
            is_city, slot_idx = empty_slots[0]
            self.game.action_mayor_place_colonist(player_idx, is_city=is_city, slot_idx=slot_idx)
            return True

        return False

    def _finalize_episode(self) -> None:
        """Populate episode stats, assign terminal rewards, and set termination flags."""
        scores = self.game.get_scores()
        for idx, p in enumerate(self.game.players):
            agent_name = f"player_{idx}"
            total_vp, tie_breaker, shipping_vp, building_vp, bonus_vp = scores[idx]
            self.episode_metrics["player_stats"][agent_name] = {
                "total_vp": total_vp,
                "tie_breaker": tie_breaker,
                "shipping_vp": shipping_vp,
                "building_vp": building_vp,
                "bonus_vp": bonus_vp,
                "goods_produced": p.goods_produced,
                "doubloons_earned": p.doubloons_earned
            }
        all_rewards = self._calculate_all_rewards()
        for idx, r in enumerate(all_rewards):
            agent_name = f"player_{idx}"
            self.rewards[agent_name] = r
            self.terminations[agent_name] = True
        final_scores = self.game.get_scores()
        for a in self.agents:
            self.infos[a]["final_scores"] = final_scores
            self.infos[a]["episode_metrics"] = self.episode_metrics

    def step(self, action):
        if (
            self.terminations.get(self.agent_selection, True)
            or self.truncations.get(self.agent_selection, True)
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        player_idx = self.agent_name_mapping[agent]
        p = self.game.players[player_idx]

        try:
            if 0 <= action <= 7:
                role = Role(action)
                # Track total role selections
                self.episode_metrics["role_selections"][role.name] += 1
                # Track role selections by round
                round_num = self.game.round_number
                if round_num not in self.episode_metrics["role_selections_by_round"]:
                    self.episode_metrics["role_selections_by_round"][round_num] = {r.name: 0 for r in Role}
                self.episode_metrics["role_selections_by_round"][round_num][role.name] += 1
                self.game.select_role(player_idx, role)

            elif 8 <= action <= 13:
                # Settler Phase (No Hacienda)
                if action <= 12:
                    target_type = TileType(action - 8)
                    idx = -1
                    for i, t in enumerate(self.game.face_up_plantations):
                        if t == target_type:
                            idx = i
                            break
                    if idx == -1:
                        raise ValueError(f"Tile {target_type.name} not available in face_up_plantations.")
                    self.game.action_settler(player_idx, tile_choice=idx)
                else:
                    self.game.action_settler(player_idx, tile_choice=-1) # Quarry

            elif action == 15:
                # Pass
                self._handle_pass(player_idx)

            elif 16 <= action <= 38:
                # Builder Phase (23 buildings)
                b_type = BuildingType(action - 16)
                self.game.action_builder(player_idx, building_choice=b_type)

            elif 39 <= action <= 43:
                # Trader Phase
                g_type = Good(action - 39)
                self.game.action_trader(player_idx, sell_good=g_type)

            elif 44 <= action <= 58:
                # Captain Load (5 goods * 3 ships)
                idx = action - 44
                ship_idx = idx // 5
                g_type = Good(idx % 5)
                self.game.action_captain_load(player_idx, ship_idx, g_type)

            elif 59 <= action <= 63:
                # Captain Load Wharf
                g_type = Good(action - 59)
                self.game.action_captain_load(player_idx, -1, g_type)

            elif 64 <= action <= 68:
                # Captain Store Windrose
                g_type = Good(action - 64)
                self.game.action_captain_store_windrose(player_idx, g_type)

            elif 106 <= action <= 110:
                # Captain Store Warehouse
                g_type = Good(action - 106)
                self.game.action_captain_store_warehouse(player_idx, g_type)

            elif action == 105:
                # Hacienda draw
                self.game.action_hacienda_draw(player_idx)

            elif 120 <= action <= 125:
                # Mayor Placement (Island) by TileType
                target_type = TileType(action - 120)
                idx = -1
                for i, t in enumerate(p.island_board):
                    if t.tile_type == target_type and not t.is_occupied:
                        idx = i
                        break
                if idx == -1:
                    raise ValueError(f"No unoccupied {target_type.name} found on island.")
                self.game.action_mayor_place_colonist(player_idx, is_city=False, slot_idx=idx)

            elif 140 <= action <= 162:
                # Mayor Placement (City) by BuildingType
                target_type = BuildingType(action - 140)
                idx = -1
                for i, b in enumerate(p.city_board):
                    if b.building_type == target_type and b.colonists < BUILDING_DATA[b.building_type][2]:
                        idx = i
                        break
                if idx == -1:
                    raise ValueError(f"No valid building of type {target_type.name} found in city.")
                self.game.action_mayor_place_colonist(player_idx, is_city=True, slot_idx=idx)

            elif 93 <= action <= 97:
                # Craftsman Privilege
                g_type = Good(action - 93)
                self.game.action_craftsman(player_idx, privilege_good=g_type)

        except ValueError as e:
            # An action the mask advertised as legal but the engine rejects is a
            # bug in this repo, not a game event. It used to be swallowed here:
            # the actor got -10 and the whole game was terminated, which quietly
            # corrupted a result instead of reporting a defect (exactly how a
            # Mayor-phase deadlock went unnoticed in the research code until it
            # was made loud). Agents cannot trigger this — the harness already
            # replaces illegal *agent* moves with a random legal one before the
            # env ever sees them.
            raise AssertionError(
                f"valid_action_mask() allowed action {action} but the engine "
                f"rejected it in phase "
                f"{self.game.current_phase.name if self.game.current_phase else 'INIT'} "
                f"for {agent}: {e}"
            ) from e

        self._game_step_count += 1
        done = self.game.check_game_end()

        if done:
            self._finalize_episode()
        else:
            for i in range(self.num_players):
                a_name = f"player_{i}"
                self.rewards[a_name] = 0.0
                self.infos[a_name] = self._get_info(game_over=False)
                # Expose phase ID as int for hierarchical agent routing
                self.infos[a_name]["current_phase_id"] = int(
                    self.game.current_phase if self.game.current_phase is not None else 9
                )

        # Execute auto-actions (pass-only, forced captain, etc.)
        while not all(self.terminations.values()) and not all(self.truncations.values()):
            if not self._execute_auto_actions():
                break
            # Check game end after each auto-action
            if self.game.check_game_end():
                self._finalize_episode()
                break

        self.agent_selection = f"player_{self.game.current_player_idx}"
        self._accumulate_rewards()

    def _handle_pass(self, player_idx: int):
        if self.game.current_phase == Phase.SETTLER:
            self.game.action_settler(player_idx, tile_choice=-2)
        elif self.game.current_phase == Phase.BUILDER:
            self.game.action_builder(player_idx, building_choice=None)
        elif self.game.current_phase == Phase.TRADER:
            self.game.action_trader(player_idx, sell_good=None)
        elif self.game.current_phase == Phase.CAPTAIN:
            self.game.action_captain_pass(player_idx)
        elif self.game.current_phase == Phase.CAPTAIN_STORE:
            self.game.action_captain_store_pass(player_idx)
        elif self.game.current_phase == Phase.MAYOR:
            raise ValueError("Cannot pass in Mayor phase - must select a strategy.")
        elif self.game.current_phase == Phase.CRAFTSMAN:
            self.game.action_craftsman(player_idx, privilege_good=None)
        else:
            raise ValueError(f"Cannot pass in phase {self.game.current_phase.name if self.game.current_phase else 'INIT'}")

    def _get_obs(self):
        """Observation dict of _define_observation_space. All float32 values are built in
        one pass into a single buffer, and every entry is a slice (view) of it; the only
        int8 entry, roles_available, is a separate array."""
        game = self.game
        vals = []

        # Global state, in key order (roles_available is int8 and kept apart).
        onehot = [0.0] * 18
        loads = [0.0] * 3
        space = [0.0] * 3
        for i, ship in enumerate(game.cargo_ships[:3]):
            onehot[i * 6 + (5 if ship.good_type is None else ship.good_type)] = 1.0  # 5 = "none"
            loads[i] = ship.current_load
            space[i] = ship.capacity - ship.current_load
        vals += onehot
        vals += loads
        vals += space
        vals.append(game.colonists_ship)
        vals.append(game.colonists_supply)
        phase = [0.0] * 10
        phase[9 if game.current_phase is None else game.current_phase] = 1.0
        vals += phase
        vals.append(game.current_player_idx)
        face_up = [0.0] * 6
        for t in game.face_up_plantations:
            if t != TileType.EMPTY:
                face_up[t] += 1.0
        vals += face_up
        progress_at = len(vals)
        vals.append(0.0)  # game_progress, filled in after the players are scanned
        vals += [game.goods_supply[g] for g in _GOODS]
        vals.append(game.governor_idx)
        vals.append(game.quarry_stack)
        role_doubloons = game.role_doubloons
        vals += [role_doubloons.get(role, 0) for role in _ROLES]
        roles_available = np.array([1 if role in game.available_roles else 0 for role in _ROLES],
                                   dtype=np.int8)
        vals.append(len(game.trading_house))
        house = [0.0] * 5
        for g in game.trading_house:
            house[g] = 1.0
        vals += house
        vals.append(game.vp_chips)

        # Players, in key order.
        max_city = 0
        for p in game.players:
            tile_count = [0.0] * 6
            tile_occupied = [0.0] * 6
            planted = [0] * 5          # occupied plantations per good (TileType 0-4 = Good 0-4)
            for t in p.island_board:
                kind = t.tile_type
                if kind != TileType.EMPTY:
                    tile_count[kind] += 1.0
                    if t.is_occupied:
                        tile_occupied[kind] += 1.0
                        if kind < 5:
                            planted[kind] += 1
            has_building = [0.0] * 23
            building_colonists = [0.0] * 23
            workers = [0] * 5          # colonists in production buildings per good
            buildings = 0
            for b in p.city_board:
                bt = b.building_type
                if bt not in _NOT_BUILDINGS:
                    has_building[bt] = 1.0
                    building_colonists[bt] = b.colonists
                    buildings += 1
                good = _PRODUCES.get(bt)
                if good is not None:
                    workers[good] += b.colonists
            max_city = max(max_city, buildings)
            capacity = [min(planted[g], workers[g]) for g in range(5)]
            capacity[Good.CORN] = planted[Good.CORN]   # corn needs no production building

            vals += building_colonists
            vals.append(p.doubloons)
            vals.append(p.empty_city_spaces)
            vals += [p.goods[g] for g in _GOODS]
            vals += has_building
            vals.append(p.empty_island_spaces)
            vals += tile_count
            vals += tile_occupied
            vals += capacity
            vals.append(p.unplaced_colonists)
            vals.append(p.vp_chips)

        vals[progress_at] = self._compute_game_progress(max_city)
        buf = np.array(vals, dtype=np.float32)
        global_state = {key: (roles_available if lo is None else buf[lo:hi])
                        for key, lo, hi in self._obs_global_slices}
        players_obs = {name: {key: buf[lo:hi] for key, lo, hi in slices}
                       for name, slices in self._obs_player_slices}
        return {
            "global_state": global_state,
            "players": players_obs
        }

    def _get_info(self, game_over=None):
        info = {"current_phase": self.game.current_phase.name if self.game.current_phase else "INIT"}
        if game_over is None:
            game_over = getattr(self, 'game', None) is not None and self.game.check_game_end()
        if game_over:
            info["final_scores"] = self.game.get_scores()
            info["episode_metrics"] = getattr(self, 'episode_metrics', {})
        return info

    def _compute_game_progress(self, max_city=None) -> float:
        game = self.game
        initial_vp = VP_CHIPS_SETUP.get(self.num_players, 75)
        vp_prog = max(0.0, (initial_vp - game.vp_chips)) / initial_vp

        if max_city is None:
            max_city = 0
            for p in game.players:
                filled = sum(1 for b in p.city_board if b.building_type not in _NOT_BUILDINGS)
                max_city = max(max_city, filled)
        city_prog = max_city / 12.0

        initial_col = COLONIST_SUPPLY_SETUP.get(self.num_players, 55)
        col_prog = max(0.0, (initial_col - game.colonists_supply)) / initial_col

        return min(1.0, max(vp_prog, city_prog, col_prog))

    def _calculate_all_rewards(self) -> list[float]:
        """Terminal reward: winner-takes-all.
        1st place (by total_vp, then tiebreaker) → +1.0.
        All other players → -1.0.
        Players sharing 1st place all receive +1.0.
        """
        scores = self.game.get_scores()
        best = max((scores[i][0], scores[i][1]) for i in range(self.num_players))
        return [1.0 if (scores[i][0], scores[i][1]) == best else -1.0
                for i in range(self.num_players)]

    def valid_action_mask(self):
        """Legal actions of the current player (docs/OBSERVATION_AND_ACTIONS.md)."""
        mask = np.zeros(200, dtype=bool)
        game = self.game
        cp = game.current_player_idx
        p = game.players[cp]
        phase = game.current_phase

        if phase == Phase.END_ROUND or phase is None:
            for r in game.available_roles:
                mask[r] = True

        elif phase == Phase.SETTLER:
            mask[15] = True # Pass
            if p.empty_island_spaces > 0:
                if (game.plantation_stack and not game._hacienda_used
                        and p.is_building_occupied(BuildingType.HACIENDA)):
                    mask[105] = True
                for tile in game.face_up_plantations:
                    if tile != TileType.EMPTY:
                        mask[8 + tile] = True
                if game.quarry_stack > 0 and (cp == game.active_role_player or
                                              p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)):
                    mask[13] = True # TileType.QUARRY

        elif phase == Phase.BUILDER:
            mask[15] = True # Pass
            empty = p.empty_city_spaces
            if empty > 0:
                discount = 1 if cp == game.active_role_player else 0
                quarries = sum(1 for t in p.island_board
                               if t.tile_type == TileType.QUARRY and t.is_occupied)
                owned = {b.building_type for b in p.city_board}
                for b_type, count in game.building_supply.items():
                    if count > 0 and b_type not in owned:
                        cost, column, _, _, large, _ = BUILDING_DATA[b_type]
                        # Quarries reduce the cost up to the building's column, which equals
                        # its VP; the privilege takes one more off; never below zero.
                        if (empty >= (2 if large else 1)
                                and p.doubloons >= max(0, cost - min(quarries, column) - discount)):
                            mask[16 + b_type] = True

        elif phase == Phase.TRADER:
            mask[15] = True # Pass
            if len(game.trading_house) < 4:
                has_office = p.is_building_occupied(BuildingType.OFFICE)
                for g in _GOODS:
                    if p.goods[g] > 0 and (has_office or g not in game.trading_house):
                        mask[39 + g] = True

        elif phase == Phase.CAPTAIN:
            ships = game.cargo_ships
            carried = [s.good_type for s in ships]
            can_load_anything = False
            for g in _GOODS:
                held = p.goods[g]
                if held == 0:
                    continue
                # A ship carrying g, or an empty ship while no other ship carries g.
                options = []
                for i, s in enumerate(ships):
                    if s.current_load >= s.capacity:
                        continue
                    if s.good_type is None:
                        if any(carried[j] == g for j in range(len(ships)) if j != i):
                            continue
                    elif s.good_type != g:
                        continue
                    options.append((i, min(held, s.capacity - s.current_load)))
                # The player chooses the KIND freely, but among legal ships must take one
                # where the most barrels fit, so only max-load ships are offered per kind.
                if options:
                    best = max(load for _, load in options)
                    for i, load in options:
                        if load == best:
                            mask[44 + 5 * i + g] = True
                            can_load_anything = True

            # Wharf
            if (p.is_building_occupied(BuildingType.WHARF)
                    and not game._wharf_used.get(cp, False)):
                for g in _GOODS:
                    if p.goods[g] > 0:
                        mask[59 + g] = True

            # Pass only allowed if cannot load anything
            if not can_load_anything:
                mask[15] = True

        elif phase == Phase.CAPTAIN_STORE:
            assign = game._storage_assignments[cp]
            windrose = assign['windrose']
            warehouses = assign['warehouses']
            slots = ((1 if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE) else 0)
                     + (2 if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE) else 0))
            windrose_free = windrose is None
            warehouse_free = len(warehouses) < slots
            unstored = False
            for g in _GOODS:
                if p.goods[g] > 0 and g != windrose and g not in warehouses:
                    unstored = True
                    if windrose_free:
                        mask[64 + g] = True
                    if warehouse_free:
                        mask[106 + g] = True
            # Finishing is legal only when nothing more can be stored.
            if not unstored or not (windrose_free or warehouse_free):
                mask[15] = True

        elif phase == Phase.MAYOR:
            if p.unplaced_colonists > 0:
                for t in p.island_board:
                    if t.tile_type != TileType.EMPTY and not t.is_occupied:
                        mask[120 + t.tile_type] = True
                for b in p.city_board:
                    bt = b.building_type
                    if bt not in _NOT_BUILDINGS and b.colonists < BUILDING_DATA[bt][2]:
                        mask[140 + bt] = True

        elif phase == Phase.CRAFTSMAN:
            mask[15] = True # Can always pass
            if cp == game.active_role_player:
                for g in getattr(game, '_craftsman_produced_kinds', []):
                    if game.goods_supply[g] > 0:
                        mask[93 + g] = True

        elif phase == Phase.PROSPECTOR:
            mask[15] = True

        return mask
