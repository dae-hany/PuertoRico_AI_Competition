from pettingzoo import AECEnv
try:
    from pettingzoo.utils.agent_selector import AgentSelector  # pettingzoo >= 1.24.4
except ImportError:  # older pettingzoo only exposes the lowercase name
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
from gymnasium import spaces
import numpy as np

from puerto_rico.engine import PuertoRicoGame
from puerto_rico.constants import Phase, Role, Good, TileType, BuildingType, BUILDING_DATA, MayorStrategy, VP_CHIPS_SETUP, COLONIST_SUPPLY_SETUP

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

    def reset(self, seed=None, options=None):
        # Seed control
        final_seed = seed if self.random_seed_mode else self.fixed_seed
        if final_seed is not None:
            import random
            random.seed(final_seed)
            np.random.seed(final_seed)

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

        self.game = PuertoRicoGame(self.num_players)
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

        # 1. Check for pass-only or single-choice situations
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

        # 3. Mayor phase auto-fill
        if phase == Phase.MAYOR:
            empty_slots = []
            for i, t in enumerate(p.island_board):
                if t.tile_type != TileType.EMPTY and not t.is_occupied:
                    empty_slots.append((False, i))
            for i, b in enumerate(p.city_board):
                if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                    from puerto_rico.constants import BUILDING_DATA
                    cap = BUILDING_DATA[b.building_type][2]
                    for _ in range(cap - b.colonists):
                        empty_slots.append((True, i))

            if p.unplaced_colonists == 0 or len(empty_slots) == 0:
                self.game._advance_phase_turn()
                return True

            if p.unplaced_colonists >= len(empty_slots) and len(empty_slots) > 0:
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
            # Invalid action penalty (reduced to -10 to avoid value function distortion)
            self.rewards[agent] = -10.0
            for a in self.agents:
                self.terminations[a] = True
                self.infos[a]["error"] = str(e)
            self._accumulate_rewards()
            self.agent_selection = f"player_{self.game.current_player_idx}"
            return

        self._game_step_count += 1
        done = self.game.check_game_end()

        if done:
            self._finalize_episode()
        else:
            for i in range(self.num_players):
                a_name = f"player_{i}"
                self.rewards[a_name] = 0.0
                self.infos[a_name] = self._get_info()
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
        game = self.game

        # ═══════════════════════════════════════════════════════════════════════
        # Global State
        # ═══════════════════════════════════════════════════════════════════════

        cargo_ships_good_onehot = np.zeros(18, dtype=np.float32)
        cargo_ships_load = np.zeros(3, dtype=np.float32)
        cargo_ships_space = np.zeros(3, dtype=np.float32)
        for i, ship in enumerate(game.cargo_ships):
            if i >= 3:
                break
            if ship.good_type is not None:
                cargo_ships_good_onehot[i * 6 + ship.good_type.value] = 1.0
            else:
                cargo_ships_good_onehot[i * 6 + 5] = 1.0  # "none" class
            cargo_ships_load[i] = float(ship.current_load)
            cargo_ships_space[i] = float(ship.capacity - ship.current_load)

        trading_house_has_good = np.zeros(5, dtype=np.float32)
        for g in game.trading_house:
            trading_house_has_good[g.value] = 1.0
        trading_house_count = np.array([len(game.trading_house)], dtype=np.float32)

        role_doubloons = np.zeros(8, dtype=np.float32)
        roles_available = np.zeros(8, dtype=np.int8)
        for i in range(8):
            try:
                role = Role(i)
                role_doubloons[i] = float(game.role_doubloons.get(role, 0))
                roles_available[i] = 1 if role in game.available_roles else 0
            except ValueError:
                pass

        face_up_plantation_counts = np.zeros(6, dtype=np.float32)
        for t in game.face_up_plantations:
            if t != TileType.EMPTY:
                face_up_plantation_counts[t.value] += 1.0

        current_phase_onehot = np.zeros(10, dtype=np.float32)
        phase_idx = int(game.current_phase) if game.current_phase is not None else 9
        current_phase_onehot[phase_idx] = 1.0

        global_state = {
            "cargo_ships_good_onehot": cargo_ships_good_onehot,
            "cargo_ships_load": cargo_ships_load,
            "cargo_ships_space": cargo_ships_space,
            "colonists_ship": np.array([game.colonists_ship], dtype=np.float32),
            "colonists_supply": np.array([game.colonists_supply], dtype=np.float32),
            "current_phase_onehot": current_phase_onehot,
            "current_player": np.array([game.current_player_idx], dtype=np.float32),
            "face_up_plantation_counts": face_up_plantation_counts,
            "game_progress": np.array([self._compute_game_progress()], dtype=np.float32),
            "goods_supply": np.array([game.goods_supply[Good(i)] for i in range(5)], dtype=np.float32),
            "governor_idx": np.array([game.governor_idx], dtype=np.float32),
            "quarry_stack": np.array([game.quarry_stack], dtype=np.float32),
            "role_doubloons": role_doubloons,
            "roles_available": roles_available,
            "trading_house_count": trading_house_count,
            "trading_house_has_good": trading_house_has_good,
            "vp_chips": np.array([game.vp_chips], dtype=np.float32),
        }

        players_obs = {}
        for i in range(self.num_players):
            p = game.players[i]

            island_tile_count = np.zeros(6, dtype=np.float32)
            island_tile_occupied = np.zeros(6, dtype=np.float32)
            for t in p.island_board:
                if t.tile_type != TileType.EMPTY:
                    idx = t.tile_type.value
                    island_tile_count[idx] += 1.0
                    if t.is_occupied:
                        island_tile_occupied[idx] += 1.0

            has_building = np.zeros(23, dtype=np.float32)
            building_colonists = np.zeros(23, dtype=np.float32)
            for b in p.city_board:
                bt = b.building_type
                if bt not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                    has_building[bt.value] = 1.0
                    building_colonists[bt.value] = float(b.colonists)

            player_dict = {
                "building_colonists": building_colonists,
                "doubloons": np.array([p.doubloons], dtype=np.float32),
                "empty_city_spaces": np.array([p.empty_city_spaces], dtype=np.float32),
                "goods": np.array([p.goods[Good(g)] for g in range(5)], dtype=np.float32),
                "has_building": has_building,
                "island_empty_spaces": np.array([p.empty_island_spaces], dtype=np.float32),
                "island_tile_count": island_tile_count,
                "island_tile_occupied": island_tile_occupied,
                "production_capacity": self._compute_production_capacity(p),
                "unplaced_colonists": np.array([p.unplaced_colonists], dtype=np.float32),
                "vp_chips": np.array([p.vp_chips], dtype=np.float32),
            }
            players_obs[f"player_{i}"] = player_dict

        return {
            "global_state": global_state,
            "players": players_obs
        }

    def _get_info(self):
        info = {"current_phase": self.game.current_phase.name if self.game.current_phase else "INIT"}
        if getattr(self, 'game', None) and self.game.check_game_end():
            info["final_scores"] = self.game.get_scores()
            info["episode_metrics"] = getattr(self, 'episode_metrics', {})
        return info

    def _compute_production_capacity(self, p) -> np.ndarray:
        capacity = np.zeros(5, dtype=np.float32)
        plan_cnt = {g: 0 for g in Good}
        for t in p.island_board:
            if t.is_occupied:
                if t.tile_type == TileType.COFFEE_PLANTATION:  plan_cnt[Good.COFFEE] += 1
                elif t.tile_type == TileType.TOBACCO_PLANTATION: plan_cnt[Good.TOBACCO] += 1
                elif t.tile_type == TileType.CORN_PLANTATION:    plan_cnt[Good.CORN] += 1
                elif t.tile_type == TileType.SUGAR_PLANTATION:   plan_cnt[Good.SUGAR] += 1
                elif t.tile_type == TileType.INDIGO_PLANTATION:  plan_cnt[Good.INDIGO] += 1

        bldg_cap = {g: 0 for g in Good}
        for b in p.city_board:
            bt = b.building_type
            if bt in (BuildingType.SMALL_INDIGO_PLANT, BuildingType.INDIGO_PLANT):
                bldg_cap[Good.INDIGO] += b.colonists
            elif bt in (BuildingType.SMALL_SUGAR_MILL, BuildingType.SUGAR_MILL):
                bldg_cap[Good.SUGAR] += b.colonists
            elif bt == BuildingType.TOBACCO_STORAGE:
                bldg_cap[Good.TOBACCO] += b.colonists
            elif bt == BuildingType.COFFEE_ROASTER:
                bldg_cap[Good.COFFEE] += b.colonists

        capacity[Good.COFFEE]  = min(plan_cnt[Good.COFFEE],  bldg_cap[Good.COFFEE])
        capacity[Good.TOBACCO] = min(plan_cnt[Good.TOBACCO], bldg_cap[Good.TOBACCO])
        capacity[Good.CORN]    = plan_cnt[Good.CORN]
        capacity[Good.SUGAR]   = min(plan_cnt[Good.SUGAR],   bldg_cap[Good.SUGAR])
        capacity[Good.INDIGO]  = min(plan_cnt[Good.INDIGO],  bldg_cap[Good.INDIGO])
        return capacity

    def _compute_game_progress(self) -> float:
        game = self.game
        initial_vp = VP_CHIPS_SETUP.get(self.num_players, 75)
        vp_prog = max(0.0, (initial_vp - game.vp_chips)) / initial_vp

        max_city = 0
        for p in game.players:
            filled = sum(1 for b in p.city_board
                         if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE))
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
        mask = np.zeros(200, dtype=bool)
        game = self.game
        p = game.players[game.current_player_idx]
        phase = game.current_phase

        if phase == Phase.END_ROUND or phase is None:
            for r in game.available_roles:
                mask[r.value] = True

        elif phase == Phase.SETTLER:
            mask[15] = True # Pass

            can_hacienda = (
                p.is_building_occupied(BuildingType.HACIENDA) and
                self.game.plantation_stack and
                not getattr(self.game, '_hacienda_used', False) and
                p.empty_island_spaces > 0
            )
            if can_hacienda:
                mask[105] = True

            if p.empty_island_spaces > 0:
                for tile in game.face_up_plantations:
                    if tile != TileType.EMPTY:
                        mask[8 + tile.value] = True

            can_quarry = (game.current_player_idx == game.active_role_player_idx()) or p.is_building_occupied(BuildingType.CONSTRUCTION_HUT)
            if can_quarry and game.quarry_stack > 0 and p.empty_island_spaces > 0:
                mask[13] = True # TileType.QUARRY

        elif phase == Phase.BUILDER:
            mask[15] = True # Pass
            has_privilege = (game.current_player_idx == game.active_role_player_idx())
            active_quarries = sum(1 for t in p.island_board if t.tile_type == TileType.QUARRY and t.is_occupied)

            for b_type, count in game.building_supply.items():
                if count > 0 and not p.has_building(b_type):
                    spaces_needed = 2 if BUILDING_DATA[b_type][4] else 1
                    if p.empty_city_spaces >= spaces_needed:
                        base_cost = BUILDING_DATA[b_type][0]
                        # Max reduction depends on building column. Base VP equals the column number (1 to 4).
                        max_q = BUILDING_DATA[b_type][1]

                        quarry_discount = min(active_quarries, max_q)
                        privilege_discount = 1 if has_privilege else 0
                        final_cost = max(0, base_cost - quarry_discount - privilege_discount)

                        if p.doubloons >= final_cost:
                            mask[16 + b_type.value] = True

        elif phase == Phase.TRADER:
            mask[15] = True # Pass
            if len(game.trading_house) < 4:
                has_office = p.is_building_occupied(BuildingType.OFFICE)
                for g in Good:
                    if p.goods[g] > 0:
                        if g not in game.trading_house or has_office:
                            mask[39 + g.value] = True

        elif phase == Phase.CAPTAIN:
            # Need to find valid ship/good combos
            can_load_anything = False

            # For each good, find the maximum loadable amount across all valid ships
            max_loadable_for_good = {g: 0 for g in Good}
            allowed_ships_for_good = {g: [] for g in Good}

            for ship_idx, ship in enumerate(game.cargo_ships):
                if not ship.is_full:
                    for g in Good:
                        if p.goods[g] > 0:
                            allowed = False
                            if ship.good_type is None:
                                other_has_it = any(os.good_type == g for i, os in enumerate(game.cargo_ships) if i != ship_idx)
                                if not other_has_it:
                                    allowed = True
                            elif ship.good_type == g:
                                allowed = True

                            if allowed:
                                potential_load = min(p.goods[g], ship.capacity - ship.current_load)
                                allowed_ships_for_good[g].append((ship_idx, potential_load))
                                max_loadable_for_good[g] = max(max_loadable_for_good[g], potential_load)

            for g, ships in allowed_ships_for_good.items():
                max_load = max_loadable_for_good[g]
                for ship_idx, potential_load in ships:
                    if potential_load == max_load:
                        mask[44 + (ship_idx * 5) + g.value] = True
                        can_load_anything = True

            # Wharf
            if p.is_building_occupied(BuildingType.WHARF) and not game._wharf_used.get(game.current_player_idx, False):
                for g in Good:
                    if p.goods[g] > 0:
                        mask[59 + g.value] = True


            # Pass only allowed if cannot load anything
            if not can_load_anything:
                mask[15] = True

        elif phase == Phase.CAPTAIN_STORE:
            assign = game._storage_assignments[game.current_player_idx]
            unstored_types = [g for g in Good if p.goods[g] > 0 and g != assign['windrose'] and g not in assign['warehouses']]

            max_wh_slots = 0
            if p.is_building_occupied(BuildingType.SMALL_WAREHOUSE): max_wh_slots += 1
            if p.is_building_occupied(BuildingType.LARGE_WAREHOUSE): max_wh_slots += 2

            has_empty_windrose = (assign['windrose'] is None)
            has_empty_wh = len(assign['warehouses']) < max_wh_slots

            can_pass = True
            if len(unstored_types) > 0:
                if has_empty_windrose or has_empty_wh:
                    can_pass = False

            if can_pass:
                mask[15] = True

            for g in Good:
                if p.goods[g] > 0:
                    if has_empty_windrose and assign['windrose'] is None and g not in assign['warehouses']:
                        mask[64 + g.value] = True
                    if has_empty_wh and g != assign['windrose'] and g not in assign['warehouses']:
                        mask[106 + g.value] = True

        elif phase == Phase.MAYOR:
            if p.unplaced_colonists > 0:
                for t in p.island_board:
                    if t.tile_type != TileType.EMPTY and not t.is_occupied:
                        mask[120 + t.tile_type.value] = True
                for b in p.city_board:
                    if b.building_type not in (BuildingType.EMPTY, BuildingType.OCCUPIED_SPACE):
                        if b.colonists < BUILDING_DATA[b.building_type][2]:
                            mask[140 + b.building_type.value] = True

        elif phase == Phase.CRAFTSMAN:
            has_privilege = (game.current_player_idx == game.active_role_player_idx())
            mask[15] = True # Can always pass
            if has_privilege:
                for g in getattr(game, '_craftsman_produced_kinds', []):
                    if game.goods_supply[g] > 0:
                        mask[93 + g.value] = True

        elif phase == Phase.PROSPECTOR:
            mask[15] = True

        return mask
