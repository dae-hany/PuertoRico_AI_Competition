"""
webui/server.py — local web UI for playing and DEBUGGING Puerto Rico agents.

Run:  python webui/server.py   then open http://127.0.0.1:5000

What it gives you:
  * configure each of the 3 seats independently — a human (you), any baseline,
    or YOUR OWN agent (auto-discovered from submissions/, or typed as
    "module:Class" / "path/to/file.py:Class");
  * watch bots play each other (set every seat to a bot), or play a seat yourself;
  * competition-rule mirroring while debugging: each agent move is timed, and a
    move that exceeds the 1 s budget, returns an illegal action, or raises is
    replaced by a random legal move — exactly as the real tournament does — and
    the substitution is shown in the log so you can see what would happen.

This is a local single-game debug tool (global state, no auth). It is not the
competition runner — that is `tournament/` (see docs/COMPETITION_RULES.md).
"""
import copy
import importlib
import importlib.util
import logging
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Make the repo root importable when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from puerto_rico import ForwardModel, flatten_observation, make_env
from puerto_rico.constants import BUILDING_DATA, BuildingType, Good, Role, TileType
from agents.base import Agent
from agents import (ActionValueAgent, FactoryAgent, MctsAgent, PpoAgent,
                    RandomAgent, ShippingRushAgent, TradeBuildingAgent)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__, static_folder="static", static_url_path="")

MOVE_TIME_LIMIT = 1.0          # seconds per move (competition rule)
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
PPO_CHECKPOINT = os.path.join(ROOT, "training", "checkpoints", "ppo_baseline.pt")

# ── agent registry ───────────────────────────────────────────────────────────

def _ppo():
    return PpoAgent(PPO_CHECKPOINT if os.path.exists(PPO_CHECKPOINT) else None)


BASELINE_FACTORIES = {
    "random":      (RandomAgent,                                      "Random (weakest)"),
    "ppo":         (_ppo,                                             "PPO (RL, ~random)"),
    "factory":     (FactoryAgent,                                     "Factory (heuristic)"),
    "trade":       (TradeBuildingAgent,                               "TradeBuilding (heuristic)"),
    "shipping":    (ShippingRushAgent,                                "ShippingRush (strong)"),
    "actionvalue": (ActionValueAgent,                                 "ActionValue (strong)"),
    "mcts":        (lambda: MctsAgent(num_simulations=30, max_rollout_depth=40), "MCTS (search, strong)"),
}


def discover_submissions():
    """Find Agent subclasses in submissions/*.py. Returns [{token, label}]."""
    found = []
    if not os.path.isdir(SUBMISSIONS_DIR):
        return found
    for fname in sorted(os.listdir(SUBMISSIONS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(SUBMISSIONS_DIR, fname)
        try:
            module = _import_from_path(path, f"_submission_{fname[:-3]}")
            for attr in vars(module).values():
                if (isinstance(attr, type) and issubclass(attr, Agent)
                        and attr is not Agent and attr.__module__ == module.__name__):
                    found.append({"token": f"submissions/{fname}:{attr.__name__}",
                                  "label": f"{fname} · {attr.__name__}"})
        except Exception as e:  # a broken file shouldn't break the whole list
            found.append({"token": f"__error__/{fname}", "label": f"{fname} (load error: {e})"})
    return found


def _import_from_path(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_custom_agent(spec: str) -> Agent:
    """Instantiate an agent from 'module:Class' or 'path/to/file.py:Class'."""
    if ":" not in spec:
        raise ValueError(f"Custom agent must be 'module:Class' or 'file.py:Class', got '{spec}'")
    mod_part, cls_name = spec.rsplit(":", 1)
    looks_like_path = mod_part.endswith(".py") or "/" in mod_part or os.sep in mod_part
    if looks_like_path:
        path = mod_part if os.path.isabs(mod_part) else os.path.join(ROOT, mod_part)
        module = _import_from_path(path, "_custom_agent_module")
    else:
        module = importlib.import_module(mod_part)
    cls = getattr(module, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise TypeError(f"{cls_name} is not an Agent subclass")
    return cls()


def resolve_agent(token: str):
    """Map a seat token to an Agent instance, or None for a human seat."""
    if token == "human":
        return None
    if token in BASELINE_FACTORIES:
        return BASELINE_FACTORIES[token][0]()
    return load_custom_agent(token)          # treat anything else as a spec


# ── global game state ────────────────────────────────────────────────────────

game_env = None
forward_model = None
agents = []            # agents[i] = Agent instance, or None for a human seat
player_types = []      # token per seat ("human" / "random" / "submissions/..:X")
player_labels = []     # display name per seat
game_log = []
undo_stack = []        # [(deepcopy(env), log_copy, chosen_roles_copy)]
last_phase = None
chosen_roles = {}
last_ai_move = None    # {agent, action, intended, ms, note, substituted}
_rng = np.random.default_rng()


def bind_agents():
    """(Re)create the forward model and hand it to every non-human agent."""
    global forward_model
    forward_model = ForwardModel(game_env)
    for ag in agents:
        if ag is not None:
            try:
                ag.on_game_start(forward_model)
            except Exception:
                pass


def _random_legal(mask):
    legal = np.where(np.asarray(mask) > 0.5)[0]
    return int(_rng.choice(legal)) if len(legal) else 15


# ── action description (English log) ─────────────────────────────────────────

def get_action_description(player_idx, action_idx, env):
    try:
        if 0 <= action_idx <= 7:
            return f"Player {player_idx} selected role {Role(action_idx).name}."
        elif 8 <= action_idx <= 12:
            return f"Player {player_idx} drafted plantation {TileType(action_idx - 8).name}."
        elif action_idx == 13:
            return f"Player {player_idx} drafted a Quarry tile."
        elif action_idx == 15:
            phase = env.game.current_phase
            return f"Player {player_idx} passed in {phase.name if phase else 'Unknown'} phase."
        elif 16 <= action_idx <= 38:
            return f"Player {player_idx} built {BuildingType(action_idx - 16).name}."
        elif 39 <= action_idx <= 43:
            return f"Player {player_idx} sold {Good(action_idx - 39).name} to Trading House."
        elif 44 <= action_idx <= 58:
            idx = action_idx - 44
            return f"Player {player_idx} loaded {Good(idx % 5).name} onto Cargo Ship {idx // 5 + 1}."
        elif 59 <= action_idx <= 63:
            return f"Player {player_idx} loaded {Good(action_idx - 59).name} via Wharf."
        elif action_idx == 105:
            return f"Player {player_idx} used Hacienda to draw an extra plantation."
        elif 64 <= action_idx <= 68:
            return f"Player {player_idx} stored {Good(action_idx - 64).name} on Windrose."
        elif 106 <= action_idx <= 110:
            return f"Player {player_idx} stored {Good(action_idx - 106).name} in Warehouse."
        elif 120 <= action_idx <= 125:
            return f"Player {player_idx} placed a colonist on {TileType(action_idx - 120).name}."
        elif 140 <= action_idx <= 162:
            return f"Player {player_idx} placed a colonist on {BuildingType(action_idx - 140).name}."
        elif 93 <= action_idx <= 97:
            return f"Player {player_idx} chose {Good(action_idx - 93).name} as Craftsman privilege."
        return f"Player {player_idx} executed action {action_idx}."
    except Exception as e:
        return f"Player {player_idx} executed action {action_idx} (desc error: {e})."


def check_phase_transition(env):
    global last_phase, game_log
    current_phase = env.game.current_phase
    if current_phase != last_phase:
        game_log.append(f"--- Phase: {current_phase.name} ---" if current_phase is not None
                        else "--- Round Over: Role Selection ---")
        last_phase = current_phase


def serialize_state(env):
    global chosen_roles
    game = env.game
    if not game.roles_in_play:
        chosen_roles.clear()

    state = {
        "current_agent": env.agent_selection,
        "current_agent_idx": env.agent_name_mapping.get(env.agent_selection, -1),
        "round_number": game.round_number,
        "current_phase": game.current_phase.name if game.current_phase is not None else "INIT/ROLE_SELECT",
        "active_role": game.active_role.name if game.active_role is not None else None,
        "active_role_player": game.active_role_player,
        "governor_idx": game.governor_idx,
        "vp_chips": game.vp_chips,
        "colonists_supply": game.colonists_supply,
        "colonists_ship": game.colonists_ship,
        "quarry_stack": game.quarry_stack,
        "trading_house": [g.name for g in game.trading_house],
        "face_up_plantations": [t.name for t in game.face_up_plantations],
        "available_roles": [r.name for r in game.available_roles],
        "role_doubloons": {r.name: game.role_doubloons[r] for r in game.role_doubloons},
        "roles_in_play": [r.name for r in game.roles_in_play],
        "cargo_ships": [
            {"capacity": s.capacity, "current_load": s.current_load,
             "good_type": s.good_type.name if s.good_type is not None else None}
            for s in game.cargo_ships
        ],
        "building_supply": {b.name: c for b, c in game.building_supply.items()},
        "game_over": game.check_game_end(),
        "game_log": game_log,
        "last_ai_move": last_ai_move,
        "players": [],
    }

    scores = game.get_scores() if state["game_over"] else None
    for i, p in enumerate(game.players):
        p_state = {
            "index": p.player_idx,
            "type": "human" if (i < len(player_types) and player_types[i] == "human") else "ai",
            "label": player_labels[i] if i < len(player_labels) else f"Player {i}",
            "agent_token": player_types[i] if i < len(player_types) else "human",
            "doubloons": p.doubloons,
            "vp_chips": p.vp_chips,
            "unplaced_colonists": p.unplaced_colonists,
            "island_board": [{"tile_type": t.tile_type.name, "is_occupied": t.is_occupied}
                             for t in p.island_board],
            "city_board": [{"building_type": b.building_type.name, "colonists": b.colonists,
                            "capacity": BUILDING_DATA[b.building_type][2] if b.building_type in BUILDING_DATA else 0}
                           for b in p.city_board],
            "goods": {g.name: c for g, c in p.goods.items()},
            "goods_produced": p.goods_produced,
            "doubloons_earned": p.doubloons_earned,
            "empty_island_spaces": p.empty_island_spaces,
            "empty_city_spaces": p.empty_city_spaces,
        }
        if scores:
            tvp, tb, svp, bvp, bonus = scores[i]
            p_state["scores"] = {"total_vp": tvp, "tie_breaker": tb, "shipping_vp": svp,
                                 "building_vp": bvp, "bonus_vp": bonus}
        state["players"].append(p_state)

    if state["game_over"]:
        state["valid_actions"] = []
    else:
        mask = env.observe(env.agent_selection)["action_mask"]
        state["valid_actions"] = [int(i) for i, v in enumerate(mask) if v == 1]
    return state


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/agents", methods=["GET"])
def list_agents():
    builtin = [{"token": "human", "label": "Human (you play this seat)"}]
    builtin += [{"token": t, "label": lbl} for t, (_, lbl) in BASELINE_FACTORIES.items()]
    return jsonify({"builtin": builtin, "submissions": discover_submissions()})


@app.route("/api/init", methods=["POST"])
def init_game():
    global game_env, agents, player_types, player_labels, game_log, undo_stack
    global last_phase, chosen_roles, last_ai_move

    data = request.json or {}
    num_players = int(data.get("num_players", 3))
    raw = data.get("players", [])

    player_types = [raw[i] if i < len(raw) else ("human" if i == 0 else "random")
                    for i in range(num_players)]
    try:
        agents = [resolve_agent(tok) for tok in player_types]
    except Exception as e:
        return jsonify({"error": f"Could not load an agent: {e}"}), 400

    player_labels = []
    for i, tok in enumerate(player_types):
        if tok == "human":
            player_labels.append(f"Human (P{i})")
        elif tok in BASELINE_FACTORIES:
            player_labels.append(BASELINE_FACTORIES[tok][1])
        else:
            player_labels.append(tok.rsplit(":", 1)[-1])

    game_env = make_env(seed=None, num_players=num_players)
    bind_agents()

    last_phase = game_env.game.current_phase
    last_ai_move = None
    game_log = ["--- Game Started ---", f"Players: {', '.join(player_labels)}."]
    game_log.append(f"--- Phase: {last_phase.name} ---" if last_phase is not None
                    else "--- Round 1: Role Selection ---")
    undo_stack = []
    chosen_roles = {}
    return jsonify(serialize_state(game_env))


@app.route("/api/state", methods=["GET"])
def get_state():
    if game_env is None:
        return jsonify({"error": "Game not initialized"}), 400
    return jsonify(serialize_state(game_env))


@app.route("/api/action", methods=["POST"])
def apply_action():
    global game_env, game_log, undo_stack, chosen_roles, last_ai_move
    if game_env is None:
        return jsonify({"error": "Game not initialized"}), 400

    action_idx = (request.json or {}).get("action")
    if action_idx is None:
        return jsonify({"error": "Action parameter is missing"}), 400
    action_idx = int(action_idx)

    active_agent = game_env.agent_selection
    active_idx = game_env.agent_name_mapping[active_agent]
    if player_types[active_idx] != "human":
        return jsonify({"error": "It is not the human player's turn"}), 400
    if game_env.observe(active_agent)["action_mask"][action_idx] == 0:
        return jsonify({"error": f"Invalid action {action_idx}"}), 400

    undo_stack.append((copy.deepcopy(game_env), list(game_log), dict(chosen_roles)))
    game_log.append(get_action_description(active_idx, action_idx, game_env))
    if 0 <= action_idx <= 7:
        chosen_roles[Role(action_idx).name] = active_idx
    last_ai_move = None
    try:
        game_env.step(action_idx)
    except Exception as e:
        game_env, game_log, chosen_roles = undo_stack.pop()
        bind_agents()
        return jsonify({"error": f"Game engine error: {e}"}), 500
    check_phase_transition(game_env)
    return jsonify(serialize_state(game_env))


@app.route("/api/ai_step", methods=["POST"])
def ai_step():
    global game_env, game_log, chosen_roles, last_ai_move
    if game_env is None:
        return jsonify({"error": "Game not initialized"}), 400
    if game_env.game.check_game_end():
        return jsonify({"error": "Game is already over"}), 400

    active_agent = game_env.agent_selection
    active_idx = game_env.agent_name_mapping[active_agent]
    agent = agents[active_idx]
    if agent is None:
        return jsonify({"error": "It is a human player's turn"}), 400

    obs = game_env.observe(active_agent)
    flat = flatten_observation(obs["observation"])
    mask = np.asarray(obs["action_mask"], dtype=np.int8)

    # Run the agent under the same rules as the tournament.
    intended, note, substituted = None, None, False
    t0 = time.perf_counter()
    try:
        intended = int(agent.act(flat, mask))
    except Exception as e:
        note, substituted = f"raised {type(e).__name__}: {e}", True
    elapsed = time.perf_counter() - t0

    if not substituted and elapsed > MOVE_TIME_LIMIT:
        note, substituted = f"timeout {elapsed:.2f}s > {MOVE_TIME_LIMIT:.0f}s", True
    elif not substituted and (intended < 0 or intended >= 200 or mask[intended] < 0.5):
        note, substituted = f"illegal action {intended}", True

    action = _random_legal(mask) if substituted else intended
    name = player_labels[active_idx]
    ms = elapsed * 1000.0

    if substituted:
        intended_txt = get_action_description(active_idx, intended, game_env) if intended is not None else "(no action)"
        game_log.append(f"[{name}] SUBSTITUTED ({note}) — intended: {intended_txt} "
                        f"→ played random: {get_action_description(active_idx, action, game_env)} ({ms:.0f} ms)")
    else:
        game_log.append(f"[{name}] {get_action_description(active_idx, action, game_env)} ({ms:.0f} ms)")

    last_ai_move = {"agent": name, "action": action, "intended": intended,
                    "ms": round(ms, 1), "note": note, "substituted": substituted}

    if 0 <= action <= 7:
        chosen_roles[Role(action).name] = active_idx
    try:
        game_env.step(action)
    except Exception as e:
        return jsonify({"error": f"Game engine error during AI step: {e}"}), 500
    check_phase_transition(game_env)
    return jsonify(serialize_state(game_env))


@app.route("/api/undo", methods=["POST"])
def undo_turn():
    global game_env, game_log, undo_stack, chosen_roles, last_phase, last_ai_move
    if not undo_stack:
        return jsonify({"error": "No actions to undo"}), 400
    game_env, game_log, chosen_roles = undo_stack.pop()
    bind_agents()                       # re-point agents at the restored game
    last_phase = game_env.game.current_phase
    last_ai_move = None
    return jsonify(serialize_state(game_env))


if __name__ == "__main__":
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    print("Puerto Rico web UI - open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
