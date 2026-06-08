// Puerto Rico Game Frontend Controller

// Mappings
const GOODS = {
    "COFFEE": { name: "Coffee", emoji: "☕", class: "good-tile-coffee", badge: "badge-COFFEE" },
    "TOBACCO": { name: "Tobacco", emoji: "🍂", class: "good-tile-tobacco", badge: "badge-TOBACCO" },
    "CORN": { name: "Corn", emoji: "🌽", class: "good-tile-corn", badge: "badge-CORN" },
    "SUGAR": { name: "Sugar", emoji: "🍬", class: "good-tile-sugar", badge: "badge-SUGAR" },
    "INDIGO": { name: "Indigo", emoji: "🔹", class: "good-tile-indigo", badge: "badge-INDIGO" },
    "QUARRY": { name: "Quarry", emoji: "🪨", class: "good-tile-quarry", badge: "badge-QUARRY" },
    "EMPTY": { name: "", emoji: "", class: "", badge: "" }
};

const BUILDINGS = [
    { id: 0, name: "Small Indigo Plant", cost: 1, vp: 1, cap: 1, type: "production" },
    { id: 1, name: "Small Sugar Mill", cost: 2, vp: 1, cap: 1, type: "production" },
    { id: 2, name: "Indigo Plant", cost: 3, vp: 2, cap: 3, type: "production" },
    { id: 3, name: "Sugar Mill", cost: 4, vp: 2, cap: 3, type: "production" },
    { id: 4, name: "Tobacco Storage", cost: 5, vp: 3, cap: 3, type: "production" },
    { id: 5, name: "Coffee Roaster", cost: 6, vp: 3, cap: 2, type: "production" },
    { id: 6, name: "Small Market", cost: 1, vp: 1, cap: 1, type: "violet" },
    { id: 7, name: "Hacienda", cost: 2, vp: 1, cap: 1, type: "violet" },
    { id: 8, name: "Construction Hut", cost: 2, vp: 1, cap: 1, type: "violet" },
    { id: 9, name: "Small Warehouse", cost: 3, vp: 1, cap: 1, type: "violet" },
    { id: 10, name: "Hospice", cost: 4, vp: 2, cap: 1, type: "violet" },
    { id: 11, name: "Office", cost: 5, vp: 2, cap: 1, type: "violet" },
    { id: 12, name: "Large Market", cost: 5, vp: 2, cap: 1, type: "violet" },
    { id: 13, name: "Large Warehouse", cost: 6, vp: 2, cap: 1, type: "violet" },
    { id: 14, name: "Factory", cost: 7, vp: 3, cap: 1, type: "violet" },
    { id: 15, name: "University", cost: 8, vp: 3, cap: 1, type: "violet" },
    { id: 16, name: "Harbor", cost: 8, vp: 3, cap: 1, type: "violet" },
    { id: 17, name: "Wharf", cost: 9, vp: 3, cap: 1, type: "violet" },
    { id: 18, name: "Guildhall", cost: 10, vp: 4, cap: 1, type: "large_violet" },
    { id: 19, name: "Residence", cost: 10, vp: 4, cap: 1, type: "large_violet" },
    { id: 20, name: "Fortress", cost: 10, vp: 4, cap: 1, type: "large_violet" },
    { id: 21, name: "Customs House", cost: 10, vp: 4, cap: 1, type: "large_violet" },
    { id: 22, name: "City Hall", cost: 10, vp: 4, cap: 1, type: "large_violet" }
];

const ENUM_ROLES = ["SETTLER", "MAYOR", "BUILDER", "CRAFTSMAN", "TRADER", "CAPTAIN", "PROSPECTOR_1", "PROSPECTOR_2"];
const ENUM_GOODS = ["COFFEE", "TOBACCO", "CORN", "SUGAR", "INDIGO"];
const ENUM_TILES = ["COFFEE_PLANTATION", "TOBACCO_PLANTATION", "CORN_PLANTATION", "SUGAR_PLANTATION", "INDIGO_PLANTATION", "QUARRY", "EMPTY"];
const ENUM_BUILDINGS = [
    "SMALL_INDIGO_PLANT", "SMALL_SUGAR_MILL", "INDIGO_PLANT", "SUGAR_MILL", "TOBACCO_STORAGE", "COFFEE_ROASTER",
    "SMALL_MARKET", "HACIENDA", "CONSTRUCTION_HUT", "SMALL_WAREHOUSE", "HOSPICE", "OFFICE", "LARGE_MARKET",
    "LARGE_WAREHOUSE", "FACTORY", "UNIVERSITY", "HARBOR", "WHARF", "GUILDHALL", "RESIDENCE", "FORTRESS",
    "CUSTOMS_HOUSE", "CITY_HALL", "EMPTY", "OCCUPIED_SPACE"
];

// App State
let gameState = null;
let selectedTabIdx = 0; // Index of the player board currently being viewed in tabs
let aiTimer = null;
let humanPlayerIdx = 0; // Seat index of the (first) human player, for board focus
let skipHaciendaThisTurn = false;
let AGENT_OPTIONS = { builtin: [], submissions: [] };

// DOM Elements
const setupScreen = document.getElementById("setup-screen");
const gameScreen = document.getElementById("game-screen");
const playerSetupList = document.getElementById("player-setup-list");
const startBtn = document.getElementById("start-game-btn");
const activePlayerBoard = document.getElementById("active-player-board");

// Setup Event Listeners for Player Count Configuration
document.querySelectorAll(".count-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".count-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderPlayerSetupRows(parseInt(btn.dataset.count));
    });
});

function agentSelectOptions() {
    const baseOpts = (AGENT_OPTIONS.builtin || [])
        .filter(a => a.token !== "human")
        .map(a => `<option value="${a.token}">${a.label}</option>`).join("");
    const subOpts = (AGENT_OPTIONS.submissions || [])
        .map(a => `<option value="${a.token}">${a.label}</option>`).join("");
    const subGroup = subOpts ? `<optgroup label="Your submissions/">${subOpts}</optgroup>` : "";
    return `
        <optgroup label="Human"><option value="human">Human (you play)</option></optgroup>
        <optgroup label="Baselines">${baseOpts}</optgroup>
        ${subGroup}
    `;
}

function renderPlayerSetupRows(count) {
    playerSetupList.innerHTML = "";
    // Default seats: you + strong baselines (change any of them below).
    const defaults = ["human", "actionvalue", "shipping", "mcts", "random"];
    for (let i = 0; i < count; i++) {
        const row = document.createElement("div");
        row.className = "player-setup-row";
        row.innerHTML = `
            <label>Player ${i + 1}</label>
            <select id="p-type-${i}">${agentSelectOptions()}</select>
            <input type="text" id="p-custom-${i}" class="custom-spec"
                   placeholder="or module:Class / file.py:Class">
        `;
        playerSetupList.appendChild(row);
        const sel = document.getElementById(`p-type-${i}`);
        const def = defaults[i] || "random";
        if ([...sel.options].some(o => o.value === def)) sel.value = def;
    }
}

// Fetch available agents (baselines + discovered submissions), then render setup.
async function fetchAgents() {
    try {
        const res = await fetch("/api/agents");
        if (res.ok) AGENT_OPTIONS = await res.json();
    } catch (e) { console.warn("Could not fetch agent list:", e); }
}
fetchAgents().then(() => {
    const active = document.querySelector(".count-btn.active");
    renderPlayerSetupRows(active ? parseInt(active.dataset.count) : 3);
});

startBtn.addEventListener("click", async () => {
    const activeCountBtn = document.querySelector(".count-btn.active");
    const numPlayers = parseInt(activeCountBtn.dataset.count);
    const players = [];
    
    // Scan configured seats
    for (let i = 0; i < numPlayers; i++) {
        const typeSelect = document.getElementById(`p-type-${i}`);
        const customSpec = (document.getElementById(`p-custom-${i}`).value || "").trim();
        const token = customSpec !== "" ? customSpec : typeSelect.value;
        players.push(token);
        if (token === "human") {
            humanPlayerIdx = i; // first human seat (for board focus)
        }
    }
    
    // Call Init Endpoint
    const res = await fetch("/api/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_players: numPlayers, players: players })
    });
    
    if (res.ok) {
        gameState = await res.json();
        setupScreen.classList.remove("active");
        gameScreen.classList.add("active");
        selectedTabIdx = humanPlayerIdx; // Focus human board on load
        renderGame();
    } else {
        const err = await res.json().catch(() => ({}));
        alert("Failed to start game: " + (err.error || res.statusText));
    }
});

// Control Button Event Listeners
document.getElementById("undo-btn").addEventListener("click", async () => {
    clearTimeout(aiTimer);
    const res = await fetch("/api/undo", { method: "POST" });
    if (res.ok) {
        gameState = await res.json();
        renderGame();
    } else {
        const err = await res.json();
        alert(err.error || "Cannot undo");
    }
});

document.getElementById("restart-btn").addEventListener("click", () => {
    clearTimeout(aiTimer);
    gameScreen.classList.remove("active");
    setupScreen.classList.add("active");
});

document.getElementById("pass-btn").addEventListener("click", async () => {
    // Pass is always action index 15
    await sendAction(15);
});

document.getElementById("ai-step-btn").addEventListener("click", async () => {
    await runAIStep();
});

document.getElementById("modal-close-btn").addEventListener("click", () => {
    document.getElementById("game-over-modal").classList.remove("active");
    gameScreen.classList.remove("active");
    setupScreen.classList.add("active");
});

async function sendAction(actionIdx) {
    clearTimeout(aiTimer);
    const res = await fetch("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: actionIdx })
    });
    
    if (res.ok) {
        gameState = await res.json();
        renderGame();
    } else {
        const err = await res.json();
        alert(err.error || "Action failed.");
    }
}

async function runAIStep() {
    clearTimeout(aiTimer);
    const res = await fetch("/api/ai_step", { method: "POST" });
    if (res.ok) {
        gameState = await res.json();
        renderGame();
    } else {
        const err = await res.json();
        alert(err.error || "AI turn failed.");
    }
}

// Rendering Logic
function renderGame() {
    if (!gameState) return;
    
    const isHumanTurn = (gameState.players[gameState.current_agent_idx].type === "human") && !gameState.game_over;
    
    // Reset skipHacienda if it's not settler phase or not human's turn
    if (gameState.current_phase !== "SETTLER" || !isHumanTurn) {
        skipHaciendaThisTurn = false;
    }
    
    // 1. Header Details
    document.getElementById("round-number").innerText = `Round: ${gameState.round_number}`;
    document.getElementById("phase-name").innerText = `Phase: ${gameState.current_phase.replace("_", " ")}`;
    document.getElementById("global-vp-chips").innerText = gameState.vp_chips;
    document.getElementById("global-colonist-ship").innerText = gameState.colonists_ship;
    document.getElementById("global-colonist-supply").innerText = gameState.colonists_supply;
    
    // 2. Undo Enable Status
    document.getElementById("undo-btn").disabled = gameState.game_over; // Let server validate stack depth
    
    // 3. Pass Button Visibility
    const passBtn = document.getElementById("pass-btn");
    if (isHumanTurn && gameState.valid_actions.includes(15)) {
        passBtn.classList.remove("hidden");
    } else {
        passBtn.classList.add("hidden");
    }
    
    // 4. Action Prompt English Instruction
    const prompt = document.getElementById("action-prompt");
    if (gameState.game_over) {
        prompt.innerHTML = "<strong>Game Over!</strong> Click 'Restart' or view final standing details.";
    } else if (isHumanTurn) {
        let text = "It is your turn! ";
        const phase = gameState.current_phase;
        if (phase === "INIT/ROLE_SELECT" || phase === "END_ROUND") {
            text += "Pick an available Role Card from the center board.";
        } else if (phase === "SETTLER") {
            text += "Select a plantation card from the market, pick a Quarry, trigger Hacienda, or Pass.";
        } else if (phase === "BUILDER") {
            text += "Select a building from the shop to build/purchase, or click Pass.";
        } else if (phase === "TRADER") {
            text += "Choose one of your goods to sell to the Trading House, or click Pass.";
        } else if (phase === "CAPTAIN") {
            text += "Choose a good to load on a valid cargo ship. You must load if possible, or click Pass.";
        } else if (phase === "CAPTAIN_STORE") {
            text += "Store remaining goods in Windrose (holds 1 barrel) or Warehouse slots. Click Pass when done.";
        } else if (phase === "MAYOR") {
            text += "Place colonists one by one onto island plantations or city buildings.";
        } else if (phase === "CRAFTSMAN") {
            text += "Select one bonus good from supply as your Craftsman privilege, or click Pass.";
        }
        prompt.innerHTML = `<strong>${text}</strong>`;
        prompt.className = "action-prompt";
    } else {
        const ap = gameState.players[gameState.current_agent_idx];
        prompt.innerHTML = `Waiting for Player ${gameState.current_agent_idx + 1} — <strong>${ap.label}</strong> to act...`;
        prompt.className = "action-prompt warning";
    }
    
    // 4b. Last AI move — decision time and competition-rule mirroring
    const lastMoveEl = document.getElementById("last-ai-move");
    if (lastMoveEl) {
        const m = gameState.last_ai_move;
        if (m) {
            let html = `Last AI move: <strong>${m.agent}</strong> <span class="lm-time">${m.ms} ms</span>`;
            if (m.substituted) html += ` <span class="lm-flag">⚠ SUBSTITUTED — ${m.note}</span>`;
            lastMoveEl.innerHTML = html;
            lastMoveEl.className = m.substituted ? "last-ai-move flagged" : "last-ai-move";
        } else {
            lastMoveEl.innerHTML = "";
            lastMoveEl.className = "last-ai-move";
        }
    }

    // 5. Activity Log Scroll
    const logBox = document.getElementById("log-container");
    logBox.innerHTML = "";
    gameState.game_log.forEach(item => {
        const el = document.createElement("div");
        if (item.startsWith("---")) {
            el.className = "log-item phase-divider";
        } else if (item.startsWith("[AI]")) {
            el.className = "log-item ai-action";
        } else {
            el.className = "log-item";
        }
        el.innerText = item;
        logBox.appendChild(el);
    });
    logBox.scrollTop = logBox.scrollHeight;
    
    // 6. Available Role Cards
    const rolesContainer = document.getElementById("roles-container");
    rolesContainer.innerHTML = "";
    ENUM_ROLES.forEach((roleName) => {
        const roleVal = ENUM_ROLES.indexOf(roleName);
        
        // Skip roles not used in setup (Prospectors for small counts)
        if (!gameState.available_roles.includes(roleName) && !gameState.roles_in_play.includes(roleName)) {
            return; 
        }
        
        const card = document.createElement("div");
        card.className = "role-card";
        
        const isTaken = gameState.roles_in_play.includes(roleName);
        if (isTaken) {
            card.classList.add("taken");
        }
        
        const isActive = gameState.active_role === roleName;
        if (isActive) {
            card.classList.add("active-role");
        }
        
        const doubloons = gameState.role_doubloons[roleName] || 0;
        
        // Show who picked the role if any
        let pickedByHtml = "";
        if (gameState.chosen_roles && gameState.chosen_roles[roleName] !== undefined) {
            const pickerIdx = gameState.chosen_roles[roleName];
            const pickerType = gameState.players[pickerIdx].type;
            const label = pickerType === "human" ? `P${pickerIdx+1} (Human)` : `P${pickerIdx+1} (AI)`;
            pickedByHtml = `<div class="picked-by">Picked: ${label}</div>`;
        }
        
        card.innerHTML = `
            <div class="role-title">${roleName.replace("_", " ")}</div>
            ${doubloons > 0 ? `<div class="role-bonus">+${doubloons} 🪙</div>` : ''}
            ${pickedByHtml}
        `;
        
        // Pick role click listener
        if (isHumanTurn && gameState.valid_actions.includes(roleVal)) {
            card.classList.add("playable");
            card.addEventListener("click", () => sendAction(roleVal));
        }
        
        rolesContainer.appendChild(card);
    });
    
    // 7. Face-Up Plantations & Quarry
    const plantationsSection = document.getElementById("plantations-section");
    // Clear any existing hacienda prompt banner first
    const existingBanner = document.getElementById("hacienda-prompt-banner");
    if (existingBanner) existingBanner.remove();
    
    const hasHaciendaOption = isHumanTurn && gameState.valid_actions.includes(105);
    const showHaciendaPrompt = hasHaciendaOption && !skipHaciendaThisTurn;
    
    if (showHaciendaPrompt) {
        const banner = document.createElement("div");
        banner.id = "hacienda-prompt-banner";
        banner.className = "hacienda-prompt-banner";
        banner.style.background = "rgba(99, 102, 241, 0.15)";
        banner.style.border = "1px solid var(--accent-indigo)";
        banner.style.textAlign = "center";
        banner.style.padding = "10px";
        banner.style.marginBottom = "8px";
        banner.style.borderRadius = "var(--radius-sm)";
        banner.innerHTML = `
            <div style="font-size:0.85rem; font-weight:700; margin-bottom:6px; color:#c7d2fe;">🏡 Hacienda active! Draw face-down from stack?</div>
            <div style="display:flex; gap:8px; justify-content:center;">
                <button type="button" class="btn btn-primary btn-small" id="hacienda-yes-btn">Yes, Draw</button>
                <button type="button" class="btn btn-secondary btn-small" id="hacienda-no-btn">No, Skip</button>
            </div>
        `;
        // Insert banner at the top (before plantations-container)
        const container = plantationsSection.querySelector(".plantations-container");
        plantationsSection.insertBefore(banner, container);
        
        document.getElementById("hacienda-yes-btn").addEventListener("click", () => sendAction(105));
        document.getElementById("hacienda-no-btn").addEventListener("click", () => {
            skipHaciendaThisTurn = true;
            renderGame();
        });
    }

    const plantationsGrid = document.getElementById("face-up-plantations");
    plantationsGrid.innerHTML = "";
    gameState.face_up_plantations.forEach((tileName, index) => {
        const tile = document.createElement("div");
        const baseGood = tileName.split("_")[0];
        const spec = GOODS[baseGood] || GOODS.EMPTY;
        
        tile.className = `plantation-tile ${spec.class}`;
        tile.innerHTML = `
            <span class="emoji">${spec.emoji}</span>
            <span class="label">${baseGood}</span>
        `;
        
        // Settler choice index matches 8 + Good value
        const tileVal = ENUM_GOODS.indexOf(baseGood);
        const actionIdx = 8 + tileVal;
        
        // Only playable if hacienda prompt is NOT shown/blocking
        if (isHumanTurn && !showHaciendaPrompt && gameState.valid_actions.includes(actionIdx)) {
            tile.classList.add("playable");
            tile.addEventListener("click", () => sendAction(actionIdx));
        }
        
        plantationsGrid.appendChild(tile);
    });
    
    // Quarry stack card
    const quarryCard = document.getElementById("quarry-card");
    document.getElementById("quarry-stock").innerText = `Quarries: ${gameState.quarry_stack}`;
    
    // Quarry action is index 13
    // Only playable if hacienda prompt is NOT shown/blocking
    if (isHumanTurn && !showHaciendaPrompt && gameState.valid_actions.includes(13)) {
        quarryCard.classList.add("playable");
        // Remove old event listeners
        const newQuarryCard = quarryCard.cloneNode(true);
        quarryCard.parentNode.replaceChild(newQuarryCard, quarryCard);
        newQuarryCard.addEventListener("click", () => sendAction(13));
    } else {
        quarryCard.classList.remove("playable");
    }
    
    // Craftsman privilege choice rendering
    // Action indices 93 to 97
    let craftsmanDiv = document.getElementById("craftsman-choices");
    if (isHumanTurn && [93, 94, 95, 96, 97].some(a => gameState.valid_actions.includes(a))) {
        if (!craftsmanDiv) {
            craftsmanDiv = document.createElement("div");
            craftsmanDiv.id = "craftsman-choices";
            craftsmanDiv.className = "board-card card";
            craftsmanDiv.innerHTML = `<h3 class="section-title">Craftsman Privilege (Choose 1 Good)</h3><div class="goods-badge-row" style="display:flex; gap:8px;"></div>`;
            document.querySelector(".board-top-grid").appendChild(craftsmanDiv);
        }
        const row = craftsmanDiv.querySelector(".goods-badge-row");
        row.innerHTML = "";
        ENUM_GOODS.forEach((gName, idx) => {
            const actionIdx = 93 + idx;
            if (gameState.valid_actions.includes(actionIdx)) {
                const spec = GOODS[gName];
                const btn = document.createElement("button");
                btn.className = "btn btn-secondary btn-small";
                btn.innerHTML = `${spec.emoji} ${gName}`;
                btn.addEventListener("click", () => sendAction(actionIdx));
                row.appendChild(btn);
            }
        });
    } else if (craftsmanDiv) {
        craftsmanDiv.remove();
    }

    // 8. Cargo Ships
    const shipsContainer = document.getElementById("ships-container");
    shipsContainer.innerHTML = "";
    gameState.cargo_ships.forEach((ship, shipIdx) => {
        const shipEl = document.createElement("div");
        shipEl.className = "cargo-ship";
        
        let barrelsHtml = "";
        for (let b = 0; b < ship.capacity; b++) {
            if (b < ship.current_load) {
                barrelsHtml += `<div class="cargo-barrel loaded-${ship.good_type}"></div>`;
            } else {
                barrelsHtml += `<div class="cargo-barrel"></div>`;
            }
        }
        
        // Captain Load Action (Indices 44 to 58: ship_idx * 5 + good_val)
        let loadBtnHtml = "";
        if (isHumanTurn) {
            ENUM_GOODS.forEach((gName, gVal) => {
                const actionIdx = 44 + (shipIdx * 5) + gVal;
                if (gameState.valid_actions.includes(actionIdx)) {
                    const spec = GOODS[gName];
                    loadBtnHtml += `<button type="button" class="ship-load-action-btn" onclick="sendAction(${actionIdx})">Load ${spec.emoji}</button> `;
                }
            });
        }
        
        shipEl.innerHTML = `
            <div class="ship-label">Ship ${shipIdx + 1} (Cap: ${ship.capacity})</div>
            <div class="ship-cargo-row">${barrelsHtml}</div>
            <div class="ship-actions">${loadBtnHtml || `<span class="ship-status">${ship.current_load}/${ship.capacity} ${ship.good_type || 'Empty'}</span>`}</div>
        `;
        shipsContainer.appendChild(shipEl);
    });
    
    // Render Wharf Imaginary Ship if Wharf is occupied
    const humanPlayer = gameState.players[humanPlayerIdx];
    const wharfOccupied = humanPlayer.city_board.some(b => b.building_type === "WHARF" && b.colonists > 0);
    
    if (wharfOccupied) {
        // Wharf Load Actions (Indices 59 to 63: 59 + good_val)
        let wharfLoadHtml = "";
        if (isHumanTurn) {
            ENUM_GOODS.forEach((gName, gVal) => {
                const actionIdx = 59 + gVal;
                if (gameState.valid_actions.includes(actionIdx)) {
                    const spec = GOODS[gName];
                    wharfLoadHtml += `<button type="button" class="ship-load-action-btn" onclick="sendAction(${actionIdx})">Wharf ${spec.emoji}</button> `;
                }
            });
        }
        
        const shipEl = document.createElement("div");
        shipEl.className = "cargo-ship";
        shipEl.innerHTML = `
            <div class="ship-label" style="color: #a7f3d0">Wharf Ship</div>
            <div style="flex:1; text-align:center; font-size:0.8rem; color:var(--text-secondary)">Unlimited Capacity</div>
            <div class="ship-actions">${wharfLoadHtml || '<span class="ship-status" style="color: #64748b">Inactive</span>'}</div>
        `;
        shipsContainer.appendChild(shipEl);
    }
    
    // 9. Trading House
    for (let slotIdx = 0; slotIdx < 4; slotIdx++) {
        const slotEl = document.getElementById(`trade-${slotIdx}`);
        slotEl.className = "trade-slot empty";
        slotEl.innerHTML = "";
        
        if (slotIdx < gameState.trading_house.length) {
            const goodName = gameState.trading_house[slotIdx];
            const spec = GOODS[goodName] || GOODS.EMPTY;
            slotEl.className = `trade-slot filled filled-${goodName}`;
            slotEl.innerHTML = spec.emoji;
        }
    }
    
    // 10. Building Shop
    const shopContainer = document.getElementById("shop-container");
    shopContainer.innerHTML = "";
    
    // Create column elements for VP 1, 2, 3, 4
    const cols = {};
    for (let vp = 1; vp <= 4; vp++) {
        const col = document.createElement("div");
        col.className = "shop-column";
        col.innerHTML = `<h4 class="shop-col-title">${vp} VP</h4>`;
        shopContainer.appendChild(col);
        cols[vp] = col;
    }
    
    // Group and sort buildings
    const grouped = { 1: [], 2: [], 3: [], 4: [] };
    BUILDINGS.forEach(b => {
        if (grouped[b.vp]) {
            grouped[b.vp].push(b);
        }
    });
    
    // Within each group, sort: production first
    Object.keys(grouped).forEach(vp => {
        grouped[vp].sort((a, b) => {
            const aIsProd = a.type === "production" ? 1 : 0;
            const bIsProd = b.type === "production" ? 1 : 0;
            return bIsProd - aIsProd; // Production first (1 comes before 0)
        });
    });
    
    // Render sorted buildings in columns
    Object.keys(grouped).forEach(vp => {
        const colEl = cols[vp];
        grouped[vp].forEach(b => {
            const item = document.createElement("div");
            item.className = "building-shop-item";
            
            const stockCount = gameState.building_supply[b.name] || 0;
            
            // Calculate Dynamic Discount
            const activeQuarries = humanPlayer.island_board.filter(t => t.tile_type === "QUARRY" && t.is_occupied).length;
            const maxQ = b.vp;
            const quarryDiscount = Math.min(activeQuarries, maxQ);
            
            const isBuilderRole = (gameState.current_phase === "BUILDER");
            const hasPrivilege = isBuilderRole && (gameState.active_role_player === humanPlayerIdx);
            const privilegeDiscount = hasPrivilege ? 1 : 0;
            
            const finalCost = Math.max(0, b.cost - quarryDiscount - privilegeDiscount);
            const costDiffers = finalCost !== b.cost;
            
            item.innerHTML = `
                <div class="b-header">
                    <span class="b-name" style="font-size:0.75rem;">${b.name}</span>
                    <span class="b-stock">${stockCount} Left</span>
                </div>
                <div class="b-details">
                    <span class="b-slots">${b.cap} 👥</span>
                    <span class="b-cost">
                        ${costDiffers ? `<span class="orig-cost">${b.cost}</span>` : ''}
                        ${finalCost} 🪙
                    </span>
                </div>
            `;
            
            const actionIdx = 16 + b.id;
            if (isHumanTurn && gameState.valid_actions.includes(actionIdx)) {
                item.classList.add("playable");
                item.addEventListener("click", () => sendAction(actionIdx));
            }
            
            colEl.appendChild(item);
        });
    });
    
    // 11. Player Board Tabs Rendering
    const tabsContainer = document.getElementById("player-tabs");
    tabsContainer.innerHTML = "";
    gameState.players.forEach((p, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "tab-btn";
        if (idx === selectedTabIdx) {
            btn.classList.add("active");
        }
        if (idx === gameState.current_agent_idx && !gameState.game_over) {
            btn.classList.add("active-turn");
        }
        
        // Distinguish human player and add Governor indicator
        const isGov = (idx === gameState.governor_idx);
        const govSuffix = isGov ? " 👑" : "";
        const label = p.type === "human" ? "Human (P" + (idx+1) + ")" + govSuffix : "AI Bot (P" + (idx+1) + ")" + govSuffix;
        btn.innerText = label;
        if (isGov) {
            btn.title = "Governor of this round";
        }
        btn.addEventListener("click", () => {
            selectedTabIdx = idx;
            renderSelectedPlayerBoard();
        });
        tabsContainer.appendChild(btn);
    });
    
    renderSelectedPlayerBoard();
    
    // 12. Captain Store Windrose / Warehouse Assignments UI
    // If Phase is CAPTAIN_STORE and human is active, render special UI
    let storeDiv = document.getElementById("store-choices");
    if (isHumanTurn && gameState.current_phase === "CAPTAIN_STORE") {
        if (!storeDiv) {
            storeDiv = document.createElement("div");
            storeDiv.id = "store-choices";
            storeDiv.className = "board-card card";
            storeDiv.innerHTML = `<h3 class="section-title">Captain Store Phase - Select Storage Actions</h3><div class="store-buttons" style="display:flex; flex-direction:column; gap:6px;"></div>`;
            document.querySelector(".board-top-grid").appendChild(storeDiv);
        }
        
        const list = storeDiv.querySelector(".store-buttons");
        list.innerHTML = "";
        
        ENUM_GOODS.forEach((gName, gVal) => {
            const windroseAction = 64 + gVal;
            const warehouseAction = 106 + gVal;
            const spec = GOODS[gName];
            
            if (gameState.valid_actions.includes(windroseAction)) {
                const btn = document.createElement("button");
                btn.className = "btn btn-secondary btn-small";
                btn.innerHTML = `Store 1 barrel of ${spec.emoji} on Windrose`;
                btn.addEventListener("click", () => sendAction(windroseAction));
                list.appendChild(btn);
            }
            if (gameState.valid_actions.includes(warehouseAction)) {
                const btn = document.createElement("button");
                btn.className = "btn btn-secondary btn-small";
                btn.innerHTML = `Store all ${spec.emoji} barrels in Warehouse`;
                btn.addEventListener("click", () => sendAction(warehouseAction));
                list.appendChild(btn);
            }
        });
    } else if (storeDiv) {
        storeDiv.remove();
    }
    
    // 13. Game Over Screen Modal
    if (gameState.game_over) {
        const scoreboardBody = document.getElementById("scoreboard-body");
        scoreboardBody.innerHTML = "";
        
        // Sort players based on rank (highest total_vp first, then tie breaker)
        const sortedPlayers = [...gameState.players].sort((a, b) => {
            if (b.scores.total_vp !== a.scores.total_vp) {
                return b.scores.total_vp - a.scores.total_vp;
            }
            return b.scores.tie_breaker - a.scores.tie_breaker;
        });
        
        sortedPlayers.forEach((p, index) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${index + 1}</strong></td>
                <td>Player ${p.index + 1} (${p.type === "human" ? "Human" : "AI"})</td>
                <td><strong>${p.scores.total_vp}</strong></td>
                <td>${p.scores.shipping_vp}</td>
                <td>${p.scores.building_vp}</td>
                <td>${p.scores.bonus_vp}</td>
                <td>${p.scores.tie_breaker}</td>
            `;
            scoreboardBody.appendChild(row);
        });
        
        document.getElementById("game-over-modal").classList.add("active");
    }
    
    // 14. Handle AI Next Step Triggers
    const activePlayerType = gameState.players[gameState.current_agent_idx].type;
    const isAITurn = activePlayerType.includes("ai") && !gameState.game_over;
    const aiBtn = document.getElementById("ai-step-btn");
    
    if (isAITurn) {
        aiBtn.classList.remove("hidden");
        const autoToggle = document.getElementById("auto-ai-toggle");
        if (autoToggle.checked) {
            aiBtn.disabled = true;
            aiTimer = setTimeout(async () => {
                await runAIStep();
            }, 800); // 800ms delay to make game flow watchable
        } else {
            aiBtn.disabled = false;
        }
    } else {
        aiBtn.classList.add("hidden");
    }
}

function renderSelectedPlayerBoard() {
    if (!gameState) return;
    
    const p = gameState.players[selectedTabIdx];
    const isHumanTurn = (gameState.players[gameState.current_agent_idx].type === "human") && !gameState.game_over;
    
    // Update Title with Governor status
    const isGov = (selectedTabIdx === gameState.governor_idx);
    const label = p.type === "human" ? `Player ${p.player_idx + 1} (Human) Board` : `Player ${p.player_idx + 1} (AI Bot) Board`;
    const govBadge = isGov ? `<span style="font-size:0.75rem; background:rgba(234,179,8,0.15); color:var(--accent-gold); padding:2px 8px; border-radius:12px; font-weight:700; border:1px solid rgba(234,179,8,0.3); margin-left:8px; display:inline-flex; align-items:center; gap:4px;">👑 Governor</span>` : '';
    document.getElementById("player-board-title").innerHTML = `${label} ${govBadge}`;
    
    // Update Meta Resources
    document.getElementById("player-doubloons").innerText = p.doubloons;
    document.getElementById("player-vp").innerText = p.vp_chips;
    document.getElementById("player-colonists").innerText = p.unplaced_colonists;
    
    // Goods List
    const goodsContainer = document.getElementById("player-goods-container");
    goodsContainer.innerHTML = "";
    Object.keys(p.goods).forEach((gName) => {
        const count = p.goods[gName];
        if (count > 0) {
            const spec = GOODS[gName] || GOODS.EMPTY;
            const badge = document.createElement("div");
            badge.className = `goods-badge ${spec.badge}`;
            badge.innerHTML = `${spec.emoji} ${gName}: ${count}`;
            
            // Sell logic during Trader phase
            const gVal = ENUM_GOODS.indexOf(gName);
            const actionIdx = 39 + gVal;
            if (isHumanTurn && (selectedTabIdx === humanPlayerIdx) && gameState.valid_actions.includes(actionIdx)) {
                badge.style.cursor = "pointer";
                badge.style.boxShadow = "0 0 8px rgba(99, 102, 241, 0.4)";
                badge.addEventListener("click", () => sendAction(actionIdx));
            }
            
            goodsContainer.appendChild(badge);
        }
    });
    
    // Island grid (12 slots)
    const islandGrid = document.getElementById("island-grid");
    islandGrid.innerHTML = "";
    
    for (let slot = 0; slot < 12; slot++) {
        const slotEl = document.createElement("div");
        slotEl.className = "grid-slot";
        
        if (slot < p.island_board.length) {
            const tile = p.island_board[slot];
            const baseType = tile.tile_type.replace("_PLANTATION", "");
            const spec = GOODS[baseType] || GOODS.EMPTY;
            
            slotEl.className = `grid-slot ${spec.class} ${tile.is_occupied ? 'colonist-active' : ''}`;
            slotEl.innerHTML = `
                <span class="slot-emoji">${spec.emoji}</span>
                <span class="slot-name">${baseType}</span>
                ${tile.is_occupied ? '<div class="colonist-indicator active" title="Occupied"></div>' : '<div class="colonist-indicator" title="Empty"></div>'}
            `;
            
            // Mayor island placement: 120 + tile_type value
            const tileTypeVal = ENUM_TILES.indexOf(tile.tile_type);
            const actionIdx = 120 + tileTypeVal;
            
            if (isHumanTurn && (selectedTabIdx === humanPlayerIdx) && !tile.is_occupied && gameState.valid_actions.includes(actionIdx)) {
                slotEl.classList.add("playable");
                slotEl.addEventListener("click", () => sendAction(actionIdx));
            }
            
        } else {
            slotEl.innerHTML = `<span style="color:var(--text-muted)">Empty</span>`;
        }
        islandGrid.appendChild(slotEl);
    }
    
    // City grid (12 slots)
    const cityGrid = document.getElementById("city-grid");
    cityGrid.innerHTML = "";
    
    for (let slot = 0; slot < 12; slot++) {
        const slotEl = document.createElement("div");
        slotEl.className = "grid-slot";
        
        if (slot < p.city_board.length) {
            const b = p.city_board[slot];
            if (b.building_type === "OCCUPIED_SPACE") {
                // Occupted space for large buildings
                slotEl.className = "grid-slot";
                slotEl.style.opacity = 0.55;
                slotEl.innerHTML = `<span style="font-size:0.6rem; color:var(--text-muted)">Large Bldg Space</span>`;
            } else if (b.building_type !== "EMPTY") {
                const bDetail = BUILDINGS.find(x => x.name.toUpperCase().replace(/ /g, "_") === b.building_type);
                const bName = bDetail ? bDetail.name : b.building_type;
                const isLarge = bName === "Guildhall" || bName === "Residence" || bName === "Fortress" || bName === "Customs House" || bName === "City Hall";
                
                slotEl.className = `grid-slot ${b.colonists > 0 ? 'colonist-active' : ''}`;
                slotEl.innerHTML = `
                    <div style="font-weight:700; color:var(--text-primary); text-align:center; font-size:0.65rem;">${bName}</div>
                    <div style="font-size:0.6rem; color:var(--text-secondary); margin-top:4px;">${b.colonists}/${b.capacity} Colonists</div>
                    ${b.colonists > 0 ? '<div class="colonist-indicator active"></div>' : '<div class="colonist-indicator"></div>'}
                `;
                
                // Mayor city placement: 140 + building_type value
                const bTypeVal = ENUM_BUILDINGS.indexOf(b.building_type);
                const actionIdx = 140 + bTypeVal;
                
                if (isHumanTurn && (selectedTabIdx === humanPlayerIdx) && (b.colonists < b.capacity) && gameState.valid_actions.includes(actionIdx)) {
                    slotEl.classList.add("playable");
                    slotEl.addEventListener("click", () => sendAction(actionIdx));
                }
            } else {
                slotEl.innerHTML = `<span style="color:var(--text-muted)">Empty</span>`;
            }
        } else {
            slotEl.innerHTML = `<span style="color:var(--text-muted)">Empty</span>`;
        }
        cityGrid.appendChild(slotEl);
    }
}

// Global functions exposed to inline onClick
window.sendAction = sendAction;

// Theme Initialization & Handler
const savedTheme = localStorage.getItem("theme") || "dark";
document.documentElement.setAttribute("data-theme", savedTheme);

document.querySelectorAll(".theme-toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });
});
