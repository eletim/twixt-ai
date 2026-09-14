const SVG_NS = "http://www.w3.org/2000/svg";
const viewerMode = window.location.pathname === "/viewer";
const $ = (selector) => document.querySelector(selector);
const boardElement = $("#board");
const statusElement = $("#status");
const messageElement = $("#message");
const inspectionElement = $("#inspection");
const candidateTableWrap = $("#candidate-table-wrap");
const candidateTable = $("#candidate-table");
const resetButton = $("#reset");
const sideSelect = $("#human-side");
const agentSelect = $("#agent");
const presetSelect = $("#board-preset");
const inspectionToggle = $("#inspection-toggle");
const viewerSetup = $("#viewer-setup");
const humanSetup = $("#human-setup");
const redAgentSelect = $("#red-agent");
const blackAgentSelect = $("#black-agent");
const redCheckpointSelect = $("#red-checkpoint");
const blackCheckpointSelect = $("#black-checkpoint");
const seedInput = $("#viewer-seed");
const generateButton = $("#generate-game");
const artifactSelect = $("#artifact");
const loadArtifactButton = $("#load-artifact");
const searchSettings = $("#search-settings");
const replaySummary = $("#replay-summary");
const replayControls = $("#replay-controls");
const firstButton = $("#replay-first");
const backButton = $("#replay-back");
const playButton = $("#replay-play");
const nextButton = $("#replay-next");
const lastButton = $("#replay-last");
const moveCounter = $("#move-counter");
const overlayModeSelect = $("#overlay-mode");

let session = null;
let requestPending = false;
let agentThinking = false;
let replay = null;
let replayIndex = 0;
let playbackTimer = null;

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function textNode(tag, value, className = "") {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
}

function point(coordinate, spacing, margin) {
  return { x: margin + coordinate.x * spacing, y: margin + coordinate.y * spacing };
}

function title(value) {
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ").replaceAll("-", " ");
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(value < 0.01 ? 1 : 0)}%` : "—";
}

function candidateLabel(candidate, overlayMode = "visits", simulations = 0) {
  if (overlayMode === "prior" && Number.isFinite(candidate.prior)) return formatPercent(candidate.prior);
  if (Number.isFinite(candidate.visits) && simulations) return formatPercent(candidate.visits / simulations);
  if (Number.isFinite(candidate.probability)) return `${Math.round(candidate.probability * 100)}%`;
  if (Number.isFinite(candidate.score)) return formatNumber(candidate.score);
  return "";
}

function summarizeMetadata(metadata) {
  const scalarTypes = ["string", "number", "boolean"];
  return Object.entries(metadata)
    .filter(([, value]) => scalarTypes.includes(typeof value))
    .map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`)
    .join(", ");
}

function decisionCandidates(decision) {
  const metadata = decision?.metadata ?? {};
  if (Array.isArray(metadata.root_moves)) return metadata.root_moves;
  return metadata.inspection?.candidates ?? [];
}

function renderBoard(game, { interactive = false, thinking = null, lastMove = null, overlayMode = "off" } = {}) {
  boardElement.replaceChildren();
  const margin = 18;
  const spacing = 28;
  const width = margin * 2 + (game.board.width - 1) * spacing;
  const height = margin * 2 + (game.board.height - 1) * spacing;
  boardElement.setAttribute("viewBox", `0 0 ${width} ${height}`);
  boardElement.setAttribute("aria-rowcount", game.board.height);
  boardElement.setAttribute("aria-colcount", game.board.width);

  const grid = svgElement("g", { class: "grid-lines", "aria-hidden": "true" });
  for (let x = 0; x < game.board.width; x += 1) {
    const start = point({ x, y: 0 }, spacing, margin);
    const end = point({ x, y: game.board.height - 1 }, spacing, margin);
    grid.append(svgElement("line", { x1: start.x, y1: start.y, x2: end.x, y2: end.y }));
  }
  for (let y = 0; y < game.board.height; y += 1) {
    const start = point({ x: 0, y }, spacing, margin);
    const end = point({ x: game.board.width - 1, y }, spacing, margin);
    grid.append(svgElement("line", { x1: start.x, y1: start.y, x2: end.x, y2: end.y }));
  }
  boardElement.append(grid);

  const links = svgElement("g", { class: "links", "aria-hidden": "true" });
  for (const link of game.links) {
    const start = point(link.start, spacing, margin);
    const end = point(link.end, spacing, margin);
    links.append(svgElement("line", { class: link.owner, x1: start.x, y1: start.y, x2: end.x, y2: end.y }));
  }
  boardElement.append(links);

  const metadata = thinking?.metadata ?? {};
  const candidatesData = decisionCandidates(thinking);
  let candidateOverlays = null;
  if (overlayMode !== "off" && candidatesData.length) {
    const candidates = svgElement("g", { class: "candidate-overlays", "aria-hidden": "true" });
    const simulations = metadata.simulations ?? metadata.inspection?.statistics?.simulations ?? 0;
    for (const candidate of candidatesData) {
      const position = point(candidate, spacing, margin);
      const strength = overlayMode === "prior"
        ? candidate.prior : simulations ? candidate.visits / simulations : candidate.probability;
      const marker = svgElement("g", { class: "candidate-overlay" });
      marker.append(svgElement("circle", {
        cx: position.x, cy: position.y, r: 10,
        style: `--strength: ${Math.max(0.15, Number.isFinite(strength) ? strength : 0.15)}`,
      }));
      const label = svgElement("text", { x: position.x, y: position.y + 3 });
      label.textContent = candidateLabel(candidate, overlayMode, simulations);
      marker.append(label);
      candidates.append(marker);
    }
    candidateOverlays = candidates;
  }

  const pegsByCoordinate = new Map(game.pegs.map((peg) => [`${peg.coordinate.x},${peg.coordinate.y}`, peg.owner]));
  const points = svgElement("g", { class: "points" });
  for (let y = 0; y < game.board.height; y += 1) {
    for (let x = 0; x < game.board.width; x += 1) {
      const position = point({ x, y }, spacing, margin);
      const owner = pegsByCoordinate.get(`${x},${y}`);
      const canPlay = interactive && !owner;
      const intersection = svgElement("circle", {
        class: owner ? `intersection peg ${owner}` : "intersection",
        cx: position.x, cy: position.y, r: owner ? 7 : 3, role: "gridcell",
        tabindex: canPlay ? "0" : "-1",
        "aria-label": owner ? `${owner} peg at column ${x + 1}, row ${y + 1}` : `Place at column ${x + 1}, row ${y + 1}`,
        "aria-disabled": canPlay ? "false" : "true",
      });
      if (canPlay) {
        intersection.classList.add("playable");
        intersection.addEventListener("click", () => placePeg(x, y));
        intersection.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            placePeg(x, y);
          }
        });
      } else if (!owner) intersection.classList.add("blocked");
      points.append(intersection);
    }
  }
  boardElement.append(points);
  if (candidateOverlays) boardElement.append(candidateOverlays);
  if (lastMove) {
    const position = point(lastMove.coordinate, spacing, margin);
    boardElement.append(svgElement("circle", { class: "last-move-overlay", cx: position.x, cy: position.y, r: 10, "aria-hidden": "true" }));
  }
  const selectedMove = thinking?.selected_move ?? thinking?.move;
  if (overlayMode !== "off" && selectedMove) {
    const position = point(selectedMove.coordinate, spacing, margin);
    boardElement.append(svgElement("circle", { class: "selected-move-overlay", cx: position.x, cy: position.y, r: 12, "aria-hidden": "true" }));
  }
}

function describeHumanStatus(game) {
  const results = { red_wins: "Red wins!", black_wins: "Black wins!", draw: "Draw" };
  if (results[game.result]) return results[game.result];
  if (agentThinking) return `${title(session.agent)} is thinking…`;
  return game.side_to_move === session.human_side ? `Your turn (${title(session.human_side)})` : `${title(session.agent)} to move`;
}

function populateSetup(view) {
  const names = [...agentSelect.options].map((option) => option.value);
  if (names.join("\0") !== view.available_agents.join("\0")) {
    agentSelect.replaceChildren(...view.available_agents.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = title(name);
      return option;
    }));
  }
  const presets = Object.entries(view.available_presets ?? {});
  if (view.preset === "custom") presets.push(["custom", view.state.board]);
  const presetNames = [...presetSelect.options].map((option) => option.value);
  if (presetNames.join("\0") !== presets.map(([name]) => name).join("\0")) {
    presetSelect.replaceChildren(...presets.map(([name, board]) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = `${title(name)} (${board.width}×${board.height})`;
      return option;
    }));
  }
  sideSelect.value = view.human_side;
  agentSelect.value = view.agent;
  presetSelect.value = view.preset;
  for (const control of [resetButton, sideSelect, agentSelect, presetSelect, inspectionToggle]) control.disabled = requestPending;
  boardElement.setAttribute("aria-busy", agentThinking ? "true" : "false");
}

function renderHuman(view) {
  session = view;
  const game = view.state;
  populateSetup(view);
  statusElement.textContent = describeHumanStatus(game);
  statusElement.dataset.player = game.result === "in_progress" ? game.side_to_move : "complete";
  const showInspection = inspectionToggle.checked && view.thinking?.metadata?.inspection;
  renderBoard(game, {
    interactive: game.result === "in_progress" && game.side_to_move === view.human_side && !requestPending,
    thinking: view.thinking, overlayMode: showInspection ? "visits" : "off",
  });
  inspectionElement.hidden = !showInspection;
  inspectionElement.replaceChildren();
  if (showInspection) {
    const metadata = view.thinking.metadata;
    inspectionElement.append(
      textNode("strong", "Last AI decision"),
      textNode("span", `Value: ${formatNumber(metadata.inspection.value)}`),
      textNode("span", Object.entries(metadata.inspection.statistics ?? {}).map(([key, item]) => `${title(key)}: ${formatNumber(item)}`).join(" · ")),
    );
  }
}

async function request(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    const reason = payload.reason?.replaceAll("_", " ") ?? payload.detail ?? "request failed";
    const error = new Error(reason);
    error.session = payload.session;
    throw error;
  }
  return payload;
}

async function playAgentIfNeeded() {
  if (!session || session.state.result !== "in_progress" || session.state.side_to_move === session.human_side) return;
  requestPending = true;
  agentThinking = true;
  messageElement.textContent = "";
  renderHuman(session);
  try {
    const next = await request("/api/session/agent-moves", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ revision: session.revision }),
    });
    const move = next.thinking.move.coordinate;
    const metadata = summarizeMetadata(next.thinking.metadata);
    messageElement.textContent = `${title(next.agent)} played column ${move.x + 1}, row ${move.y + 1}${metadata ? ` (${metadata})` : ""}.`;
    session = next;
  } catch (error) {
    if (error.session) session = error.session;
    messageElement.textContent = `Agent could not move: ${error.message}.`;
  } finally {
    requestPending = false;
    agentThinking = false;
    renderHuman(session);
  }
}

async function placePeg(x, y) {
  if (requestPending || !session || session.state.result !== "in_progress" || session.state.side_to_move !== session.human_side) return;
  requestPending = true;
  messageElement.textContent = "";
  renderHuman(session);
  try {
    session = await request("/api/session/human-moves", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ x, y, revision: session.revision }),
    });
  } catch (error) {
    if (error.session) session = error.session;
    messageElement.textContent = `Cannot place there: ${error.message}.`;
  } finally {
    requestPending = false;
    renderHuman(session);
  }
  await playAgentIfNeeded();
}

function stopPlayback() {
  if (playbackTimer !== null) window.clearInterval(playbackTimer);
  playbackTimer = null;
  playButton.textContent = "Play";
}

function renderCandidates(decision) {
  const candidates = [...decisionCandidates(decision)].sort((left, right) =>
    (right.visits ?? 0) - (left.visits ?? 0) || (right.prior ?? 0) - (left.prior ?? 0));
  const simulations = decision?.metadata?.simulations ?? decision?.metadata?.inspection?.statistics?.simulations ?? 0;
  candidateTable.replaceChildren(...candidates.map((candidate) => {
    const row = document.createElement("tr");
    const values = [
      `(${candidate.x + 1}, ${candidate.y + 1})`, formatPercent(candidate.prior),
      Number.isFinite(candidate.visits) ? String(candidate.visits) : "—",
      simulations && Number.isFinite(candidate.visits) ? formatPercent(candidate.visits / simulations) : "—",
      formatNumber(candidate.value ?? candidate.score),
    ];
    row.replaceChildren(...values.map((value) => textNode("td", value)));
    return row;
  }));
  candidateTableWrap.hidden = candidates.length === 0;
}

function renderReplay() {
  if (!replay) return;
  const frame = replay.frames[replayIndex];
  const game = frame.state;
  const total = replay.frames.length - 1;
  const complete = replayIndex === total;
  statusElement.textContent = complete
    ? ({ red_wins: "Red wins!", black_wins: "Black wins!", draw: "Draw" }[game.result] ?? title(game.result))
    : `Side to move: ${title(game.side_to_move)}`;
  statusElement.dataset.player = game.result === "in_progress" ? game.side_to_move : "complete";
  renderBoard(game, { thinking: frame.decision, lastMove: frame.last_move, overlayMode: overlayModeSelect.value });
  moveCounter.textContent = `move ${replayIndex} / ${total}`;
  firstButton.disabled = backButton.disabled = replayIndex === 0;
  nextButton.disabled = lastButton.disabled = replayIndex === total;
  replayControls.hidden = false;
  const result = replay.result;
  replaySummary.replaceChildren(
    textNode("strong", `Result: ${title(result.status)}`),
    textNode("span", `Winner: ${result.winner ? title(result.winner) : "None"}`),
    textNode("span", `Draw: ${result.draw ? "yes" : "no"}`),
    textNode("span", `Total moves: ${result.move_count}`),
    textNode("span", result.draw
      ? (result.filled_to_limit ? `Board filled to the 96-move limit (${result.move_count}/${result.board_capacity}).` : `Draw before board capacity (${result.move_count}/${result.board_capacity}).`)
      : `Board occupancy: ${result.move_count}/${result.board_capacity}`, result.draw ? "draw-warning" : ""),
  );
  replaySummary.hidden = false;
  const metadata = frame.decision?.metadata;
  inspectionElement.replaceChildren();
  inspectionElement.hidden = !metadata;
  if (metadata) {
    const selected = frame.decision.selected_move.coordinate;
    inspectionElement.append(
      textNode("strong", `Move ${replayIndex}: ${title(frame.decision.selected_move.player)} (${selected.x + 1}, ${selected.y + 1})`),
      textNode("span", `Guidance: ${title(metadata.guidance_mode ?? "artifact")}`),
      textNode("span", `Simulations: ${formatNumber(metadata.simulations)}`),
      textNode("span", `Maximum depth: ${formatNumber(metadata.maximum_depth)}`),
      textNode("span", `Selected Q: ${formatNumber(metadata.inspection?.value)}`),
    );
  }
  renderCandidates(frame.decision);
}

function setReplayIndex(index) {
  if (!replay) return;
  replayIndex = Math.max(0, Math.min(replay.frames.length - 1, index));
  if (replayIndex === replay.frames.length - 1) stopPlayback();
  renderReplay();
}

function updateCheckpointAvailability() {
  redCheckpointSelect.disabled = requestPending || redAgentSelect.value === "non-neural-mcts";
  blackCheckpointSelect.disabled = requestPending || blackAgentSelect.value === "non-neural-mcts";
}

function setViewerPending(pending) {
  requestPending = pending;
  for (const control of [redAgentSelect, blackAgentSelect, seedInput, generateButton, artifactSelect, loadArtifactButton]) control.disabled = pending;
  updateCheckpointAvailability();
  boardElement.setAttribute("aria-busy", pending ? "true" : "false");
}

async function generateReplay() {
  if (requestPending) return;
  stopPlayback();
  setViewerPending(true);
  messageElement.textContent = "Generating the complete game…";
  try {
    replay = await request("/api/viewer/games", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        red: { mode: redAgentSelect.value, checkpoint: redCheckpointSelect.value || null },
        black: { mode: blackAgentSelect.value, checkpoint: blackCheckpointSelect.value || null },
        seed: Number(seedInput.value),
      }),
    });
    replayIndex = replay.frames.length - 1;
    messageElement.textContent = "Game generated. Use the replay controls to inspect every move.";
    renderReplay();
  } catch (error) {
    messageElement.textContent = `Could not generate game: ${error.message}.`;
  } finally {
    setViewerPending(false);
  }
}

async function loadReplayArtifact() {
  if (requestPending || !artifactSelect.value) return;
  stopPlayback();
  setViewerPending(true);
  messageElement.textContent = "Loading saved game…";
  try {
    replay = await request("/api/viewer/artifacts", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ artifact: artifactSelect.value }),
    });
    replayIndex = 0;
    messageElement.textContent = `Loaded ${replay.source.path}.`;
    renderReplay();
  } catch (error) {
    messageElement.textContent = `Could not load artifact: ${error.message}.`;
  } finally {
    setViewerPending(false);
  }
}

async function initializeViewer() {
  $("#eyebrow").textContent = "AI vs AI · Mini 10×10";
  $("#mode-link").textContent = "Human vs AI";
  $("#mode-link").href = "/";
  resetButton.hidden = true;
  humanSetup.hidden = true;
  viewerSetup.hidden = false;
  const config = await request("/api/viewer/config");
  const agentOptions = config.agent_modes.map((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = title(mode);
    return option;
  });
  redAgentSelect.replaceChildren(...agentOptions.map((option) => option.cloneNode(true)));
  blackAgentSelect.replaceChildren(...agentOptions.map((option) => option.cloneNode(true)));
  redAgentSelect.value = blackAgentSelect.value = "learned-policy-value";
  const checkpointOptions = config.checkpoints.map((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint.id;
    option.textContent = checkpoint.label;
    return option;
  });
  redCheckpointSelect.replaceChildren(...checkpointOptions.map((option) => option.cloneNode(true)));
  blackCheckpointSelect.replaceChildren(...checkpointOptions.map((option) => option.cloneNode(true)));
  const preferred = config.checkpoints.find((item) => item.id.includes("issue-151/5k/candidate")) ?? config.checkpoints.at(-1);
  if (preferred) redCheckpointSelect.value = blackCheckpointSelect.value = preferred.id;
  else redAgentSelect.value = blackAgentSelect.value = "non-neural-mcts";
  const placeholder = textNode("option", "Select a saved game…");
  placeholder.value = "";
  artifactSelect.replaceChildren(placeholder, ...config.artifacts.map((artifact) => {
    const option = document.createElement("option");
    option.value = artifact.id;
    option.textContent = artifact.label;
    return option;
  }));
  searchSettings.textContent = `Mini 10×10 · ${config.search.simulations} simulations/move · rollout limit ${config.search.rollout_limit}`;
  setViewerPending(false);
  statusElement.textContent = "Configure both agents, then generate a game.";
  statusElement.dataset.player = "complete";
  renderBoard({ board: config.board, pegs: [], links: [], side_to_move: "red", result: "in_progress" });
}

resetButton.addEventListener("click", async () => {
  if (requestPending || !session) return;
  const reset = { human_side: sideSelect.value, agent: agentSelect.value };
  if (presetSelect.value !== "custom") reset.preset = presetSelect.value;
  requestPending = true;
  messageElement.textContent = "";
  renderHuman(session);
  try {
    session = await request("/api/session/reset", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(reset),
    });
  } catch (error) {
    messageElement.textContent = `Could not start game: ${error.message}.`;
  } finally {
    requestPending = false;
    renderHuman(session);
  }
  await playAgentIfNeeded();
});
inspectionToggle.addEventListener("change", () => { if (session) renderHuman(session); });
redAgentSelect.addEventListener("change", updateCheckpointAvailability);
blackAgentSelect.addEventListener("change", updateCheckpointAvailability);
generateButton.addEventListener("click", generateReplay);
loadArtifactButton.addEventListener("click", loadReplayArtifact);
firstButton.addEventListener("click", () => { stopPlayback(); setReplayIndex(0); });
backButton.addEventListener("click", () => { stopPlayback(); setReplayIndex(replayIndex - 1); });
nextButton.addEventListener("click", () => { stopPlayback(); setReplayIndex(replayIndex + 1); });
lastButton.addEventListener("click", () => { stopPlayback(); setReplayIndex(replay.frames.length - 1); });
playButton.addEventListener("click", () => {
  if (!replay) return;
  if (playbackTimer !== null) { stopPlayback(); return; }
  if (replayIndex === replay.frames.length - 1) replayIndex = 0;
  playButton.textContent = "Pause";
  renderReplay();
  playbackTimer = window.setInterval(() => setReplayIndex(replayIndex + 1), 650);
});
overlayModeSelect.addEventListener("change", renderReplay);

try {
  if (viewerMode) await initializeViewer();
  else {
    session = await request("/api/session");
    renderHuman(session);
    await playAgentIfNeeded();
  }
} catch (error) {
  statusElement.textContent = "Game unavailable";
  messageElement.textContent = error.message;
}
