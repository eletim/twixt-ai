const SVG_NS = "http://www.w3.org/2000/svg";
const boardElement = document.querySelector("#board");
const statusElement = document.querySelector("#status");
const messageElement = document.querySelector("#message");
const resetButton = document.querySelector("#reset");
const sideSelect = document.querySelector("#human-side");
const agentSelect = document.querySelector("#agent");
const inspectionToggle = document.querySelector("#inspection-toggle");
const inspectionElement = document.querySelector("#inspection");

let session = null;
let requestPending = false;
let agentThinking = false;

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function point(coordinate, spacing, margin) {
  return { x: margin + coordinate.x * spacing, y: margin + coordinate.y * spacing };
}

function title(value) {
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function candidateLabel(candidate) {
  if (Number.isFinite(candidate.probability)) {
    return `${Math.round(candidate.probability * 100)}%`;
  }
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

function describeStatus(game) {
  const results = { red_wins: "Red wins!", black_wins: "Black wins!", draw: "Draw" };
  if (results[game.result]) return results[game.result];
  if (agentThinking) return `${title(session.agent)} is thinking…`;
  return game.side_to_move === session.human_side
    ? `Your turn (${title(session.human_side)})`
    : `${title(session.agent)} to move`;
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
  sideSelect.value = view.human_side;
  agentSelect.value = view.agent;
  resetButton.disabled = requestPending;
  sideSelect.disabled = requestPending;
  agentSelect.disabled = requestPending;
  inspectionToggle.disabled = requestPending;
  boardElement.setAttribute("aria-busy", agentThinking ? "true" : "false");
}

function render(view) {
  session = view;
  const game = view.state;
  populateSetup(view);
  boardElement.replaceChildren();
  statusElement.textContent = describeStatus(game);
  statusElement.dataset.player = game.result === "in_progress" ? game.side_to_move : "complete";

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
    links.append(svgElement("line", {
      class: link.owner, x1: start.x, y1: start.y, x2: end.x, y2: end.y,
    }));
  }
  boardElement.append(links);

  const inspection = view.thinking?.metadata?.inspection;
  const showInspection = inspectionToggle.checked && inspection;
  let candidateOverlays = null;
  if (showInspection) {
    const candidates = svgElement("g", { class: "candidate-overlays", "aria-hidden": "true" });
    const scores = inspection.candidates
      .map((candidate) => candidate.score)
      .filter(Number.isFinite);
    const minimum = scores.length ? Math.min(...scores) : 0;
    const maximum = scores.length ? Math.max(...scores) : 0;
    for (const candidate of inspection.candidates) {
      const position = point(candidate, spacing, margin);
      const strength = Number.isFinite(candidate.probability)
        ? candidate.probability
        : (maximum === minimum ? 1 : (candidate.score - minimum) / (maximum - minimum));
      const marker = svgElement("g", { class: "candidate-overlay" });
      marker.append(svgElement("circle", {
        cx: position.x,
        cy: position.y,
        r: 10,
        style: `--strength: ${Math.max(0.15, strength)}`,
      }));
      const label = svgElement("text", { x: position.x, y: position.y + 3 });
      label.textContent = candidateLabel(candidate);
      marker.append(label);
      candidates.append(marker);
    }
    candidateOverlays = candidates;
  }

  const pegsByCoordinate = new Map(
    game.pegs.map((peg) => [`${peg.coordinate.x},${peg.coordinate.y}`, peg.owner]),
  );
  const humanCanPlay = game.result === "in_progress"
    && game.side_to_move === view.human_side && !requestPending;
  const points = svgElement("g", { class: "points" });
  for (let y = 0; y < game.board.height; y += 1) {
    for (let x = 0; x < game.board.width; x += 1) {
      const position = point({ x, y }, spacing, margin);
      const owner = pegsByCoordinate.get(`${x},${y}`);
      const intersection = svgElement("circle", {
        class: owner ? `intersection peg ${owner}` : "intersection",
        cx: position.x,
        cy: position.y,
        r: owner ? 7 : 3,
        role: "gridcell",
        tabindex: !owner && humanCanPlay ? "0" : "-1",
        "aria-label": owner
          ? `${owner} peg at column ${x + 1}, row ${y + 1}`
          : `Place at column ${x + 1}, row ${y + 1}`,
        "aria-disabled": !owner && !humanCanPlay ? "true" : "false",
      });
      if (!owner && humanCanPlay) {
        intersection.classList.add("playable");
        intersection.addEventListener("click", () => placePeg(x, y));
        intersection.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            placePeg(x, y);
          }
        });
      } else if (!owner) {
        intersection.classList.add("blocked");
      }
      points.append(intersection);
    }
  }
  boardElement.append(points);
  if (candidateOverlays) boardElement.append(candidateOverlays);

  if (showInspection && view.thinking.move) {
    const position = point(view.thinking.move.coordinate, spacing, margin);
    boardElement.append(svgElement("circle", {
      class: "selected-move-overlay",
      cx: position.x,
      cy: position.y,
      r: 12,
      "aria-hidden": "true",
    }));
  }

  inspectionElement.hidden = !showInspection;
  inspectionElement.replaceChildren();
  if (showInspection) {
    const heading = document.createElement("strong");
    heading.textContent = "Last AI decision";
    const value = document.createElement("span");
    value.textContent = `Value: ${formatNumber(inspection.value)}`;
    const statistics = document.createElement("span");
    statistics.textContent = Object.entries(inspection.statistics ?? {})
      .map(([key, item]) => `${title(key)}: ${formatNumber(item)}`)
      .join(" · ");
    inspectionElement.append(heading, value, statistics);
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
  if (!session || session.state.result !== "in_progress"
      || session.state.side_to_move === session.human_side) return;
  requestPending = true;
  agentThinking = true;
  messageElement.textContent = "";
  render(session);
  try {
    const next = await request("/api/session/agent-moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: session.revision }),
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
    render(session);
  }
}

async function placePeg(x, y) {
  if (requestPending || !session || session.state.result !== "in_progress"
      || session.state.side_to_move !== session.human_side) return;
  requestPending = true;
  messageElement.textContent = "";
  render(session);
  try {
    session = await request("/api/session/human-moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, revision: session.revision }),
    });
  } catch (error) {
    if (error.session) session = error.session;
    messageElement.textContent = `Cannot place there: ${error.message}.`;
  } finally {
    requestPending = false;
    render(session);
  }
  await playAgentIfNeeded();
}

resetButton.addEventListener("click", async () => {
  if (requestPending || !session) return;
  const humanSide = sideSelect.value;
  const agent = agentSelect.value;
  requestPending = true;
  messageElement.textContent = "";
  render(session);
  try {
    session = await request("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ human_side: humanSide, agent }),
    });
  } catch (error) {
    messageElement.textContent = `Could not start game: ${error.message}.`;
  } finally {
    requestPending = false;
    render(session);
  }
  await playAgentIfNeeded();
});

inspectionToggle.addEventListener("change", () => {
  if (session) render(session);
});

try {
  session = await request("/api/session");
  render(session);
  await playAgentIfNeeded();
} catch (error) {
  statusElement.textContent = "Game unavailable";
  messageElement.textContent = error.message;
}
