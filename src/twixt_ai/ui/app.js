const SVG_NS = "http://www.w3.org/2000/svg";
const boardElement = document.querySelector("#board");
const statusElement = document.querySelector("#status");
const messageElement = document.querySelector("#message");
const resetButton = document.querySelector("#reset");

let state = null;
let requestPending = false;

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function point(coordinate, spacing, margin) {
  return {
    x: margin + coordinate.x * spacing,
    y: margin + coordinate.y * spacing,
  };
}

function describeStatus(game) {
  const results = {
    red_wins: "Red wins!",
    black_wins: "Black wins!",
    draw: "Draw",
  };
  return results[game.result] ?? `${game.side_to_move === "red" ? "Red" : "Black"} to move`;
}

function render(game) {
  state = game;
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
      class: link.owner,
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
    }));
  }
  boardElement.append(links);

  const pegsByCoordinate = new Map(
    game.pegs.map((peg) => [`${peg.coordinate.x},${peg.coordinate.y}`, peg.owner]),
  );
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
        tabindex: owner || game.result !== "in_progress" ? "-1" : "0",
        "aria-label": owner ? `${owner} peg at column ${x + 1}, row ${y + 1}` : `Place at column ${x + 1}, row ${y + 1}`,
      });
      if (!owner && game.result === "in_progress") {
        intersection.classList.add("playable");
        intersection.addEventListener("click", () => placePeg(x, y));
        intersection.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            placePeg(x, y);
          }
        });
      }
      points.append(intersection);
    }
  }
  boardElement.append(points);
}

async function request(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    if (payload.state) render(payload.state);
    const reason = payload.reason?.replaceAll("_", " ") ?? payload.detail ?? "request failed";
    throw new Error(reason);
  }
  return payload;
}

async function placePeg(x, y) {
  if (requestPending || !state || state.result !== "in_progress") return;
  requestPending = true;
  messageElement.textContent = "";
  try {
    render(await request("/api/game/moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y }),
    }));
  } catch (error) {
    messageElement.textContent = `Cannot place there: ${error.message}.`;
  } finally {
    requestPending = false;
  }
}

resetButton.addEventListener("click", async () => {
  if (requestPending) return;
  requestPending = true;
  messageElement.textContent = "";
  try {
    render(await request("/api/game/reset", { method: "POST" }));
  } catch (error) {
    messageElement.textContent = `Could not reset: ${error.message}.`;
  } finally {
    requestPending = false;
  }
});

try {
  render(await request("/api/game"));
} catch (error) {
  statusElement.textContent = "Game unavailable";
  messageElement.textContent = error.message;
}
