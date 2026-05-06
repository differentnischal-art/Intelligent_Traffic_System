const dashboardGrid = document.getElementById("dashboardGrid");
const laneElements = new Map();

function laneLabel(laneId) {
  return laneId.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function createSignalLight(color) {
  const light = document.createElement("span");
  light.className = `signal-light ${color}`;
  light.dataset.color = color.toUpperCase();
  return light;
}

function createLanePanel(laneId) {
  const panel = document.createElement("article");
  panel.className = "lane-panel";
  panel.dataset.laneId = laneId;

  const signalColumn = document.createElement("div");
  signalColumn.className = "signal-column";

  const housing = document.createElement("div");
  housing.className = "signal-housing";

  const red = createSignalLight("red");
  const yellow = createSignalLight("yellow");
  const green = createSignalLight("green");
  housing.append(red, yellow, green);
  signalColumn.appendChild(housing);

  const feedPanel = document.createElement("div");
  feedPanel.className = "feed-panel";

  const image = document.createElement("img");
  image.className = "camera-image";
  image.id = `img-${laneId}`;
  image.alt = `${laneLabel(laneId)} live camera feed`;

  const placeholder = document.createElement("div");
  placeholder.className = "feed-placeholder";
  placeholder.textContent = "WAITING FOR LIVE FEED";

  const title = document.createElement("div");
  title.className = "lane-title";
  title.textContent = laneLabel(laneId);

  const meta = document.createElement("div");
  meta.className = "feed-meta";

  feedPanel.append(image, placeholder, title, meta);
  panel.append(signalColumn, feedPanel);
  dashboardGrid.appendChild(panel);

  const refs = {
    panel,
    image,
    placeholder,
    meta,
    lights: { RED: red, YELLOW: yellow, GREEN: green }
  };
  laneElements.set(laneId, refs);
  return refs;
}

function renderLanes(lanes) {
  Object.keys(lanes).forEach((laneId) => {
    if (!laneElements.has(laneId)) {
      createLanePanel(laneId);
    }
  });

  laneElements.forEach((refs, laneId) => {
    const shouldShow = Object.prototype.hasOwnProperty.call(lanes, laneId);
    refs.panel.hidden = !shouldShow;
  });
}

function updateSignal(refs, signal) {
  const activeSignal = String(signal || "RED").toUpperCase();
  Object.entries(refs.lights).forEach(([color, light]) => {
    light.classList.toggle("active", color === activeSignal);
  });
}

function updateMeta(refs, lane) {
  const density = lane.density ?? lane.count ?? 0;
  const ambulance = lane.ambulance ? "Yes" : "No";
  const time = lane.timer ?? lane.green_time ?? 0;

  refs.meta.innerHTML = `
    <span>Density: ${density}</span>
    <span class="divider">|</span>
    <span>Ambulance: ${ambulance}</span>
    <span class="divider">|</span>
    <span>Time: ${time}</span>
  `;

  if (lane.active === false) {
    refs.placeholder.textContent = "FEED ENDED";
  }
}

async function updateState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) return;

    const state = await response.json();
    const lanes = state.lanes || {};
    renderLanes(lanes);

    Object.entries(lanes).forEach(([laneId, lane]) => {
      const refs = laneElements.get(laneId);
      if (!refs) return;
      updateSignal(refs, lane.signal);
      updateMeta(refs, lane);
      refs.panel.classList.toggle("ambulance-active", Boolean(lane.ambulance));
    });
  } catch (error) {
    console.error("Failed to update state", error);
  }
}

async function updateFrame(laneId, refs) {
  try {
    const response = await fetch(`/api/frame/${laneId}`, { cache: "no-store" });
    if (!response.ok) return;

    const data = await response.json();
    console.log(`/api/frame/${laneId}`, {
      active: data.active,
      frameLength: data.frame ? data.frame.length : 0
    });

    if (data.frame && data.frame.length > 100) {
      refs.image.src = `data:image/jpeg;base64,${data.frame}`;
      refs.image.classList.add("visible");
      refs.placeholder.hidden = true;
      refs.placeholder.style.display = "none";
    } else {
      refs.image.removeAttribute("src");
      refs.image.classList.remove("visible");
      refs.placeholder.textContent = data.active ? "WAITING FOR LIVE FEED" : "FEED ENDED";
      refs.placeholder.hidden = false;
      refs.placeholder.style.display = "flex";
    }
  } catch (error) {
    console.error(`Failed to update frame for ${laneId}`, error);
  }
}

function updateFrames() {
  laneElements.forEach((refs, laneId) => {
    if (!refs.panel.hidden) {
      updateFrame(laneId, refs);
    }
  });
}

updateState().then(updateFrames);
setInterval(updateState, 150);
setInterval(updateFrames, 150);
