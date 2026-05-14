const dashboardGrid = document.getElementById("dashboardGrid");
const decisionSnapshot = document.getElementById("decisionSnapshot");
const emergencyStatus = document.getElementById("emergencyStatus");
const laneElements = new Map();
const STATE_POLL_MS = 250;
const FRAME_POLL_MS = 250;
const DEBUG_FEEDS = window.ITMS_DEBUG_FEEDS === true;

function laneLabel(laneId) {
  return String(laneId).replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[char]));
}

function snapshotValue(value) {
  return value === null || value === undefined || value === "" ? "N/A" : value;
}

function updateDecisionSnapshot(snapshot) {
  if (!decisionSnapshot) return;

  if (!snapshot) {
    decisionSnapshot.hidden = true;
    decisionSnapshot.textContent = "";
    return;
  }

  const laneId = snapshot.selected_lane_id || snapshot.lane_id || "N/A";
  const density = snapshot.decision_density ?? snapshot.stable_density;
  const greenTime = snapshot.assigned_green_time_seconds ?? snapshot.assigned_green_time;

  decisionSnapshot.hidden = false;
  decisionSnapshot.innerHTML = `
    <span class="snapshot-title">Decision Snapshot</span>
    <span>Lane: ${escapeHtml(laneLabel(laneId))}</span>
    <span>Decision Density: ${escapeHtml(snapshotValue(density))}</span>
    <span>Weighted: ${escapeHtml(snapshotValue(snapshot.weighted_density))}</span>
    <span>Green: ${escapeHtml(snapshotValue(greenTime))}s</span>
    <span>Reason: ${escapeHtml(snapshotValue(snapshot.decision_reason))}</span>
    <span>Time: ${escapeHtml(snapshotValue(snapshot.decision_timestamp))}</span>
  `;
}

function updateEmergencyStatus(status) {
  if (!emergencyStatus) return;

  if (!status || !status.active) {
    emergencyStatus.hidden = true;
    emergencyStatus.textContent = "";
    return;
  }

  const lane = status.lane_id ? laneLabel(status.lane_id) : "Pending";
  const remaining = Number(status.remaining_seconds || 0);
  const suffix = remaining > 0 ? ` | ${remaining}s` : "";

  emergencyStatus.hidden = false;
  emergencyStatus.innerHTML = `
    <span class="emergency-title">Emergency</span>
    <span>${escapeHtml(status.message || "Emergency Vehicle Detected")}</span>
    <span>Lane: ${escapeHtml(lane)}${escapeHtml(suffix)}</span>
  `;
}

function shouldShowEmergencyTransitionWarning(laneId, lane, emergencyStatus) {
  const emergencyLaneId = emergencyStatus?.lane_id || lane.emergency_lane_id;
  const emergencyState = emergencyStatus?.state || lane.emergency_state;
  const signal = String(lane.signal || "RED").toUpperCase();

  return (
    signal === "GREEN" &&
    emergencyState === "EMERGENCY_CLEARANCE_MODE" &&
    Boolean(emergencyLaneId) &&
    emergencyLaneId !== laneId
  );
}

function updateEmergencyTransitionWarning(refs, laneId, lane, emergencyStatus) {
  const showWarning = shouldShowEmergencyTransitionWarning(laneId, lane, emergencyStatus);
  refs.emergencyWarning.hidden = !showWarning;
  refs.panel.classList.toggle("emergency-transition", showWarning);
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

  const emergencyWarning = document.createElement("div");
  emergencyWarning.className = "lane-emergency-warning";
  emergencyWarning.setAttribute("aria-live", "polite");
  emergencyWarning.hidden = true;
  emergencyWarning.innerHTML = `
    <span>Ambulance Priority Incoming</span>
    <span>Preparing Safe Transition...</span>
  `;

  feedPanel.append(image, placeholder, title, emergencyWarning, meta);
  panel.append(signalColumn, feedPanel);
  dashboardGrid.appendChild(panel);

  const refs = {
    panel,
    image,
    placeholder,
    emergencyWarning,
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
    const laneSnapshot = Object.values(lanes).find((lane) => lane.decision_snapshot)?.decision_snapshot;
    updateEmergencyStatus(state.emergency_status);
    updateDecisionSnapshot(state.decision_snapshot || laneSnapshot);
    renderLanes(lanes);

    Object.entries(lanes).forEach(([laneId, lane]) => {
      const refs = laneElements.get(laneId);
      if (!refs) return;
      updateSignal(refs, lane.signal);
      updateMeta(refs, lane);
      updateEmergencyTransitionWarning(refs, laneId, lane, state.emergency_status);
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
    if (DEBUG_FEEDS) {
      console.log(`/api/frame/${laneId}`, {
        active: data.active,
        frameLength: data.frame ? data.frame.length : 0
      });
    }

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
setInterval(updateState, STATE_POLL_MS);
setInterval(updateFrames, FRAME_POLL_MS);
