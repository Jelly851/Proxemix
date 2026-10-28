// Proxemix demo UI — video player + canvas overlay + event/report/chat
// panels, all driven by one jump-to-frame handler per the build spec's
// stage 10: "event list, chat panel, and report citations all drive one
// handler (currentTime = (f + 0.5) / fps)".

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const PALETTE = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#95e1d3", "#f38181", "#aa96da", "#6c5ce7", "#00cec9"];
const trackColor = (() => {
  const assigned = new Map();
  let i = 0;
  return (trackId) => {
    if (!assigned.has(trackId)) {
      assigned.set(trackId, PALETTE[i % PALETTE.length]);
      i += 1;
    }
    return assigned.get(trackId);
  };
})();

let BUNDLE = null;
let FRAMES_BY_INDEX = []; // FRAMES_BY_INDEX[frame] = [{track_id, x, y}, ...]

function seekToFrame(frame) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  video.currentTime = (frame + 0.5) / fps;
  if (video.paused) drawOverlayForFrame(frame);
}

function nearestFrame(mediaTimeSeconds) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  const maxFrame = FRAMES_BY_INDEX.length - 1;
  let f = Math.round(mediaTimeSeconds * fps);
  if (f < 0) f = 0;
  if (f > maxFrame) f = maxFrame;
  return f;
}

function drawOverlayForFrame(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const points = FRAMES_BY_INDEX[frame];
  if (!points) return;
  for (const p of points) {
    const color = trackColor(p.track_id);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.font = "12px system-ui";
    ctx.fillStyle = color;
    ctx.fillText(p.track_id, p.x + 10, p.y - 10);
  }
}

function startOverlayLoop() {
  if ("requestVideoFrameCallback" in video) {
    const onFrame = (_now, metadata) => {
      drawOverlayForFrame(nearestFrame(metadata.mediaTime));
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    // fallback for browsers without rVFC: redraw on timeupdate.
    video.addEventListener("timeupdate", () => drawOverlayForFrame(nearestFrame(video.currentTime)));
  }
  video.addEventListener("seeked", () => drawOverlayForFrame(nearestFrame(video.currentTime)));
}

function citeButton(citation) {
  const btn = document.createElement("button");
  btn.className = "cite-btn";
  btn.textContent = citation.citation || `frame ${citation.frame_start}`;
  btn.addEventListener("click", () => seekToFrame(citation.frame_start));
  return btn;
}

async function loadBundle() {
  const res = await fetch("/api/bundle");
  BUNDLE = await res.json();
  video.width = BUNDLE.width;
  video.height = BUNDLE.height;
  canvas.width = BUNDLE.width;
  canvas.height = BUNDLE.height;
  video.src = BUNDLE.video_url;
  document.getElementById("bundle-meta").textContent =
    `${BUNDLE.width}x${BUNDLE.height} @ ${BUNDLE.fps}fps, ${BUNDLE.duration_s}s`;
}

async function loadTracks() {
  const res = await fetch("/api/tracks");
  const rows = await res.json();
  FRAMES_BY_INDEX = [];
  for (const r of rows) {
    if (!FRAMES_BY_INDEX[r.frame]) FRAMES_BY_INDEX[r.frame] = [];
    FRAMES_BY_INDEX[r.frame].push({ track_id: r.track_id, x: r.x, y: r.y });
  }
}

async function loadEvents() {
  const res = await fetch("/api/events");
  const events = await res.json();
  const list = document.getElementById("event-list");
  list.innerHTML = "";
  for (const ev of events) {
    const li = document.createElement("li");
    li.textContent = ev.label;
    if (ev.citation) {
      const cite = document.createElement("span");
      cite.className = "cite";
      cite.textContent = ev.citation;
      li.appendChild(cite);
      li.addEventListener("click", () => seekToFrame(ev.frame_start));
    }
    list.appendChild(li);
  }
}

async function loadReport() {
  const res = await fetch("/api/report");
  const data = await res.json();
  const body = document.getElementById("report-body");
  body.innerHTML = "";
  for (const stmt of data.statements) {
    const div = document.createElement("div");
    div.className = "statement";
    div.appendChild(document.createTextNode(stmt.text + " "));
    for (const c of stmt.citations) {
      div.appendChild(citeButton(c));
    }
    body.appendChild(div);
  }
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
      document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
    });
  });
}

function setupChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const log = document.getElementById("chat-log");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    const qDiv = document.createElement("div");
    qDiv.className = "chat-msg q";
    qDiv.textContent = "You: " + question;
    log.appendChild(qDiv);
    input.value = "";

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    const aDiv = document.createElement("div");
    aDiv.className = "chat-msg a";
    aDiv.appendChild(document.createTextNode(data.answer + " "));
    for (const c of data.citations) {
      aDiv.appendChild(citeButton(c));
    }
    log.appendChild(aDiv);
    log.scrollTop = log.scrollHeight;
  });
}

async function main() {
  setupTabs();
  setupChat();
  await loadBundle();
  startOverlayLoop();
  await Promise.all([loadTracks(), loadEvents(), loadReport()]);
}

main();
