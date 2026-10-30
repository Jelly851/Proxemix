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

// Promptable-search hits, drawn as boxes over the same canvas as the tracks.
// Keyed by canonical frame so the overlay loop can look them up with no search.
let SEARCH_BOXES = new Map(); // frame -> [{x1,y1,x2,y2,conf}, ...]
let SEARCH_CONCEPT = "";

function seekToFrame(frame) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  video.currentTime = (frame + 0.5) / fps;
  if (video.paused) drawOverlayForFrame(frame);
}

function nearestFrame(mediaTimeSeconds) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  const maxFrame = FRAMES_BY_INDEX.length - 1;
  // floor, not round. seekToFrame puts the playhead at (f + 0.5) / fps — the
  // middle of frame f — so rounding came back with f + 1 and the overlay drew
  // the frame after the one that was clicked. Flooring inverts the seek exactly,
  // and is also the right reading mid-playback: time t belongs to frame ⌊t·fps⌋.
  let f = Math.floor(mediaTimeSeconds * fps);
  if (f < 0) f = 0;
  if (f > maxFrame) f = maxFrame;
  return f;
}

// The canvas is the video's own pixel size (2560x1440 on the plaza bundle) but
// is displayed at 640px wide, so the browser scales it down by ~4x. A 3px line
// drawn in canvas pixels arrives on screen as less than one — which is why the
// overlay read as invisible specks. Everything drawn is sized in screen pixels
// and multiplied up by that ratio.
function overlayScale() {
  const shown = canvas.clientWidth || canvas.width;
  return shown ? canvas.width / shown : 1;
}

function drawSearchBoxesForFrame(frame, k) {
  const boxes = SEARCH_BOXES.get(frame);
  if (!boxes) return;
  const pad = 4 * k;
  const lineH = 15 * k;
  ctx.save();
  ctx.strokeStyle = "#ffd93d";
  ctx.lineWidth = Math.max(1, 2 * k);
  ctx.font = `600 ${Math.round(12 * k)}px system-ui`;
  for (const b of boxes) {
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const label = b.conf == null ? SEARCH_CONCEPT : `${SEARCH_CONCEPT} ${b.conf.toFixed(2)}`;
    const w = ctx.measureText(label).width + pad * 2;
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(b.x1, Math.max(0, b.y1 - lineH), w, lineH);
    ctx.fillStyle = "#ffd93d";
    ctx.fillText(label, b.x1 + pad, Math.max(lineH - pad, b.y1 - pad));
  }
  ctx.restore();
}

function drawOverlayForFrame(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const k = overlayScale();
  const points = FRAMES_BY_INDEX[frame];
  if (!points) {
    drawSearchBoxesForFrame(frame, k);
    return;
  }
  // With a search showing, 102 labelled track rings bury the handful of boxes
  // the question was actually about. The tracks stay — they are the context —
  // but they recede.
  const searching = SEARCH_BOXES.size > 0;
  ctx.save();
  if (searching) ctx.globalAlpha = 0.3;
  for (const p of points) {
    const color = trackColor(p.track_id);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7 * k, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, 1.5 * k);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5 * k, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    if (!searching) {
      ctx.font = `${Math.round(11 * k)}px system-ui`;
      ctx.fillStyle = color;
      ctx.fillText(p.track_id, p.x + 9 * k, p.y - 9 * k);
    }
  }
  ctx.restore();
  // Search hits paint last so they sit on top of the tracks, never under them.
  drawSearchBoxesForFrame(frame, k);
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

  // The player box was a fixed 640x480 while the footage is 16:9, so the video
  // letterboxed inside it — but the overlay canvas filled the whole box. Every
  // dot and box was therefore stretched down the frame against the image it was
  // meant to sit on, and hits near the bottom landed on the black bar. The box
  // takes the clip's own aspect ratio instead.
  const wrap = document.getElementById("player-wrap");
  const shownW = wrap.clientWidth || 640;
  const shownH = Math.round(shownW * BUNDLE.height / BUNDLE.width);
  for (const el of [wrap, video, canvas]) el.style.height = `${shownH}px`;

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

async function refreshSearchHistory() {
  const res = await fetch("/api/searches");
  const data = await res.json();
  const el = document.getElementById("search-history");
  if (!data.searches || !data.searches.length) {
    el.textContent = "";
    return;
  }
  const names = data.searches.map((s) => `${s.concept} (${s.n_detections})`).join(", ");
  el.textContent = `already run, free to repeat: ${names} — spent $${data.spent_usd.toFixed(3)} of $${data.session_cap_usd.toFixed(2)}`;
}

function renderSearchResult(data) {
  const status = document.getElementById("search-status");
  const list = document.getElementById("search-list");
  const coverage = document.getElementById("search-coverage");
  list.innerHTML = "";
  coverage.textContent = "";

  if (data.error) {
    status.textContent = data.error;
    return;
  }

  SEARCH_CONCEPT = data.concept;
  SEARCH_BOXES = new Map();
  for (const d of data.detections || []) {
    if (!SEARCH_BOXES.has(d.frame)) SEARCH_BOXES.set(d.frame, []);
    SEARCH_BOXES.get(d.frame).push({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, conf: d.conf });
  }

  const paid = data.cached ? "cached, nothing sent" : `$${data.cost_usd.toFixed(4)}`;
  status.textContent = data.found
    ? `${data.n_detections} hit(s) for "${data.concept}" across ${data.frames_with_hits} of ${data.n_frames_sampled} sampled frames — peak ${data.peak_count} in one frame (${paid})`
    : `No "${data.concept}" found on the sampled frames (${paid})`;

  for (const s of data.spans) {
    const li = document.createElement("li");
    li.textContent = `peak ${s.peak_count} at frame ${s.peak_frame.toLocaleString()}`;
    const cite = document.createElement("span");
    cite.className = "cite";
    cite.textContent = s.citation;
    li.appendChild(cite);
    li.addEventListener("click", () => seekToFrame(s.peak_frame));
    list.appendChild(li);
  }
  coverage.textContent = data.coverage || "";
  drawOverlayForFrame(nearestFrame(video.currentTime));
  refreshSearchHistory();
}

function setupSearch() {
  const form = document.getElementById("search-form");
  const input = document.getElementById("search-input");
  const status = document.getElementById("search-status");
  const button = form.querySelector("button");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const concept = input.value.trim();
    if (!concept) return;
    button.disabled = true;
    status.textContent = `Looking for "${concept}" across the clip…`;
    document.getElementById("search-list").innerHTML = "";
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concept }),
      });
      renderSearchResult(await res.json());
    } catch (err) {
      status.textContent = `The search failed to run: ${err}`;
    } finally {
      button.disabled = false;
    }
  });

  refreshSearchHistory();
}

async function main() {
  setupTabs();
  setupChat();
  setupSearch();
  await loadBundle();
  startOverlayLoop();
  await Promise.all([loadTracks(), loadEvents(), loadReport()]);
}

main();
