// Proxemix — project manager plus a five-stage workspace.
//
// Two screens. The project list, and a workspace whose bottom bar moves between
// Media, Calibrate, Analyze, Review and Report — the pipeline's own steps, in
// order, with the ones you haven't earned greyed out.
//
// Legacy mode matters: launched with `--bundle DIR`, the server reports
// has_launch_bundle and this file opens straight into Review of that bundle, so
// the existing browser proofs keep passing with no project at all.

const $ = (id) => document.getElementById(id);

let PROJECT = null;        // the open project record, or null in legacy mode
let BUNDLE = null;         // bundle_meta for whatever Review is showing
let FRAMES_BY_INDEX = [];  // FRAMES_BY_INDEX[frame] = [{track_id, x, y}, ...]
let EVENTS = [];
let SEARCH_BOXES = new Map();
let SEARCH_CONCEPT = "";
let CALIB_POINTS = [];
let POLL_TIMER = null;
let REVIEW_LOADED_FOR = null;

const CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"];

function q(extra) {
  // Every bundle-reading call carries its project, or none at all in legacy mode.
  return PROJECT ? `${extra ? extra + "&" : "?"}project=${encodeURIComponent(PROJECT.slug)}`
                 : (extra ? "?" + extra : "");
}

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return { error: `${res.status}: ${text.slice(0, 200)}` };
  }
}

// ===================== PROJECT MANAGER =====================

async function loadProjects() {
  const data = await getJSON("/api/projects");
  const grid = $("project-grid");
  grid.innerHTML = "";
  const rows = data.projects || [];
  $("projects-empty").classList.toggle("hidden", rows.length > 0 || !!data.demo);

  if (data.demo) grid.appendChild(demoCard(data.demo));

  for (const p of rows) {
    const card = document.createElement("div");
    card.className = "project-card";

    const thumb = document.createElement("div");
    thumb.className = "card-thumb";
    if (p.video) {
      const img = document.createElement("img");
      img.src = `/api/projects/${p.slug}/poster`;
      img.alt = "";
      thumb.appendChild(img);
    } else {
      thumb.textContent = "no footage yet";
    }

    const name = document.createElement("div");
    name.className = "card-name";
    name.textContent = p.name;

    const state = document.createElement("div");
    state.className = "card-state";
    state.textContent = describeProject(p);

    card.append(thumb, name, state);
    card.addEventListener("click", () => openProject(p.slug));

    const del = document.createElement("button");
    del.className = "card-del";
    del.textContent = "×";
    del.title = "Delete this project";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${p.name}"? The footage and results go with it.`)) return;
      await fetch(`/api/projects/${p.slug}`, { method: "DELETE" });
      loadProjects();
    });
    card.appendChild(del);

    grid.appendChild(card);
  }
}

function demoCard(demo) {
  // The plaza demo, offered rather than forced. It is read-only: the footage is
  // already analyzed, so Media, Calibrate and Analyze stay shut on it.
  const card = document.createElement("div");
  card.className = "project-card demo";
  const thumb = document.createElement("div");
  thumb.className = "card-thumb";
  thumb.textContent = "built-in example";
  const name = document.createElement("div");
  name.className = "card-name";
  name.textContent = demo.name;
  const state = document.createElement("div");
  state.className = "card-state";
  state.textContent = "analyzed — open to explore";
  card.append(thumb, name, state);
  card.addEventListener("click", () => openDemo(demo));
  return card;
}

function openDemo(demo) {
  PROJECT = null;
  REVIEW_LOADED_FOR = null;
  $("screen-projects").classList.add("hidden");
  $("screen-workspace").classList.remove("hidden");
  $("back-to-projects").classList.remove("hidden");
  $("work-title").textContent = demo.name;
  $("work-sub").textContent = "built-in example, read-only";
  renderWorkspace();
  showStage("review");
}

function describeProject(p) {
  const a = p.analysis || {};
  if (a.state === "running") return "analyzing now";
  if (a.state === "failed") return "analysis failed";
  if (p.stages && p.stages.review === "done") {
    const n = (p.analysis && p.analysis.counts) ? Object.values(p.analysis.counts).reduce((x, y) => x + y, 0) : null;
    return n ? `analyzed — ${n.toLocaleString()} events` : "analyzed";
  }
  if (p.video) return "footage ready, not analyzed";
  return "empty";
}

function setupProjectForm() {
  $("new-project-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("new-project-name").value.trim();
    if (!name) return;
    const made = await getJSON("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    $("new-project-name").value = "";
    if (made.error) { alert(made.error); return; }
    await loadProjects();
    openProject(made.slug);
  });

  $("back-to-projects").addEventListener("click", () => {
    PROJECT = null;
    REVIEW_LOADED_FOR = null;
    stopPolling();
    $("screen-workspace").classList.add("hidden");
    $("screen-projects").classList.remove("hidden");
    loadProjects();
  });
}

// ===================== WORKSPACE SHELL =====================

async function openProject(slug) {
  const record = await getJSON(`/api/projects/${slug}`);
  if (record.error) { alert(record.error); return; }
  PROJECT = record;
  REVIEW_LOADED_FOR = null;
  $("screen-projects").classList.add("hidden");
  $("screen-workspace").classList.remove("hidden");
  $("work-title").textContent = record.name;
  renderWorkspace();
  const first = record.stages.review === "done" ? "review"
              : record.video ? "analyze" : "media";
  showStage(first);
}

async function refreshProject() {
  if (!PROJECT) return;
  const record = await getJSON(`/api/projects/${PROJECT.slug}`);
  if (!record.error) {
    PROJECT = record;
    renderWorkspace();
  }
}

function renderWorkspace() {
  const stages = PROJECT ? PROJECT.stages
                         : { media: "locked", calibrate: "locked", analyze: "locked",
                             review: "done", report: "done" };
  for (const btn of document.querySelectorAll(".stage-btn")) {
    const s = stages[btn.dataset.stage] || "locked";
    btn.classList.toggle("locked", s === "locked");
    btn.classList.toggle("done", s === "done");
    btn.disabled = s === "locked";
  }
  if (PROJECT) {
    const v = PROJECT.video;
    const cal = PROJECT.calibration;
    const bits = [];
    if (v) bits.push(`${v.width}×${v.height}, ${Math.round(v.duration_s)}s`);
    bits.push(cal ? (cal.uncalibrated ? "uncalibrated — pixels" : "calibrated") : "not calibrated");
    $("work-sub").textContent = bits.join("  ·  ");
  }
}

function showStage(stage) {
  for (const btn of document.querySelectorAll(".stage-btn")) {
    btn.classList.toggle("active", btn.dataset.stage === stage);
  }
  for (const page of document.querySelectorAll(".page")) {
    page.classList.toggle("hidden", page.id !== `page-${stage}`);
  }
  if (stage === "calibrate") loadCalibration();
  if (stage === "analyze") loadEstimate();
  if (stage === "review") loadReview();
  if (stage === "report") loadReport();
}

function setupStageBar() {
  for (const btn of document.querySelectorAll(".stage-btn")) {
    btn.addEventListener("click", () => {
      if (!btn.disabled) showStage(btn.dataset.stage);
    });
  }
}

// ===================== MEDIA =====================

function setupMedia() {
  const zone = $("drop-zone");
  const input = $("file-input");

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) uploadVideo(input.files[0]);
  });
  for (const ev of ["dragenter", "dragover"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); });
  }
  zone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) uploadVideo(f);
  });
}

function uploadVideo(file) {
  if (!PROJECT) return;
  const bar = $("upload-bar");
  const fill = $("upload-fill");
  bar.classList.remove("hidden");
  $("drop-label").textContent = `Copying ${file.name}…`;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/projects/${PROJECT.slug}/video`);
  // The clip goes up as the raw body with its name in a header: no multipart
  // envelope, so the server needs no extra dependency to unwrap it.
  xhr.setRequestHeader("X-Filename", file.name);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.addEventListener("progress", (e) => {
    if (e.lengthComputable) fill.style.width = `${(e.loaded / e.total) * 100}%`;
  });
  xhr.addEventListener("load", async () => {
    bar.classList.add("hidden");
    fill.style.width = "0%";
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (e) { /* shown below */ }
    if (data.error) {
      $("drop-label").textContent = data.error;
      return;
    }
    $("drop-label").textContent = "Drop a video here, or click to choose one";
    PROJECT = data;
    renderWorkspace();
    renderMediaPool();
  });
  xhr.addEventListener("error", () => {
    bar.classList.add("hidden");
    $("drop-label").textContent = "The upload failed. Try again.";
  });
  xhr.send(file);
}

function renderMediaPool() {
  const pool = $("media-pool");
  pool.innerHTML = "";
  if (!PROJECT || !PROJECT.video) return;
  const v = PROJECT.video;

  const card = document.createElement("div");
  card.className = "pool-item";
  const img = document.createElement("img");
  img.src = `/api/projects/${PROJECT.slug}/poster?t=${Date.now()}`;
  img.alt = "";
  const info = document.createElement("div");
  info.className = "pool-info";
  const mb = v.size_bytes ? (v.size_bytes / 1048576).toFixed(1) + " MB" : "";
  info.innerHTML = "";
  const line1 = document.createElement("div");
  line1.className = "pool-name";
  line1.textContent = v.filename;
  const line2 = document.createElement("div");
  line2.className = "meta-line";
  line2.textContent = `${v.width}×${v.height}  ·  ${Math.round(v.duration_s)}s  ·  ${mb}`;
  info.append(line1, line2);
  card.append(img, info);
  pool.appendChild(card);
}

// ===================== CALIBRATE =====================

function loadCalibration() {
  if (!PROJECT || !PROJECT.video) {
    $("calib-status").textContent = "Add a video on the Media page first.";
    return;
  }
  const canvas = $("calib-canvas");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas._img = img;
    drawCalibration();
  };
  img.src = `/api/projects/${PROJECT.slug}/poster?t=${Date.now()}`;

  if (PROJECT.calibration && PROJECT.calibration.image_points && !PROJECT.calibration.uncalibrated) {
    CALIB_POINTS = PROJECT.calibration.image_points.map((p) => ({ x: p[0], y: p[1] }));
    $("calib-status").textContent = "This project is calibrated. Click again to redo it.";
  }
  renderCalibPoints();
}

function drawCalibration() {
  const canvas = $("calib-canvas");
  const img = canvas._img;
  if (!img) return;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const k = canvas.width / (canvas.clientWidth || canvas.width);
  ctx.save();
  ctx.strokeStyle = "#ffd93d";
  ctx.fillStyle = "#ffd93d";
  ctx.lineWidth = Math.max(1, 2 * k);
  ctx.font = `600 ${Math.round(12 * k)}px system-ui`;

  if (CALIB_POINTS.length > 1) {
    ctx.beginPath();
    ctx.moveTo(CALIB_POINTS[0].x, CALIB_POINTS[0].y);
    for (const p of CALIB_POINTS.slice(1)) ctx.lineTo(p.x, p.y);
    if (CALIB_POINTS.length === 4) ctx.closePath();
    ctx.stroke();
  }
  CALIB_POINTS.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6 * k, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#16181c";
    ctx.fillText(String(i + 1), p.x - 3 * k, p.y + 4 * k);
    ctx.fillStyle = "#ffd93d";
  });
  ctx.restore();
}

function renderCalibPoints() {
  const list = $("calib-points");
  list.innerHTML = "";
  CORNER_NAMES.forEach((name, i) => {
    const li = document.createElement("li");
    const p = CALIB_POINTS[i];
    li.textContent = p ? `${name} — ${Math.round(p.x)}, ${Math.round(p.y)}`
                       : `${name} — click it`;
    li.className = p ? "set" : "unset";
    list.appendChild(li);
  });
}

function setupCalibrate() {
  const canvas = $("calib-canvas");
  canvas.addEventListener("click", (e) => {
    if (CALIB_POINTS.length >= 4) return;
    const r = canvas.getBoundingClientRect();
    CALIB_POINTS.push({
      x: (e.clientX - r.left) * (canvas.width / r.width),
      y: (e.clientY - r.top) * (canvas.height / r.height),
    });
    drawCalibration();
    renderCalibPoints();
  });

  $("calib-reset").addEventListener("click", () => {
    CALIB_POINTS = [];
    drawCalibration();
    renderCalibPoints();
    $("calib-status").textContent = "";
  });

  $("calib-save").addEventListener("click", async () => {
    if (CALIB_POINTS.length !== 4) {
      $("calib-status").textContent = "Click all four corners first.";
      return;
    }
    const w = parseFloat($("calib-width").value);
    const d = parseFloat($("calib-depth").value);
    if (!(w > 0) || !(d > 0)) {
      $("calib-status").textContent = "Give the width and depth in metres.";
      return;
    }
    const canvas = $("calib-canvas");
    const data = await getJSON(`/api/projects/${PROJECT.slug}/calibration`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_points: CALIB_POINTS.map((p) => [p.x, p.y]),
        rect_size_m: [w, d],
        frame_width: canvas.width,
        frame_height: canvas.height,
      }),
    });
    if (data.error) { $("calib-status").textContent = data.error; return; }
    PROJECT = data;
    renderWorkspace();
    $("calib-status").textContent = "Saved. Distances will come out in metres.";
  });

  $("calib-skip").addEventListener("click", () => {
    $("calib-status").textContent =
      "Skipped. Analysis will run and every distance will be labelled uncalibrated.";
    showStage("analyze");
  });
}

// ===================== ANALYZE =====================

async function loadEstimate() {
  if (!PROJECT || !PROJECT.video) {
    $("analyze-status").textContent = "Add a video on the Media page first.";
    return;
  }
  const fps = parseFloat($("analyze-fps").value) || 5;
  const est = await getJSON(`/api/projects/${PROJECT.slug}/estimate?fps=${fps}`);
  if (est.error) { $("analyze-estimate").textContent = est.error; return; }
  $("analyze-estimate").textContent =
    `${est.n_frames.toLocaleString()} frames — about $${est.usd.toFixed(2)}`;
  if (!$("analyze-steps").children.length) renderSteps(est.steps.map(
    (s) => ({ ...s, state: "waiting" })));

  const a = PROJECT.analysis;
  if (a && a.state === "running") startPolling(a.job_id);
}

function renderSteps(steps) {
  const list = $("analyze-steps");
  list.innerHTML = "";
  for (const s of steps) {
    const li = document.createElement("li");
    li.className = `step ${s.state || "waiting"}`;
    const mark = document.createElement("span");
    mark.className = "step-mark";
    mark.textContent = s.state === "done" ? "✓"
                     : s.state === "failed" ? "✗"
                     : s.state === "running" ? "…" : "";
    const label = document.createElement("span");
    label.textContent = s.label;
    li.append(mark, label);
    if (s.detail) {
      const d = document.createElement("span");
      d.className = "step-detail";
      d.textContent = s.detail;
      li.appendChild(d);
    }
    list.appendChild(li);
  }
}

function setupAnalyze() {
  $("analyze-fps").addEventListener("change", loadEstimate);

  $("analyze-start").addEventListener("click", async () => {
    if (!PROJECT) return;
    const fps = parseFloat($("analyze-fps").value) || 5;
    $("analyze-start").disabled = true;
    $("analyze-status").textContent = "Starting…";
    const data = await getJSON(`/api/projects/${PROJECT.slug}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fps }),
    });
    if (data.error) {
      $("analyze-status").textContent = data.error;
      $("analyze-start").disabled = false;
      return;
    }
    startPolling(data.job_id);
  });
}

function startPolling(jobId) {
  stopPolling();
  $("analyze-start").disabled = true;
  POLL_TIMER = setInterval(async () => {
    const s = await getJSON(`/api/jobs/${jobId}`);
    if (s.error) { stopPolling(); $("analyze-status").textContent = s.error; return; }
    renderSteps(s.steps || []);
    const spent = s.cost_usd ? ` — $${s.cost_usd.toFixed(3)} spent` : "";
    if (s.state === "running") {
      $("analyze-status").textContent = `Running${spent}`;
    } else if (s.state === "done") {
      stopPolling();
      $("analyze-status").textContent = `Finished${spent}. Open Review.`;
      await refreshProject();
      REVIEW_LOADED_FOR = null;
    } else if (s.state === "failed") {
      stopPolling();
      $("analyze-status").textContent = `Failed: ${s.error || "unknown"}`;
      await refreshProject();
    }
  }, 1000);
}

function stopPolling() {
  if (POLL_TIMER) clearInterval(POLL_TIMER);
  POLL_TIMER = null;
  const btn = $("analyze-start");
  if (btn) btn.disabled = false;
}

// ===================== REVIEW — player, overlay, ribbon =====================

const video = () => $("video");
const canvas = () => $("overlay");

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

function seekToFrame(frame) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  video().currentTime = (frame + 0.5) / fps;
  if (video().paused) drawOverlayForFrame(frame);
}

function nearestFrame(mediaTimeSeconds) {
  const fps = BUNDLE ? BUNDLE.fps : 25;
  const maxFrame = FRAMES_BY_INDEX.length - 1;
  // floor, not round. seekToFrame puts the playhead at (f + 0.5) / fps — the
  // middle of frame f — so rounding came back with f + 1 and the overlay drew
  // the frame after the one that was clicked.
  let f = Math.floor(mediaTimeSeconds * fps);
  if (f < 0) f = 0;
  if (f > maxFrame) f = maxFrame;
  return f;
}

// The canvas is the video's own pixel size but is displayed far smaller, so a
// 3px line arrives on screen as less than one. Everything is sized in screen
// pixels and multiplied up by this ratio.
function overlayScale() {
  const c = canvas();
  const shown = c.clientWidth || c.width;
  return shown ? c.width / shown : 1;
}

function drawSearchBoxesForFrame(frame, k) {
  const boxes = SEARCH_BOXES.get(frame);
  if (!boxes) return;
  const ctx = canvas().getContext("2d");
  const pad = 4 * k, lineH = 15 * k;
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
  const c = canvas();
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const k = overlayScale();
  const points = FRAMES_BY_INDEX[frame];
  if (!points) { drawSearchBoxesForFrame(frame, k); drawRibbonPlayhead(frame); return; }

  // With a search showing, 100 labelled track rings bury the handful of boxes
  // the question was about. The tracks stay as context, but they recede.
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
  drawSearchBoxesForFrame(frame, k);
  drawRibbonPlayhead(frame);
}

function startOverlayLoop() {
  const v = video();
  if ("requestVideoFrameCallback" in v) {
    const onFrame = (_now, metadata) => {
      drawOverlayForFrame(nearestFrame(metadata.mediaTime));
      v.requestVideoFrameCallback(onFrame);
    };
    v.requestVideoFrameCallback(onFrame);
  } else {
    v.addEventListener("timeupdate", () => drawOverlayForFrame(nearestFrame(v.currentTime)));
  }
  v.addEventListener("seeked", () => drawOverlayForFrame(nearestFrame(v.currentTime)));
}

// --- the event ribbon: density over time, with markers and search pins ------

const RIBBON_COLORS = { contact: "#ff6b6b", group: "#4ecdc4", zone_dwell: "#aa96da" };

function drawRibbon() {
  const c = $("ribbon");
  const wrap = $("ribbon-wrap");
  if (!BUNDLE || !FRAMES_BY_INDEX.length) return;
  c.width = wrap.clientWidth || 640;
  c.height = 54;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const maxFrame = FRAMES_BY_INDEX.length - 1;
  const xOf = (f) => (f / Math.max(1, maxFrame)) * c.width;

  // headcount per frame, as a filled area — the clip's shape at a glance
  let peak = 1;
  for (const pts of FRAMES_BY_INDEX) if (pts && pts.length > peak) peak = pts.length;
  ctx.beginPath();
  ctx.moveTo(0, c.height);
  for (let f = 0; f <= maxFrame; f++) {
    const n = FRAMES_BY_INDEX[f] ? FRAMES_BY_INDEX[f].length : 0;
    ctx.lineTo(xOf(f), c.height - (n / peak) * (c.height - 16));
  }
  ctx.lineTo(c.width, c.height);
  ctx.closePath();
  ctx.fillStyle = "rgba(78,205,196,0.22)";
  ctx.fill();

  for (const ev of EVENTS) {
    const color = RIBBON_COLORS[ev.type];
    if (!color || ev.frame_start == null) continue;
    ctx.fillStyle = color;
    ctx.fillRect(xOf(ev.frame_start), 2, 2, 8);
  }

  ctx.fillStyle = "#ffd93d";
  for (const frame of SEARCH_BOXES.keys()) {
    ctx.fillRect(xOf(frame) - 1, c.height - 7, 3, 7);
  }
  c._maxFrame = maxFrame;
}

function drawRibbonPlayhead(frame) {
  const c = $("ribbon");
  if (!c || !c._maxFrame) return;
  if (c._last === frame) return;
  c._last = frame;
  drawRibbonStatic();
  const ctx = c.getContext("2d");
  const x = (frame / Math.max(1, c._maxFrame)) * c.width;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x - 1, 0, 2, c.height);
}

let RIBBON_CACHE = null;
function drawRibbonStatic() {
  const c = $("ribbon");
  if (RIBBON_CACHE && RIBBON_CACHE.width === c.width) {
    c.getContext("2d").putImageData(RIBBON_CACHE.data, 0, 0);
    return;
  }
  drawRibbon();
  RIBBON_CACHE = { width: c.width, data: c.getContext("2d").getImageData(0, 0, c.width, c.height) };
}

function setupRibbon() {
  $("ribbon").addEventListener("click", (e) => {
    const c = $("ribbon");
    const r = c.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    seekToFrame(Math.round(frac * (c._maxFrame || 0)));
  });
}

// --- loading Review ---------------------------------------------------------

async function loadReview() {
  const key = PROJECT ? PROJECT.slug : "__launch__";
  if (REVIEW_LOADED_FOR === key) return;

  const bundle = await getJSON(`/api/bundle${q()}`);
  if (bundle.error) {
    $("bundle-meta").textContent = bundle.error;
    return;
  }
  BUNDLE = bundle;
  REVIEW_LOADED_FOR = key;
  RIBBON_CACHE = null;

  const v = video(), c = canvas();
  v.width = BUNDLE.width; v.height = BUNDLE.height;
  c.width = BUNDLE.width; c.height = BUNDLE.height;

  // The player box was a fixed 640x480 while footage is 16:9, so the video
  // letterboxed inside it and the overlay canvas did not — every dot sat lower
  // in the frame than the person it marked.
  const wrap = $("player-wrap");
  const shownW = wrap.clientWidth || 640;
  const shownH = Math.round(shownW * BUNDLE.height / BUNDLE.width);
  for (const el of [wrap, v, c]) el.style.height = `${shownH}px`;

  v.src = BUNDLE.video_url;
  $("bundle-meta").textContent =
    `${BUNDLE.width}×${BUNDLE.height} @ ${BUNDLE.fps}fps, ${BUNDLE.duration_s}s`;

  await Promise.all([loadTracks(), loadEvents()]);
  drawRibbon();
  refreshSearchHistory();
}

async function loadTracks() {
  const rows = await getJSON(`/api/tracks${q()}`);
  FRAMES_BY_INDEX = [];
  if (!Array.isArray(rows)) return;
  for (const r of rows) {
    if (!FRAMES_BY_INDEX[r.frame]) FRAMES_BY_INDEX[r.frame] = [];
    FRAMES_BY_INDEX[r.frame].push({ track_id: r.track_id, x: r.x, y: r.y });
  }
}

async function loadEvents() {
  const events = await getJSON(`/api/events${q()}`);
  EVENTS = Array.isArray(events) ? events : [];
  const list = $("event-list");
  list.innerHTML = "";
  for (const ev of EVENTS) {
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

// ===================== REPORT =====================

async function loadReport() {
  const data = await getJSON(`/api/report${q()}`);
  const body = $("report-body");
  const warn = $("report-warning");
  body.innerHTML = "";
  if (data.error) { body.textContent = data.error; return; }

  warn.classList.toggle("hidden", !data.uncalibrated);
  if (data.uncalibrated) {
    warn.textContent = "This project was never calibrated. Every distance below is "
      + "in pixels treated as metres, so the numbers are relative, not real.";
  }
  for (const stmt of data.statements) {
    const div = document.createElement("div");
    div.className = "statement";
    div.appendChild(document.createTextNode(stmt.text + " "));
    for (const c of stmt.citations) div.appendChild(citeButton(c));
    body.appendChild(div);
  }
}

function citeButton(citation) {
  const btn = document.createElement("button");
  btn.className = "cite-btn";
  btn.textContent = citation.citation || `frame ${citation.frame_start}`;
  btn.addEventListener("click", () => {
    showStage("review");
    seekToFrame(citation.frame_start);
  });
  return btn;
}

// ===================== TABS, SEARCH, CHAT =====================

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
      $(`tab-${btn.dataset.tab}`).classList.remove("hidden");
    });
  });
}

async function refreshSearchHistory() {
  const data = await getJSON(`/api/searches${q()}`);
  const el = $("search-history");
  if (!data.searches || !data.searches.length) { el.textContent = ""; return; }
  const names = data.searches.map((s) => `${s.concept} (${s.n_detections})`).join(", ");
  el.textContent = `already run, free to repeat: ${names} — spent $${data.spent_usd.toFixed(3)} of $${data.session_cap_usd.toFixed(2)}`;
}

function renderSearchResult(data) {
  const status = $("search-status"), list = $("search-list"), coverage = $("search-coverage");
  list.innerHTML = "";
  coverage.textContent = "";
  if (data.error) { status.textContent = data.error; return; }

  SEARCH_CONCEPT = data.concept;
  SEARCH_BOXES = new Map();
  for (const d of data.detections || []) {
    if (!SEARCH_BOXES.has(d.frame)) SEARCH_BOXES.set(d.frame, []);
    SEARCH_BOXES.get(d.frame).push({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, conf: d.conf });
  }

  // A result that came back without a cost is a result something went wrong in.
  // Reading .toFixed off it turned a server error into "Cannot read properties
  // of undefined", which told Jayden nothing about what actually failed.
  const paid = data.cached ? "cached, nothing sent"
             : typeof data.cost_usd === "number" ? `$${data.cost_usd.toFixed(4)}`
             : "cost unknown";
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
  RIBBON_CACHE = null;
  drawRibbon();
  drawOverlayForFrame(nearestFrame(video().currentTime));
  refreshSearchHistory();
}

function setupSearch() {
  const form = $("search-form"), input = $("search-input"), status = $("search-status");
  const button = form.querySelector("button");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const concept = input.value.trim();
    if (!concept) return;
    button.disabled = true;
    status.textContent = `Looking for "${concept}" across the clip…`;
    $("search-list").innerHTML = "";
    try {
      renderSearchResult(await getJSON("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concept, project: PROJECT ? PROJECT.slug : null }),
      }));
    } catch (err) {
      status.textContent = `The search failed to run: ${err}`;
    } finally {
      button.disabled = false;
    }
  });
}

function setupChat() {
  const form = $("chat-form"), input = $("chat-input"), log = $("chat-log");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    const qDiv = document.createElement("div");
    qDiv.className = "chat-msg q";
    qDiv.textContent = "You: " + question;
    log.appendChild(qDiv);
    input.value = "";

    const data = await getJSON("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, project: PROJECT ? PROJECT.slug : null }),
    });
    const aDiv = document.createElement("div");
    aDiv.className = "chat-msg a";
    aDiv.appendChild(document.createTextNode((data.answer || data.error || "") + " "));
    for (const c of data.citations || []) aDiv.appendChild(citeButton(c));
    log.appendChild(aDiv);
    log.scrollTop = log.scrollHeight;
  });
}

// ===================== BOOT =====================

async function main() {
  setupProjectForm();
  setupStageBar();
  setupMedia();
  setupCalibrate();
  setupAnalyze();
  setupTabs();
  setupSearch();
  setupChat();
  setupRibbon();
  startOverlayLoop();

  const data = await getJSON("/api/projects");
  if (data.auto_open) {
    // Opened on one bundle: skip the project manager entirely.
    $("screen-projects").classList.add("hidden");
    $("screen-workspace").classList.remove("hidden");
    $("work-title").textContent = "Demo bundle";
    $("back-to-projects").classList.add("hidden");
    renderWorkspace();
    showStage("review");
  } else {
    loadProjects();
  }
}

main();
