// 360° turnaround viewer: one sprite sheet per frame set, drawn on a canvas.
// Angle is a continuous float; the frame shown is the nearest one (or a blend
// of the two neighbours when "smoothing" is on).
(function () {
  "use strict";

  const manifests = window.COYOTE_MANIFESTS || {};
  const $ = (id) => document.getElementById(id);
  const stage = $("stage"), canvas = $("view"), ctx = canvas.getContext("2d");
  const ui = {
    angle: $("angle"), source: $("source"), badge: $("badge"), dial: $("dial"),
    play: $("play"), dir: $("dir"), speed: $("speed"), speedOut: $("speedOut"),
    set: $("set"), blend: $("blend"), snap: $("snap"), loading: $("loading"),
  };

  const state = {
    angle: 0,            // degrees, kept in [0, 360)
    velocity: 0,         // deg/s, inertia after a drag
    playing: false,
    direction: 1,
    speed: 45,           // deg/s for auto-rotation
    dragging: false,
    snapTarget: null,    // angle we ease to after a drag
    set: null,           // active manifest
    image: null,         // its sprite sheet
    dirty: true,
  };
  const images = {};     // set key -> HTMLImageElement (cached)

  const wrap = (a) => ((a % 360) + 360) % 360;
  // shortest signed difference b - a in degrees
  const delta = (a, b) => ((b - a + 540) % 360) - 180;

  function loadSet(key) {
    const m = manifests[key];
    if (!m) return;
    ui.loading.hidden = false;
    const done = (img) => {
      state.set = m;
      state.image = img;
      ui.loading.hidden = true;
      state.dirty = true;
    };
    if (images[key] && images[key].complete) return done(images[key]);
    const img = new Image();
    img.decoding = "async";
    img.onload = () => done(img);
    img.onerror = () => { ui.loading.textContent = "Не удалось загрузить " + m.sprite; };
    img.src = m.sprite;
    images[key] = img;
  }

  // ---------- drawing ----------
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const r = stage.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    state.dirty = true;
  }

  function drawFrame(m, img, index, alpha, scale, ox, oy) {
    const sx = (index % m.columns) * m.frameWidth;
    const sy = Math.floor(index / m.columns) * m.frameHeight;
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, sx, sy, m.frameWidth, m.frameHeight,
                  ox, oy, m.frameWidth * scale, m.frameHeight * scale);
    ctx.globalAlpha = 1;
  }

  function render() {
    const m = state.set, img = state.image;
    if (!m || !img) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const scale = Math.min(W / m.frameWidth, (H * 0.94) / m.frameHeight);
    // put the rotation axis at the horizontal centre, the soles near the bottom
    const ox = W / 2 - m.axisX * scale;
    const oy = H * 0.97 - m.baselineY * scale;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    // floor shadow (constant, so it never jitters between frames)
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.beginPath();
    ctx.ellipse(W / 2, oy + m.baselineY * scale, 52 * scale, 6 * scale, 0, 0, Math.PI * 2);
    ctx.fill();

    const pos = state.angle / m.stepDegrees;
    let shown;
    if (ui.blend.checked && state.snapTarget === null) {
      const i0 = Math.floor(pos) % m.count, i1 = (i0 + 1) % m.count;
      const f = pos - Math.floor(pos);
      drawFrame(m, img, i0, 1, scale, ox, oy);
      if (f > 0.02) drawFrame(m, img, i1, f, scale, ox, oy);
      shown = f < 0.5 ? i0 : i1;
    } else {
      shown = Math.round(pos) % m.count;
      drawFrame(m, img, shown, 1, scale, ox, oy);
    }

    const info = m.frames[shown];
    ui.angle.textContent = Math.round(state.angle) % 360 + "°";
    ui.source.textContent = "кадр " + info.angle + "° · " + info.source +
      (info.confidence ? " · уверенность: " + info.confidence : "");
    ui.badge.hidden = !(info.interpolated || info.redrawn);
    ui.badge.textContent = info.interpolated ? "интерполированный кадр" : "дорисованный кадр";
    ui.badge.classList.toggle("redrawn", !info.interpolated && !!info.redrawn);
    if (document.activeElement !== ui.dial) ui.dial.value = state.angle.toFixed(1);
  }

  // ---------- animation loop ----------
  let last = performance.now();
  function tick(now) {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (!state.dragging) {
      if (state.playing) {
        state.angle = wrap(state.angle + state.direction * state.speed * dt);
        state.dirty = true;
      } else if (Math.abs(state.velocity) > 1) {
        // inertia after a flick
        state.angle = wrap(state.angle + state.velocity * dt);
        state.velocity *= Math.exp(-4 * dt);
        state.dirty = true;
        if (Math.abs(state.velocity) <= 1) beginSnap();
      } else if (state.snapTarget !== null) {
        const d = delta(state.angle, state.snapTarget);
        if (Math.abs(d) < 0.05) {
          state.angle = wrap(state.snapTarget);
          state.snapTarget = null;
        } else {
          state.angle = wrap(state.angle + d * Math.min(1, dt * 12));
        }
        state.dirty = true;
      }
    }
    if (state.dirty) { render(); state.dirty = false; }
    requestAnimationFrame(tick);
  }

  function beginSnap() {
    state.velocity = 0;
    if (!ui.snap.checked || !state.set) { state.snapTarget = null; return; }
    const step = state.set.stepDegrees;
    state.snapTarget = wrap(Math.round(state.angle / step) * step);
  }

  function setPlaying(on) {
    state.playing = on;
    ui.play.textContent = on ? "❚❚ Стоп" : "▶ Авто";
    ui.play.setAttribute("aria-pressed", String(on));
    if (on) state.snapTarget = null; else beginSnap();
  }

  // ---------- input: drag with mouse / finger / pen ----------
  let lastX = 0, lastT = 0, samples = [];
  const degPerPx = () => 360 / Math.max(stage.clientWidth * 1.4, 300);

  stage.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button > 0) return;
    stage.setPointerCapture(e.pointerId);
    state.dragging = true;
    state.velocity = 0;
    state.snapTarget = null;
    if (state.playing) setPlaying(false);
    state.snapTarget = null;
    lastX = e.clientX; lastT = e.timeStamp; samples = [];
    stage.classList.add("dragging");
    stage.focus({ preventScroll: true });
  });
  stage.addEventListener("pointermove", (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - lastX;
    const d = dx * degPerPx();
    state.angle = wrap(state.angle + d);
    const dt = (e.timeStamp - lastT) / 1000;
    if (dt > 0) samples.push({ v: d / dt, t: e.timeStamp });
    samples = samples.filter((s) => e.timeStamp - s.t < 100);
    lastX = e.clientX; lastT = e.timeStamp;
    state.dirty = true;
  });
  const endDrag = (e) => {
    if (!state.dragging) return;
    state.dragging = false;
    stage.classList.remove("dragging");
    const recent = samples.filter((s) => e.timeStamp - s.t < 100);
    const v = recent.length ? recent.reduce((a, s) => a + s.v, 0) / recent.length : 0;
    state.velocity = Math.max(-720, Math.min(720, v));
    if (Math.abs(state.velocity) <= 1) beginSnap();
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);

  // keyboard
  stage.addEventListener("keydown", (e) => {
    const step = state.set ? state.set.stepDegrees : 10;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      setPlaying(false);
      const s = e.key === "ArrowRight" ? 1 : -1;
      state.snapTarget = wrap(Math.round(state.angle / step) * step + s * step);
      state.velocity = 0;
      e.preventDefault();
    } else if (e.key === " ") {
      setPlaying(!state.playing);
      e.preventDefault();
    }
  });

  // ---------- controls ----------
  ui.play.addEventListener("click", () => setPlaying(!state.playing));
  ui.dir.addEventListener("click", () => {
    state.direction *= -1;
    ui.dir.textContent = state.direction > 0 ? "⟳" : "⟲";
  });
  ui.speed.addEventListener("input", () => {
    state.speed = +ui.speed.value;
    ui.speedOut.textContent = state.speed + "°/с";
  });
  ui.dial.addEventListener("input", () => {
    if (state.playing) setPlaying(false);
    state.velocity = 0;
    state.snapTarget = null;
    state.angle = wrap(+ui.dial.value);
    state.dirty = true;
  });
  ui.dial.addEventListener("change", beginSnap);
  ui.set.addEventListener("change", () => loadSet(ui.set.value));
  ui.blend.addEventListener("change", () => (state.dirty = true));
  ui.snap.addEventListener("change", () => { if (!state.playing) beginSnap(); });

  // preload the other frame set in the background so switching is instant
  window.addEventListener("resize", resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(stage);
  resize();
  if (!manifests["10"]) {
    ui.loading.textContent = "Нет assets/manifest.js — запустите tools/build_frames.py";
    return;
  }
  if (!manifests["5"]) ui.set.querySelector('option[value="5"]').disabled = true;
  loadSet("10");
  if (manifests["5"]) { const pre = new Image(); pre.src = manifests["5"].sprite; images["5"] = pre; }
  requestAnimationFrame(tick);

  // debug / automation hook
  window.coyoteViewer = {
    setAngle(a) { setPlaying(false); state.snapTarget = null; state.velocity = 0; state.angle = wrap(a); state.dirty = true; },
    play: () => setPlaying(true),
    stop: () => setPlaying(false),
    get angle() { return state.angle; },
  };
})();
