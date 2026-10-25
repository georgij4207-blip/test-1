// 360° turnaround viewer.
// The angle is continuous. Between two drawn frames the image is morphed on
// the GPU: both neighbours are warped along precomputed optical-flow fields
// (tools/flows.py) toward the in-between angle and blended, so the turn reads
// as one continuous motion instead of a flip-book. Without WebGL the viewer
// falls back to a cross-fade on a 2D canvas.
(function () {
  "use strict";

  const M = window.COYOTE_MANIFEST;
  const $ = (id) => document.getElementById(id);
  const stage = $("stage"), floor = $("floor");
  let canvas = $("view");
  const ui = {
    angle: $("angle"), source: $("source"), badge: $("badge"), dial: $("dial"),
    play: $("play"), dir: $("dir"), speed: $("speed"), speedOut: $("speedOut"),
    morph: $("morph"), morphLabel: $("morphLabel"), snap: $("snap"), loading: $("loading"),
  };

  const state = {
    angle: 0,            // degrees, kept in [0, 360)
    velocity: 0,         // deg/s, inertia after a drag
    playing: false,
    direction: 1,
    speed: 45,           // deg/s for auto-rotation
    dragging: false,
    snapTarget: null,    // angle we ease to after a drag
    dirty: true,
  };

  const wrap = (a) => ((a % 360) + 360) % 360;
  // shortest signed difference b - a in degrees
  const delta = (a, b) => ((b - a + 540) % 360) - 180;
  const smoothstep = (e0, e1, x) => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Не удалось загрузить " + src));
      img.src = src;
    });
  }

  // ---------- layout ----------
  let view = { W: 1, H: 1, scale: 1, ox: 0, oy: 0 };

  function layout() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const r = stage.getBoundingClientRect();
    const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
    for (const c of [canvas, floor]) { c.width = W; c.height = H; }
    const scale = Math.min(W / M.frameWidth, (H * 0.94) / M.frameHeight);
    view = {
      W, H, scale,
      ox: W / 2 - M.axisX * scale,               // rotation axis in the middle
      oy: H * 0.97 - M.baselineY * scale,        // soles near the bottom
    };
    // constant floor shadow, so it never jitters
    const g = floor.getContext("2d");
    g.clearRect(0, 0, W, H);
    g.fillStyle = "rgba(0,0,0,0.28)";
    g.beginPath();
    g.ellipse(W / 2, view.oy + M.baselineY * scale, 52 * scale, 6 * scale, 0, 0, Math.PI * 2);
    g.fill();
    state.dirty = true;
  }

  // ---------- WebGL morph renderer ----------
  const VERT = `
    attribute vec2 aPos;            // frame pixel coords of the quad corner
    uniform vec2 uOrigin, uCanvas;
    uniform float uDraw;
    varying vec2 vPos;
    void main() {
      vPos = aPos;
      vec2 px = uOrigin + aPos * uDraw;
      vec2 clip = px / uCanvas * 2.0 - 1.0;
      gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    }`;
  const FRAG = `
    precision highp float;
    uniform sampler2D uFrames, uFlow;
    uniform vec2 uFrameSize, uSpriteSize, uFlowSize;
    uniform vec2 uCellA, uCellB, uFlowFwd, uFlowBwd;
    uniform float uT, uW, uFlowScale, uFlowOffset;
    varying vec2 vPos;
    vec2 flowAt(vec2 cell, vec2 p) {
      return (texture2D(uFlow, (cell + p) / uFlowSize).rg * 255.0 - uFlowOffset) / uFlowScale;
    }
    vec4 frameAt(vec2 cell, vec2 p) {
      if (p.x < 0.5 || p.y < 0.5 || p.x > uFrameSize.x - 0.5 || p.y > uFrameSize.y - 0.5) return vec4(0.0);
      return texture2D(uFrames, (cell + p) / uSpriteSize);
    }
    void main() {
      vec2 p = vPos;
      // a pixel of the in-between came from A at p + t*f(B->A) and from B at p + (1-t)*f(A->B)
      vec4 a = frameAt(uCellA, p + uT * flowAt(uFlowBwd, p));
      vec4 b = frameAt(uCellB, p + (1.0 - uT) * flowAt(uFlowFwd, p));
      gl_FragColor = mix(a, b, uW);     // premultiplied alpha
    }`;

  function createGL(frames, flow) {
    const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) return null;
    const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (Math.max(frames.width, frames.height, flow.width, flow.height) > max) return null;

    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const U = {};
    for (const n of ["uOrigin", "uCanvas", "uDraw", "uFrames", "uFlow", "uFrameSize", "uSpriteSize",
                     "uFlowSize", "uCellA", "uCellB", "uFlowFwd", "uFlowBwd", "uT", "uW",
                     "uFlowScale", "uFlowOffset"]) U[n] = gl.getUniformLocation(prog, n);

    const fw = M.frameWidth, fh = M.frameHeight;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, fw, 0, 0, fh, fw, fh]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const texture = (unit, img, premultiply) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premultiply);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);   // flow values must stay exact
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };
    texture(0, frames, true);
    texture(1, flow, false);
    gl.uniform1i(U.uFrames, 0);
    gl.uniform1i(U.uFlow, 1);
    gl.uniform2f(U.uFrameSize, fw, fh);
    gl.uniform2f(U.uSpriteSize, frames.width, frames.height);
    gl.uniform2f(U.uFlowSize, flow.width, flow.height);
    gl.uniform1f(U.uFlowScale, M.flow.scale);
    gl.uniform1f(U.uFlowOffset, M.flow.offset);

    const cell = (k) => [(k % M.columns) * fw, Math.floor(k / M.columns) * fh];
    return function draw(i, j, t, w) {
      gl.viewport(0, 0, view.W, view.H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(U.uOrigin, view.ox, view.oy);
      gl.uniform2f(U.uCanvas, view.W, view.H);
      gl.uniform1f(U.uDraw, view.scale);
      const [ax, ay] = cell(i), [bx, by] = cell(j);
      gl.uniform2f(U.uCellA, ax, ay);
      gl.uniform2f(U.uCellB, bx, by);
      gl.uniform2f(U.uFlowFwd, ax, ay);
      gl.uniform2f(U.uFlowBwd, ax, ay + M.flow.backwardOffsetY);
      gl.uniform1f(U.uT, t);
      gl.uniform1f(U.uW, w);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };
  }

  // ---------- 2D fallback: cross-fade ----------
  function create2D(frames) {
    let ctx = canvas.getContext("2d");
    if (!ctx) {                        // canvas already claimed by a failed WebGL setup
      const fresh = canvas.cloneNode(false);
      canvas.replaceWith(fresh);
      canvas = fresh;
      ctx = canvas.getContext("2d");
      layout();
    }
    const fw = M.frameWidth, fh = M.frameHeight;
    const blit = (k, alpha) => {
      ctx.globalAlpha = alpha;
      ctx.drawImage(frames, (k % M.columns) * fw, Math.floor(k / M.columns) * fh, fw, fh,
                    view.ox, view.oy, fw * view.scale, fh * view.scale);
      ctx.globalAlpha = 1;
    };
    return function draw(i, j, t, w) {
      ctx.clearRect(0, 0, view.W, view.H);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      blit(i, 1);
      if (w > 0.01) blit(j, w);
    };
  }

  let draw = null, gpu = false;

  function render() {
    const step = M.stepDegrees, n = M.count;
    const pos = state.angle / step;
    let i = Math.floor(pos) % n, t = pos - Math.floor(pos);
    if (!ui.morph.checked) { i = Math.round(pos) % n; t = 0; }
    const j = (i + 1) % n;
    const w = gpu ? smoothstep(M.flow.blend[0], M.flow.blend[1], t) : t;
    draw(i, j, t, w);

    const near = t < 0.5 ? i : j;
    const info = M.frames[near];
    ui.angle.textContent = Math.round(state.angle) % 360 + "°";
    const between = t > 0.02 && t < 0.98;
    ui.source.textContent = between
      ? `переход ${M.frames[i].angle}° → ${M.frames[j].angle}°${gpu ? " · морфинг" : " · наплыв"}`
      : `кадр ${info.angle}° · ${info.source}`;
    ui.badge.hidden = between || !(info.interpolated || info.redrawn);
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
        state.velocity *= Math.exp(-3 * dt);
        state.dirty = true;
        if (Math.abs(state.velocity) <= 1) beginSnap();
      } else if (state.snapTarget !== null) {
        const d = delta(state.angle, state.snapTarget);
        if (Math.abs(d) < 0.05) {
          state.angle = wrap(state.snapTarget);
          state.snapTarget = null;
        } else {
          state.angle = wrap(state.angle + d * Math.min(1, dt * 8));
        }
        state.dirty = true;
      }
    }
    if (state.dirty && draw) { render(); state.dirty = false; }
    requestAnimationFrame(tick);
  }

  function beginSnap() {
    state.velocity = 0;
    if (!ui.snap.checked) { state.snapTarget = null; return; }
    const step = M.stepDegrees;
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
    if (state.playing) setPlaying(false);
    state.dragging = true;
    state.velocity = 0;
    state.snapTarget = null;
    lastX = e.clientX; lastT = e.timeStamp; samples = [];
    stage.classList.add("dragging");
    stage.focus({ preventScroll: true });
  });
  stage.addEventListener("pointermove", (e) => {
    if (!state.dragging) return;
    const d = (e.clientX - lastX) * degPerPx();
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

  // keyboard: arrows glide to the neighbouring drawn frame
  stage.addEventListener("keydown", (e) => {
    const step = M.stepDegrees;
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
  ui.morph.addEventListener("change", () => (state.dirty = true));
  ui.snap.addEventListener("change", () => { if (!state.playing) beginSnap(); });

  // ---------- start ----------
  if (!M) {
    ui.loading.textContent = "Нет assets/manifest.js — запустите tools/build_frames.py и tools/flows.py";
    return;
  }
  window.addEventListener("resize", layout);
  if (window.ResizeObserver) new ResizeObserver(layout).observe(stage);
  layout();

  // From file:// the browser won't let WebGL read image files; use the inlined copies.
  const inlined = () => new Promise((resolve) => {
    if (location.protocol !== "file:" || window.COYOTE_SPRITES) return resolve();
    const s = document.createElement("script");
    s.src = "assets/sprites.js";
    s.onload = s.onerror = () => resolve();
    document.head.appendChild(s);
  });
  const src = (path) => (window.COYOTE_SPRITES && window.COYOTE_SPRITES[path]) || path;

  inlined()
    .then(() => Promise.all([loadImage(src(M.sprite)),
                             M.flow ? loadImage(src(M.flow.sprite)) : Promise.resolve(null)]))
    .then(([frames, flow]) => {
      try { draw = flow && createGL(frames, flow); } catch (err) { console.warn(err); draw = null; }
      gpu = !!draw;
      if (!draw) {
        draw = create2D(frames);
        ui.morphLabel.textContent = "Плавный наплыв (без WebGL)";
      }
      ui.loading.hidden = true;
      state.dirty = true;
      setPlaying(true);
    })
    .catch((err) => { ui.loading.textContent = err.message; });
  requestAnimationFrame(tick);

  // debug / automation hook
  window.coyoteViewer = {
    setAngle(a) { setPlaying(false); state.snapTarget = null; state.velocity = 0; state.angle = wrap(a); state.dirty = true; },
    play: () => setPlaying(true),
    stop: () => setPlaying(false),
    get angle() { return state.angle; },
    get gpu() { return gpu; },
  };
})();
