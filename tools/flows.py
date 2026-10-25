"""Optical-flow fields for seamless rotation between the 36 drawn frames.

For every neighbouring pair (k, k+1) two dense flow fields are computed with
OpenCV's DIS optical flow: forward (k -> k+1) and backward (k+1 -> k). The
viewer warps both frames toward the requested in-between angle and blends
them on the GPU, so any angle, not just multiples of 10°, gets an image.

The fields are packed into one PNG (web/assets/coyote_flow.png):
  rows 0 .. 6*FH      forward flows,  same 6x6 layout as the frame sprite
  rows 6*FH .. 12*FH  backward flows
  R = dx, G = dy, encoded as 128 + 4*d (0.25 px steps, range ±32 px), B = 0

`render(frames, flows, angle)` is the same maths as the WebGL shader in
web/viewer.js, used here for the preview GIF and for checking results.

Run after build_frames.py:  python3 tools/flows.py
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

ASSETS = "web/assets"
SCALE, OFFSET, RANGE = 4.0, 128.0, 31.75
# blend window: the warped frames swap over the middle of the step, so most
# of the time one drawing dominates and double images stay short-lived
BLEND_LO, BLEND_HI = 0.2, 0.8


def _feature(img):
    """Grey image on white: carries both the silhouette and inner line work."""
    a = img[..., 3:].astype(np.float32) / 255.0
    g = (img[..., :3].astype(np.float32) * a + 255 * (1 - a)).mean(2)
    return g.astype(np.uint8)


def _dis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.setFinestScale(0)
    d.setPatchSize(8)
    d.setPatchStride(3)
    d.setGradientDescentIterations(25)
    d.setVariationalRefinementIterations(10)
    d.setVariationalRefinementAlpha(20)
    return d


def flow(a, b, dis=None):
    """Dense flow from frame a to frame b (where each pixel of a moves to)."""
    dis = dis or _dis()
    f = dis.calc(_feature(a), _feature(b), None)
    return np.clip(f, -RANGE, RANGE)


def encode(f):
    q = np.clip(np.round(OFFSET + f * SCALE), 0, 255).astype(np.uint8)
    return np.dstack([q, np.zeros(q.shape[:2], np.uint8)])


def decode(rgb):
    return (rgb[..., :2].astype(np.float32) - OFFSET) / SCALE


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _sample(img_pm, f, t):
    h, w = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    t = np.float32(t)
    return cv2.remap(img_pm, (gx + f[..., 0] * t).astype(np.float32),
                     (gy + f[..., 1] * t).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def premul(img):
    a = img[..., 3:].astype(np.float32) / 255.0
    return np.dstack([img[..., :3].astype(np.float32) * a, img[..., 3:].astype(np.float32)])


def unpremul(pm):
    a = pm[..., 3:]
    rgb = np.where(a > 0, pm[..., :3] * 255.0 / np.maximum(a, 1e-3), 0)
    return np.dstack([rgb, a]).clip(0, 255).astype(np.uint8)


def morph(a, b, f_ab, f_ba, t):
    """In-between of a and b at fraction t, given both flow directions."""
    # a pixel at p in the in-between came from a at p + t*f_ba and from b at p + (1-t)*f_ab
    wa = _sample(premul(a), f_ba, t)
    wb = _sample(premul(b), f_ab, 1 - t)
    w = smoothstep(BLEND_LO, BLEND_HI, t)
    return unpremul(wa * (1 - w) + wb * w)


def render(frames, fwd, bwd, angle, step=10):
    n = len(frames)
    pos = (angle % 360) / step
    i = int(np.floor(pos)) % n
    t = pos - np.floor(pos)
    j = (i + 1) % n
    if t < 1e-4:
        return frames[i]
    return morph(frames[i], frames[j], fwd[i], bwd[i], t)


def flow_error(a, b, f_ab):
    """Mean silhouette mismatch (0..255) after carrying b back onto a with f_ab."""
    back = _sample(premul(b), f_ab, 1.0)
    return float(np.abs(back[..., 3] - a[..., 3].astype(np.float32)).mean())


def mirror_flow(f):
    """Flow field of the horizontally mirrored image pair."""
    m = f[:, ::-1].copy()
    m[..., 0] *= -1
    return m


def load_frames(man):
    return [np.asarray(Image.open(f"{ASSETS}/frames/coyote_{f['angle']:03d}.png").convert("RGBA"))
            for f in man["frames"]]


def load_flows(man):
    fl = man["flow"]
    img = np.asarray(Image.open(fl["sprite"].replace("assets/", ASSETS + "/", 1)).convert("RGB"))
    fw, fh, cols, n = man["frameWidth"], man["frameHeight"], man["columns"], man["count"]
    half = fl["backwardOffsetY"]
    fwd, bwd = [], []
    for k in range(n):
        x, y = (k % cols) * fw, (k // cols) * fh
        fwd.append(decode(img[y:y + fh, x:x + fw]))
        bwd.append(decode(img[half + y:half + y + fh, x:x + fw]))
    return fwd, bwd


def main():
    with open(f"{ASSETS}/manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    fw, fh_, cols, n = man["frameWidth"], man["frameHeight"], man["columns"], man["count"]
    rows = (n + cols - 1) // cols
    frames = load_frames(man)
    dis = _dis()
    fwd, bwd, err = [], [], []
    for k in range(n):
        a, b = frames[k], frames[(k + 1) % n]
        fwd.append(flow(a, b, dis))
        bwd.append(flow(b, a, dis))
        err.append(flow_error(a, b, fwd[k]) + flow_error(b, a, bwd[k]))
    # The second half of the turn is the mirror image of the first: pair k
    # (k -> k+1) mirrors pair n-1-k traversed backwards. Use whichever of the
    # two computed fields fits better for both. Pairs touching 0° or 180° are
    # left alone: those two drawings are not exact mirrors of themselves.
    half = n // 2
    for k in range(n):
        m = n - 1 - k
        if {k, (k + 1) % n, m, (m + 1) % n} & {0, half} or err[m] <= err[k]:
            continue
        fwd[m], bwd[m] = mirror_flow(bwd[k]), mirror_flow(fwd[k])
        err[m] = err[k]
    sheet = np.zeros((2 * rows * fh_, cols * fw, 3), np.uint8)
    for k in range(n):
        a, b = frames[k], frames[(k + 1) % n]
        x, y = (k % cols) * fw, (k // cols) * fh_
        sheet[y:y + fh_, x:x + fw] = encode(fwd[k])
        sheet[rows * fh_ + y:rows * fh_ + y + fh_, x:x + fw] = encode(bwd[k])
        e = flow_error(a, b, decode(encode(fwd[k])))
        man["frames"][k]["to_next"]["flow_error"] = round(e, 2)
        print(f"{man['frames'][k]['angle']:3d}° -> {(man['frames'][k]['angle'] + 10) % 360:3d}°  "
              f"max |flow| {np.abs(fwd[k]).max():5.1f}px  silhouette error {e:4.1f}")
    Image.fromarray(sheet, "RGB").save(f"{ASSETS}/coyote_flow.png", optimize=True)
    man["flow"] = {"sprite": "assets/coyote_flow.png", "scale": SCALE, "offset": OFFSET,
                   "backwardOffsetY": rows * fh_, "blend": [BLEND_LO, BLEND_HI]}
    with open(f"{ASSETS}/manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    from build_frames import write_js
    write_js()
    write_sprites_js(man)


def write_sprites_js(man):
    """Sprites as data URIs for index.html opened via file://, where WebGL may
    not read image files (the browser treats them as cross-origin)."""
    import base64
    urls = {}
    for key in (man["sprite"], man["flow"]["sprite"]):
        with open(key.replace("assets/", ASSETS + "/", 1), "rb") as fh:
            urls[key] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    with open(f"{ASSETS}/sprites.js", "w", encoding="utf-8") as fh:
        fh.write("// Generated by tools/flows.py for file:// use. Do not edit.\n")
        fh.write("window.COYOTE_SPRITES = " + json.dumps(urls) + ";\n")


if __name__ == "__main__":
    main()
