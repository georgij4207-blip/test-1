"""Build the canonical 36-frame turnaround (10° step) from the source sheet.

Pipeline
  1. Cut the 9x4 sheet into cells, cut the coyote out of the backdrop
     (removes the angle label and the drawn floor shadow).
  2. Normalise scale and position: frame 0 is centred on its feet; every next
     frame of the 0..180° half is registered to the previous one (scale + shift,
     feet locked to one baseline) by maximising the overlap of the lower body.
  3. Build the 360° loop: 0..180 come from the sheet, 190..350 are horizontal
     mirrors of 170..10 (the sheet's own 240..340 cells face the wrong way).
     The sheet has no 80° drawing; it is synthesised by optical-flow morphing
     between 70° and 90° and flagged as interpolated.
  4. Measure every neighbouring pair and write a QA report.

Run from the repo root:  python3 tools/build_frames.py
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from common import cells, load_sheet, segment  # noqa: E402
from morph import flow_morph  # noqa: E402

OUT = "web/assets"
FW, FH = 200, 270          # output frame size
BASELINE = FH - 14         # y of the soles
CX = FW // 2               # rotation axis
TARGET_H = 236             # height of frame 0 (ear tip to sole) after scaling


def cutout(cell):
    alpha, bg = segment(cell)
    img = cell.astype(np.float32)
    d = np.linalg.norm(img - bg, axis=2)
    warm = (img[..., 0] - img[..., 2]) - (bg[..., 0] - bg[..., 2])
    m = alpha > 128
    base = np.nonzero(m.any(1))[0].max()
    # floor shadow: faint, cool, flat smear at the very bottom, mostly beside the feet
    band = np.zeros_like(m)
    band[base - 14:, :] = True
    shadow = band & (d < 62) & (warm < 5)
    alpha = alpha.copy()
    alpha[shadow] = 0
    # remove now-isolated crumbs
    n, lbl, st, _ = cv2.connectedComponentsWithStats((alpha > 128).astype(np.uint8), 8)
    if n > 1:
        big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        for i in range(1, n):
            if i != big and st[i, cv2.CC_STAT_AREA] < 40:
                alpha[lbl == i] = 0
    return np.dstack([cell, alpha])


def place(rgba, scale, ax, base):
    """Scale about (ax, base) and put that point at (CX, BASELINE)."""
    M = np.float32([[scale, 0, CX - scale * ax], [0, scale, BASELINE - scale * base]])
    rgb = rgba[..., :3].astype(np.float32) * (rgba[..., 3:] / 255.0)   # premultiply
    pm = np.dstack([rgb, rgba[..., 3].astype(np.float32)])
    out = cv2.warpAffine(pm, M, (FW, FH), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    out = np.clip(out, 0, 255)
    a = out[..., 3:]
    rgb = np.where(a > 0, out[..., :3] * 255.0 / np.maximum(a, 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)


def feet_center(rgba, base):
    feet = rgba[int(base) - 12:int(base) + 1, :, 3] > 128
    xs = np.nonzero(feet.any(0))[0]
    return (xs.min() + xs.max()) / 2.0


def symmetry_axis(rgba):
    """x of the vertical mirror axis that best maps the silhouette onto itself."""
    m = rgba[..., 3] > 128
    w = m.shape[1]
    xs = np.arange(w)
    best = (-1.0, w / 2.0)
    for ax2 in range(w // 2, w + w // 2):
        src = ax2 - xs
        ok = (src >= 0) & (src < w)
        sh = np.zeros_like(m)
        sh[:, ok] = m[:, src[ok]]
        v = (sh & m).sum() / (sh | m).sum()
        if v > best[0]:
            best = (v, ax2 / 2.0)
    return best[1]


def body_mask(rgba):
    """Lower 72% of the silhouette: legs + torso, which keep their height while turning."""
    m = (rgba[..., 3] > 128)
    ys = np.nonzero(m.any(1))[0]
    cut = int(ys.max() - 0.72 * (ys.max() - ys.min()))
    m[:cut] = False
    return m


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def register(ref_placed, rgba, base, guess_scale, guess_ax):
    """Search scale/axis so that rgba, once placed, best overlaps ref_placed's body."""
    ref = body_mask(ref_placed)
    best = (-1, guess_scale, guess_ax)
    for s in np.linspace(guess_scale * 0.9, guess_scale * 1.1, 21):
        for ax in np.arange(guess_ax - 20, guess_ax + 21, 1.0):
            m = place(rgba, s, ax, base)[..., 3] > 128
            ys = np.nonzero(m.any(1))[0]
            cut = int(ys.max() - 0.72 * (ys.max() - ys.min()))
            m[:cut] = False
            v = iou(ref, m)
            if v > best[0]:
                best = (v, s, ax)
    return best


def metrics(a, b):
    ma, mb = a[..., 3] > 128, b[..., 3] > 128
    both = ma & mb
    col = float(np.abs(a[..., :3].astype(int) - b[..., :3].astype(int))[both].mean()) if both.any() else 255.0
    return {"iou": round(float(iou(ma, mb)), 3), "color_diff": round(col, 1)}


def base_of(rgba):
    return float(np.nonzero((rgba[..., 3] > 128).any(1))[0].max())


def main():
    os.makedirs(f"{OUT}/frames", exist_ok=True)
    sheet = load_sheet()
    src = {lab: cutout(c) for lab, c in cells(sheet)}

    # --- frame 0: centre on the feet, fixed height
    f0 = src[0]
    m0 = f0[..., 3] > 128
    ys = np.nonzero(m0.any(1))[0]
    b0 = float(ys.max())
    s0 = TARGET_H / (ys.max() - ys.min())
    # Front (0°) and back (180°) views are symmetric, so their true axis is the
    # mirror axis. Other views use the middle of the soles, corrected by the
    # sole-vs-mirror offset interpolated between 0° and 180°.
    ax0 = symmetry_axis(f0)
    d0 = ax0 - feet_center(f0, b0)
    f180 = src[180]
    d180 = symmetry_axis(f180) - feet_center(f180, base_of(f180))
    placed = {0: place(f0, s0, ax0, b0)}
    params = {0: (1.0, s0, ax0)}

    order = [0, 10, 20, 30, 40, 50, 60, 70, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    prev = 0
    for lab in order[1:]:
        rgba = src[lab]
        b = base_of(rgba)
        _, ps, pax = params[prev]
        # rough guess: same relative position as previous frame
        score, s, _ = register(placed[prev], rgba, b, ps, pax)
        # Horizontal anchor is measured per frame instead of chained, so the
        # axis cannot drift: the middle of the soles stays on the rotation axis
        # (front/back: between both feet; profile: middle of heel..toe).
        ax = feet_center(rgba, b) + d0 + (d180 - d0) * lab / 180.0
        params[lab] = (score, s, ax)
        placed[lab] = place(rgba, s, ax, b)
        print(f"{lab:3d}°  overlap {score:.3f}  scale {s:.3f}  axis x {ax:.1f}")
        prev = lab

    # --- 80°: optical-flow in-between of 70° and 90°
    placed[80] = flow_morph(placed[70], placed[90], 0.5)

    # --- second half: mirror
    frames, info = [], []
    for k in range(36):
        ang = k * 10
        if ang <= 180:
            img = placed[ang]
            origin = "интерполяция (оптический поток 70°↔90°)" if ang == 80 else f"лист, {ang}°"
        else:
            img = placed[360 - ang][:, ::-1].copy()
            origin = f"зеркало {360 - ang}°" + (" (интерполяция)" if 360 - ang == 80 else "")
        frames.append(img)
        info.append({"angle": ang, "source": origin,
                     "interpolated": ang in (80, 280)})
        Image.fromarray(img, "RGBA").save(f"{OUT}/frames/coyote_{ang:03d}.png", optimize=True)

    # --- sprite sheet 6x6
    cols = 6
    sprite = Image.new("RGBA", (FW * cols, FH * 6), (0, 0, 0, 0))
    for k, img in enumerate(frames):
        sprite.paste(Image.fromarray(img, "RGBA"), ((k % cols) * FW, (k // cols) * FH))
    sprite.save(f"{OUT}/coyote_sprite.png", optimize=True)

    # --- QA: neighbour metrics
    for k in range(36):
        m = metrics(frames[k], frames[(k + 1) % 36])
        info[k]["to_next"] = m
    ious = [i["to_next"]["iou"] for i in info]
    med = float(np.median(ious))
    for i in info:
        v = i["to_next"]["iou"]
        i["to_next"]["flag"] = ("jump" if v < med - 0.15 else
                                "near-duplicate" if v > 0.93 else "ok")

    manifest = {
        "frameWidth": FW, "frameHeight": FH, "columns": cols, "count": 36,
        "stepDegrees": 10, "baselineY": BASELINE, "axisX": CX,
        "sprite": "assets/coyote_sprite.png", "frames": info,
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    from interpolate import write_js
    write_js()
    print("\nneighbour check (IoU of silhouettes, mean RGB diff on overlap):")
    for i in info:
        t = i["to_next"]
        print(f"  {i['angle']:3d}° -> {(i['angle'] + 10) % 360:3d}°  iou {t['iou']:.3f}  "
              f"col {t['color_diff']:5.1f}  {t['flag']:14s} [{i['source']}]")


if __name__ == "__main__":
    main()
