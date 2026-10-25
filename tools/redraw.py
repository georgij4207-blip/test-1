"""Redrawn back-quarter of the turn (120..180°), built from parts of the sheet.

The source sheet has no true back view: its "180°" cell is really a
three-quarter back view (feet point right, one cheek tuft visible, tail out to
the left) and 100..170° are near-identical profiles. This module rebuilds the
quarter so the head turns away, the tail swings in behind the legs and the
turn ends on a symmetric back view that mirrors cleanly into 190°.

Everything here works on frames already aligned by build_frames.py
(same size, baseline and rotation axis). All new frames are composites of
the user's own drawings (moved, mirrored and repainted parts), not freehand
art; build_frames.py labels them as such.

  120°  sheet 120°                       (unchanged)
  130°  sheet 140°                       (re-timed: 130° was a near-duplicate)
  140°  sheet 160°                       (re-timed: 150° was a near-duplicate)
  150°  sheet 170°                       (re-timed)
  160°  sheet "180°" cell, tail swung halfway to the centre
  170°  back view: head still slightly turned, tail almost centred
  180°  true back view: front silhouette repainted as back fur, symmetric
        back of head from the sheet's "180°" cell, tail hanging centred
"""
import cv2
import numpy as np

AXIS_X = 100          # rotation axis in aligned frames
NECK_Y = 118          # where the head meets the shoulders in the front view
HEAD_CUT = 128        # rows of the back-view cell taken as "head" (with neck fur)
HEAD_SCALE = 1.1      # the back-view cell is drawn slightly smaller than the front
TAIL_ROOT = (76.0, 166.0)
# outline of the tail in the aligned sheet "180°" cell (left of the far leg)
TAIL_POLY = [(20, 160), (70, 160), (80, 163), (79, 210), (62, 240), (20, 252)]

RETIME = {120: 120, 130: 140, 140: 160, 150: 170}


# ---------- small RGBA helpers (straight alpha, uint8) ----------

def over(dst, src):
    a = src[..., 3:] / 255.0
    b = dst[..., 3:] / 255.0
    oa = a + b * (1 - a)
    rgb = (src[..., :3] * a + dst[..., :3] * b * (1 - a)) / np.maximum(oa, 1e-6)
    return np.dstack([rgb, oa * 255]).clip(0, 255).astype(np.uint8)


def affine(img, M):
    a = img[..., 3:] / 255.0
    pm = np.dstack([img[..., :3] * a, img[..., 3:]]).astype(np.float32)
    o = cv2.warpAffine(pm, M, (img.shape[1], img.shape[0]),
                       flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0)).clip(0, 255)
    al = o[..., 3:]
    rgb = np.where(al > 0, o[..., :3] * 255 / np.maximum(al, 1e-3), 0)
    return np.dstack([rgb.clip(0, 255), al]).astype(np.uint8)


def with_alpha(img, factor):
    out = img.copy()
    out[..., 3] = (out[..., 3] * factor).clip(0, 255).astype(np.uint8)
    return out


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m


def mirror_axis(rgba):
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
        v = (sh & m).sum() / max((sh | m).sum(), 1)
        if v > best[0]:
            best = (v, ax2 / 2.0)
    return best[1]


def torso_x(img):
    """Mean x of the chest/belly band, a stable marker of where the body sits."""
    band = img[125:160, :, 3] > 128
    return float(np.nonzero(band)[1].mean())


def shift_x(img, dx):
    return affine(img, np.float32([[1, 0, dx], [0, 1, 0]]))


# ---------- parts ----------

def split_tail(back34):
    """Tail (feathered at the root) and the rest of the three-quarter back cell."""
    h, w = back34.shape[:2]
    yy = np.mgrid[0:h, 0:w][0]
    tm = poly_mask(back34.shape, TAIL_POLY)
    tail = with_alpha(back34, tm * np.clip((yy - 160) / 14.0, 0, 1))
    rest = with_alpha(back34, 1 - tm)
    return tail, rest


def swing_tail(tail, t, target_x=96.0):
    """Swing the tail from its drawn position (t=0) to hanging centred (t=1)."""
    ang = 20.0 * t
    sc = 1.0 - 0.08 * t                       # slight foreshortening behind the legs
    R = cv2.getRotationMatrix2D(TAIL_ROOT, ang, sc)
    R[0, 2] += (target_x - TAIL_ROOT[0]) * t
    return affine(tail, R)


def _fur_fill(img, fur, ksize):
    """Smooth fur colour at every pixel, averaged from nearby `fur` pixels only."""
    w = cv2.GaussianBlur(fur.astype(np.float32), (0, 0), ksize)
    c = cv2.GaussianBlur(img[..., :3].astype(np.float32) * fur[..., None], (0, 0), ksize)
    return c / np.maximum(w[..., None], 1e-3)


def back_body(front):
    """Front view below the head, repainted as seen from behind.

    - chest stripe (cream) and its outline -> back fur
    - cool grey-blue inner shadows (far side between arms/legs) -> dark warm fur
    - toe strokes on the feet -> smooth (from behind we see the heels)
    """
    body = front.copy()
    body[:NECK_Y - 6, :, 3] = 0
    rgb = body[..., :3].astype(np.float32)
    lum = rgb.mean(2)
    solid = body[..., 3] > 200
    warm = rgb[..., 0] - rgb[..., 2]
    fur = solid & (lum > 50) & (lum < 110) & (warm > 12)

    cream = solid & (lum > 115)
    region = cv2.dilate(cream.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & solid
    region[:NECK_Y] = False
    fill = _fur_fill(body, fur & ~region, 5)
    out = np.where(region[..., None], fill, rgb)

    cool = solid & (warm < 8) & (lum > 45) & (lum < 130)
    cool[:NECK_Y + 10] = False
    cool = cv2.dilate(cool.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & solid & ~fur
    out = np.where(cool[..., None], _fur_fill(body, fur, 4) * 0.72, out)

    base = int(np.nonzero(solid.any(1))[0].max())
    feet = np.zeros_like(solid)
    feet[base - 16:base + 1] = True
    inner = cv2.erode(solid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    toes = feet & inner & (lum < 60)
    out = np.where(toes[..., None], _fur_fill(body, feet & fur, 3), out)

    body[..., :3] = out.clip(0, 255).astype(np.uint8)
    return body


def _head_crop(back34):
    head = back34.copy()
    head[HEAD_CUT:, :, 3] = 0
    head[:, :65, 3] = 0                       # drop the tail tip / far side
    fade = np.clip((HEAD_CUT - np.arange(head.shape[0])) / 10.0, 0, 1)
    return with_alpha(head, fade[:, None])


def back_heads(back34):
    """(symmetric back of head, as-drawn turned head), both placed on the axis."""
    head = _head_crop(back34)
    ax = mirror_axis(head)
    w = head.shape[1]
    xs = np.arange(w)
    src = np.round(2 * ax - xs).astype(int)
    ok = (src >= 0) & (src < w)
    mir = np.zeros_like(head)
    mir[:, ok] = head[:, src[ok]]
    sym = head.copy()
    sym[:, xs < ax] = mir[:, xs < ax]         # right half carries the visible cheek tuft
    # clear the stray strokes along the centre of the skull left by the turn
    strip = np.zeros(sym.shape[:2], np.uint8)
    ai = int(round(ax))
    strip[92:HEAD_CUT, ai - 9:ai + 10] = 1
    strip &= (sym[..., 3] > 200)
    sym[..., :3] = cv2.inpaint(sym[..., :3].copy(), strip, 5, cv2.INPAINT_TELEA)
    M = np.float32([[HEAD_SCALE, 0, AXIS_X - HEAD_SCALE * ax],
                    [0, HEAD_SCALE, NECK_Y + 3 - HEAD_SCALE * 121]])
    return affine(sym, M), affine(head, M)


# ---------- frames ----------

def build(placed):
    """placed: aligned sheet frames keyed by sheet label. Returns {angle: (rgba, note)}."""
    out = {}
    for ang, src in RETIME.items():
        note = f"лист, {src}°" + ("" if ang == src else f" (перенесён на {ang}°)")
        out[ang] = (placed[src], note)

    back34 = placed[180]
    tail, rest = split_tail(back34)
    body = back_body(placed[0])
    head_sym, head_turned = back_heads(back34)

    # 160°: the sheet's three-quarter back view, tail swung halfway in
    f160 = over(rest, swing_tail(tail, 0.3))
    out[160] = (f160, "дорисован: ячейка «180°» листа, хвост смещён к центру")

    # 170°: back-view body narrowed slightly, head as drawn in the turned cell
    squash = np.float32([[0.95, 0, AXIS_X * 0.05], [0, 1, 0]])
    body170 = affine(body, squash)
    shift = np.float32([[1, 0, 3], [0, 1, 0]])
    head170 = affine(head_turned, shift)
    f170 = over(over(body170, head170), swing_tail(tail, 0.7))
    out[170] = (f170, "дорисован: тело вида сзади + полуповёрнутая голова, хвост почти по центру")

    # 180°: symmetric back view
    f180 = over(over(body, head_sym), swing_tail(tail, 1.0))
    out[180] = (f180, "дорисован: силуэт анфаса в окраске спины, симметричный затылок, хвост по центру")

    # The sheet's axis estimates for its back-quarter cells drift (its "180°"
    # cell is not symmetric). Walk the body smoothly from where it sits at 120°
    # to the centre of the symmetric back view.
    x0, x1 = torso_x(out[120][0]), torso_x(out[180][0])
    for ang in range(130, 180, 10):
        img, note = out[ang]
        target = x0 + (x1 - x0) * (ang - 120) / 60.0
        out[ang] = (shift_x(img, target - torso_x(img)), note)
    return out
