"""Shared helpers: grid layout of the source turnaround sheet and cell segmentation."""
import numpy as np
from PIL import Image
import cv2

SHEET = "source/turnaround_sheet.png"

# Pixel bounds of the 9x4 grid, measured from the thin light separator lines.
COLS = [(0, 186), (188, 354), (356, 522), (524, 688), (690, 863),
        (864, 1033), (1034, 1196), (1198, 1358), (1360, 1536)]
ROWS = [(0, 257), (259, 510), (512, 765), (767, 1024)]

# Angle printed in each cell, row by row. Note: the sheet has no 80° cell
# (row 1 jumps 70 -> 90) and ends with a duplicate 360° cell.
LABELS = [0, 10, 20, 30, 40, 50, 60, 70, 90,
          100, 110, 120, 130, 140, 150, 160, 170, 180,
          190, 200, 210, 220, 230, 240, 250, 260, 270,
          280, 290, 300, 310, 320, 330, 340, 350, 360]


def load_sheet():
    return np.asarray(Image.open(SHEET).convert("RGB"))


def cells(sheet):
    i = 0
    for (y0, y1) in ROWS:
        for (x0, x1) in COLS:
            # inset by 2px to drop separator lines
            yield LABELS[i], sheet[y0 + 2:y1 - 2, x0 + 2:x1 - 2].copy()
            i += 1


def _fit_background(img, mask_bg):
    """Least-squares quadratic surface per channel over known-background pixels."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / w - 0.5
    y = yy / h - 0.5
    basis = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    A = basis[mask_bg]
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, img[..., c][mask_bg].astype(np.float32), rcond=None)
        out[..., c] = basis @ coef
    return out


def segment(cell):
    """Return an alpha mask (uint8 0..255) separating the coyote from the backdrop."""
    h, w, _ = cell.shape
    img = cell.astype(np.float32)
    # start from the cell border (minus the angle label corner) as background
    border = np.zeros((h, w), bool)
    border[:6, :] = border[-6:, :] = True
    border[:, :6] = border[:, -6:] = True
    border[:34, :60] = False
    bg = _fit_background(img, border)
    for _ in range(2):
        d = np.linalg.norm(img - bg, axis=2)
        warm = (img[..., 0] - img[..., 2]) - (bg[..., 0] - bg[..., 2])
        fg = (d > 16) | (warm > 10)
        fg[:34, :60] &= (d[:34, :60] > 16) & (warm[:34, :60] > 4)  # angle label is gray/white, not warm
        bg = _fit_background(img, ~cv2.dilate(fg.astype(np.uint8), np.ones((9, 9))).astype(bool))
    d = np.linalg.norm(img - bg, axis=2)
    warm = (img[..., 0] - img[..., 2]) - (bg[..., 0] - bg[..., 2])
    fg = ((d > 14) | (warm > 8)).astype(np.uint8)
    # drop the angle label text
    lab = np.zeros_like(fg)
    lab[:34, :60] = 1
    fg[(lab == 1) & (warm < 4)] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # keep the largest connected component plus anything big attached nearby
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep = np.zeros_like(fg)
        bx, by, bw, bh = stats[big, :4]
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            near = (x < bx + bw + 4 and x + ww > bx - 4 and y < by + bh + 4 and y + hh > by - 4)
            if i == big or (area > 25 and near):
                keep[lbl == i] = 1
        fg = keep
    # fill interior holes (e.g. gaps between arm and body stay open: only fully enclosed holes)
    flood = fg.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (flood == 0)
    small_holes = np.zeros_like(fg)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 4)
    for i in range(1, n2):
        if s2[i, cv2.CC_STAT_AREA] < 60:
            small_holes[l2 == i] = 1
    fg = fg | small_holes
    # soft edge
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (3, 3), 0.7)
    return (np.clip(alpha, 0, 1) * 255).astype(np.uint8), bg
