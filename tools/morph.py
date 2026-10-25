"""Optical-flow in-betweening of two RGBA frames (used for missing / extra angles).

Results are synthetic: they are a warp + cross-dissolve of the two neighbours,
not new drawings, and must be labelled as interpolated.
"""
import cv2
import numpy as np


def _gray(rgba):
    a = rgba[..., 3:].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255 * (1 - a)   # on white so edges matter
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _flow(a, b):
    return cv2.calcOpticalFlowFarneback(_gray(a), _gray(b), None,
                                        pyr_scale=0.5, levels=5, winsize=21,
                                        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)


def _warp(img, flow, t):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = gx + flow[..., 0] * t
    mapy = gy + flow[..., 1] * t
    a = img[..., 3:].astype(np.float32) / 255.0
    pm = np.dstack([img[..., :3].astype(np.float32) * a, a * 255])
    return cv2.remap(pm, mapx, mapy, cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def flow_morph(a, b, t):
    """Frame at fraction t (0..1) between RGBA frames a and b."""
    f_ab = _flow(a, b)   # a -> b
    f_ba = _flow(b, a)
    # backward warping: sample a at p - t*f_ab ≈ p + t*f_ba, b at p + (1-t)*f_ab
    wa = _warp(a, f_ba, t)
    wb = _warp(b, f_ab, 1 - t)
    pm = wa * (1 - t) + wb * t
    alpha = pm[..., 3:]
    rgb = np.where(alpha > 0, pm[..., :3] * 255.0 / np.maximum(alpha, 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 255)]).astype(np.uint8)
