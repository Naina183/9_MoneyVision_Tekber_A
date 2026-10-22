
"""Uji pengurutan sudut dan perspective warp, termasuk uang yang diputar ±45°."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from moneyvision.preprocessing import order_corners, warp_quad


def _rotated_rect(cx, cy, w, h, deg):
    return cv2.boxPoints(((cx, cy), (w, h), deg)).astype(np.float32)


@pytest.mark.parametrize("deg", [0, 15, 30, 44, 45, 46, 60, 89, 120, 135, 170])
def test_order_corners_is_clockwise_and_unique(deg):
    quad = _rotated_rect(400, 300, 300, 140, deg)
    rng = np.random.default_rng(deg)
    ordered = order_corners(quad[rng.permutation(4)])
    assert len({tuple(p) for p in ordered.round(3)}) == 4
    # Luas bertanda positif = searah jarum jam pada koordinat citra (y ke bawah)
    x, y = ordered[:, 0], ordered[:, 1]
    signed = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed > 0


@pytest.mark.parametrize("deg", [0, 45, 135])
def test_warp_keeps_note_aspect(deg):
    img = np.zeros((600, 800, 3), np.uint8)
    quad = _rotated_rect(400, 300, 300, 140, deg)
    cv2.fillConvexPoly(img, quad.astype(np.int32), (0, 0, 255))
    out = warp_quad(img, quad)
    long_side, short_side = max(out.shape[:2]), min(out.shape[:2])
    assert 1.9 < long_side / short_side < 2.4
    assert out[..., 2].mean() > 200