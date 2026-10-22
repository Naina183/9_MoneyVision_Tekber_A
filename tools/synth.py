"""Generator scene sintetis multi-uang untuk pengujian deteksi kontur & penjumlahan.

Uang (citra scan Dataset A yang BUKAN referensi) ditempel pada latar belakang
buatan (kertas putih, meja gelap, kayu, kain) dengan rotasi, skala, perspektif,
pencahayaan, blur, dan noise acak. Ground truth (nominal & posisi) diketahui.
"""
from __future__ import annotations

import cv2
import numpy as np

SCENE_W, SCENE_H = 1280, 720


def _background(rng: np.random.Generator, kind: str) -> np.ndarray:
    h, w = SCENE_H, SCENE_W
    if kind == "kertas_putih":
        base = np.array([232, 234, 236], np.float32) + rng.uniform(-12, 8, 3)
        bg = np.ones((h, w, 3), np.float32) * base
    elif kind == "meja_gelap":
        base = rng.uniform(35, 75) + rng.uniform(-6, 6, 3)
        bg = np.ones((h, w, 3), np.float32) * base
    elif kind == "kayu":
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        n = cv2.resize(rng.random((8, 40)).astype(np.float32), (w, h))
        stripes = 0.5 + 0.5 * np.sin((y * 60 + n * 6 + x * 2) * np.pi)
        tone = (0.75 + 0.25 * stripes)[..., None]
        bg = tone * np.array([60, 105, 150], np.float32) * rng.uniform(0.8, 1.15)
    else:  # kain berwarna polos
        base = rng.choice([[150, 90, 40], [60, 60, 140], [70, 120, 70], [110, 110, 110]])
        bg = np.ones((h, w, 3), np.float32) * np.array(base, np.float32)
        bg += cv2.resize(rng.normal(0, 8, (h // 8, w // 8, 3)).astype(np.float32), (w, h))[..., :3]
    return bg


def _lighting(rng, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    gain = rng.uniform(0.75, 1.15) * (1.0 - rng.uniform(0.1, 0.35) * d)
    return img * gain[..., None]


def make_scene(notes: list[tuple[int, np.ndarray]], rng: np.random.Generator,
               bg_kind: str, max_tries: int = 200):
    """notes: list (nominal, citra BGR). Mengembalikan (scene, list ground-truth)."""
    scene = _background(rng, bg_kind)
    occupied = np.zeros((SCENE_H, SCENE_W), np.uint8)
    gts = []
    for denom, img in notes:
        for _ in range(max_tries):
            target_w = rng.uniform(0.28, 0.42) * SCENE_W
            s = target_w / img.shape[1]
            nh, nw = img.shape[0] * s, img.shape[1] * s
            src = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
            ang = rng.uniform(0, 2 * np.pi)
            cx, cy = rng.uniform(0.15, 0.85) * SCENE_W, rng.uniform(0.2, 0.8) * SCENE_H
            rect = np.float32([[-nw / 2, -nh / 2], [nw / 2, -nh / 2], [nw / 2, nh / 2], [-nw / 2, nh / 2]])
            rect += rng.normal(0, 0.025 * nw, rect.shape)             # perspektif ringan
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
            dst = rect @ R.T + np.float32([cx, cy])
            if dst[:, 0].min() < 5 or dst[:, 1].min() < 5 or dst[:, 0].max() > SCENE_W - 5 or dst[:, 1].max() > SCENE_H - 5:
                continue
            m = np.zeros_like(occupied)
            cv2.fillConvexPoly(m, dst.astype(np.int32), 255)
            grown = cv2.dilate(m, np.ones((25, 25), np.uint8))
            if (grown & occupied).any():
                continue
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (SCENE_W, SCENE_H)).astype(np.float32)
            a = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, (SCENE_W, SCENE_H))
            a = cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32)[..., None] / 255.0
            # bayangan tipis di bawah uang
            sh = cv2.GaussianBlur(cv2.warpAffine(m, np.float32([[1, 0, 6], [0, 1, 8]]), (SCENE_W, SCENE_H)), (31, 31), 0)
            scene *= (1 - 0.25 * sh.astype(np.float32)[..., None] / 255.0)
            scene = scene * (1 - a) + warped * a
            occupied |= m
            gts.append({"denom": denom, "quad": dst})
            break
    scene = _lighting(rng, scene)
    k = int(rng.choice([1, 3, 5]))
    if k > 1:
        scene = cv2.GaussianBlur(scene, (k, k), 0)
    scene += rng.normal(0, rng.uniform(2, 7), scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8), gts
