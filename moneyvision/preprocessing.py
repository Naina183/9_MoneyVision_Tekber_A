
"""Tahap image processing: resize, grayscale, peningkatan kualitas, dan pelurusan."""
from __future__ import annotations

import cv2
import numpy as np

from ..data.MoneyVision import config as C

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def resize_to_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Mengubah lebar citra (menjaga rasio). Mengembalikan (citra, skala)."""
    h, w = img.shape[:2]
    if w == width:
        return img, 1.0
    scale = width / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=interp), scale


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    """CLAHE (equalisasi histogram adaptif) + Gaussian blur ringan.

    CLAHE menormalkan kontras lokal sehingga fitur tetap muncul pada uang yang
    kusam/lusuh atau pencahayaan yang tidak merata; blur meredam noise sensor.
    """
    eq = _CLAHE.apply(gray)
    return cv2.GaussianBlur(eq, (3, 3), 0)


def gray_world(img: np.ndarray) -> np.ndarray:
    """Koreksi white balance sederhana (asumsi gray-world) untuk fitur warna."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def canonical(note_bgr: np.ndarray) -> np.ndarray:
    """Menyamakan orientasi (landscape) dan ukuran citra uang ke ukuran kanonik."""
    h, w = note_bgr.shape[:2]
    if h > w:
        note_bgr = cv2.rotate(note_bgr, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(note_bgr, (C.CANON_W, C.CANON_H), interpolation=cv2.INTER_AREA)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Mengurutkan 4 titik sudut: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    # Diurutkan menurut sudut terhadap titik pusat (searah jarum jam pada
    # koordinat citra), lalu dimulai dari titik dengan x+y terkecil. Cara ini
    # tetap benar untuk uang yang diputar ±45°, kasus di mana heuristik
    # "jumlah/selisih koordinat" yang umum dipakai menghasilkan urutan rusak.
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def warp_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective transform: memotong area uang menjadi persegi panjang lurus."""
    tl, tr, br, bl = order_corners(quad)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(img, M, (w, h))
