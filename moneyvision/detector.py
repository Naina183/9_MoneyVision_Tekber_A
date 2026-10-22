
"""Contour detection: menemukan dan memisahkan area setiap lembar uang pada frame.

Alur:
 1. frame diperkecil ke lebar kerja (WORK_WIDTH) agar cepat;
 2. mask latar depan dibuat dari dua petunjuk:
    a. jarak warna (CIE Lab) terhadap warna latar yang diperkirakan dari tepi frame,
    b. kerapatan tepi Canny (uang bertekstur rapat, latar polos relatif halus);
 3. operasi morfologi (opening + closing) dan pengisian lubang kontur;
 4. cv2.findContours -> setiap kontur disaring berdasarkan luas, rasio sisi,
    dan solidity (kemiripan dengan persegi panjang). Kontur besar yang gagal
    disaring dicoba dipisah dengan erosi bertahap (uang yang saling berdekatan);
 5. sudut kontur (approxPolyDP / minAreaRect) dipakai untuk perspective warp
    sehingga setiap uang menjadi citra persegi panjang yang lurus.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..data.MoneyVision import config as C
from .preprocessing import resize_to_width, warp_quad


@dataclass
class Detection:
    quad: np.ndarray          # 4 titik sudut (koordinat citra asli), float32 (4, 2)
    note: np.ndarray          # potongan uang hasil perspective warp (BGR)
    area_frac: float          # luas relatif terhadap frame
    full_frame: bool = False  # True bila tidak ada kontur & seluruh frame dipakai


def _border_values(img: np.ndarray, t: int = 6) -> np.ndarray:
    ch = 1 if img.ndim == 2 else img.shape[2]
    return np.concatenate([img[:t].reshape(-1, ch), img[-t:].reshape(-1, ch),
                           img[:, :t].reshape(-1, ch), img[:, -t:].reshape(-1, ch)])


def _kernel(size: int, shape=cv2.MORPH_RECT) -> np.ndarray:
    return cv2.getStructuringElement(shape, (size, size))


def foreground_mask(work: np.ndarray) -> np.ndarray:
    """Mask biner area yang berbeda dari latar (uang = 255)."""
    blur = cv2.GaussianBlur(work, (5, 5), 0)

    # (a) jarak warna terhadap latar
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(_border_values(lab), axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = (dist > C.BG_DIST_THRESH).astype(np.uint8) * 255

    # (b) kerapatan tepi (tekstur cetakan uang)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90).astype(np.float32) / 255.0
    dens = cv2.blur(edges, (21, 21))
    bg_dens = float(np.median(_border_values(dens[..., None])))
    tex = (dens > max(0.12, bg_dens * 2.5 + 0.03)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, tex)

    # (c) morfologi + isi lubang
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(9), iterations=2)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, _kernel(15))


def _rect_ok(cnt: np.ndarray, frame_area: float) -> bool:
    area = cv2.contourArea(cnt)
    frac = area / frame_area
    if not (C.MIN_AREA_FRAC <= frac <= C.MAX_AREA_FRAC):
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    if min(w, h) < 20:
        return False
    aspect = max(w, h) / min(w, h)
    return C.ASPECT_MIN <= aspect <= C.ASPECT_MAX and area / (w * h) >= C.MIN_SOLIDITY


def _split_blob(blob: np.ndarray, frame_area: float) -> list[np.ndarray]:
    """Memisahkan beberapa uang yang menempel/berdekatan dengan erosi bertahap."""
    for r in (4, 8, 12, 16, 22, 28):
        k = _kernel(2 * r + 1, cv2.MORPH_ELLIPSE)
        er = cv2.erode(blob, k)
        n, lab = cv2.connectedComponents(er)
        if n - 1 >= 2:
            parts = []
            for i in range(1, n):
                comp = cv2.dilate((lab == i).astype(np.uint8) * 255, k) & blob
                c = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if c:
                    c = max(c, key=cv2.contourArea)
                    if _rect_ok(c, frame_area):
                        parts.append(c)
            if parts:
                return parts
        if n - 1 == 0:
            break
    return []


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float32)
        a = cv2.contourArea(approx)
        if 0.85 < a / max(cv2.contourArea(box), 1) < 1.15 and _near_rectangle(quad):
            return quad   # menangkap efek perspektif
    return box


def _near_rectangle(quad: np.ndarray, max_cos: float = 0.25) -> bool:
    """True bila keempat sudut mendekati 90° dan rasio sisinya wajar untuk uang.

    approxPolyDP kadang memilih sudut yang salah (mis. ikut bayangan) sehingga
    hasilnya jajaran genjang; warp dari sudut seperti itu merusak citra uang.
    """
    pts = quad.reshape(4, 2)
    for i in range(4):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        cos = abs(float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        if cos > max_cos:
            return False
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    w, h = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    aspect = max(w, h) / max(min(w, h), 1)
    return C.ASPECT_MIN <= aspect <= C.ASPECT_MAX


def detect_notes(frame: np.ndarray, allow_full_frame: bool = True,
                 return_mask: bool = False):
    """Mendeteksi semua lembar uang pada frame BGR.

    Mengembalikan list Detection (diurutkan dari kiri ke kanan); bila
    return_mask=True juga mengembalikan mask hasil segmentasi.
    """
    work, scale = resize_to_width(frame, C.WORK_WIDTH)
    mask = foreground_mask(work)
    frame_area = float(work.shape[0] * work.shape[1])
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    accepted = []
    for c in cnts:
        if _rect_ok(c, frame_area):
            accepted.append(c)
        elif cv2.contourArea(c) / frame_area >= 1.6 * C.MIN_AREA_FRAC:
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
            accepted.extend(_split_blob(blob, frame_area))

    dets: list[Detection] = []
    for c in accepted:
        quad = _quad_from_contour(c) / scale
        dets.append(Detection(quad=quad, note=warp_quad(frame, quad),
                              area_frac=cv2.contourArea(c) / frame_area))

    if not dets and allow_full_frame:
        h, w = frame.shape[:2]
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        dets.append(Detection(quad=quad, note=frame.copy(), area_frac=1.0, full_frame=True))

    dets.sort(key=lambda d: d.quad[:, 0].min())
    return (dets, mask) if return_mask else dets
