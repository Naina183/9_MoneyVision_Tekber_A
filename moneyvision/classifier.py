"""Klasifikasi nominal: ORB feature matching + verifikasi geometri (RANSAC) + warna.

Untuk setiap citra uang yang sudah dipotong oleh detector:
 1. citra diseragamkan (canonical size), di-grayscale, lalu CLAHE;
 2. keypoint & descriptor ORB diekstrak;
 3. descriptor dicocokkan ke setiap citra referensi (BFMatcher Hamming, k=2)
    lalu disaring dengan Lowe ratio test;
 4. pasangan yang lolos diverifikasi dengan homografi RANSAC; jumlah inlier
    menjadi skor kemiripan geometri terhadap referensi tersebut;
 5. skor kelas = inlier maksimum di antara referensi kelas itu;
 6. histogram warna HSV dipakai sebagai petunjuk tambahan saat dua kelas
    bersaing ketat, dan sebagai cadangan bila fitur ORB terlalu sedikit.
Jika bukti tidak cukup, hasilnya UNKNOWN ("tidak dikenali") — untuk pengguna
tunanetra, menolak lebih aman daripada menyebut nominal yang salah.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from .preprocessing import canonical, enhance_gray, gray_world, resize_to_width, to_gray

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class Prediction:
    denom: int                       # nominal; 0 (C.UNKNOWN) bila tidak dikenali
    confidence: float                # 0..1 (perkiraan, bukan probabilitas terkalibrasi)
    method: str                      # 'orb', 'orb+warna', 'warna', '-'
    inliers: dict = field(default_factory=dict)    # {nominal: inlier terbaik}
    color_sim: dict = field(default_factory=dict)  # {nominal: kemiripan warna}
    best_ref: str = ""

    @property
    def known(self) -> bool:
        return self.denom != C.UNKNOWN


@dataclass
class _Ref:
    denom: int
    name: str
    pts: np.ndarray       # koordinat keypoint (N, 2)
    des: np.ndarray       # descriptor ORB (N, 32)
    hist: np.ndarray      # histogram warna
    size: tuple           # (w, h) citra referensi
    aug: bool = False     # True = hasil augmentasi (dipakai di tahap 2 saja)


class NoteClassifier:
    def __init__(self, ref_dir: str = "references", use_orb: bool = True,
                 use_color: bool = True):
        self.use_orb = use_orb
        self.use_color = use_color
        self.orb = cv2.ORB_create(nfeatures=C.ORB_FEATURES, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=15, patchSize=31, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.refs: list[_Ref] = []
        self._load(ref_dir)
        if not self.refs:
            raise RuntimeError(f"Tidak ada citra referensi di '{ref_dir}'. "
                               "Jalankan build_references.py terlebih dahulu.")
        self.classes = sorted({r.denom for r in self.refs})

    # ------------------------------------------------------------ fitur
    def _gray_for_orb(self, bgr: np.ndarray, full_frame: bool) -> np.ndarray:
        img = resize_to_width(bgr, 900)[0] if full_frame else canonical(bgr)
        return enhance_gray(to_gray(img))

    def extract(self, bgr: np.ndarray, full_frame: bool = False):
        gray = self._gray_for_orb(bgr, full_frame)
        kps, des = self.orb.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        return pts, des, gray.shape[::-1]

    @staticmethod
    def color_hist(bgr: np.ndarray) -> np.ndarray:
        img = gray_world(canonical(bgr))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # abaikan piksel sangat terang & tak berwarna (kertas latar / bagian sobek)
        mask = ~((hsv[..., 2] > 235) & (hsv[..., 1] < 30))
        mask = mask.astype(np.uint8) * 255
        h = cv2.calcHist([hsv], [0, 1], mask, list(C.HIST_BINS), [0, 180, 0, 256])
        cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
        return h.flatten().astype(np.float32)

    @staticmethod
    def _augment(img: np.ndarray) -> list[tuple[np.ndarray, str]]:
        """Buat variasi citra referensi untuk menangani uang lusuh/kusut."""
        augs = []
        # 1. Blur ringan (simulasi uang kusut/tidak fokus)
        blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
        augs.append((blurred, "blur"))
        # 2. Brightness rendah (pencahayaan kurang)
        dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)
        augs.append((dark, "dark"))
        # 3. Brightness tinggi (cahaya berlebih)
        bright = cv2.convertScaleAbs(img, alpha=1.3, beta=25)
        augs.append((bright, "bright"))
        # 4. Kontras rendah (uang pudar/lusuh)
        faded = cv2.convertScaleAbs(img, alpha=0.8, beta=30)
        augs.append((faded, "faded"))
        # 5. Noise (simulasi kamera murah / kondisi buruk)
        noisy = img.copy().astype(np.float32)
        noise = np.random.normal(0, 12, noisy.shape).astype(np.float32)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        augs.append((noisy, "noisy"))
        return augs

    def _load(self, ref_dir: str) -> None:
        if not os.path.isdir(ref_dir):
            return
        for sub in sorted(os.listdir(ref_dir)):
            path = os.path.join(ref_dir, sub)
            if not (os.path.isdir(path) and sub.isdigit()):
                continue
            for fn in sorted(os.listdir(path)):
                if not fn.lower().endswith(IMG_EXT):
                    continue
                img = cv2.imread(os.path.join(path, fn))
                if img is None:
                    continue
                # referensi asli
                pts, des, size = self.extract(img)
                if des is None or len(des) < 10:
                    continue
                self.refs.append(_Ref(int(sub), f"{sub}/{fn}", pts, des,
                                      self.color_hist(img), size))
                # referensi augmentasi (uang lusuh/kondisi buruk)
                for aug_img, aug_tag in self._augment(img):
                    a_pts, a_des, a_size = self.extract(aug_img)
                    if a_des is None or len(a_des) < 10:
                        continue
                    self.refs.append(_Ref(int(sub), f"{sub}/{fn}_{aug_tag}",
                                          a_pts, a_des,
                                          self.color_hist(aug_img), a_size, aug=True))

    # ------------------------------------------------------------ pencocokan
    def _inliers(self, q_pts, q_des, ref: _Ref) -> int:
        if q_des is None or len(q_des) < 8:
            return 0
        pairs = self.matcher.knnMatch(q_des, ref.des, k=2)
        good = [m for m, *rest in (p for p in pairs if len(p) == 2)
                if m.distance < C.LOWE_RATIO * rest[0].distance]
        if len(good) < 8:
            return 0
        src = ref.pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        dst = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, C.RANSAC_REPROJ)
        if H is None or mask is None:
            return 0
        # tolak homografi yang tidak masuk akal (terlipat/terbalik/terlalu kecil)
        w, h = ref.size
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        if not cv2.isContourConvex(proj.astype(np.int32)) or cv2.contourArea(proj) < 0.02 * w * h:
            return 0
        return int(mask.sum())

    def scores(self, note_bgr: np.ndarray, full_frame: bool = False):
        inl = {c: 0 for c in self.classes}
        best_ref = {c: "" for c in self.classes}
        if self.use_orb:
            q_pts, q_des, _ = self.extract(note_bgr, full_frame)
            for ref in self.refs:
                n = self._inliers(q_pts, q_des, ref)
                if n > inl[ref.denom]:
                    inl[ref.denom], best_ref[ref.denom] = n, ref.name
        csim = {c: 0.0 for c in self.classes}
        if self.use_color and not full_frame:
            qh = self.color_hist(note_bgr)
            for ref in self.refs:
                d = cv2.compareHist(qh, ref.hist, cv2.HISTCMP_BHATTACHARYYA)
                csim[ref.denom] = max(csim[ref.denom], 1.0 - float(d))
        return inl, csim, best_ref

    # ------------------------------------------------------------ keputusan
    def classify(self, note_bgr: np.ndarray, full_frame: bool = False) -> Prediction:
        """Klasifikasi dua tahap supaya cepat (dipakai aplikasi live).

        Tahap 1 hanya mencocokkan ke referensi ASLI. Kalau pemenangnya sudah
        jelas (metode 'orb'), langsung selesai. Kalau belum yakin, tahap 2
        menambahkan referensi augmentasi (uang lusuh/gelap/noise).
        Untuk evaluasi penuh tanpa tahap, pakai scores() + decide().
        """
        inl = {c: 0 for c in self.classes}
        best_ref = {c: "" for c in self.classes}
        q_pts = q_des = None
        if self.use_orb:
            q_pts, q_des, _ = self.extract(note_bgr, full_frame)
            self._accumulate(q_pts, q_des, [r for r in self.refs if not r.aug], inl, best_ref)
        csim = self._color_scores(note_bgr, full_frame)
        pred = self.decide(inl, csim, best_ref, full_frame)
        if not self.use_orb or pred.method == "orb":
            return pred
        self._accumulate(q_pts, q_des, [r for r in self.refs if r.aug], inl, best_ref)
        return self.decide(inl, csim, best_ref, full_frame)

    def _accumulate(self, q_pts, q_des, refs, inl, best_ref) -> None:
        for ref in refs:
            n = self._inliers(q_pts, q_des, ref)
            if n > inl[ref.denom]:
                inl[ref.denom], best_ref[ref.denom] = n, ref.name

    def _color_scores(self, note_bgr: np.ndarray, full_frame: bool) -> dict:
        csim = {c: 0.0 for c in self.classes}
        if self.use_color and not full_frame:
            qh = self.color_hist(note_bgr)
            for ref in self.refs:
                d = cv2.compareHist(qh, ref.hist, cv2.HISTCMP_BHATTACHARYYA)
                csim[ref.denom] = max(csim[ref.denom], 1.0 - float(d))
        return csim

    def decide(self, inl: dict, csim: dict, best_ref: dict | None = None,
               full_frame: bool = False, use_orb: bool | None = None,
               use_color: bool | None = None) -> Prediction:
        """Aturan keputusan (dipisah agar bisa diuji ulang / ablation tanpa
        menghitung ulang fitur)."""
        use_orb = self.use_orb if use_orb is None else use_orb
        use_color = (self.use_color if use_color is None else use_color) and not full_frame
        best_ref = best_ref or {}
        ranked = sorted(inl, key=inl.get, reverse=True)
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]
        n1, n2 = inl[top], inl[second]

        if use_orb and n1 >= C.MIN_INLIERS:
            if n1 >= C.MARGIN_RATIO * max(n2, 1):          # pemenang jelas
                conf = min(1.0, n1 / 60.0) * (1 - 0.5 * n2 / max(n1, 1))
                return Prediction(top, round(conf, 3), "orb", inl, csim, best_ref.get(top, ""))
            if use_color:                                   # dua kelas bersaing ketat
                comb = {c: (1 - C.COLOR_WEIGHT) * inl[c] / n1 + C.COLOR_WEIGHT * csim[c]
                        for c in inl}
                best = max(comb, key=comb.get)
                return Prediction(best, round(comb[best] * 0.6, 3), "orb+warna",
                                  inl, csim, best_ref.get(best, ""))
            return Prediction(C.UNKNOWN, 0.0, "-", inl, csim, "")

        if use_color:
            cr = sorted(csim, key=csim.get, reverse=True)
            c1, c2 = csim[cr[0]], csim[cr[1]]
            if not use_orb or (c1 >= C.COLOR_MIN_SIM and c1 - c2 >= C.COLOR_MARGIN):
                return Prediction(cr[0], round(c1 * 0.5, 3), "warna", inl, csim, "")

        return Prediction(C.UNKNOWN, 0.0, "-", inl, csim, "")