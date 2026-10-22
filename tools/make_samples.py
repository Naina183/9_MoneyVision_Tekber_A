"""Membuat gambar contoh di folder samples/ (untuk mencoba aplikasi tanpa webcam).

    python tools/make_samples.py
"""
from __future__ import annotations

import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_scene  # noqa: E402

A = "data/datasetUang"
B = "data/DATASET_UANG"
OUT = "samples"

SCENES = {
    # nama file: (latar, [(nominal, nama file Dataset A)], seed)
    "scene_demo_meja.jpg": ("meja_gelap", [(50000, "50k_d_2022_4.png"), (20000, "20k_b_2016_8.png"),
                                           (5000, "5k_d_2016_3.png")], 7),
    "scene_demo_kertas.jpg": ("kertas_putih", [(100000, "100k_d_2016_5.png"),
                                               (2000, "2k_b_2022_7.png")], 11),
    "scene_demo_kayu.jpg": ("kayu", [(10000, "10k_d_2022_3.png"), (10000, "10k_b_2016_9.png"),
                                     (1000, "1k_d_2022_4.png"), (50000, "50k_b_2016_23.png")], 3),
    # uang kembalian Rp15.000 untuk demo mode cek kembalian (harga 60.000, bayar 75.000)
    "scene_kembalian.jpg": ("kain", [(10000, "10k_b_2022_5.png"), (5000, "5k_b_2022_8.png")], 5),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (bg, notes, seed) in SCENES.items():
        imgs = [(d, cv2.imread(os.path.join(A, str(d), fn))) for d, fn in notes]
        scene, gts = make_scene(imgs, np.random.default_rng(seed), bg)
        cv2.imwrite(os.path.join(OUT, name), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
        total = sum(g["denom"] for g in gts)
        print(f"{name}: {len(gts)} lembar, total Rp" + f"{total:,}".replace(",", "."))
    for sub, fn in (("20RIBU", "3.jpg"), ("100RIBU", "7.jpg")):
        dst = os.path.join(OUT, f"foto_{sub.lower()}_{fn}")
        shutil.copy(os.path.join(B, sub, fn), dst)
        print(f"{dst}: salinan foto Dataset B")


if __name__ == "__main__":
    main()
