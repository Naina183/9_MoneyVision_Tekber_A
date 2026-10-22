"""Membangun basis citra referensi dari dataset.

Mengambil citra uang KONDISI NORMAL (indeks 1–2) dari Dataset A untuk setiap
nominal, sisi (depan/belakang) dan tahun emisi (2016/2022), lalu menyimpannya
ke references/<nominal>/ dalam ukuran kanonik.

Contoh:
    python build_references.py --dataset data/datasetUang --out references

Citra referensi tambahan (mis. foto webcam kelompok sendiri) cukup disalin ke
references/<nominal>/ — classifier otomatis memuat semua citra di folder itu.
"""
from __future__ import annotations

import argparse
import os
import re

import cv2

from moneyvision import config as C
from moneyvision.preprocessing import canonical

# Nama file Dataset A: <nominal>_<d|b>_<tahun>_<indeks>.png
#   d = depan, b = belakang; indeks 1–5 normal, 6–10 lusuh, 11–15 kotor,
#   16–20 sobek, 21–25 buram
NAME_RE = re.compile(r"^(\d+k)_([db])_(\d{4})_(\d+)\.(png|jpg|jpeg)$", re.I)
REF_INDICES = {1, 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/datasetUang",
                    help="folder Dataset A (berisi subfolder 1000, 2000, ...)")
    ap.add_argument("--out", default="references")
    args = ap.parse_args()

    total = 0
    for denom in C.DENOMINATIONS:
        src_dir = os.path.join(args.dataset, str(denom))
        if not os.path.isdir(src_dir):
            print(f"[!] folder {src_dir} tidak ditemukan, dilewati")
            continue
        dst_dir = os.path.join(args.out, str(denom))
        os.makedirs(dst_dir, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(src_dir)):
            m = NAME_RE.match(fn)
            if not m or int(m.group(4)) not in REF_INDICES:
                continue
            img = cv2.imread(os.path.join(src_dir, fn))
            if img is None:
                continue
            out = os.path.join(dst_dir, os.path.splitext(fn)[0] + ".jpg")
            cv2.imwrite(out, canonical(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        print(f"Rp{denom:>7,}: {n} referensi".replace(",", "."))
        total += n
    print(f"Selesai: {total} citra referensi di '{args.out}'")


if __name__ == "__main__":
    main()
