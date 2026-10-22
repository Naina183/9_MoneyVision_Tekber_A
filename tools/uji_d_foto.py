"""Uji D — foto ponsel nyata (dataset mkaspulanwar/rupiah-banknotes-dataset, lisensi MIT).

Foto diambil dengan kamera ponsel (4080×3060) di latar putih; banyak uang yang
sengaja dilipat, digulung, diremas, atau sebagian keluar dari bingkai.
Skrip mengambil N foto acak per nominal (seed tetap) lalu menjalankan pipeline
lengkap aplikasi (deteksi kontur -> klasifikasi), sama seperti mode --image.

    python tools/uji_d_foto.py --dataset data/rupiah-banknotes-dataset --per-class 15

Hasil: results/uji_D_foto_ponsel.csv dan kunci "D" di results/ringkasan.json.
Kategori per foto (tiap foto berisi satu lembar):
  benar            tepat satu nominal dikenali dan nominalnya benar
  salah            ada nominal yang disebut keliru (kegagalan paling berbahaya)
  ganda            nominal benar tetapi satu lembar terhitung lebih dari sekali
  ditolak          uang terdeteksi tetapi dijawab "tidak dikenali"
  tidak_terdeteksi tidak ada uang yang ditemukan
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moneyvision import config as C  # noqa: E402
from moneyvision.app import analyze  # noqa: E402
from moneyvision.classifier import NoteClassifier  # noqa: E402
from moneyvision.preprocessing import resize_to_width  # noqa: E402

OUTCOMES = ["benar", "salah", "ganda", "ditolak", "tidak_terdeteksi"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/rupiah-banknotes-dataset")
    ap.add_argument("--refs", default="references")
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    clf = NoteClassifier(args.refs)
    rng = random.Random(args.seed)
    rows = []
    for denom in C.DENOMINATIONS:
        files = sorted(glob.glob(os.path.join(args.dataset, f"idr_{denom}", "*.jpg")))
        if not files:
            sys.exit(f"Folder idr_{denom} tidak ditemukan di {args.dataset}")
        for path in rng.sample(files, min(args.per_class, len(files))):
            img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)   # 4080x3060 -> 2040x1530
            img, _ = resize_to_width(img, 1600)
            results, ms = analyze(img, clf)
            known = [r.pred.denom for r in results if r.pred.known]
            unknown = sum(1 for r in results if not r.pred.known)
            if any(k != denom for k in known):
                out = "salah"
            elif len(known) == 1:
                out = "benar"
            elif len(known) > 1:
                out = "ganda"        # satu lembar terhitung lebih dari sekali
            elif unknown:
                out = "ditolak"
            else:
                out = "tidak_terdeteksi"
            rows.append(dict(file=f"idr_{denom}/{os.path.basename(path)}", y=denom,
                             prediksi=" ".join(str(k) for k in known), tidak_dikenali=unknown,
                             full_frame=int(any(r.det.full_frame for r in results)),
                             hasil=out, ms=round(ms)))
        print(f"Rp{denom}: {dict(Counter(r['hasil'] for r in rows if r['y'] == denom))}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "uji_D_foto_ponsel.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = Counter(r["hasil"] for r in rows)
    summary = dict(
        n=len(rows), per_kelas=args.per_class, seed=args.seed,
        **{k: total.get(k, 0) for k in OUTCOMES},
        akurasi=round(total.get("benar", 0) / len(rows), 4),
        per_nominal={d: {k: sum(1 for r in rows if r["y"] == d and r["hasil"] == k) for k in OUTCOMES}
                     for d in C.DENOMINATIONS},
        kasus_salah=[(r["file"], r["prediksi"]) for r in rows if r["hasil"] == "salah"],
        ms_per_foto=round(sum(r["ms"] for r in rows) / len(rows)),
    )
    path = os.path.join(args.out, "ringkasan.json")
    data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    data["D"] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nUji D: {len(rows)} foto -> " + ", ".join(f"{k} {total.get(k, 0)}" for k in OUTCOMES)
          + f" (akurasi {summary['akurasi'] * 100:.1f}%)")
    print("Kasus salah:", summary["kasus_salah"])


if __name__ == "__main__":
    main()
