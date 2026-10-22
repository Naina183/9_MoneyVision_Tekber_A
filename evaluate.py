"""Evaluasi MoneyVision (pengujian awal Week 8).

Tiga pengujian:
  A. Dataset A (scan, 700 citra) — klasifikasi per lembar pada citra NON-referensi
     (indeks 3–25) per kondisi: normal, lusuh, kotor, sobek, buram.
     Dataset ini juga dipakai untuk memilih parameter (data pengembangan).
  B. Dataset B (foto kamera, 120 citra, sumber berbeda) — pipeline lengkap
     (deteksi kontur -> klasifikasi). Tidak pernah dipakai untuk referensi atau
     tuning, sehingga menjadi uji independen. Kelas Rp75.000 (uang peringatan)
     tidak ada di referensi -> diuji sebagai "open-set": idealnya ditolak.
  C. Scene sintetis multi-uang — 1–4 lembar dari citra uji Dataset A ditempel
     acak di latar kertas putih / meja gelap / kayu / kain; menguji deteksi
     kontur, penghitungan jumlah lembar, dan total nominal.
Ditambah ablation: ORB saja vs warna saja vs ORB+warna (metode usulan).

Contoh:
    python evaluate.py                       # lengkap (± 10 menit di laptop biasa)
    python evaluate.py --limit-a 100 --scenes 30   # cepat
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from moneyvision import config as C  # noqa: E402
from moneyvision.app import NoteResult, draw_panel, draw_results  # noqa: E402
from moneyvision.classifier import NoteClassifier  # noqa: E402
from moneyvision.detector import detect_notes  # noqa: E402
from moneyvision.money import TransactionSession, format_rupiah  # noqa: E402
from synth import make_scene  # noqa: E402

CONDITIONS = ["normal", "lusuh", "kotor", "sobek", "buram"]
NAME_RE = re.compile(r"^(\d+k)_([db])_(\d{4})_(\d+)\.(png|jpg|jpeg)$", re.I)
B_MAP = {"1RIBU": 1000, "2RIBU": 2000, "5RIBU": 5000, "10RIBU": 10000, "20RIBU": 20000,
         "50RIBU": 50000, "100RIBU": 100000, "75RIBU": 75000}
VARIANTS = {
    "ORB+warna (usulan)": dict(),
    "ORB saja": dict(use_color=False),
    "Warna saja": dict(use_orb=False),
}


# --------------------------------------------------------------------------- data
def collect_a(root: str, exclude_idx=(1, 2)):
    items = []
    for denom in C.DENOMINATIONS:
        d = os.path.join(root, str(denom))
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            m = NAME_RE.match(fn)
            if not m or int(m.group(4)) in exclude_idx:
                continue
            idx = int(m.group(4))
            items.append(dict(path=os.path.join(d, fn), y=denom, side=m.group(2),
                              year=m.group(3), cond=CONDITIONS[(idx - 1) // 5]))
    return items


def collect_b(root: str):
    items = []
    for sub, denom in B_MAP.items():
        d = os.path.join(root, sub)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append(dict(path=os.path.join(d, fn), y=denom))
    return items


# --------------------------------------------------------------------------- metrik
def summarize(rows, key_true="y", key_pred="pred"):
    n = len(rows)
    correct = sum(r[key_pred] == r[key_true] for r in rows)
    rejected = sum(r[key_pred] == C.UNKNOWN for r in rows)
    wrong = n - correct - rejected
    acc_known = correct / max(correct + wrong, 1)
    return dict(n=n, benar=correct, salah=wrong, ditolak=rejected,
                akurasi=round(correct / max(n, 1), 4), error=round(wrong / max(n, 1), 4),
                tolak=round(rejected / max(n, 1), 4), presisi_diterima=round(acc_known, 4))


def confusion(rows, labels):
    cols = labels + [C.UNKNOWN]
    M = np.zeros((len(labels), len(cols)), int)
    for r in rows:
        if r["y"] in labels:
            M[labels.index(r["y"]), cols.index(r["pred"])] += 1
    return M, cols


def save_confusion_png(M, rows_lbl, cols_lbl, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    lbl = lambda v: "Tidak\ndikenali" if v == C.UNKNOWN else f"{v // 1000}rb"  # noqa: E731
    norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(cols_lbl)), [lbl(c) for c in cols_lbl], fontsize=8.5, color="#52514e")
    ax.set_yticks(range(len(rows_lbl)), [lbl(c) for c in rows_lbl], fontsize=8.5, color="#52514e")
    ax.set_xlabel("Prediksi", color="#52514e")
    ax.set_ylabel("Nominal sebenarnya", color="#52514e")
    ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_condition_png(per_cond: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_cond)
    vals = [per_cond[k]["akurasi"] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(names, vals, color="#2a78d6", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                fontsize=9, color="#0b0b0b")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Akurasi (%)", color="#52514e")
    ax.set_title("Akurasi klasifikasi per kondisi uang (Dataset A, citra non-referensi)",
                 fontsize=10.5, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b8b7b2")
    ax.tick_params(colors="#52514e", length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- uji
def run_test_a(clf, items, out):
    print(f"\n[A] Dataset A: {len(items)} citra non-referensi")
    t0 = time.perf_counter()
    for k, it in enumerate(items, 1):
        img = cv2.imread(it["path"])
        ts = time.perf_counter()
        it["inl"], it["csim"], it["best_ref"] = clf.scores(img)
        it["ms"] = (time.perf_counter() - ts) * 1000
        if k % 50 == 0:
            print(f"    {k}/{len(items)}")
    print(f"    selesai dalam {time.perf_counter() - t0:.0f} s")
    return items


def run_test_b(clf, items):
    print(f"\n[B] Dataset B: {len(items)} foto (pipeline lengkap)")
    for it in items:
        img = cv2.imread(it["path"])
        ts = time.perf_counter()
        dets = detect_notes(img)
        det = max(dets, key=lambda d: d.area_frac)
        it["full_frame"] = det.full_frame
        it["n_det"] = 0 if det.full_frame else len(dets)
        it["inl"], it["csim"], it["best_ref"] = clf.scores(det.note, det.full_frame)
        it["ms"] = (time.perf_counter() - ts) * 1000
        it["det"] = det
    return items


def _center_in(quad, poly):
    c = tuple(map(float, quad.mean(axis=0)))
    return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), c, False) >= 0


def run_test_c(clf, a_items, n_scenes, seed, out_dir):
    print(f"\n[C] {n_scenes} scene sintetis multi-uang")
    rng = np.random.default_rng(seed)
    kinds = ["kertas_putih", "meja_gelap", "kayu", "kain"]
    scenes = []
    n_fail = 0
    ex_dir = os.path.join(out_dir, "contoh_output")
    os.makedirs(ex_dir, exist_ok=True)
    for s in range(n_scenes):
        k = int(rng.integers(1, 5))
        pick = rng.choice(len(a_items), k, replace=False)
        notes = [(a_items[j]["y"], cv2.imread(a_items[j]["path"])) for j in pick]
        kind = kinds[s % len(kinds)]
        scene, gts = make_scene(notes, rng, kind)
        ts = time.perf_counter()
        dets = detect_notes(scene, allow_full_frame=False)
        preds = []
        for d in dets:
            inl, csim, br = clf.scores(d.note, d.full_frame)
            preds.append(clf.decide(inl, csim, br, d.full_frame))
        ms = (time.perf_counter() - ts) * 1000
        # pencocokan deteksi <-> ground truth (pusat deteksi di dalam poligon GT)
        matched_gt, fp = set(), 0
        per_note = []
        for d, p in zip(dets, preds):
            hit = next((i for i, g in enumerate(gts) if i not in matched_gt and _center_in(d.quad, g["quad"])), None)
            if hit is None:
                fp += 1
                continue
            matched_gt.add(hit)
            per_note.append((gts[hit]["denom"], p.denom))
        true_total = sum(g["denom"] for g in gts)
        pred_total = sum(p.denom for p in preds if p.known)
        scenes.append(dict(scene=s, latar=kind, n_uang=len(gts), n_deteksi=len(dets),
                           fp=fp, terdeteksi=len(matched_gt),
                           dikenali_benar=sum(1 for t, p in per_note if t == p),
                           salah=sum(1 for t, p in per_note if p not in (t, C.UNKNOWN)),
                           ditolak=sum(1 for t, p in per_note if p == C.UNKNOWN),
                           total_benar=true_total, total_prediksi=pred_total,
                           total_tepat=int(true_total == pred_total), ms=round(ms, 1)))
        failed = true_total != pred_total
        if s < 8 or (failed and n_fail < 6):  # simpan contoh output (dan contoh gagal)
            n_fail += int(failed and s >= 8)
            sess = TransactionSession()
            sess.add_scan([p.denom for p in preds if p.known])
            res = [NoteResult(d, p) for d, p in zip(dets, preds)]
            status = (f"Terdeteksi {len(dets)} lembar. Total {format_rupiah(pred_total)} "
                      f"(ground truth {format_rupiah(true_total)}).")
            view = draw_panel(draw_results(scene, res), sess, status, extra=f"proses {ms:.0f} ms")
            tag = "_GAGAL" if failed else ""
            cv2.imwrite(os.path.join(ex_dir, f"scene_{s:02d}_{kind}{tag}.jpg"), view,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        if (s + 1) % 20 == 0:
            print(f"    {s + 1}/{n_scenes}")
    return scenes


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-a", default="data/datasetUang")
    ap.add_argument("--dataset-b", default="data/DATASET_UANG")
    ap.add_argument("--refs", default="references")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--limit-a", type=int, default=0, help="batasi jumlah citra uji A (0 = semua)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clf = NoteClassifier(args.refs)
    print(f"Referensi: {len(clf.refs)} citra, {len(clf.classes)} kelas, ORB {C.ORB_FEATURES} fitur")
    summary = dict(referensi=len(clf.refs), orb_features=C.ORB_FEATURES,
                   parameter=dict(MIN_INLIERS=C.MIN_INLIERS, MARGIN_RATIO=C.MARGIN_RATIO,
                                  LOWE_RATIO=C.LOWE_RATIO, COLOR_WEIGHT=C.COLOR_WEIGHT))

    # ---------------- A
    a_items = collect_a(args.dataset_a)
    if args.limit_a:
        rng = np.random.default_rng(args.seed)
        a_items = [a_items[i] for i in sorted(rng.choice(len(a_items), args.limit_a, replace=False))]
    run_test_a(clf, a_items, args.out)
    summary["A"] = {}
    for vname, kw in VARIANTS.items():
        rows = [dict(it, pred=clf.decide(it["inl"], it["csim"], it["best_ref"], **kw).denom)
                for it in a_items]
        s = summarize(rows)
        s["per_kondisi"] = {c: summarize([r for r in rows if r["cond"] == c]) for c in CONDITIONS}
        summary["A"][vname] = s
        if vname.endswith("(usulan)"):
            a_rows = rows
    for it, r in zip(a_items, a_rows):
        p = clf.decide(it["inl"], it["csim"], it["best_ref"])
        r.update(metode=p.method, conf=p.confidence, inlier_terbaik=max(it["inl"].values()),
                 ms=round(it["ms"], 1), file=os.path.basename(it["path"]))
    write_csv(os.path.join(args.out, "uji_A_dataset_scan.csv"), a_rows,
              ["file", "y", "side", "year", "cond", "pred", "metode", "conf", "inlier_terbaik", "ms"])
    M, cols = confusion(a_rows, clf.classes)
    save_confusion_png(M, clf.classes, cols, "Confusion matrix — Dataset A (scan, non-referensi)",
                       os.path.join(args.out, "confusion_A.png"))
    summary["A"]["confusion"] = dict(labels=clf.classes, cols=cols, matrix=M.tolist())
    save_condition_png(summary["A"]["ORB+warna (usulan)"]["per_kondisi"],
                       os.path.join(args.out, "akurasi_per_kondisi_A.png"))
    summary["A"]["ms_per_lembar"] = round(float(np.mean([it["ms"] for it in a_items])), 1)

    # ---------------- B
    b_items = run_test_b(clf, collect_b(args.dataset_b))
    summary["B"] = {}
    in_scope = [it for it in b_items if it["y"] in clf.classes]
    open_set = [it for it in b_items if it["y"] not in clf.classes]
    for vname, kw in VARIANTS.items():
        rows = [dict(it, pred=clf.decide(it["inl"], it["csim"], it["best_ref"], it["full_frame"], **kw).denom)
                for it in in_scope]
        summary["B"][vname] = summarize(rows)
        if vname.endswith("(usulan)"):
            b_rows = rows
    for it, r in zip(in_scope, b_rows):
        p = clf.decide(it["inl"], it["csim"], it["best_ref"], it["full_frame"])
        r.update(metode=p.method, conf=p.confidence, inlier_terbaik=max(it["inl"].values()),
                 ms=round(it["ms"], 1), file="/".join(it["path"].split(os.sep)[-2:]))
    os_preds = [clf.decide(it["inl"], it["csim"], it["best_ref"], it["full_frame"]).denom for it in open_set]
    summary["B"]["open_set_75rb"] = dict(n=len(open_set),
                                         ditolak=sum(p == C.UNKNOWN for p in os_preds),
                                         salah_diterima=dict(Counter(p for p in os_preds if p != C.UNKNOWN)))
    summary["B"]["kontur_terdeteksi"] = dict(n=len(in_scope),
                                             kontur=sum(not it["full_frame"] for it in in_scope),
                                             fallback_full_frame=sum(it["full_frame"] for it in in_scope))
    summary["B"]["ms_per_foto"] = round(float(np.mean([it["ms"] for it in in_scope])), 1)
    write_csv(os.path.join(args.out, "uji_B_foto_kamera.csv"), b_rows,
              ["file", "y", "pred", "metode", "conf", "inlier_terbaik", "full_frame", "ms"])
    M, cols = confusion(b_rows, clf.classes)
    save_confusion_png(M, clf.classes, cols, "Confusion matrix — Dataset B (foto kamera, uji independen)",
                       os.path.join(args.out, "confusion_B.png"))
    summary["B"]["confusion"] = dict(labels=clf.classes, cols=cols, matrix=M.tolist())

    # ---------------- C
    scenes = run_test_c(clf, a_items, args.scenes, args.seed, args.out)
    write_csv(os.path.join(args.out, "uji_C_scene_sintetis.csv"), scenes, list(scenes[0].keys()))
    n_notes = sum(s["n_uang"] for s in scenes)
    by_bg = defaultdict(list)
    for s in scenes:
        by_bg[s["latar"]].append(s)
    summary["C"] = dict(
        scene=len(scenes), lembar=n_notes,
        jumlah_lembar_tepat=round(np.mean([s["n_deteksi"] == s["n_uang"] for s in scenes]), 4),
        recall_deteksi=round(sum(s["terdeteksi"] for s in scenes) / n_notes, 4),
        false_positive=sum(s["fp"] for s in scenes),
        akurasi_per_lembar=round(sum(s["dikenali_benar"] for s in scenes) / n_notes, 4),
        lembar_terdeteksi=sum(s["terdeteksi"] for s in scenes),
        terdeteksi_benar=sum(s["dikenali_benar"] for s in scenes),
        terdeteksi_salah=sum(s["salah"] for s in scenes),
        terdeteksi_ditolak=sum(s["ditolak"] for s in scenes),
        total_tepat=round(np.mean([s["total_tepat"] for s in scenes]), 4),
        mae_total=round(float(np.mean([abs(s["total_benar"] - s["total_prediksi"]) for s in scenes])), 1),
        ms_per_scene=round(float(np.mean([s["ms"] for s in scenes])), 1),
        per_latar={k: dict(scene=len(v),
                           lembar=sum(s["n_uang"] for s in v),
                           terdeteksi=sum(s["terdeteksi"] for s in v),
                           benar=sum(s["dikenali_benar"] for s in v),
                           jumlah_lembar_tepat=round(np.mean([s["n_deteksi"] == s["n_uang"] for s in v]), 3),
                           total_tepat=round(np.mean([s["total_tepat"] for s in v]), 3))
                   for k, v in by_bg.items()},
    )

    with open(os.path.join(args.out, "ringkasan.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=int)
    print_report(summary)


def print_report(s):
    pct = lambda v: f"{v * 100:5.1f}%"  # noqa: E731
    print("\n================ RINGKASAN ================")
    for test, label in (("A", "Dataset A (scan)"), ("B", "Dataset B (foto)")):
        print(f"\n{label}")
        for v in VARIANTS:
            m = s[test][v]
            print(f"  {v:<20} akurasi {pct(m['akurasi'])}  salah {pct(m['error'])}  "
                  f"ditolak {pct(m['tolak'])}  (n={m['n']})")
    print("\nPer kondisi (A, metode usulan):")
    for c, m in s["A"]["ORB+warna (usulan)"]["per_kondisi"].items():
        print(f"  {c:<7} akurasi {pct(m['akurasi'])}  salah {pct(m['error'])}  ditolak {pct(m['tolak'])}  (n={m['n']})")
    print(f"\nOpen-set Rp75.000: {s['B']['open_set_75rb']}")
    print(f"Kontur Dataset B: {s['B']['kontur_terdeteksi']}")
    c = s["C"]
    print(f"\nScene sintetis: {c['scene']} scene / {c['lembar']} lembar")
    print(f"  jumlah lembar tepat {pct(c['jumlah_lembar_tepat'])}  recall deteksi {pct(c['recall_deteksi'])}  "
          f"FP {c['false_positive']}")
    print(f"  dari {c['lembar_terdeteksi']} lembar terdeteksi: benar {c['terdeteksi_benar']}, "
          f"salah {c['terdeteksi_salah']}, ditolak {c['terdeteksi_ditolak']}")
    print(f"  akurasi per lembar {pct(c['akurasi_per_lembar'])}  total tepat {pct(c['total_tepat'])}  "
          f"MAE total Rp{c['mae_total']:.0f}")
    print(f"\nWaktu: {s['A']['ms_per_lembar']} ms/lembar (A), {s['B']['ms_per_foto']} ms/foto (B), "
          f"{c['ms_per_scene']} ms/scene (C)")


if __name__ == "__main__":
    main()
