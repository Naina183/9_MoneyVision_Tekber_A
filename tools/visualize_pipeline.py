"""Menyimpan gambar tiap tahap pipeline (untuk laporan / slide).

    python tools/visualize_pipeline.py --image samples/scene_demo.jpg --out results/pipeline

Menghasilkan: 1_input, 2_mask, 3_kontur, 4_warp, 5_clahe_orb, 6_matching (+ gabungan).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moneyvision import config as C  # noqa: E402
from moneyvision.classifier import NoteClassifier  # noqa: E402
from moneyvision.detector import detect_notes  # noqa: E402
from moneyvision.preprocessing import canonical, enhance_gray, to_gray  # noqa: E402


def _title(img, text):
    out = cv2.copyMakeBorder(img, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(252, 252, 251))
    cv2.putText(out, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (11, 11, 11), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--refs", default="references")
    ap.add_argument("--out", default="results/pipeline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(args.image)
    clf = NoteClassifier(args.refs)
    dets, mask = detect_notes(img, allow_full_frame=False, return_mask=True)
    if not dets:
        sys.exit("Tidak ada uang terdeteksi pada gambar ini.")

    steps = {}
    steps["1_input"] = img
    steps["2_mask"] = cv2.cvtColor(cv2.resize(mask, (img.shape[1], img.shape[0])), cv2.COLOR_GRAY2BGR)
    cont = img.copy()
    for i, d in enumerate(dets, 1):
        q = d.quad.astype(np.int32)
        cv2.polylines(cont, [q], True, (0, 220, 0), 4, cv2.LINE_AA)
        for p in q:
            cv2.circle(cont, tuple(int(v) for v in p), 9, (0, 0, 255), -1)
        cv2.putText(cont, f"#{i}", tuple(int(v) for v in q[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 3, cv2.LINE_AA)
    steps["3_kontur"] = cont

    # tampilkan lembar dengan bukti ORB terkuat
    preds = [clf.classify(x.note) for x in dets]
    k = max(range(len(dets)), key=lambda i: max(preds[i].inliers.values()))
    d, pred = dets[k], preds[k]
    warp = canonical(d.note)
    steps["4_warp"] = warp
    gray = enhance_gray(to_gray(warp))
    kps = clf.orb.detect(gray, None)
    steps["5_clahe_orb"] = cv2.drawKeypoints(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), kps, None,
                                             color=(0, 140, 255))

    # matching dengan referensi terbaik
    ref_img = canonical(cv2.imread(os.path.join(args.refs, pred.best_ref))) if pred.best_ref else None
    if ref_img is not None:
        rg = enhance_gray(to_gray(ref_img))
        k1, d1 = clf.orb.detectAndCompute(gray, None)
        k2, d2 = clf.orb.detectAndCompute(rg, None)
        pairs = clf.matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < C.LOWE_RATIO * n.distance]
        src = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        _, inl = cv2.findHomography(src, dst, cv2.RANSAC, C.RANSAC_REPROJ)
        inl_matches = [m for m, ok in zip(good, inl.ravel()) if ok]
        steps["6_matching"] = cv2.drawMatches(warp, k1, ref_img, k2, inl_matches, None,
                                              matchColor=(0, 200, 0), singlePointColor=None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        rp = f"{pred.denom:,}".replace(",", ".")
        print(f"Lembar #{k + 1}: prediksi Rp{rp} | metode {pred.method} | referensi {pred.best_ref} | "
              f"{len(good)} match lolos ratio test, {len(inl_matches)} inlier RANSAC")
        print("Inlier per kelas:", pred.inliers)

    titles = {"1_input": "1. Input", "2_mask": "2. Mask segmentasi", "3_kontur": "3. Kontur & sudut",
              "4_warp": "4. Perspective warp", "5_clahe_orb": "5. Grayscale+CLAHE & keypoint ORB",
              "6_matching": "6. Inlier RANSAC vs referensi"}
    for k, v in steps.items():
        cv2.imwrite(os.path.join(args.out, f"{k}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # gabungan 2 baris x 3 kolom
    tiles = [_title(cv2.resize(steps[k], (640, int(640 * steps[k].shape[0] / steps[k].shape[1]))), titles[k])
             for k in steps]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(252, 252, 251))
             for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 252))
    rows = [np.hstack([cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(252, 252, 251))
                       for t in tiles[i:i + 3]]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(args.out, "pipeline_gabungan.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Disimpan di {args.out}/")


if __name__ == "__main__":
    main()
