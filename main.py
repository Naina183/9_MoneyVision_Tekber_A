"""MoneyVision — pengenal nominal uang kertas rupiah dengan umpan balik suara.

Contoh pemakaian:
    python main.py                          # webcam 0, pindai dengan SPASI
    python main.py --camera 1 --auto        # webcam 1, pindai otomatis saat uang diam
    python main.py --image contoh.jpg --price 50000
    python main.py --video demo.mp4
    python main.py --tts gtts               # suara Bahasa Indonesia via Google TTS

Kontrol keyboard (jendela OpenCV harus aktif):
    SPASI pindai & tambahkan ke total   H  input harga (angka / V untuk suara)
    K     umumkan kembalian             C  ganti mode bayar <-> cek kembalian
    U     batalkan pindaian terakhir    R  reset transaksi
    S     ulangi pesan suara terakhir   P  simpan screenshot
    Q/Esc keluar
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from moneyvision.app import MoneyVisionApp, analyze, draw_panel, draw_results
from moneyvision.audio import Speaker
from moneyvision.classifier import NoteClassifier
from moneyvision.detector import detect_notes
from moneyvision.money import format_rupiah, msg_change

WINDOW = "MoneyVision"


def run_image(app: MoneyVisionApp, path: str, price: int | None, show: bool, save: str | None):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Gagal membaca gambar: {path}")
    results = app.scan(frame)
    if price:
        app.set_price(price)
    print(f"\nHasil ({app.last_ms:.0f} ms):")
    for i, r in enumerate(results, 1):
        p = r.pred
        print(f"  {i}. {format_rupiah(p.denom) if p.known else 'tidak dikenali':>14}  "
              f"metode={p.method:<9} conf={p.confidence:.2f}  inlier_terbaik={max(p.inliers.values())}")
    print(f"  Total: {format_rupiah(app.session.total_paid)}")
    if price:
        print("  " + msg_change(app.session))
    view = draw_panel(draw_results(frame, results), app.session, app.status,
                      extra=f"proses {app.last_ms:.0f} ms")
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        cv2.imwrite(save, view)
        print(f"[i] output disimpan: {save}")
    if show:
        cv2.imshow(WINDOW, view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    time.sleep(0.5)  # beri waktu TTS


def run_stream(app: MoneyVisionApp, source, is_video: bool):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Gagal membuka sumber video/kamera: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    app.say("MoneyVision siap. Letakkan uang di depan kamera lalu tekan spasi.")
    delay = 30 if is_video else 1
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_video:  # ulangi video dari awal
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        dets = detect_notes(frame, allow_full_frame=False)
        if app.auto:
            app.auto_step(frame, dets)
        show_res = (time.time() - app.last_scan_t) < 4.0
        view = app.render(frame, None if show_res else dets, show_results=show_res)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(delay)
        if not app.handle_key(key & 0xFF if key >= 0 else -1, frame):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="indeks webcam (default 0)")
    src.add_argument("--image", help="path gambar input")
    src.add_argument("--video", help="path video input")
    ap.add_argument("--refs", default="references", help="folder citra referensi")
    ap.add_argument("--price", type=int, help="harga barang (opsional, mode gambar)")
    ap.add_argument("--tts", default="auto", choices=["auto", "pyttsx3", "gtts", "none"])
    ap.add_argument("--auto", action="store_true", help="pindai otomatis saat uang diam")
    ap.add_argument("--save", help="simpan output gambar ke path ini (mode gambar)")
    ap.add_argument("--no-window", action="store_true", help="jangan tampilkan jendela")
    args = ap.parse_args()

    print("[i] memuat citra referensi ...")
    clf = NoteClassifier(args.refs)
    print(f"[i] {len(clf.refs)} referensi dimuat untuk nominal: "
          + ", ".join(format_rupiah(c) for c in clf.classes))
    speaker = Speaker(args.tts)
    app = MoneyVisionApp(clf, speaker, auto=args.auto)
    try:
        if args.image:
            run_image(app, args.image, args.price, not args.no_window, args.save)
        else:
            run_stream(app, args.video if args.video else args.camera, bool(args.video))
    finally:
        speaker.close()


if __name__ == "__main__":
    main()
    