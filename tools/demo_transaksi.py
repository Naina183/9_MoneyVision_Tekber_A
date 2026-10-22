"""Simulasi satu transaksi lengkap tanpa webcam (bukti alur & bahan screenshot).

    python tools/demo_transaksi.py

Langkah: pindai uang bayar -> input harga -> umumkan kembalian -> mode cek
kembalian -> pindai uang kembalian. Setiap langkah disimpan sebagai gambar
di results/demo_transaksi/ dan semua kalimat suara dicetak ke konsol.
"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moneyvision.app import MoneyVisionApp  # noqa: E402
from moneyvision.audio import Speaker  # noqa: E402
from moneyvision.classifier import NoteClassifier  # noqa: E402

OUT = "results/demo_transaksi"


def main():
    os.makedirs(OUT, exist_ok=True)
    app = MoneyVisionApp(NoteClassifier("references"), Speaker("none"), save_dir=OUT)
    bayar = cv2.imread("samples/scene_demo_meja.jpg")        # 50.000 + 20.000 + 5.000
    kembali = cv2.imread("samples/scene_kembalian.jpg")      # 10.000 + 5.000

    def step(name, frame, keys):
        for k in keys:
            app.handle_key(ord(k) if isinstance(k, str) else k, frame)
        app.render(frame)
        app.save_view(name)

    step("1_pindai_bayar.jpg", bayar, [32])
    step("2_input_harga.jpg", bayar, ["h", "6", "0", "0", "0", "0"])
    step("3_harga_tersimpan.jpg", bayar, [13])
    step("4_umumkan_kembalian.jpg", bayar, ["k"])
    app.last_results = []
    step("5_mode_cek_kembalian.jpg", kembali, ["c"])
    step("6_pindai_kembalian.jpg", kembali, [32])
    print(f"Total dibayar Rp{app.session.total_paid}, harga Rp{app.session.price}, "
          f"kembalian seharusnya Rp{app.session.expected_change}, diterima Rp{app.session.change_received}")


if __name__ == "__main__":
    main()
