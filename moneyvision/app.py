"""Inti aplikasi: analisis frame, sesi transaksi, kontrol keyboard, dan tampilan."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from .classifier import NoteClassifier, Prediction
from .detector import Detection, detect_notes
from .money import (MODE_BAYAR, MODE_KEMBALIAN, TransactionSession, format_rupiah,
                    msg_change, msg_scan, msg_verify, rupiah_words)

GREEN, ORANGE, GRAY, WHITE = (60, 200, 60), (0, 165, 255), (160, 160, 160), (255, 255, 255)
HELP = "[SPASI] pindai  [H] harga  [K] kembalian  [C] mode cek kembalian  [U] batal  [R] reset  [S] ulangi  [P] screenshot  [Q] keluar"


@dataclass
class NoteResult:
    det: Detection
    pred: Prediction


def analyze(frame: np.ndarray, clf: NoteClassifier) -> tuple[list[NoteResult], float]:
    """Deteksi kontur + klasifikasi setiap uang. Mengembalikan (hasil, waktu ms)."""
    t0 = time.perf_counter()
    results = []
    for det in detect_notes(frame):
        pred = clf.classify(det.note, full_frame=det.full_frame)
        if det.full_frame and not pred.known:
            continue  # fallback seluruh frame tanpa bukti kuat = tidak ada uang
        results.append(NoteResult(det, pred))
    return results, (time.perf_counter() - t0) * 1000


def draw_results(frame: np.ndarray, results: list[NoteResult]) -> np.ndarray:
    out = frame.copy()
    th = max(2, frame.shape[1] // 400)
    for r in results:
        color = GREEN if r.pred.known else ORANGE
        quad = r.det.quad.astype(np.int32)
        cv2.polylines(out, [quad], True, color, th, cv2.LINE_AA)
        label = (f"{format_rupiah(r.pred.denom)} ({r.pred.method} {r.pred.confidence:.2f})"
                 if r.pred.known else "tidak dikenali")
        x, y = int(quad[:, 0].min()), int(quad[:, 1].min())
        _label(out, label, (x, max(28, y - 8)), color, scale=max(0.6, frame.shape[1] / 1400))
    return out


def draw_contours(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        if not d.full_frame:
            cv2.polylines(out, [d.quad.astype(np.int32)], True, GRAY, 2, cv2.LINE_AA)
    return out


def _label(img, text, org, color, scale=0.7):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y - h - b - 4), (x + w + 8, y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)


def draw_panel(img: np.ndarray, s: TransactionSession, status: str,
               input_buf: str | None = None, extra: str = "") -> np.ndarray:
    """Panel informasi di bagian bawah frame."""
    h, w = img.shape[:2]
    scale = max(0.5, w / 1700)
    lh = int(34 * scale / 0.75)
    lines = [
        f"MoneyVision | Mode: {'BAYAR' if s.mode == MODE_BAYAR else 'CEK KEMBALIAN'}"
        + (f" | {extra}" if extra else ""),
        f"Total dibayar: {format_rupiah(s.total_paid)}   Harga: {format_rupiah(s.price)}   "
        + (f"Uang kurang: {format_rupiah(-s.expected_change)}" if (s.expected_change or 0) < 0
           else f"Kembalian seharusnya: {format_rupiah(s.expected_change)}"),
    ]
    if s.mode == MODE_KEMBALIAN:
        lines.append(f"Kembalian diterima: {format_rupiah(s.change_received)}")
    if input_buf is not None:
        lines.append(f"Input harga: Rp {input_buf}_   [Enter] simpan  [V] suara  [Esc] batal")
    lines.extend(_wrap(status, 100, 3))
    lines.append(HELP)
    ph = lh * len(lines) + 12
    canvas = np.zeros((h + ph, w, 3), np.uint8)
    canvas[:h] = img
    canvas[h:] = (32, 32, 32)
    for i, t in enumerate(lines):
        sc = scale * (0.8 if t is HELP else 1.0)
        col = (200, 200, 200) if t is HELP else WHITE
        cv2.putText(canvas, t, (10, h + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                    1 if t is HELP else 2, cv2.LINE_AA)
    return canvas


def _wrap(t: str, width: int, max_lines: int) -> list[str]:
    """Memecah teks status menjadi beberapa baris (tanpa memotong kata)."""
    out, cur = [], ""
    for w in t.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    if len(out) > max_lines:
        out = out[:max_lines]
        out[-1] = out[-1][: width - 3] + "..."
    return out


class MoneyVisionApp:
    """Menghubungkan kamera/gambar, classifier, sesi transaksi, dan suara."""

    def __init__(self, clf: NoteClassifier, speaker, save_dir: str = "results/screenshots",
                 auto: bool = False):
        self.clf = clf
        self.speaker = speaker
        self.session = TransactionSession()
        self.save_dir = save_dir
        self.auto = auto
        self.status = "Arahkan uang ke kamera lalu tekan SPASI."
        self.last_msg = self.status
        self.last_results: list[NoteResult] = []
        self.last_ms = 0.0
        self.last_scan_t = 0.0
        self.input_buf: str | None = None
        self._stable = 0
        self._armed = True
        self._last_view = None

    # ------------------------------------------------------------ aksi
    def say(self, msg: str) -> None:
        self.status = self.last_msg = msg
        self.speaker.say(msg)

    def scan(self, frame: np.ndarray) -> list[NoteResult]:
        results, ms = analyze(frame, self.clf)
        self.last_results, self.last_ms = results, ms
        self.last_scan_t = time.time()
        notes = [r.pred.denom for r in results if r.pred.known]
        unknown = sum(1 for r in results if not r.pred.known)
        self.session.add_scan(notes)
        self.say(msg_scan(notes, unknown, self.session))
        return results

    def set_price(self, price: int | None) -> None:
        if price is None or price <= 0:
            self.say("Harga tidak valid, silakan ulangi.")
            return
        self.session.price = int(price)
        self.say(msg_change(self.session) if self.session.total_paid
                 else f"Harga diatur {rupiah_words(price)}. Silakan pindai uang Anda.")

    def handle_key(self, key: int, frame: np.ndarray) -> bool:
        """Memproses tombol. Mengembalikan False bila aplikasi harus berhenti."""
        if key < 0:
            return True
        if self.input_buf is not None:
            return self._handle_price_key(key)
        ch = chr(key).lower() if key < 256 else ""
        if ch == "q" or key == 27:
            return False
        if key == 32:
            self.scan(frame)
        elif ch == "h":
            self.input_buf = ""
            self.say("Masukkan harga dengan angka lalu tekan Enter, atau tekan V untuk input suara.")
        elif ch == "k":
            self.say(msg_change(self.session))
        elif ch == "c":
            mode = self.session.toggle_mode()
            if mode == MODE_KEMBALIAN:
                exp = self.session.expected_change
                extra = (f" Kembalian seharusnya {rupiah_words(exp)}." if exp and exp > 0 else "")
                self.say("Mode cek kembalian. Pindai uang kembalian yang Anda terima." + extra)
            else:
                self.say("Mode bayar.")
        elif ch == "u":
            undone = self.session.undo()
            self.say(("Pindaian terakhir dibatalkan. " if undone else "Tidak ada pindaian untuk dibatalkan. ")
                     + f"Total uang sekarang {rupiah_words(self.session.total_paid)}.")
        elif ch == "r":
            self.session.reset()
            self.last_results = []
            self.say("Transaksi direset.")
        elif ch == "s":
            self.speaker.say(self.last_msg)
        elif ch == "p":
            self.save_view()
        return True

    def _handle_price_key(self, key: int) -> bool:
        if key == 27:
            self.input_buf = None
            self.say("Input harga dibatalkan.")
        elif key in (13, 10):
            buf, self.input_buf = self.input_buf, None
            self.set_price(int(buf) if buf else None)
        elif key in (8, 127):
            self.input_buf = self.input_buf[:-1]
        elif chr(key & 0xFF).isdigit():
            self.input_buf += chr(key & 0xFF)
        elif chr(key & 0xFF).lower() == "v":
            self.input_buf = None
            self.voice_price()
        return True

    def voice_price(self) -> None:
        try:
            from .audio import listen_nominal
            self.speaker.say("Sebutkan harga sekarang.")
            time.sleep(1.2)
            value, text = listen_nominal()
            print(f"[STT] '{text}' -> {value}")
            self.set_price(value)
        except Exception as e:
            self.say(f"Input suara gagal ({type(e).__name__}). Gunakan angka di keyboard.")

    def auto_step(self, frame: np.ndarray, dets: list[Detection]) -> None:
        """Mode otomatis: pindai sekali ketika uang diam beberapa frame."""
        real = [d for d in dets if not d.full_frame]
        if not real:
            self._stable, self._armed = 0, True
            return
        self._stable += 1
        if self._armed and self._stable >= 12:
            self._armed = False
            self.scan(frame)

    # ------------------------------------------------------------ tampilan
    def render(self, frame: np.ndarray, live_dets: list[Detection] | None = None,
               show_results: bool = True) -> np.ndarray:
        view = draw_contours(frame, live_dets) if live_dets else frame
        if self.last_results and show_results:
            view = draw_results(view, self.last_results)
        extra = f"proses {self.last_ms:.0f} ms" if self.last_ms else ""
        self._last_view = draw_panel(view, self.session, self.status, self.input_buf, extra)
        return self._last_view

    def save_view(self, name: str | None = None) -> str | None:
        if self._last_view is None:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, name or time.strftime("mv_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, self._last_view)
        print(f"[i] screenshot disimpan: {path}")
        return path