"""MoneyVision — Web UI (Flask).

Jalankan:
    python web_app.py                    # webcam 0, auto-scan aktif
    python web_app.py --camera 1         # webcam 1
    python web_app.py --tts none         # tanpa suara

Buka browser di http://localhost:5000
"""
from __future__ import annotations

import argparse
import os
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from moneyvision.app import NoteResult, draw_contours, draw_results
from moneyvision.audio import Speaker
from moneyvision.classifier import NoteClassifier, Prediction
from moneyvision.detector import Detection, detect_notes
from moneyvision.money import format_rupiah, rupiah_words

app = Flask(__name__)

# --- state global ---
clf: NoteClassifier | None = None
speaker: Speaker | None = None
camera: cv2.VideoCapture | None = None
lock = threading.Lock()
latest_frame: np.ndarray | None = None

# auto-scan state
auto_scan_enabled = True
_stable_count = 0          # berapa frame uang sudah diam
_armed = True              # siap scan (reset setelah uang diangkat)
_last_scan_t = 0.0         # waktu scan terakhir
_last_results: list[NoteResult] = []
_status_msg = "Arahkan uang ke kamera. Scan otomatis aktif."
_last_spoken = ""          # nominal terakhir yang diucapkan
STABLE_FRAMES = 15         # frame stabil sebelum auto-scan
COOLDOWN_SEC = 2.0         # jeda minimum antar scan


def _analyze(frame: np.ndarray) -> tuple[list[NoteResult], float]:
    """Deteksi + klasifikasi semua uang di frame."""
    t0 = time.perf_counter()
    results = []
    for det in detect_notes(frame):
        pred = clf.classify(det.note, full_frame=det.full_frame)
        if det.full_frame and not pred.known:
            continue
        results.append(NoteResult(det, pred))
    return results, (time.perf_counter() - t0) * 1000


def _announce(results: list[NoteResult]) -> str:
    """Umumkan nominal yang terdeteksi lewat suara. Return pesan teks."""
    global _last_spoken
    known = [r for r in results if r.pred.known]
    unknown_count = sum(1 for r in results if not r.pred.known)

    if not known and not unknown_count:
        msg = "Uang tidak terdeteksi."
        speaker.say(msg)
        return msg

    parts = []
    for r in known:
        parts.append(rupiah_words(r.pred.denom))

    if parts:
        voice = ", ".join(parts) + "."
        # Hanya ucapkan jika berbeda dari terakhir kali (hindari spam suara)
        if voice != _last_spoken:
            speaker.say(voice)
            _last_spoken = voice
    else:
        voice = ""

    display_parts = []
    if known:
        display_parts.append(f"Terdeteksi: {', '.join(format_rupiah(r.pred.denom) for r in known)}")
    if unknown_count:
        display_parts.append(f"{unknown_count} lembar tidak dikenali")

    return ". ".join(display_parts) + "."


def gen_frames():
    """Generator MJPEG: kirim frame webcam + overlay hasil."""
    global latest_frame, _stable_count, _armed, _last_scan_t, _last_results, _status_msg, _last_spoken

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.05)
            continue

        with lock:
            latest_frame = frame.copy()

        # deteksi kontur (tanpa klasifikasi — ringan)
        dets = detect_notes(frame, allow_full_frame=False)
        real_dets = [d for d in dets if not d.full_frame]
        show_results = (time.time() - _last_scan_t) < 4.0

        # --- auto-scan logic ---
        if auto_scan_enabled:
            if not real_dets:
                # uang tidak ada → reset, siap scan lagi
                _stable_count = 0
                _armed = True
                _last_spoken = ""
            else:
                _stable_count += 1
                now = time.time()
                if (_armed
                        and _stable_count >= STABLE_FRAMES
                        and (now - _last_scan_t) > COOLDOWN_SEC):
                    # uang diam cukup lama → scan
                    results, ms = _analyze(frame)
                    _last_results = results
                    _last_scan_t = now
                    _status_msg = _announce(results)
                    _armed = False  # tunggu uang diangkat dulu

        # --- render overlay ---
        out = frame.copy()
        if _last_results and show_results:
            out = draw_results(out, _last_results)
        elif real_dets:
            out = draw_contours(out, dets)

        _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


# --- routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/scan', methods=['POST'])
def api_scan():
    """Pindai frame terakhir secara manual."""
    global _last_results, _last_scan_t, _status_msg, _armed
    with lock:
        frame = latest_frame
    if frame is None:
        return jsonify(status="error", message="Kamera belum siap.")

    results, ms = _analyze(frame)
    _last_results = results
    _last_scan_t = time.time()
    _armed = False
    _status_msg = _announce(results)

    return jsonify(status="ok", message=_status_msg, notes=_format_notes(results))


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Pindai dari gambar yang diupload."""
    global _last_results, _last_scan_t, _status_msg

    if 'image' not in request.files:
        return jsonify(status="error", message="Tidak ada file gambar.")

    file = request.files['image']
    if file.filename == '':
        return jsonify(status="error", message="Nama file kosong.")

    # baca gambar dari upload
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(status="error", message="Gagal membaca gambar.")

    results, ms = _analyze(frame)
    _last_results = results
    _last_scan_t = time.time()
    _status_msg = _announce(results)

    # simpan gambar hasil untuk ditampilkan
    out = draw_results(frame, results)
    result_filename = f"upload_result_{uuid.uuid4().hex[:8]}.jpg"
    result_path = os.path.join("results", "uploads", result_filename)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path, out)

    return jsonify(
        status="ok",
        message=_status_msg,
        notes=_format_notes(results),
        result_image=f"/api/result_image/{result_filename}",
        processing_ms=round(ms, 1),
    )


@app.route('/api/result_image/<filename>')
def api_result_image(filename):
    """Serve gambar hasil upload yang sudah diproses."""
    path = os.path.join("results", "uploads", filename)
    if not os.path.isfile(path):
        return "Not found", 404
    with open(path, 'rb') as f:
        data = f.read()
    return Response(data, mimetype='image/jpeg')


@app.route('/api/status')
def api_status():
    return jsonify(
        status="ok",
        message=_status_msg,
        notes=_format_notes(_last_results),
        auto_scan=auto_scan_enabled,
    )


def _format_notes(results: list[NoteResult]) -> list[dict]:
    notes = []
    for r in results:
        notes.append({
            "nominal": format_rupiah(r.pred.denom) if r.pred.known else "Tidak dikenali",
            "nominal_raw": r.pred.denom if r.pred.known else 0,
            "method": r.pred.method,
            "confidence": round(r.pred.confidence, 2),
            "known": r.pred.known,
        })
    return notes


def main():
    global clf, speaker, camera, auto_scan_enabled

    ap = argparse.ArgumentParser(description="MoneyVision Web UI")
    ap.add_argument("--camera", type=int, default=0, help="indeks webcam (default 0)")
    ap.add_argument("--refs", default="references", help="folder citra referensi")
    ap.add_argument("--tts", default="auto", choices=["auto", "pyttsx3", "gtts", "none"])
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    print("[i] Memuat citra referensi (termasuk augmentasi) ...")
    clf = NoteClassifier(args.refs)
    print(f"[i] {len(clf.refs)} referensi dimuat untuk nominal: "
          + ", ".join(format_rupiah(c) for c in clf.classes))
    speaker = Speaker(args.tts)

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not camera.isOpened():
        print("[!] Gagal membuka kamera. Pastikan webcam terhubung.")
        return

    print(f"\n  MoneyVision siap di http://localhost:{args.port}")
    print(f"  Auto-scan: AKTIF (scan otomatis saat uang diam di depan kamera)\n")
    speaker.say("MoneyVision siap. Arahkan uang ke kamera.")

    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    finally:
        camera.release()
        speaker.close()


if __name__ == '__main__':
    main()
