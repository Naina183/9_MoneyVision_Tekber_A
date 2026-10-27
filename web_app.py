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

lock = threading.Lock()          # melindungi latest_frame
scan_lock = threading.Lock()     # satu scan pada satu waktu (ORB tidak thread-safe)
latest_frame: np.ndarray | None = None
frame_seq = 0                    # nomor urut frame (untuk deteksi frame baru)

# auto-scan state
auto_scan_enabled = True
_armed = True              # siap scan (reset setelah uang diangkat)
_scanning = False          # True selama scan berjalan di thread latar
_last_scan_t = 0.0         # waktu scan terakhir
_last_results: list[NoteResult] = []
_live_dets: list[Detection] = []   # kontur terbaru untuk overlay
_status_msg = "Arahkan uang ke kamera. Scan otomatis aktif."
_last_spoken = ""          # nominal terakhir yang diucapkan
STABLE_SEC = 0.6           # uang harus diam selama ini sebelum auto-scan
MOVE_PX = 25               # gerak pusat uang (px) yang masih dianggap diam
COOLDOWN_SEC = 2.0         # jeda minimum antar scan
DETECT_FPS = 8             # kontur live dihitung segini kali per detik
STREAM_FPS = 25            # frame MJPEG dikirim segini kali per detik
STREAM_WIDTH = 960         # lebar frame yang dikirim ke browser


def _analyze(frame: np.ndarray) -> tuple[list[NoteResult], float]:
    """Deteksi + klasifikasi semua uang di frame."""
    t0 = time.perf_counter()
    results = []
    with scan_lock:
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


def _camera_loop():
    """Thread 1: baca webcam terus-menerus, simpan frame terakhir saja."""
    global latest_frame, frame_seq
    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.05)
            continue
        with lock:
            latest_frame = frame
            frame_seq += 1


def _get_frame() -> tuple[int, np.ndarray | None]:
    with lock:
        return frame_seq, latest_frame


def _scan_worker(frame: np.ndarray):
    """Scan berat (klasifikasi) di thread sendiri supaya video tidak membeku."""
    global _last_results, _last_scan_t, _status_msg, _scanning
    try:
        results, _ = _analyze(frame)
        _last_results = results
        _last_scan_t = time.time()
        _status_msg = _announce(results)
    finally:
        _scanning = False


def _detect_loop():
    """Thread 2: deteksi kontur ringan + logika auto-scan (~DETECT_FPS kali/detik)."""
    global _live_dets, _armed, _scanning, _last_spoken
    stable_since = None
    last_center = None
    last_id = None
    while True:
        time.sleep(1.0 / DETECT_FPS)
        seq, frame = _get_frame()
        if frame is None or seq == last_id:
            continue
        last_id = seq

        dets = [d for d in detect_notes(frame, allow_full_frame=False) if not d.full_frame]
        _live_dets = dets

        if not auto_scan_enabled:
            continue
        if not dets:
            # uang tidak ada -> reset, siap scan lagi
            stable_since, last_center = None, None
            _armed = True
            _last_spoken = ""
            continue

        center = np.mean([d.quad.mean(axis=0) for d in dets], axis=0)
        now = time.time()
        if last_center is None or np.linalg.norm(center - last_center) > MOVE_PX:
            stable_since = now          # masih bergerak -> mulai hitung ulang
        last_center = center

        if (_armed and not _scanning
                and now - stable_since >= STABLE_SEC
                and now - _last_scan_t > COOLDOWN_SEC):
            _armed = False              # tunggu uang diangkat dulu
            _scanning = True
            threading.Thread(target=_scan_worker, args=(frame.copy(),), daemon=True).start()


def gen_frames():
    """Generator MJPEG: hanya ambil frame terbaru + gambar overlay + encode. Ringan."""
    last_id = None
    while True:
        seq, frame = _get_frame()
        if frame is None or seq == last_id:
            time.sleep(0.01)
            continue
        last_id = seq

        out = frame
        if _last_results and (time.time() - _last_scan_t) < 4.0:
            out = draw_results(frame, _last_results)
        elif _live_dets:
            out = draw_contours(frame, _live_dets)

        if out.shape[1] > STREAM_WIDTH:
            h = int(out.shape[0] * STREAM_WIDTH / out.shape[1])
            out = cv2.resize(out, (STREAM_WIDTH, h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(1.0 / STREAM_FPS)


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
    _, frame = _get_frame()
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
    global clf, speaker, camera

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

    # CAP_DSHOW jauh lebih cepat dibuka di Windows; MJPG membuat webcam USB bisa
    # 30 fps di 720p (format default YUY2 sering mentok ~10 fps = terlihat lag).
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("[!] Gagal membuka kamera. Pastikan webcam terhubung.")
        return

    print(f"\n  MoneyVision siap di http://localhost:{args.port}")
    print(f"  Auto-scan: AKTIF (scan otomatis saat uang diam di depan kamera)\n")
    speaker.say("MoneyVision siap. Arahkan uang ke kamera.")

    threading.Thread(target=_camera_loop, daemon=True).start()
    threading.Thread(target=_detect_loop, daemon=True).start()

    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    finally:
        camera.release()
        speaker.close()


if __name__ == '__main__':
    main()