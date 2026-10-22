"""Uji integrasi alur aplikasi tanpa kamera (simulasi tombol keyboard).

Butuh folder references/ dan samples/ (lihat README). Dilewati bila tidak ada.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import pytest

from moneyvision.app import MoneyVisionApp
from moneyvision.audio import Speaker

REFS = os.path.join(ROOT, "references")
SCENE = os.path.join(ROOT, "samples", "scene_demo_meja.jpg")      # 50.000 + 20.000 + 5.000
PHOTO = os.path.join(ROOT, "samples", "foto_20ribu_3.jpg")        # 20.000

pytestmark = pytest.mark.skipif(not (os.path.isdir(REFS) and os.path.exists(SCENE)),
                                reason="references/ atau samples/ belum tersedia")


@pytest.fixture(scope="module")
def app():
    from moneyvision.classifier import NoteClassifier
    return MoneyVisionApp(NoteClassifier(REFS), Speaker("none", verbose=False))


def keys(app, frame, seq):
    for k in seq:
        assert app.handle_key(ord(k) if isinstance(k, str) else k, frame)


def test_full_transaction(app):
    scene, photo = cv2.imread(SCENE), cv2.imread(PHOTO)
    keys(app, scene, ["r", 32])                       # reset, pindai scene
    assert sorted(r.pred.denom for r in app.last_results) == [5000, 20000, 50000]
    assert app.session.total_paid == 75000
    keys(app, scene, ["h", "6", "0", "0", "0", "0", 13])  # harga 60.000
    assert app.session.price == 60000
    assert app.session.expected_change == 15000
    keys(app, scene, ["k"])
    assert "lima belas ribu" in app.last_msg
    keys(app, photo, ["c", 32])                       # mode cek kembalian, pindai 20.000
    assert app.session.change_received == 20000
    assert "berlebih lima ribu" in app.last_msg
    keys(app, photo, ["u"])                           # batalkan
    assert app.session.change_received == 0
    keys(app, photo, ["r"])
    assert app.session.total_paid == 0 and app.session.price is None


def test_price_input_cancel_and_backspace(app):
    frame = cv2.imread(PHOTO)
    keys(app, frame, ["r", "h", "1", "2", 8, "5", 27])    # Esc membatalkan
    assert app.session.price is None and app.input_buf is None
    keys(app, frame, ["h", "1", "2", 8, "5", "0", "0", "0", 13])
    assert app.session.price == 15000


def test_quit(app):
    assert app.handle_key(ord("q"), cv2.imread(PHOTO)) is False
