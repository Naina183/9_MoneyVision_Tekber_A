"""Unit test logika uang: terbilang, parsing harga, total, dan kembalian.

Jalankan:  python -m pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from moneyvision.money import (MODE_KEMBALIAN, TransactionSession, format_rupiah,
                               msg_change, msg_verify, parse_nominal, terbilang)


@pytest.mark.parametrize("n,words", [
    (1000, "seribu"),
    (2000, "dua ribu"),
    (11000, "sebelas ribu"),
    (15000, "lima belas ribu"),
    (75000, "tujuh puluh lima ribu"),
    (100000, "seratus ribu"),
    (125500, "seratus dua puluh lima ribu lima ratus"),
    (1250000, "satu juta dua ratus lima puluh ribu"),
    (0, "nol"),
])
def test_terbilang(n, words):
    assert terbilang(n) == words


@pytest.mark.parametrize("text,value", [
    ("25000", 25000),
    ("Rp25.000", 25000),
    ("rp 1.250.000", 1250000),
    ("25 ribu", 25000),
    ("1,5 juta", 1500000),
    ("dua puluh lima ribu", 25000),
    ("seratus dua puluh ribu lima ratus", 120500),
    ("tujuh ribu lima ratus rupiah", 7500),
    ("sebelas ribu", 11000),
    ("seribu", 1000),
    ("dua belas ribu", 12000),
    ("harganya empat puluh delapan ribu", 48000),
    ("halo", None),
])
def test_parse_nominal(text, value):
    assert parse_nominal(text) == value


@pytest.mark.parametrize("n", [1000, 7500, 48000, 99999, 125500, 2_750_000])
def test_terbilang_roundtrip(n):
    assert parse_nominal(terbilang(n)) == n


def test_format_rupiah():
    assert format_rupiah(50000) == "Rp 50.000"
    assert format_rupiah(-2000) == "-Rp 2.000"
    assert format_rupiah(None) == "-"


def test_session_change_flow():
    s = TransactionSession()
    s.add_scan([50000, 20000, 0])          # 0 = tidak dikenali -> diabaikan
    s.add_scan([10000])
    assert s.total_paid == 80000
    s.price = 63500
    assert s.expected_change == 16500
    assert "enam belas ribu lima ratus" in msg_change(s)
    s.toggle_mode()
    assert s.mode == MODE_KEMBALIAN
    s.add_scan([10000, 5000])
    assert s.change_received == 15000
    assert "kurang seribu lima ratus" in msg_verify(s)
    s.add_scan([1000, 1000])
    assert "berlebih lima ratus" in msg_verify(s)
    s.undo()
    s.add_scan([1000])
    assert s.change_received == 16000
    s.undo(); s.undo(); s.add_scan([10000, 5000, 1000, 500])
    assert msg_verify(s) == "Kembalian sesuai."


def test_insufficient_and_exact():
    s = TransactionSession(price=50000)
    s.add_scan([20000, 20000])
    assert "kurang sepuluh ribu" in msg_change(s)
    s.add_scan([10000])
    assert "Uang pas" in msg_change(s)


def test_undo_reset():
    s = TransactionSession()
    assert s.undo() is None
    s.add_scan([2000]); s.add_scan([5000])
    s.undo()
    assert s.total_paid == 2000
    s.reset()
    assert s.total_paid == 0 and s.price is None
