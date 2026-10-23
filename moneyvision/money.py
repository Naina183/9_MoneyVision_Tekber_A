# money.py
# bagian logika uang: ubah angka jadi kata (terbilang), baca input harga,
# dan menyimpan transaksi (total uang, harga, kembalian)
# file ini tidak memakai opencv sama sekali, jadi gampang dites pakai pytest

import re

# terbilang (angka -> kata) untuk dibacakan oleh tts

_SATUAN = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
           "delapan", "sembilan", "sepuluh", "sebelas"]


def _terbilang(n):
    # dipanggil berulang (rekursif) sampai angkanya kecil
    if n < 12:
        return _SATUAN[n]
    if n < 20:
        return _terbilang(n - 10) + " belas"
    if n < 100:
        return _terbilang(n // 10) + " puluh " + _terbilang(n % 10)
    if n < 200:
        return "seratus " + _terbilang(n - 100)
    if n < 1000:
        return _terbilang(n // 100) + " ratus " + _terbilang(n % 100)
    if n < 2000:
        return "seribu " + _terbilang(n - 1000)
    if n < 1_000_000:
        return _terbilang(n // 1000) + " ribu " + _terbilang(n % 1000)
    if n < 1_000_000_000:
        return _terbilang(n // 1_000_000) + " juta " + _terbilang(n % 1_000_000)
    return _terbilang(n // 1_000_000_000) + " miliar " + _terbilang(n % 1_000_000_000)


# contoh: 125500 -> "seratus dua puluh lima ribu lima ratus"
def terbilang(n):
    n = int(n)
    if n == 0:
        return "nol"
    if n < 0:
        return "minus " + terbilang(-n)
    return " ".join(_terbilang(n).split())  # rapikan spasi dobel


# contoh: 50000 -> "lima puluh ribu rupiah"
def rupiah_words(n):
    return f"{terbilang(n)} rupiah"


# contoh: 50000 -> "Rp 50.000", untuk ditampilkan di layar
def format_rupiah(n):
    if n is None:
        return "-"
    tanda = "-" if n < 0 else ""
    return f"{tanda}Rp {abs(int(n)):,}".replace(",", ".")


# baca input harga (dari keyboard atau hasil suara)
_KATA_ANGKA = {
    "nol": 0, "kosong": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
}
_KATA_SE = {"sepuluh": 10, "sebelas": 11, "seratus": 100}
_PENGALI = {"ribu": 1_000, "juta": 1_000_000, "miliar": 1_000_000_000,
            "milyar": 1_000_000_000}
_PENGALI_SE = {"seribu": 1_000, "sejuta": 1_000_000, "semiliar": 1_000_000_000}


# ubah teks harga jadi angka, bisa angka biasa atau kata
# contoh: "25000", "Rp25.000", "25 ribu", "1,5 juta", "dua puluh lima ribu"
# kalau tidak ada angka sama sekali hasilnya None
def parse_nominal(text):
    if not text:
        return None
    t = text.lower()
    t = re.sub(r"\brp\.?", " ", t)
    t = t.replace("rupiah", " ")
    t = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", t)  # hapus titik ribuan: 25.000 -> 25000
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)       # koma desimal: 1,5 -> 1.5
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", t)

    total = 0.0
    grup = 0.0     # nilai di bawah 1000 yang sedang dihitung
    satuan = 0.0   # angka terakhir yang dibaca
    ketemu = False
    for tok in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            satuan += float(tok)
            ketemu = True
        elif tok in _KATA_ANGKA:
            satuan += _KATA_ANGKA[tok]
            ketemu = True
        elif tok in _KATA_SE:
            grup += _KATA_SE[tok]
            ketemu = True
        elif tok == "belas":
            grup += satuan + 10
            satuan = 0
        elif tok == "puluh":
            grup += satuan * 10
            satuan = 0
        elif tok == "ratus":
            grup += (satuan or 1) * 100
            satuan = 0
        elif tok in _PENGALI:
            g = grup + satuan
            total += (g if g else 1) * _PENGALI[tok]
            grup = satuan = 0
            ketemu = True
        elif tok in _PENGALI_SE:
            g = grup + satuan
            total += (g * _PENGALI_SE[tok]) if g else _PENGALI_SE[tok]
            grup = satuan = 0
            ketemu = True
        # kata lain seperti "harga" atau "nya" dilewati saja
    total += grup + satuan
    if not ketemu:
        return None
    return int(round(total))


# sesi transaksi

MODE_BAYAR = "bayar"          # memindai uang yang mau dibayarkan
MODE_KEMBALIAN = "kembalian"  # memindai uang kembalian dari kasir


# satu kali pindai: daftar nominal + mode saat dipindai
class Scan:
    def __init__(self, notes, mode):
        self.notes = notes
        self.mode = mode


class TransactionSession:
    def __init__(self, price=None, mode=MODE_BAYAR):
        self.price = price
        self.mode = mode
        self.scans = []  # riwayat pindaian, dipakai juga untuk tombol undo

    def add_scan(self, notes):
        # nominal 0 artinya tidak dikenali, jadi tidak ikut dihitung
        valid = [int(n) for n in notes if n and n > 0]
        if valid:
            self.scans.append(Scan(valid, self.mode))
        return valid

    def undo(self):
        return self.scans.pop() if self.scans else None

    def reset(self):
        self.price = None
        self.mode = MODE_BAYAR
        self.scans.clear()

    def toggle_mode(self):
        self.mode = MODE_KEMBALIAN if self.mode == MODE_BAYAR else MODE_BAYAR
        return self.mode

    def _jumlah(self, mode):
        return sum(sum(s.notes) for s in self.scans if s.mode == mode)

    @property
    def total_paid(self):
        return self._jumlah(MODE_BAYAR)

    @property
    def change_received(self):
        return self._jumlah(MODE_KEMBALIAN)

    @property
    def expected_change(self):
        # kembalian seharusnya = uang dibayar - harga (minus berarti uangnya kurang)
        if self.price is None:
            return None
        return self.total_paid - self.price


# kalimat yang dibacakan ke user

# kalimat setelah pindai, contoh: "Terdeteksi dua lembar. Lima ribu rupiah, ..."
def msg_scan(notes, n_unknown, session):
    if not notes and not n_unknown:
        return "Uang tidak terdeteksi. Letakkan uang di atas permukaan polos dan coba lagi."
    parts = []
    if notes:
        parts.append(f"Terdeteksi {terbilang(len(notes))} lembar.")
        parts.append((", ".join(rupiah_words(n) for n in notes) + ".").capitalize())
    if n_unknown:
        parts.append(f"{terbilang(n_unknown).capitalize()} lembar tidak dikenali, "
                     "coba posisikan ulang.")
    if session.mode == MODE_BAYAR:
        parts.append(f"Total uang sekarang {rupiah_words(session.total_paid)}.")
    else:
        parts.append(f"Kembalian diterima sejauh ini {rupiah_words(session.change_received)}.")
        parts.append(msg_verify(session))
    return " ".join(p for p in parts if p)


# kalimat untuk tombol K: kembalian seharusnya atau uang yang masih kurang
def msg_change(session):
    if session.price is None:
        return "Harga belum dimasukkan. Tekan H untuk memasukkan harga."
    selisih = session.expected_change
    awal = (f"Harga {rupiah_words(session.price)}. "
            f"Uang dibayar {rupiah_words(session.total_paid)}. ")
    if selisih < 0:
        return awal + f"Uang kurang {rupiah_words(-selisih)}."
    if selisih == 0:
        return awal + "Uang pas, tidak ada kembalian."
    return awal + f"Kembalian seharusnya {rupiah_words(selisih)}."


# kalimat di mode cek kembalian: bandingkan kembalian yang diterima dengan yang seharusnya
def msg_verify(session):
    seharusnya = session.expected_change
    if seharusnya is None:
        return "Harga belum dimasukkan, kembalian belum dapat diperiksa."
    if seharusnya <= 0:
        return "Transaksi ini tidak memerlukan kembalian."
    diterima = session.change_received
    if diterima == seharusnya:
        return "Kembalian sesuai."
    if diterima < seharusnya:
        return f"Kembalian masih kurang {rupiah_words(seharusnya - diterima)}."
    return f"Kembalian berlebih {rupiah_words(diterima - seharusnya)}."