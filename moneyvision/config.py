"""Konfigurasi global MoneyVision. Semua parameter yang memengaruhi deteksi dan klasifikasi.
"""

# Nominal uang kertas rupiah yang dikenali sistem (emisi 2016 dan 2022).
DENOMINATIONS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

# Ukuran kanonik citra uang setelah dipotong & diluruskan (rasio ~2,13 : 1,
# mendekati rasio fisik uang rupiah yang berkisar 2,0–2,3).
CANON_W = 640
CANON_H = 300

# ---------------------------------------------------------------- deteksi kontur
WORK_WIDTH = 960          # frame diperkecil ke lebar ini sebelum deteksi kontur
MIN_AREA_FRAC = 0.015     # luas kontur minimal (fraksi luas frame)
MAX_AREA_FRAC = 0.97      # luas kontur maksimal (hindari tepi frame)
ASPECT_MIN = 1.45         # rasio sisi panjang/pendek minimum (toleransi uang terlipat/sobek)
ASPECT_MAX = 2.90         # rasio maksimum
MIN_SOLIDITY = 0.72       # luas kontur / luas kotak minAreaRect
BG_DIST_THRESH = 28       # jarak warna Lab minimum dari latar agar dianggap objek

# ------------------------------------------------------------------- klasifikasi
ORB_FEATURES = 800        # jumlah keypoint ORB per citra (diturunkan dari 1200 demi kecepatan)
LOWE_RATIO = 0.75         # ambang ratio test Lowe
RANSAC_REPROJ = 6.0       # ambang reprojeksi RANSAC (piksel)
MIN_INLIERS = 14          # inlier minimum agar hasil ORB dianggap meyakinkan
MARGIN_RATIO = 1.5        # inlier kelas terbaik harus >= 1,5x kelas kedua
COLOR_MIN_SIM = 0.65      # kemiripan histogram warna minimum untuk fallback warna
COLOR_MARGIN = 0.10       # selisih minimum kemiripan warna kelas 1 vs kelas 2
                          # (dinaikkan dari 0,55/0,05: versi awal salah menerima uang
                          #  Rp75.000 sebagai Rp100.000 karena warnanya mirip)
COLOR_WEIGHT = 0.25       # bobot skor warna pada skor gabungan
HIST_BINS = (18, 8)       # jumlah bin histogram H dan S (HSV)

UNKNOWN = 0               # label untuk "tidak dikenali"
