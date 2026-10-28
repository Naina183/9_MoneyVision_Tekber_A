# MoneyVision

Aplikasi web berbasis Computer Vision (OpenCV) untuk mengenali nominal uang kertas rupiah lewat
kamera, dengan hasil dibacakan dalam **suara Bahasa Indonesia**. Dibuat untuk membantu pengguna
tunanetra dan low vision.

## Fitur

- Mengenali nominal Rp1.000, 2.000, 5.000, 10.000, 20.000, 50.000, dan 100.000 (emisi 2016 & 2022).
- Mendeteksi beberapa lembar uang sekaligus dalam satu frame.
- **Scan otomatis**: uang yang diam sebentar di depan kamera langsung dipindai.
- Upload foto uang dari browser.
- Hasil dibacakan lewat suara (offline dengan pyttsx3, atau Google TTS).
- Menjawab "tidak dikenali" bila tidak yakin, bukan menebak.

## Instalasi

Butuh Python 3.9+ dan webcam.

```bash
python -m pip install -r requirements.txt
```

Opsional, suara Bahasa Indonesia yang lebih natural (butuh internet):

```bash
python -m pip install gTTS pygame
```

## Menjalankan

```bash
python web_app.py
```

Lalu buka `http://localhost:5000` di browser.

| Opsi | Fungsi |
|---|---|
| `--camera 1` | Pakai webcam lain |
| `--tts gtts` | Suara Google TTS (Bahasa Indonesia, butuh internet) |
| `--tts none` | Tanpa suara |
| `--port 8000` | Ganti port |

**Cara pakai**
- Tab **Webcam**: letakkan uang di depan kamera dan diamkan sekitar 0,6 detik. Sistem memindai
  otomatis dan membacakan hasilnya. Angkat uang sebelum memindai lembar berikutnya. Tombol
  **Scan Manual** memindai saat itu juga.
- Tab **Upload Gambar**: pilih foto uang, hasilnya tampil dengan label nominal.

**Tips**
- Letakkan uang di permukaan polos yang kontras (meja gelap paling bagus), jarak kamera 30-50 cm.
- Jangan menumpuk uang, beri jarak antarlembar.
- Pencahayaan cukup dan merata, hindari pantulan lampu.

## Masalah umum

| Gejala | Solusi |
|---|---|
| Tidak ada suara | `python -m pip install pyttsx3`, atau pasang gTTS dan jalankan dengan `--tts gtts` |
| Suara berbahasa Inggris | Pasang suara Bahasa Indonesia di sistem operasi, atau pakai `--tts gtts` |
| Kamera tidak terbuka | Coba `--camera 1`; tutup aplikasi lain yang memakai kamera |
| Video tersendat | Turunkan resolusi kamera di `web_app.py` (1280×720 jadi 640×480) |
| Uang sering "tidak dikenali" | Pakai latar polos gelap, cahaya cukup, uang tidak terlipat |

## Struktur folder

```
├── web_app.py          # aplikasi web (Flask)
├── templates/          # halaman web
├── moneyvision/        # modul deteksi, klasifikasi, suara, dan konfigurasi
├── references/         # citra referensi uang
├── samples/            # gambar contoh untuk dicoba tanpa webcam
├── tests/              # pengujian otomatis (pytest)
└── tools/              # skrip bantu pengembangan
```

Foto uang sendiri bisa ditambahkan sebagai referensi dengan menyalinnya ke
`references/<nominal>/`, misalnya `references/50000/foto_1.jpg`.

## Keterbatasan

- Uang yang terlipat, tergulung, terpotong tepi bingkai, atau bertumpuk belum dikenali dengan andal.
- Latar yang warnanya mirip uang (misalnya kertas putih) menurunkan akurasi deteksi.
- Uang logam, uang asing, dan uang edisi khusus (misalnya Rp75.000) tidak dikenali.
- Belum ada deteksi keaslian uang.

## Bagaimana MoneyVision Mengenali Uang

Contour detection memisahkan tiap lembar uang, perspective warp meluruskannya, lalu ORB feature
matching dengan verifikasi RANSAC mencocokkannya ke citra referensi. Histogram warna dipakai
sebagai pemecah seri.

## Kredit

- OpenCV: https://opencv.org
- Dataset: ardinadnn (*Rupiah-PCR-Using-Image-Processing-and-CNN*), weniariska
  (*rupiah-currency-image-recognition-system*), mkaspulanwar (*rupiah-banknotes-dataset*, MIT)
- Rublee dkk. (2011), *ORB: An efficient alternative to SIFT or SURF*, ICCV.