# MoneyVision

Prototipe Computer Vision (OpenCV) untuk membantu pengguna **tunanetra dan low vision**
mengenali nominal uang kertas rupiah, menghitung total beberapa lembar uang, dan
memeriksa kembalian — seluruh hasil disampaikan lewat **suara Bahasa Indonesia**.

> Status: progres Week 8 (ETS). Fungsi utama sudah berjalan dan teruji; lihat
> `results/` untuk hasil evaluasi dan bagian *Keterbatasan* di bawah.

## Fitur

| Fitur | Keterangan |
|---|---|
| Deteksi banyak lembar sekaligus | Contour detection memisahkan tiap uang pada frame, lalu perspective warp meluruskannya |
| Pengenalan nominal | ORB feature matching + verifikasi homografi RANSAC terhadap 56 citra referensi; histogram warna HSV sebagai pemecah seri/cadangan |
| Penolakan aman | Bila bukti kurang, sistem menjawab "tidak dikenali" alih-alih menebak |
| Total & kembalian | Total uang dibayar, harga (input angka atau suara), kembalian seharusnya |
| Cek kembalian | Mode khusus untuk memindai uang kembalian yang diterima dan membandingkannya dengan yang seharusnya |
| Umpan balik suara | pyttsx3 (offline) atau Google TTS (`--tts gtts`); angka dibacakan dalam kata ("tujuh puluh lima ribu rupiah") |
| Input | Webcam (real-time), gambar, atau file video |

Nominal yang dikenali: Rp1.000, 2.000, 5.000, 10.000, 20.000, 50.000, 100.000 (emisi 2016 & 2022, sisi depan & belakang).

## Struktur folder

```
MoneyVision/
├── main.py                 # aplikasi (webcam / gambar / video)
├── build_references.py     # membuat basis citra referensi dari dataset
├── evaluate.py             # pengujian A (scan), B (foto), C (scene sintetis) + ablation
├── moneyvision/
│   ├── config.py           # semua parameter
│   ├── preprocessing.py    # resize, grayscale, CLAHE, white balance, perspective warp
│   ├── detector.py         # contour detection & pemisahan uang
│   ├── classifier.py       # ORB + RANSAC + histogram warna, aturan keputusan
│   ├── money.py            # terbilang, parsing harga, sesi transaksi, kalimat suara
│   ├── audio.py            # text-to-speech & speech-to-text
│   └── app.py              # alur aplikasi, kontrol keyboard, tampilan panel
├── tools/                  # generator scene sintetis, contoh gambar, visualisasi pipeline
├── tests/                  # unit test & uji alur aplikasi (pytest)
├── references/             # 56 citra referensi (7 nominal x 2 sisi x 2 emisi x 2)
├── samples/                # gambar contoh untuk mencoba tanpa webcam
├── data/README.md          # cara mengunduh dataset
└── results/                # output evaluasi (CSV, JSON, grafik, contoh output)
```

## Instalasi

Butuh Python 3.9+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Opsional:

```bash
pip install gTTS pygame                 # suara Bahasa Indonesia via Google TTS (butuh internet)
pip install SpeechRecognition PyAudio   # input harga dengan suara (butuh internet + mikrofon)
```

Catatan suara: pyttsx3 memakai suara bawaan sistem operasi. Di Windows, pasang paket
bahasa Indonesia (Settings → Time & Language → Speech) agar suara "Microsoft Andika"
tersedia. Bila tidak ada suara Indonesia, jalankan dengan `--tts gtts`.

## Menjalankan

```bash
python main.py                              # webcam 0, tekan SPASI untuk memindai
python main.py --camera 1 --auto            # webcam 1, pindai otomatis saat uang diam ±0,5 detik
python main.py --image samples/scene_demo_meja.jpg --price 60000
python main.py --video rekaman.mp4
python main.py --tts gtts                   # suara Google TTS
python main.py --image samples/scene_demo_kayu.jpg --no-window --tts none --save results/out.png
```

### Kontrol keyboard (jendela "MoneyVision" harus aktif)

| Tombol | Fungsi |
|---|---|
| `SPASI` | Pindai uang di depan kamera dan tambahkan ke total |
| `H` | Input harga: ketik angka lalu `Enter` (`Backspace` hapus, `Esc` batal, `V` input suara) |
| `K` | Umumkan kembalian seharusnya (atau kekurangan uang) |
| `C` | Ganti mode **bayar** ↔ **cek kembalian** |
| `U` | Batalkan pindaian terakhir |
| `R` | Reset transaksi |
| `S` | Ulangi pesan suara terakhir |
| `P` | Simpan screenshot ke `results/screenshots/` |
| `Q` / `Esc` | Keluar |

Contoh alur: pindai uang yang akan dibayar (SPASI) → `H` masukkan harga → `K` dengar
kembalian seharusnya → `C` lalu pindai uang kembalian dari kasir → sistem menyebut
"Kembalian sesuai" / "masih kurang …" / "berlebih …".

### Tips agar deteksi andal
- Letakkan uang di atas permukaan **polos yang kontras** (meja gelap ideal), jarak kamera ±30–50 cm.
- Uang **tidak saling menumpuk**; beri jarak beberapa cm antarlembar.
- Pencahayaan cukup dan merata; hindari pantulan lampu langsung.
- Bila uang memenuhi seluruh frame, sistem tetap mencoba mengenali seluruh frame (mode *full frame*).

### Mengatasi masalah umum

| Gejala | Solusi |
|---|---|
| Tidak ada suara / suara berbahasa Inggris | Pasang suara Bahasa Indonesia di OS, atau jalankan `--tts gtts` (butuh `pip install gTTS pygame` dan internet) |
| macOS: pyttsx3 diam | Gunakan `--tts gtts` |
| Kamera tidak terbuka | Coba `--camera 1` (atau 2); tutup aplikasi lain yang memakai kamera |
| Tombol tidak bereaksi | Klik jendela "MoneyVision" dulu agar aktif |
| Uang sering "tidak dikenali" | Pakai latar gelap polos, cahaya cukup, uang tidak terlipat; tambahkan foto uang Anda ke `references/<nominal>/` |
| Pesan `QFontDatabase` di terminal (Linux) | Tidak berpengaruh, abaikan |

## Dataset & referensi

Lihat [`data/README.md`](data/README.md). Ringkasnya: Dataset A (700 scan, GitHub ardinadnn)
untuk referensi dan data pengembangan, Dataset B (120 foto kamera, GitHub weniariska) untuk
uji independen. Folder `references/` sudah disertakan; untuk membuat ulang:

```bash
python build_references.py --dataset data/datasetUang --out references
```

Foto uang milik sendiri dapat ditambahkan sebagai referensi dengan menyalinnya ke
`references/<nominal>/` (mis. `references/50000/foto_kami_1.jpg`).

## Evaluasi & pengujian

```bash
python evaluate.py                 # butuh dataset A & B di folder data/
python tools/uji_d_foto.py         # Uji D: foto ponsel nyata (butuh Dataset D, lihat data/README.md)
python evaluate.py --limit-a 100 --scenes 30   # versi cepat
python tools/visualize_pipeline.py --image samples/scene_demo_meja.jpg
python -m pytest -q                # 49 pengujian: logika uang, sudut & warp, alur aplikasi
python tools/demo_transaksi.py     # simulasi satu transaksi lengkap + screenshot tiap langkah
```

Output di `results/`: `ringkasan.json`, `uji_A_dataset_scan.csv`, `uji_B_foto_kamera.csv`,
`uji_C_scene_sintetis.csv`, `confusion_A.png`, `confusion_B.png`,
`akurasi_per_kondisi_A.png`, `contoh_output/` (termasuk contoh gagal), `pipeline/`, `demo_transaksi/`.

Ringkasan hasil (seed 42): Uji A 99,2% (644 scan, 0 salah), Uji B 100% (105 foto, 0 salah), 15/15 uang Rp75.000 ditolak, Uji C total tepat di 81 dari 100 scene (0 salah, 10 ditolak, 14 tidak terdeteksi). Evaluasi dijalankan dengan OpenCV 5.0; unit test dan mode gambar juga lulus di OpenCV 4.8.

Uji D (foto ponsel nyata, 105 foto acak dari Dataset D, seed 0) jauh lebih berat: 38 benar (36,2%), 25 ditolak, 40 tidak terdeteksi, 1 salah sebut dan 1 terhitung ganda — keduanya pada uang yang terpotong tepi bingkai. Lihat `results/uji_D_foto_ponsel.csv`.

## Keterbatasan (prototipe Week 8)

- Uang yang terpotong tepi bingkai bisa salah dibaca atau terhitung dua kali (Uji D); uang terlipat atau tergulung biasanya tidak terdeteksi.
- Uang yang saling menumpuk atau menempel rapat belum dapat dipisahkan dengan andal.
- Latar yang warnanya mirip uang (mis. kertas putih untuk uang Rp1.000/Rp2.000 yang pucat) menurunkan akurasi segmentasi.
- Waktu klasifikasi ±0,4 detik per lembar di mesin uji (Xeon 2,8 GHz, 2 thread; bergantung CPU) karena dicocokkan ke 56 referensi; kontur tampil real-time, klasifikasi berjalan saat pindai.
- Uang logam, uang peringatan (mis. Rp75.000), dan uang asing tidak dikenali (ditolak).
- Belum ada deteksi keaslian uang.

## Kredit

- OpenCV — https://opencv.org
- Dataset A: ardinadnn, *Rupiah-PCR-Using-Image-Processing-and-CNN* (GitHub)
- Dataset B: weniariska, *rupiah-currency-image-recognition-system* (GitHub)
- Dataset D: mkaspulanwar, *rupiah-banknotes-dataset* (GitHub, lisensi MIT)
- Rublee dkk. (2011), *ORB: An efficient alternative to SIFT or SURF*, ICCV.
