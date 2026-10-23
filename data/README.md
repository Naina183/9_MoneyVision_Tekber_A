# Dataset

Dataset mentah **tidak disertakan** di repository/zip (ukuran besar dan milik pihak lain).
Unduh dari sumber aslinya:

```bash
# Dataset A — 700 citra scan uang emisi 2016 & 2022 (7 nominal x 100)
git clone --depth 1 https://github.com/ardinadnn/Rupiah-PCR-Using-Image-Processing-and-CNN.git tmp_a
mv tmp_a/datasetUang data/datasetUang

# Dataset B — 120 foto kamera uang emisi 2016 (8 nominal x 15, termasuk Rp75.000)
git clone --depth 1 https://github.com/weniariska/rupiah-currency-image-recognition-system.git tmp_b
mv tmp_b/DATASET_UANG data/DATASET_UANG

# Dataset D — 1.392 foto ponsel (±7,5 GB), dipakai tools/uji_d_foto.py
git clone --depth 1 https://github.com/mkaspulanwar/rupiah-banknotes-dataset.git data/rupiah-banknotes-dataset
```

| | Dataset A | Dataset B |
|---|---|---|
| Sumber | github.com/ardinadnn/Rupiah-PCR-Using-Image-Processing-and-CNN | github.com/weniariska/rupiah-currency-image-recognition-system |
| Akuisisi | scanner, sudah terpotong rapat | foto kamera di atas kertas HVS |
| Isi | 1.000–100.000; emisi 2016 & 2022; depan & belakang | 1.000–100.000 + 75.000; emisi 2016; depan |
| Jumlah | 700 (100 per nominal) | 120 (15 per nominal) |
| Kondisi | indeks 1–5 normal, 6–10 lusuh, 11–15 kotor, 16–20 sobek, 21–25 buram | normal |
| Peran di MoneyVision | indeks 1–2 = referensi (56 citra); sisanya = data pengembangan & uji A | uji independen B (tidak dipakai untuk referensi/tuning) |

| Dataset D | |
|---|---|
| Sumber | github.com/mkaspulanwar/rupiah-banknotes-dataset (lisensi MIT) |
| Akuisisi | kamera ponsel 4080×3060, latar putih; banyak uang dilipat, digulung, diremas, atau terpotong tepi |
| Isi | folder idr_1000 … idr_100000 (156–202 foto per nominal) + idr_mix (143 foto multi-lembar, belum berlabel) |
| Peran di MoneyVision | Uji D: 15 foto acak per nominal (seed 0) |
| Catatan | Foto memuat lokasi GPS di EXIF: jangan salin foto ke repository tanpa menghapus EXIF |

Dataset A dan B tidak mencantumkan lisensi eksplisit. Data hanya dipakai untuk keperluan
akademik non-komersial, sumber dicantumkan, dan dataset tidak didistribusikan ulang.
Folder `references/` berisi 56 citra turunan Dataset A (diperkecil ke 640x300) agar
prototipe dapat langsung dijalankan.
